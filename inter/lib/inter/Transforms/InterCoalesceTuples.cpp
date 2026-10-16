// Factor common XeMachine tuple updates before register allocation.

#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Dialect/XeMachine/IR/XeMachineRegionFlow.h"
#include "inter/Transforms/Passes.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace inter {
#define GEN_PASS_DEF_COALESCETUPLES
#include "inter/Transforms/Passes.h.inc"
} // namespace inter

using namespace mlir;
using namespace inter::xemachine;

namespace {

static bool areEquivalentValues(Value lhs, Value rhs,
                                DenseSet<std::pair<Value, Value>> &visiting) {
  if (lhs == rhs)
    return true;
  Operation *lhsDefinition = lhs.getDefiningOp();
  Operation *rhsDefinition = rhs.getDefiningOp();
  if (!lhsDefinition || !rhsDefinition || lhsDefinition->getNumResults() != 1 ||
      rhsDefinition->getNumResults() != 1 ||
      lhsDefinition->getNumRegions() != 0 ||
      rhsDefinition->getNumRegions() != 0 ||
      !isMemoryEffectFree(lhsDefinition) || !isMemoryEffectFree(rhsDefinition))
    return false;
  std::pair<Value, Value> pair{lhs, rhs};
  if (!visiting.insert(pair).second)
    return true;
  bool equivalent = OperationEquivalence::isEquivalentTo(
      lhsDefinition, rhsDefinition,
      [&](Value lhsOperand, Value rhsOperand) {
        return success(areEquivalentValues(lhsOperand, rhsOperand, visiting));
      },
      nullptr,
      OperationEquivalence::IgnoreLocations |
          OperationEquivalence::IgnoreDiscardableAttrs);
  visiting.erase(pair);
  return equivalent;
}

static bool areEquivalentValues(Value lhs, Value rhs) {
  DenseSet<std::pair<Value, Value>> visiting;
  return areEquivalentValues(lhs, rhs, visiting);
}

static SmallVector<unsigned> getCommonUpdateIndices(UpdateTupleOp lhs,
                                                    UpdateTupleOp rhs) {
  SmallVector<unsigned> common;
  for (auto [lhsIndex, lhsOffset] : llvm::enumerate(lhs.getOffsets())) {
    int64_t offset = cast<IntegerAttr>(lhsOffset).getInt();
    for (auto [rhsIndex, rhsOffset] : llvm::enumerate(rhs.getOffsets())) {
      if (cast<IntegerAttr>(rhsOffset).getInt() != offset)
        continue;
      Value lhsValue = lhs.getUpdates()[lhsIndex];
      Value rhsValue = rhs.getUpdates()[rhsIndex];
      if (lhsValue.getType() == rhsValue.getType() &&
          areEquivalentValues(lhsValue, rhsValue))
        common.push_back(lhsIndex);
      break;
    }
  }
  return common;
}

static bool hasCommonUpdates(UpdateTupleOp reference, UpdateTupleOp candidate,
                             ArrayRef<unsigned> commonIndices) {
  for (unsigned referenceIndex : commonIndices) {
    int64_t offset =
        cast<IntegerAttr>(reference.getOffsets()[referenceIndex]).getInt();
    bool found = false;
    for (auto [candidateIndex, candidateOffset] :
         llvm::enumerate(candidate.getOffsets())) {
      if (cast<IntegerAttr>(candidateOffset).getInt() != offset)
        continue;
      Value referenceValue = reference.getUpdates()[referenceIndex];
      Value candidateValue = candidate.getUpdates()[candidateIndex];
      found = referenceValue.getType() == candidateValue.getType() &&
              areEquivalentValues(referenceValue, candidateValue);
      break;
    }
    if (!found)
      return false;
  }
  return true;
}

static void eraseDeadProducerTree(Value value) {
  Operation *operation = value.getDefiningOp();
  if (!operation || !isOpTriviallyDead(operation))
    return;
  SmallVector<Value> operands(operation->getOperands());
  operation->erase();
  for (Value operand : operands)
    eraseDeadProducerTree(operand);
}

static bool crossesRepetitiveRegion(UpdateTupleOp from, UpdateTupleOp to,
                                    const RegionFlow &regionFlow) {
  for (Operation *operation = from->getNextNode(); operation && operation != to;
       operation = operation->getNextNode()) {
    WalkResult found = operation->walk([&](Operation *nested) {
      const RegionFlow::Branch *branch = regionFlow.lookup(nested);
      if (branch && llvm::any_of(branch->regions, [&](Region *region) {
            return regionFlow.isRepetitive(region);
          }))
        return WalkResult::interrupt();
      return WalkResult::advance();
    });
    if (found.wasInterrupted())
      return true;
  }
  return false;
}

static void collectSendDescriptors(Value value, DenseSet<Value> &visited,
                                   DenseSet<int64_t> &descriptors) {
  if (!visited.insert(value).second)
    return;
  for (Operation *user : value.getUsers()) {
    if (SendOp send = dyn_cast<SendOp>(user)) {
      if (send.getAddrPayload() == value)
        descriptors.insert(send.getDesc());
      continue;
    }
    if (UpdateTupleOp update = dyn_cast<UpdateTupleOp>(user))
      collectSendDescriptors(update.getResult(), visited, descriptors);
  }
}

static bool haveSameSendDescriptors(UpdateTupleOp lhs, UpdateTupleOp rhs) {
  DenseSet<Value> lhsVisited;
  DenseSet<Value> rhsVisited;
  DenseSet<int64_t> lhsDescriptors;
  DenseSet<int64_t> rhsDescriptors;
  collectSendDescriptors(lhs.getResult(), lhsVisited, lhsDescriptors);
  collectSendDescriptors(rhs.getResult(), rhsVisited, rhsDescriptors);
  return lhsDescriptors.size() == rhsDescriptors.size() &&
         llvm::all_of(lhsDescriptors, [&](int64_t descriptor) {
           return rhsDescriptors.contains(descriptor);
         });
}

static bool reachesDestinationlessRead(Value value, DenseSet<Value> &visited) {
  if (!visited.insert(value).second)
    return false;
  for (Operation *user : value.getUsers()) {
    if (SendOp send = dyn_cast<SendOp>(user)) {
      if (send.getAddrPayload() == value && !send.getDataPayload() &&
          cast<RegType>(send.getDst().getType()).getWidthDwords() == 0)
        return true;
      continue;
    }
    if (UpdateTupleOp update = dyn_cast<UpdateTupleOp>(user))
      if (reachesDestinationlessRead(update.getResult(), visited))
        return true;
  }
  return false;
}

static bool shouldPreferSameSendDescriptors(UpdateTupleOp lhs,
                                            UpdateTupleOp rhs) {
  DenseSet<Value> lhsVisited;
  DenseSet<Value> rhsVisited;
  bool hasDestinationlessRead =
      reachesDestinationlessRead(lhs.getResult(), lhsVisited) ||
      reachesDestinationlessRead(rhs.getResult(), rhsVisited);
  return hasDestinationlessRead && haveSameSendDescriptors(lhs, rhs);
}

static void factorBlock(Block &block, const RegionFlow &regionFlow) {
  SmallVector<UpdateTupleOp> updates;
  for (Operation &operation : block)
    if (UpdateTupleOp update = dyn_cast<UpdateTupleOp>(operation))
      updates.push_back(update);

  DenseSet<Operation *> consumed;
  for (UpdateTupleOp reference : updates) {
    if (consumed.contains(reference))
      continue;
    UpdateTupleOp bestMatch;
    SmallVector<unsigned> commonIndices;
    bool bestHasPreferredDescriptors = false;
    for (UpdateTupleOp candidate : updates) {
      if (candidate == reference || consumed.contains(candidate) ||
          !reference->isBeforeInBlock(candidate) ||
          crossesRepetitiveRegion(reference, candidate, regionFlow) ||
          candidate.getResult().getType() != reference.getResult().getType() ||
          !areEquivalentValues(reference.getBase(), candidate.getBase()))
        continue;
      SmallVector<unsigned> candidateCommon =
          getCommonUpdateIndices(reference, candidate);
      if (candidateCommon.size() < 2)
        continue;
      bool hasPreferredDescriptors =
          shouldPreferSameSendDescriptors(reference, candidate);
      // TODO: Represent payload ownership directly instead of using equal send
      // descriptors as the proxy for compatible physical materialization.
      if (!bestMatch ||
          (hasPreferredDescriptors && !bestHasPreferredDescriptors) ||
          (hasPreferredDescriptors == bestHasPreferredDescriptors &&
           candidateCommon.size() > commonIndices.size())) {
        bestMatch = candidate;
        commonIndices = std::move(candidateCommon);
        bestHasPreferredDescriptors = hasPreferredDescriptors;
      }
    }
    if (!bestMatch)
      continue;

    SmallVector<UpdateTupleOp> group{reference, bestMatch};
    for (UpdateTupleOp candidate : updates)
      if (candidate != reference && candidate != bestMatch &&
          !consumed.contains(candidate) &&
          reference->isBeforeInBlock(candidate) &&
          !crossesRepetitiveRegion(reference, candidate, regionFlow) &&
          candidate.getResult().getType() == reference.getResult().getType() &&
          shouldPreferSameSendDescriptors(reference, candidate) ==
              bestHasPreferredDescriptors &&
          areEquivalentValues(reference.getBase(), candidate.getBase()) &&
          hasCommonUpdates(reference, candidate, commonIndices))
        group.push_back(candidate);

    OpBuilder builder(reference);
    SmallVector<Value> commonValues;
    SmallVector<Attribute> commonOffsets;
    DenseSet<int64_t> factoredOffsets;
    for (unsigned index : commonIndices) {
      commonValues.push_back(reference.getUpdates()[index]);
      Attribute offset = reference.getOffsets()[index];
      commonOffsets.push_back(offset);
      factoredOffsets.insert(cast<IntegerAttr>(offset).getInt());
    }
    Value tupleTemplate =
        UpdateTupleOp::create(builder, reference.getLoc(),
                              reference.getResult().getType(),
                              reference.getBase(), commonValues,
                              builder.getArrayAttr(commonOffsets))
            .getResult();

    for (UpdateTupleOp update : group) {
      SmallVector<Value> remainingValues;
      SmallVector<Attribute> remainingOffsets;
      SmallVector<Value> removedValues;
      for (auto [index, offset] : llvm::enumerate(update.getOffsets())) {
        if (factoredOffsets.contains(cast<IntegerAttr>(offset).getInt())) {
          removedValues.push_back(update.getUpdates()[index]);
          continue;
        }
        remainingValues.push_back(update.getUpdates()[index]);
        remainingOffsets.push_back(offset);
      }
      Value replacement = tupleTemplate;
      if (!remainingValues.empty()) {
        OpBuilder updateBuilder(update);
        replacement =
            UpdateTupleOp::create(updateBuilder, update.getLoc(),
                                  update.getResult().getType(), tupleTemplate,
                                  remainingValues,
                                  updateBuilder.getArrayAttr(remainingOffsets))
                .getResult();
      }
      update.getResult().replaceAllUsesWith(replacement);
      consumed.insert(update);
      update.erase();
      for (Value value : removedValues)
        if (!llvm::is_contained(commonValues, value))
          eraseDeadProducerTree(value);
    }
  }
}

class CoalesceTuplesPass
    : public inter::impl::CoalesceTuplesBase<CoalesceTuplesPass> {
public:
  void runOnOperation() override {
    RegionFlow regionFlow(getOperation());
    getOperation().walk([&](Block *block) { factorBlock(*block, regionFlow); });
  }
};

} // namespace
