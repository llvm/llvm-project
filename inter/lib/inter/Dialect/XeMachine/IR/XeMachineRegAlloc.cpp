// Allocate XeMachine GRFs with transactional retries and ordered relief.

#include "inter/Dialect/XeMachine/IR/XeMachine.h"
#include "inter/Dialect/XeMachine/IR/XeMachineAliasAnalysis.h"
#include "inter/Dialect/XeMachine/IR/XeMachineRegAllocPreparation.h"
#include "inter/Dialect/XeMachine/IR/XeMachineRegionFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"
#include "mlir/Dialect/Transform/Interfaces/TransformInterfaces.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/SymbolTable.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <functional>
#include <limits>
#include <optional>

#define GET_OP_CLASSES
#include "inter/Dialect/XeMachine/IR/XeMachineTransformOps.cpp.inc"

using namespace mlir;
using namespace inter::xemachine;

namespace {

constexpr StringLiteral kStateAttr = "xemachine.regalloc_state";
constexpr StringLiteral kIterationAttr = "xemachine.regalloc_iterations";
constexpr StringLiteral kRematerializedAttr = "xemachine.rematerialized";
constexpr StringLiteral kSpilledAttr = "xemachine.spilled";
constexpr StringLiteral kScratchSetupAttr = "xemachine.scratch_setup";
constexpr StringLiteral kLoopIterationAttr =
    "xemachine.regalloc_loop_iteration";
constexpr StringLiteral kStageAttr = "stage";
constexpr StringLiteral kArfBuildStage = "arf-live-ranges";
constexpr StringLiteral kBuildStage = "alias-state";
constexpr StringLiteral kSuccessStage = "linear-scan-success";
constexpr StringLiteral kFailureStage = "linear-scan-failure";
constexpr unsigned kMaxSwsbDistance = 7;

struct AllocationComponent {
  SmallVector<Value> values;
  int64_t minOffset = 0;
  int64_t maxOffset = 0;
  int64_t start = std::numeric_limits<int64_t>::max();
  int64_t end = 0;
  std::optional<int64_t> fixedBase;
  int64_t assignment = -1;
  bool allowFixedOverlap = false;

  int64_t widthGRFs() const {
    return llvm::divideCeil(maxOffset - minOffset, int64_t{16});
  }
};

struct AllocationState {
  RegisterAliasAnalysis aliases;
  SmallVector<AllocationComponent> components;
  DenseMap<Operation *, int64_t> positions;
};

static bool isRegister(Value value) { return isa<RegType>(value.getType()); }

static std::optional<uint64_t> getElementBytes(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return llvm::divideCeil(integer.getWidth(), 8u);
  if (type.isF32())
    return 4;
  return std::nullopt;
}

static uint64_t getDestinationStorageDwords(Operation *operation,
                                            RegType destinationType) {
  uint64_t storageDwords = destinationType.getWidthDwords();
  for (OpOperand &use : operation->getResult(0).getUses()) {
    UpdateTupleOp update = dyn_cast<UpdateTupleOp>(use.getOwner());
    if (!update || use.getOperandNumber() == 0)
      continue;
    RegType baseType = cast<RegType>(update.getBase().getType());
    unsigned updateIndex = use.getOperandNumber() - 1;
    int64_t offset =
        cast<IntegerAttr>(update.getOffsets()[updateIndex]).getInt();
    assert(offset >= 0 &&
           static_cast<uint64_t>(offset) <= baseType.getWidthDwords() &&
           "verified tuple update offset must fit its base storage");
    storageDwords =
        std::max<uint64_t>(storageDwords, baseType.getWidthDwords() - offset);
  }
  return storageDwords;
}

static int64_t getGRFSpan(int64_t firstElement, int64_t lastElement,
                          uint64_t elementBytes) {
  constexpr int64_t bytesPerGRF = 64;
  int64_t firstByte = firstElement * elementBytes;
  int64_t lastByte = (lastElement + 1) * elementBytes - 1;
  return lastByte / bytesPerGRF - firstByte / bytesPerGRF + 1;
}

static FailureOr<int64_t> getMessageGRFLength(Operation *operation, Type type,
                                              const Twine &name) {
  int64_t width = cast<RegType>(type).getWidthDwords();
  if (width % 16 != 0)
    return operation->emitError() << name << " must occupy whole GRFs";
  return width / 16;
}

static LogicalResult validateSendFootprint(SendOp send) {
  FailureOr<int64_t> source0 =
      getMessageGRFLength(send, send.getAddrPayload().getType(), "source 0");
  FailureOr<int64_t> destination =
      getMessageGRFLength(send, send.getDst().getType(), "destination");
  if (failed(source0) || failed(destination))
    return failure();

  uint32_t descriptor = static_cast<uint32_t>(send.getDesc());
  int64_t encodedSource0 = (descriptor >> 25) & 0xf;
  int64_t encodedDestination = (descriptor >> 20) & 0x1f;
  if (*source0 != encodedSource0)
    return send.emitOpError("source 0 width does not match the descriptor");
  bool block2DArrayResponse = (descriptor & 0x3f) == 0x3 &&
                              encodedDestination == 31 && *destination == 32;
  if (*destination != encodedDestination && !block2DArrayResponse)
    return send.emitOpError("destination width does not match the descriptor");

  int64_t source1 = 0;
  if (Value data = send.getDataPayload()) {
    FailureOr<int64_t> length =
        getMessageGRFLength(send, data.getType(), "source 1");
    if (failed(length))
      return failure();
    source1 = *length;
  }
  if (source1 > 31)
    return send.emitOpError("source 1 exceeds the 31-GRF encoding limit");
  if (*source0 + source1 >= 32)
    return send.emitOpError("combined source payload exceeds 31 GRFs");
  return success();
}

static LogicalResult validateMessageFootprint(Operation *operation) {
  InstructionIssueOpInterface issue =
      dyn_cast<InstructionIssueOpInterface>(operation);
  if (!issue || issue.getInstructionKind() != MachineInstructionKind::send)
    return success();

  FailureOr<int64_t> source0 = getMessageGRFLength(
      operation, operation->getOperand(0).getType(), "source 0");
  if (failed(source0))
    return failure();
  if (*source0 > 15)
    return operation->emitError("source 0 exceeds the 15-GRF encoding limit");

  int64_t destination = 0;
  if (operation->getNumResults() != 0 &&
      isa<RegType>(operation->getResult(0).getType())) {
    FailureOr<int64_t> length = getMessageGRFLength(
        operation, operation->getResult(0).getType(), "destination");
    if (failed(length))
      return failure();
    destination = *length;
  }
  if (destination > 31)
    return operation->emitError(
        "destination exceeds the 31-GRF encoding limit");

  Value data;
  if (StoreA64Op store = dyn_cast<StoreA64Op>(operation))
    data = store.getDataPayload();
  else if (StoreSLMOp store = dyn_cast<StoreSLMOp>(operation))
    data = store.getDataPayload();
  else if (AtomicIAddA64Op atomic = dyn_cast<AtomicIAddA64Op>(operation))
    data = atomic.getDataPayload();
  int64_t source1 = 0;
  if (data) {
    FailureOr<int64_t> length =
        getMessageGRFLength(operation, data.getType(), "source 1");
    if (failed(length))
      return failure();
    source1 = *length;
  }
  if (source1 > 31)
    return operation->emitError("source 1 exceeds the 31-GRF encoding limit");
  if (*source0 + source1 >= 32)
    return operation->emitError("combined source payload exceeds 31 GRFs");
  return success();
}

static LogicalResult validateAluFootprint(Operation *operation) {
  if (SendOp send = dyn_cast<SendOp>(operation))
    return validateSendFootprint(send);
  if (failed(validateMessageFootprint(operation)))
    return failure();
  ALUOpInterface alu = dyn_cast<ALUOpInterface>(operation);
  if (!alu)
    return success();
  Type elementType = alu.getInstructionElementType();
  int64_t executionSize = alu.getExecutionSize();
  if (executionSize <= 0)
    return operation->emitError("execution size must be positive");

  if (operation->getNumResults() != 0) {
    auto destinationType = dyn_cast<RegType>(operation->getResult(0).getType());
    if (destinationType && destinationType.getWidthDwords() != 0) {
      std::optional<uint64_t> bytes = getElementBytes(elementType);
      if (!bytes)
        return operation->emitError("unsupported destination element type");
      int64_t sub = alu.getDestinationSubregister();
      DstRegionAttr region = alu.getDestinationRegion();
      int64_t stride = region ? region.getHstride() : 1;
      int64_t last = sub + (executionSize - 1) * stride;
      if (sub < 0 || stride < 0 ||
          static_cast<uint64_t>(last + 1) * *bytes >
              getDestinationStorageDwords(operation, destinationType) * 4)
        return operation->emitError(
            "destination region exceeds declared register storage");
      if (getGRFSpan(sub, last, *bytes) > 2)
        return operation->emitError(
            "destination region spans more than two GRFs");
    }
  }

  for (auto [index, operand] : llvm::enumerate(operation->getOperands())) {
    auto registerType = dyn_cast<RegType>(operand.getType());
    if (!registerType)
      continue;
    Type sourceType = elementType;
    if (std::optional<Type> explicitType =
            alu.getExplicitSourceElementType(index))
      sourceType = *explicitType;
    std::optional<uint64_t> bytes = getElementBytes(sourceType);
    if (!bytes)
      return operation->emitError("unsupported source element type");
    int64_t sub = alu.getSourceSubregister(index);
    RegionAttr region = alu.getSourceRegion(index);
    int64_t vertical = region ? region.getVstride() : 1;
    int64_t width = region ? region.getWidth() : 1;
    int64_t horizontal = region ? region.getHstride() : 0;
    if (sub < 0 || vertical < 0 || width <= 0 || horizontal < 0)
      return operation->emitError("invalid source register region");
    int64_t first = sub;
    int64_t last = sub;
    for (int64_t lane : llvm::seq<int64_t>(0, executionSize)) {
      int64_t element =
          sub + lane / width * vertical + lane % width * horizontal;
      first = std::min(first, element);
      last = std::max(last, element);
    }
    if (static_cast<uint64_t>(last + 1) * *bytes >
        registerType.getWidthDwords() * 4)
      return operation->emitError(
          "source region exceeds declared register storage");
    if (getGRFSpan(first, last, *bytes) > 2)
      return operation->emitError()
             << "source " << index << " region spans more than two GRFs";
    for (int64_t row = 0; row < executionSize; row += width) {
      int64_t rowLanes = std::min(width, executionSize - row);
      int64_t rowFirst = sub + row / width * vertical;
      int64_t rowLast = rowFirst + (rowLanes - 1) * horizontal;
      if (horizontal != 0 && getGRFSpan(rowFirst, rowLast, *bytes) > 1)
        return operation->emitError()
               << "source " << index << " row crosses a GRF boundary";
    }
  }
  return success();
}

static int64_t getFirstMachineUsePosition(Value value,
                                          const AllocationState &state,
                                          SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(value).second)
    return state.positions.size();
  int64_t first = state.positions.size();
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (!owner->hasTrait<OpTrait::xemachine::NoMachineInst>()) {
      first = std::min(first, state.positions.lookup(owner));
      continue;
    }
    bool forwarded = false;
    for (Value result : owner->getResults()) {
      if (!isRegister(result))
        continue;
      first =
          std::min(first, getFirstMachineUsePosition(result, state, visited));
      forwarded = true;
    }
    if (!forwarded)
      first = std::min(first, state.positions.lookup(owner));
  }
  visited.erase(value);
  return first;
}

static int64_t getDefinitionPosition(Value value,
                                     const AllocationState &state) {
  if (Operation *definition = value.getDefiningOp()) {
    if (!definition->hasTrait<OpTrait::xemachine::NoMachineInst>())
      return state.positions.lookup(definition);
    SmallPtrSet<Value, 16> visited;
    int64_t first = getFirstMachineUsePosition(value, state, visited);
    return first == static_cast<int64_t>(state.positions.size())
               ? state.positions.lookup(definition)
               : first;
  }
  Block *block = cast<BlockArgument>(value).getOwner();
  Operation *parent = block->getParentOp();
  return parent ? state.positions.lookup(parent) : 0;
}

static LogicalResult finalizeComponents(func::FuncOp function,
                                        AllocationState &state) {
  ArrayRef<RegisterAliasAnalysis::Component> aliasComponents =
      state.aliases.getComponents();
  state.components.resize(aliasComponents.size());
  for (auto [index, aliasComponent] : llvm::enumerate(aliasComponents)) {
    AllocationComponent &component = state.components[index];
    component.values.assign(aliasComponent.members.begin(),
                            aliasComponent.members.end());
    component.minOffset = 0;
    component.maxOffset = aliasComponent.widthDwords;
    if (aliasComponent.fixedOriginDwords) {
      if (*aliasComponent.fixedOriginDwords % 16 != 0)
        return function.emitError(
            "fixed register-storage alias is not GRF-aligned");
      component.fixedBase = *aliasComponent.fixedOriginDwords / 16;
    }
    for (Value value : component.values) {
      const RegisterAliasAnalysis::ValueInfo *valueInfo =
          state.aliases.lookup(value);
      assert(valueInfo && "register value is missing alias information");
      Operation *definition = value.getDefiningOp();
      ALUOpInterface alu = dyn_cast_or_null<ALUOpInterface>(definition);
      std::optional<uint64_t> elementBytes =
          alu ? getElementBytes(alu.getInstructionElementType()) : std::nullopt;
      bool placedAtAliasOffset =
          elementBytes && alu.getDestinationSubregister() *
                                  static_cast<int64_t>(*elementBytes) ==
                              (valueInfo->offsetDwords % 16) * 4;
      if (valueInfo->offsetDwords % 16 != 0 && !placedAtAliasOffset)
        return function.emitError()
               << "register-storage alias at dword offset "
               << valueInfo->offsetDwords
               << " is not GRF-aligned after selection; value: " << value;
      if (definition)
        component.allowFixedOverlap |=
            definition->hasAttr(kAllowFixedOverlapAttrName);
      int64_t definitionPosition = getDefinitionPosition(value, state);
      component.start = std::min(component.start, definitionPosition);
      component.end = std::max(component.end, definitionPosition);
      for (OpOperand &use : value.getUses())
        component.end =
            std::max(component.end, state.positions.lookup(use.getOwner()));
    }
  }

  DenseMap<Value, int64_t> tokenEnds;
  SmallPtrSet<Value, 8> visiting;
  std::function<int64_t(Value)> getTokenEnd = [&](Value token) -> int64_t {
    auto cached = tokenEnds.find(token);
    if (cached != tokenEnds.end())
      return cached->second;
    if (!visiting.insert(token).second)
      return state.positions.size();
    int64_t end = 0;
    for (OpOperand &use : token.getUses()) {
      Operation *owner = use.getOwner();
      end = std::max(end, state.positions.lookup(owner));
      if (!owner->hasTrait<OpTrait::xemachine::NoMachineInst>())
        continue;
      bool forwarded = false;
      for (Value result : owner->getResults()) {
        if (!isa<MemTokenType>(result.getType()))
          continue;
        end = std::max(end, getTokenEnd(result));
        forwarded = true;
      }
      if (!forwarded && isa<RegionBranchTerminatorOpInterface>(owner))
        end = std::max<int64_t>(end, state.positions.size());
    }
    visiting.erase(token);
    tokenEnds[token] = end;
    return end;
  };

  function.walk([&](Operation *operation) {
    if (!isa<AsyncScoreboardOpInterface>(operation))
      return;
    int64_t completion = state.positions.lookup(operation);
    for (Value result : operation->getResults()) {
      if (isa<MemTokenType>(result.getType()))
        completion = std::max(completion, getTokenEnd(result));
    }
    for (Value operand : operation->getOperands()) {
      if (!isRegister(operand))
        continue;
      const RegisterAliasAnalysis::ValueInfo *valueInfo =
          state.aliases.lookup(operand);
      assert(valueInfo && "register operand is missing alias information");
      AllocationComponent &component = state.components[valueInfo->component];
      component.end = std::max(component.end, completion);
    }
  });

  RegionFlow regionFlow(function);
  for (const RegionFlow::Branch &branch : regionFlow.getBranches()) {
    for (Region *region : branch.regions) {
      if (!regionFlow.isRepetitive(region))
        continue;
      int64_t loopEnd = state.positions.lookup(branch.operation);
      region->walk([&](Operation *operation) {
        loopEnd = std::max(loopEnd, state.positions.lookup(operation));
      });
      region->walk([&](Operation *operation) {
        for (Value operand : operation->getOperands()) {
          if (!isRegister(operand))
            continue;
          Operation *definition = operand.getDefiningOp();
          if (definition && region->isAncestor(definition->getParentRegion()))
            continue;
          if (BlockArgument argument = dyn_cast<BlockArgument>(operand)) {
            if (region->isAncestor(argument.getOwner()->getParent()))
              continue;
          }
          const RegisterAliasAnalysis::ValueInfo *valueInfo =
              state.aliases.lookup(operand);
          assert(valueInfo && "register operand is missing alias information");
          AllocationComponent &component =
              state.components[valueInfo->component];
          component.end = std::max(component.end, loopEnd);
        }
      });
    }
  }
  return success();
}

static LogicalResult buildState(func::FuncOp function, AllocationState &state) {
  if (function
          .walk([&](Operation *operation) {
            return failed(validateAluFootprint(operation))
                       ? WalkResult::interrupt()
                       : WalkResult::advance();
          })
          .wasInterrupted())
    return failure();
  int64_t nextPosition = 0;
  function.walk([&](Operation *operation) {
    state.positions[operation] = nextPosition++;
  });
  FailureOr<RegisterAliasAnalysis> aliases =
      RegisterAliasAnalysis::create(function);
  if (failed(aliases))
    return failure();
  state.aliases = std::move(*aliases);
  return finalizeComponents(function, state);
}

static bool intervalsOverlap(const AllocationComponent &lhs,
                             const AllocationComponent &rhs) {
  return lhs.start <= rhs.end && rhs.start <= lhs.end;
}

static bool registersOverlap(int64_t lhsBase, int64_t lhsWidth, int64_t rhsBase,
                             int64_t rhsWidth) {
  return lhsBase < rhsBase + rhsWidth && rhsBase < lhsBase + lhsWidth;
}

struct AllocationFailure {
  unsigned component;
  int64_t position;
};

static DictionaryAttr packState(Builder &builder, unsigned iteration,
                                StringRef stage, const AllocationState &state,
                                std::optional<AllocationFailure> failure = {}) {
  SmallVector<int64_t> intervals;
  for (const AllocationComponent &component : state.components) {
    intervals.push_back(component.start);
    intervals.push_back(component.end);
    intervals.push_back(component.minOffset);
    intervals.push_back(component.maxOffset);
    intervals.push_back(component.fixedBase.value_or(-1));
    intervals.push_back(component.assignment);
  }
  SmallVector<int64_t> aliases;
  for (auto [index, value] : llvm::enumerate(state.aliases.getValues())) {
    const RegisterAliasAnalysis::ValueInfo *valueInfo =
        state.aliases.lookup(value);
    assert(valueInfo && "register value is missing alias information");
    aliases.push_back(index);
    aliases.push_back(valueInfo->component);
    aliases.push_back(valueInfo->offsetDwords);
  }
  SmallVector<NamedAttribute> attributes = {
      builder.getNamedAttr("iteration", builder.getI32IntegerAttr(iteration)),
      builder.getNamedAttr(kStageAttr, builder.getStringAttr(stage)),
      builder.getNamedAttr(
          "component_intervals",
          DenseI64ArrayAttr::get(builder.getContext(), intervals)),
      builder.getNamedAttr(
          "value_aliases",
          DenseI64ArrayAttr::get(builder.getContext(), aliases))};
  if (failure) {
    attributes.push_back(builder.getNamedAttr(
        "failed_component", builder.getI32IntegerAttr(failure->component)));
    attributes.push_back(builder.getNamedAttr(
        "failure_position", builder.getI64IntegerAttr(failure->position)));
  }
  return builder.getDictionaryAttr(attributes);
}

static std::optional<AllocationFailure> tryAllocate(AllocationState &state,
                                                    unsigned grfLimit,
                                                    unsigned reservedGrfCount) {
  SmallVector<unsigned> order;
  llvm::append_range(order, llvm::seq<unsigned>(0, state.components.size()));
  llvm::sort(order, [&](unsigned lhs, unsigned rhs) {
    const AllocationComponent &left = state.components[lhs];
    const AllocationComponent &right = state.components[rhs];
    if (left.start != right.start)
      return left.start < right.start;
    return left.fixedBase.has_value() > right.fixedBase.has_value();
  });

  SmallVector<unsigned> active;
  for (unsigned index : order) {
    AllocationComponent &component = state.components[index];
    llvm::erase_if(active, [&](unsigned other) {
      return state.components[other].end < component.start;
    });

    auto isAvailable = [&](int64_t base) {
      if (base < 0 || base + component.widthGRFs() > grfLimit)
        return false;
      for (unsigned otherIndex : active) {
        const AllocationComponent &other = state.components[otherIndex];
        if (component.fixedBase && other.fixedBase &&
            (component.allowFixedOverlap || other.allowFixedOverlap))
          continue;
        if (registersOverlap(base, component.widthGRFs(), other.assignment,
                             other.widthGRFs()))
          return false;
      }
      for (const AllocationComponent &fixed : state.components) {
        if (!fixed.fixedBase || &fixed == &component ||
            !intervalsOverlap(component, fixed))
          continue;
        if (component.fixedBase &&
            (component.allowFixedOverlap || fixed.allowFixedOverlap))
          continue;
        if (registersOverlap(base, component.widthGRFs(), *fixed.fixedBase,
                             fixed.widthGRFs()))
          return false;
      }
      return true;
    };

    if (component.fixedBase) {
      if (!isAvailable(*component.fixedBase))
        return AllocationFailure{index, component.start};
      component.assignment = *component.fixedBase;
    } else {
      for (int64_t base = reservedGrfCount;
           base + component.widthGRFs() <= grfLimit; ++base) {
        if (!isAvailable(base))
          continue;
        component.assignment = base;
        break;
      }
      if (component.assignment < 0)
        return AllocationFailure{index, component.start};
    }
    active.push_back(index);
  }
  return std::nullopt;
}

static bool isRematerializable(Operation *operation) {
  if (!operation || operation->hasAttr(kRematerializedAttr) ||
      operation->getNumResults() != 1)
    return false;
  if (llvm::any_of(operation->getOperandTypes(),
                   [](Type type) { return isa<ARFType>(type); }))
    return false;
  return operation->hasTrait<OpTrait::xemachine::Rematerializable>();
}

static bool hasUseAtOrAfter(Value value, const AllocationState &state,
                            int64_t position) {
  return llvm::any_of(value.getUses(), [&](OpOperand &use) {
    return state.positions.lookup(use.getOwner()) >= position;
  });
}

static bool rematerialize(AllocationState &state,
                          const AllocationFailure &failure) {
  AllocationComponent *candidate = nullptr;
  for (AllocationComponent &component : state.components) {
    if (component.fixedBase || component.values.size() != 1 ||
        component.start >= failure.position || component.end < failure.position)
      continue;
    Value value = component.values.front();
    if (!isRematerializable(value.getDefiningOp()))
      continue;
    if (!hasUseAtOrAfter(value, state, failure.position))
      continue;
    if (!candidate || candidate->end < component.end)
      candidate = &component;
  }
  if (!candidate)
    return false;

  Value value = candidate->values.front();
  Operation *definition = value.getDefiningOp();
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : value.getUses()) {
    if (state.positions.lookup(use.getOwner()) >= failure.position)
      uses.push_back(&use);
  }
  if (uses.empty())
    return false;

  definition->setAttr(kRematerializedAttr,
                      UnitAttr::get(definition->getContext()));
  DenseMap<Operation *, Value> clones;
  for (OpOperand *use : uses) {
    Operation *owner = use->getOwner();
    Value replacement = clones.lookup(owner);
    if (!replacement) {
      OpBuilder builder(owner);
      Operation *clone = builder.clone(*definition);
      clone->setAttr(kRematerializedAttr,
                     UnitAttr::get(definition->getContext()));
      replacement = clone->getResult(0);
      clones[owner] = replacement;
    }
    use->set(replacement);
  }
  return true;
}

static bool rematerializeOwnershipPredecessor(AllocationComponent &component,
                                              AllocationState &state) {
  if (component.fixedBase || component.values.size() != 1)
    return false;
  Value value = component.values.front();
  Operation *definition = value.getDefiningOp();
  if (!isRematerializable(definition) || !value.hasOneUse())
    return false;
  OpOperand &use = *value.use_begin();
  Operation *consumer = use.getOwner();
  if (definition->getBlock() != consumer->getBlock() ||
      state.positions.lookup(definition) + 1 >=
          state.positions.lookup(consumer))
    return false;

  OpBuilder builder(consumer);
  IRMapping mapping;
  bool clonedAdapter = false;
  for (Value operand : definition->getOperands()) {
    MovOp adapter = operand.getDefiningOp<MovOp>();
    if (!adapter || !operand.hasOneUse())
      continue;
    RegType sourceType = dyn_cast<RegType>(adapter.getSrc().getType());
    if (!sourceType || sourceType.getBaseGRF() < 0)
      continue;
    Operation *adapterClone = builder.clone(*adapter);
    adapter->setAttr(kRematerializedAttr,
                     UnitAttr::get(definition->getContext()));
    adapterClone->setAttr(kRematerializedAttr,
                          UnitAttr::get(definition->getContext()));
    mapping.map(operand, adapterClone->getResult(0));
    clonedAdapter = true;
  }
  if (!clonedAdapter)
    return false;

  Operation *clone = builder.clone(*definition, mapping);
  definition->setAttr(kRematerializedAttr,
                      UnitAttr::get(definition->getContext()));
  clone->setAttr(kRematerializedAttr, UnitAttr::get(definition->getContext()));
  use.set(clone->getResult(0));
  return true;
}

static bool repairTupleOwnershipHandoff(func::FuncOp function,
                                        AllocationState &state) {
  bool repaired = false;
  function
      .walk([&](UpdateTupleOp update) {
        if (repaired || !hasPreparedUpdateBaseCopy(update))
          return WalkResult::advance();
        const RegisterAliasAnalysis::ValueInfo *resultInfo =
            state.aliases.lookup(update.getResult());
        assert(resultInfo &&
               "prepared tuple result is missing alias information");
        const AllocationComponent &successor =
            state.components[resultInfo->component];

        AllocationComponent *predecessor = nullptr;
        for (AllocationComponent &component : state.components) {
          if (&component == &successor || component.fixedBase ||
              component.values.size() != 1 ||
              component.end >= successor.start ||
              !registersOverlap(component.assignment, component.widthGRFs(),
                                successor.assignment, successor.widthGRFs()))
            continue;
          if (!predecessor || predecessor->end < component.end)
            predecessor = &component;
        }
        if (predecessor)
          if (predecessor->values.front().getDefiningOp()->getBlock() ==
              update->getBlock())
            repaired = rematerializeOwnershipPredecessor(*predecessor, state);
        return repaired ? WalkResult::interrupt() : WalkResult::advance();
      })
      .wasInterrupted();
  return repaired;
}

static Value getScratchSurfaceOffset(func::FuncOp function, OpBuilder &builder,
                                     Location location,
                                     Operation *spillDefinition) {
  Value surfaceOffset;
  function.walk([&](ShrOp shift) {
    if (shift->hasAttr(kScratchSetupAttr))
      surfaceOffset = shift.getDst();
  });
  if (surfaceOffset)
    return surfaceOffset;

  MLIRContext *context = function.getContext();
  // Separate the mask producer from the first a0 write so Xe2 can use F@1.
  Operation *maskInsertion = spillDefinition;
  unsigned machineInstructions = 0;
  for (Operation *previous = spillDefinition->getPrevNode(); previous;
       previous = previous->getPrevNode()) {
    maskInsertion = previous;
    if (!previous->hasTrait<OpTrait::xemachine::NoAsmEmission>() &&
        ++machineInstructions == kMaxSwsbDistance + 1)
      break;
  }
  OpBuilder maskBuilder(maskInsertion);
  Type i32 = maskBuilder.getI32Type();
  RegionAttr uniform = RegionAttr::get(context, 0, 1, 0);
  DstRegionAttr canonical = DstRegionAttr::get(context, 1);
  bool canDelayAddressWrite = machineInstructions > kMaxSwsbDistance;
  Value r0 =
      ArchRegOp::create(maskBuilder, location, RegType::get(context, 16, 0),
                        maskBuilder.getI32IntegerAttr(0))
          .getResult();
  Value mask = ImmOp::create(maskBuilder, location, ImmType::get(context),
                             KernelABI::get().getScratchSurfaceMask(), i32)
                   .getResult();
  Type maskType = canDelayAddressWrite
                      ? Type(RegType::get(context, 16, -1))
                      : Type(ARFType::get(context, ARFFile::a0, 16, 0));
  IntegerAttr maskDestinationSub =
      canDelayAddressWrite ? IntegerAttr() : maskBuilder.getI32IntegerAttr(2);
  AndOp maskSetup =
      AndOp::create(maskBuilder, location, maskType, i32, /*execSize=*/1,
                    canonical, uniform, RegionAttr(), maskDestinationSub,
                    maskBuilder.getI32IntegerAttr(
                        KernelABI::get().getScratchSurfaceSourceSubregister()),
                    IntegerAttr(), TypeAttr(), TypeAttr(), /*noMask=*/true,
                    /*maskOffset=*/0, r0, mask);
  maskSetup->setAttr(kScratchSetupAttr, builder.getUnitAttr());
  Value shift = ImmOp::create(builder, location, ImmType::get(context),
                              KernelABI::get().getScratchSurfaceShift(), i32)
                    .getResult();
  IntegerAttr maskSourceSub =
      canDelayAddressWrite ? IntegerAttr() : builder.getI32IntegerAttr(2);
  ShrOp setup = ShrOp::create(
      builder, location, ARFType::get(context, ARFFile::a0, 16, 0), i32,
      /*execSize=*/1, canonical, uniform, RegionAttr(),
      builder.getI32IntegerAttr(2), maskSourceSub, IntegerAttr(), TypeAttr(),
      TypeAttr(), /*noMask=*/true, /*maskOffset=*/0, maskSetup.getResult(),
      shift);
  setup->setAttr(kScratchSetupAttr, builder.getUnitAttr());
  return setup.getDst();
}

static Value createScratchAddress(OpBuilder &builder, Location location,
                                  int64_t offset) {
  MLIRContext *context = builder.getContext();
  Type i32 = builder.getI32Type();
  Value immediate =
      ImmOp::create(builder, location, ImmType::get(context),
                    offset + KernelABI::get().getScratchAddressBias(), i32)
          .getResult();
  return MovOp::create(builder, location, RegType::get(context, 16, -1), i32,
                       /*execSize=*/1, DstRegionAttr::get(context, 1),
                       RegionAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(),
                       /*noMask=*/true, /*maskOffset=*/0, immediate)
      .getResult();
}

static uint32_t getScratchDescriptor(uint32_t widthDwords, bool load) {
  uint32_t vectorEncoding = widthDwords == 16 ? 5 : 6;
  uint32_t descriptor = (1u << 30) | (1u << 25) | (2u << 7) | (2u << 9) |
                        (1u << 15) | (vectorEncoding << 12);
  if (load)
    descriptor |= llvm::divideCeil(widthDwords, 16u) << 20;
  else
    descriptor |= 4;
  return descriptor;
}

static SendOp createScratchSend(OpBuilder &builder, Location location,
                                Type destinationType, Value address, Value data,
                                Value surfaceOffset, Value dependency,
                                uint32_t descriptor) {
  MLIRContext *context = builder.getContext();
  SendOp send = SendOp::create(
      builder, location, destinationType, MemTokenType::get(context),
      SendFn::ugm, /*sfid=*/0, descriptor,
      /*exdesc=*/static_cast<int32_t>(KernelABI::get().getScratchExdescBias()),
      /*execSize=*/1,
      /*noMask=*/true, /*eot=*/false, address, data, surfaceOffset, dependency);
  send->setAttr(kScratchAccessAttrName, UnitAttr::get(context));
  return send;
}

static bool spillToScratch(func::FuncOp function, AllocationState &state,
                           const AllocationFailure &failure,
                           int64_t &nextScratchOffset) {
  bool conflictingAddressRegister = false;
  function.walk([&](Operation *operation) {
    if (operation->hasAttr(kScratchSetupAttr))
      return;
    for (Type type : operation->getResultTypes()) {
      auto arf = dyn_cast<ARFType>(type);
      conflictingAddressRegister |= arf && arf.getFile() == ARFFile::a0;
    }
  });
  if (conflictingAddressRegister)
    return false;

  AllocationComponent *candidate = nullptr;
  for (AllocationComponent &component : state.components) {
    if (component.fixedBase || component.values.size() != 1 ||
        component.start >= failure.position || component.end < failure.position)
      continue;
    Value value = component.values.front();
    Operation *definition = value.getDefiningOp();
    RegType type = cast<RegType>(value.getType());
    if (!definition || definition->hasAttr(kSpilledAttr) ||
        definition->hasAttr(kScratchSetupAttr) || isa<SendOp>(definition) ||
        (type.getWidthDwords() != 16 && type.getWidthDwords() != 32))
      continue;
    if (!hasUseAtOrAfter(value, state, failure.position))
      continue;
    if (!candidate || candidate->end < component.end)
      candidate = &component;
  }
  if (!candidate)
    return false;

  Value value = candidate->values.front();
  Operation *definition = value.getDefiningOp();
  RegType type = cast<RegType>(value.getType());
  SmallVector<OpOperand *> uses;
  for (OpOperand &use : value.getUses()) {
    if (state.positions.lookup(use.getOwner()) >= failure.position)
      uses.push_back(&use);
  }
  assert(!uses.empty() && "scratch candidate must have a rewritable use");

  int64_t slot = llvm::alignTo(
      nextScratchOffset,
      static_cast<int64_t>(KernelABI::get().getScratchSlotAlignment()));
  nextScratchOffset = slot + type.getWidthDwords() * 4;
  OpBuilder storeBuilder(definition);
  storeBuilder.setInsertionPointAfter(definition);
  Value surfaceOffset = getScratchSurfaceOffset(
      function, storeBuilder, definition->getLoc(), definition);
  Value storeAddress =
      createScratchAddress(storeBuilder, definition->getLoc(), slot);
  SendOp store = createScratchSend(
      storeBuilder, definition->getLoc(),
      RegType::get(function.getContext(), 0, -1), storeAddress, value,
      surfaceOffset, Value(),
      getScratchDescriptor(type.getWidthDwords(), /*load=*/false));
  SyncOp storeDrain =
      SyncOp::create(storeBuilder, definition->getLoc(),
                     MemTokenType::get(function.getContext()),
                     SyncKindAttr::get(function.getContext(), SyncKind::allrd),
                     store.getToken());
  definition->setAttr(kSpilledAttr, UnitAttr::get(function.getContext()));

  DenseMap<Operation *, Value> reloads;
  for (OpOperand *use : uses) {
    Operation *owner = use->getOwner();
    Value reload = reloads.lookup(owner);
    if (!reload) {
      OpBuilder loadBuilder(owner);
      Value loadAddress =
          createScratchAddress(loadBuilder, owner->getLoc(), slot);
      SendOp load = createScratchSend(
          loadBuilder, owner->getLoc(),
          RegType::get(function.getContext(), type.getWidthDwords(), -1),
          loadAddress, Value(), surfaceOffset, storeDrain.getToken(),
          getScratchDescriptor(type.getWidthDwords(), /*load=*/true));
      reload = load.getDst();
      reloads[owner] = reload;
    }
    use->set(reload);
  }
  function->setAttr(
      kScratchSizeAttrName,
      IntegerAttr::get(IntegerType::get(function.getContext(), 64),
                       nextScratchOffset));
  return true;
}

static void commitAllocation(AllocationState &state) {
  for (Value value : state.aliases.getValues()) {
    const RegisterAliasAnalysis::ValueInfo *valueInfo =
        state.aliases.lookup(value);
    assert(valueInfo && "register value is missing alias information");
    const AllocationComponent &component =
        state.components[valueInfo->component];
    RegType oldType = cast<RegType>(value.getType());
    value.setType(
        RegType::get(value.getContext(), oldType.getWidthDwords(),
                     component.assignment + valueInfo->offsetDwords / 16));
  }
}

struct RegAllocConfig {
  unsigned grfLimit;
  unsigned reservedGrfCount;
  int64_t scratchSize;
};

struct ArfLiveRange {
  Value value;
  ARFType type;
  int64_t start;
  int64_t end;
  int32_t assignment;
  bool reference;
};

static FailureOr<SmallVector<ArfLiveRange>>
buildArfState(func::FuncOp function) {
  DenseMap<Operation *, int64_t> positions;
  int64_t nextPosition = 0;
  function.walk(
      [&](Operation *operation) { positions[operation] = nextPosition++; });

  SmallVector<ArfLiveRange> ranges;
  auto addRange = [&](Value value) -> LogicalResult {
    ARFType type = dyn_cast<ARFType>(value.getType());
    if (!type)
      return success();
    if (type.getIndex() < 0 && type.getFile() != ARFFile::f)
      return emitError(value.getLoc())
                 << "virtual " << stringifyARFFile(type.getFile())
                 << " ARF allocation is unsupported",
             failure();
    if (type.getFile() == ARFFile::f && type.getWidthDwords() != 2)
      return emitError(value.getLoc())
                 << "flag allocation requires a 2-dword ARF footprint",
             failure();
    if (type.getFile() == ARFFile::f && type.getIndex() >= 2)
      return emitError(value.getLoc())
                 << "flag register index exceeds the f0/f1 register file",
             failure();

    int64_t start = 0;
    if (Operation *definition = value.getDefiningOp())
      start = positions.lookup(definition);
    else if (BlockArgument argument = dyn_cast<BlockArgument>(value)) {
      Operation *parent = argument.getOwner()->getParentOp();
      start = parent ? positions.lookup(parent) : 0;
    }
    int64_t end = start;
    for (OpOperand &use : value.getUses())
      end = std::max(end, positions.lookup(use.getOwner()));
    ranges.push_back({value, type, start, end, type.getIndex(),
                      isa_and_nonnull<ArfRegOp>(value.getDefiningOp())});
    return success();
  };

  WalkResult walkResult = function.walk([&](Operation *operation) {
    for (Value result : operation->getResults())
      if (failed(addRange(result)))
        return WalkResult::interrupt();
    for (Region &region : operation->getRegions())
      for (Block &block : region)
        for (BlockArgument argument : block.getArguments())
          if (failed(addRange(argument)))
            return WalkResult::interrupt();
    return WalkResult::advance();
  });
  if (walkResult.wasInterrupted())
    return failure();

  RegionFlow regionFlow(function);
  for (const RegionFlow::Branch &branch : regionFlow.getBranches()) {
    for (Region *region : branch.regions) {
      if (!regionFlow.isRepetitive(region))
        continue;
      int64_t loopEnd = positions.lookup(branch.operation);
      region->walk([&](Operation *operation) {
        loopEnd = std::max(loopEnd, positions.lookup(operation));
      });
      for (ArfLiveRange &range : ranges) {
        bool captured =
            llvm::any_of(range.value.getUses(), [&](OpOperand &use) {
              Operation *owner = use.getOwner();
              if (!region->isAncestor(owner->getParentRegion()))
                return false;
              Operation *definition = range.value.getDefiningOp();
              if (definition)
                return !region->isAncestor(definition->getParentRegion());
              BlockArgument argument = cast<BlockArgument>(range.value);
              return !region->isAncestor(argument.getOwner()->getParent());
            });
        if (captured)
          range.end = std::max(range.end, loopEnd);
      }
    }
  }

  llvm::stable_sort(ranges,
                    [](const ArfLiveRange &lhs, const ArfLiveRange &rhs) {
                      if (lhs.start != rhs.start)
                        return lhs.start < rhs.start;
                      return lhs.assignment >= 0 && rhs.assignment < 0;
                    });
  for (auto [index, range] : llvm::enumerate(ranges)) {
    if (range.type.getFile() != ARFFile::f || range.assignment < 0)
      continue;
    for (const ArfLiveRange &other : ArrayRef(ranges).take_front(index)) {
      if (range.type.getFile() != other.type.getFile() ||
          range.assignment != other.assignment)
        continue;
      if (range.reference && other.reference)
        continue;
      if (range.start <= other.end && other.start <= range.end)
        return emitError(range.value.getLoc())
                   << "fixed " << stringifyARFFile(range.type.getFile())
                   << range.assignment << " live ranges overlap",
               failure();
    }
  }
  return ranges;
}

static LogicalResult tryAllocateArfs(MutableArrayRef<ArfLiveRange> ranges) {
  for (ArfLiveRange &range : ranges) {
    if (range.type.getFile() != ARFFile::f || range.assignment >= 0)
      continue;
    for (int32_t candidate : llvm::seq<int32_t>(0, 2)) {
      bool available = true;
      for (const ArfLiveRange &other : ranges) {
        if (other.type.getFile() != ARFFile::f || other.assignment != candidate)
          continue;
        if (range.start <= other.end && other.start <= range.end) {
          available = false;
          break;
        }
      }
      if (!available)
        continue;
      range.assignment = candidate;
      break;
    }
    if (range.assignment < 0)
      return emitError(range.value.getLoc())
                 << "flag allocation exhausted f0/f1 for overlapping live "
                    "ranges",
             failure();
  }
  return success();
}

static void commitArfAllocation(ArrayRef<ArfLiveRange> ranges) {
  for (const ArfLiveRange &range : ranges) {
    if (range.type.getIndex() >= 0)
      continue;
    Value value = range.value;
    value.setType(ARFType::get(range.value.getContext(), range.type.getFile(),
                               range.type.getWidthDwords(), range.assignment));
  }
}

static DictionaryAttr packArfState(Builder &builder,
                                   ArrayRef<ArfLiveRange> ranges) {
  SmallVector<int64_t> intervals;
  for (const ArfLiveRange &range : ranges) {
    intervals.push_back(range.start);
    intervals.push_back(range.end);
    intervals.push_back(static_cast<int64_t>(range.type.getFile()));
    intervals.push_back(range.assignment);
  }
  return builder.getDictionaryAttr({
      builder.getNamedAttr(kStageAttr, builder.getStringAttr(kArfBuildStage)),
      builder.getNamedAttr(
          "arf_intervals",
          DenseI64ArrayAttr::get(builder.getContext(), intervals)),
  });
}

static FailureOr<RegAllocConfig>
validateRegAllocFunction(func::FuncOp function) {
  FunctionType functionType = function.getFunctionType();
  auto isRegisterType = [](Type type) { return isa<RegType, ARFType>(type); };
  if (llvm::any_of(functionType.getInputs(), isRegisterType) ||
      llvm::any_of(functionType.getResults(), isRegisterType))
    return function.emitError("register allocation does not support "
                              "register-valued signatures"),
           failure();

  if (function
          .walk([&](Operation *operation) {
            SmallVector<Type> types(operation->getResultTypes());
            for (Region &region : operation->getRegions())
              for (Block &block : region)
                llvm::append_range(types, block.getArgumentTypes());
            for (Type type : types) {
              auto arf = dyn_cast<ARFType>(type);
              if (!arf || arf.getIndex() >= 0)
                continue;
              operation->emitError(
                  "GRF allocation requires ARF values to be allocated first");
              return WalkResult::interrupt();
            }
            return WalkResult::advance();
          })
          .wasInterrupted())
    return failure();

  IntegerAttr grfCount =
      function->getAttrOfType<IntegerAttr>(kGrfCountAttrName);
  if (!grfCount || grfCount.getInt() <= 0)
    return function.emitError("register allocation requires a positive ")
               << kGrfCountAttrName << " function attribute",
           failure();
  IntegerAttr reserved =
      function->getAttrOfType<IntegerAttr>(kReservedGrfCountAttrName);
  if (!reserved || reserved.getInt() < 0 ||
      reserved.getInt() > grfCount.getInt())
    return function.emitError("register allocation requires a valid ")
               << kReservedGrfCountAttrName << " function attribute",
           failure();
  IntegerAttr scratch =
      function->getAttrOfType<IntegerAttr>(kScratchSizeAttrName);
  if (scratch && scratch.getInt() < 0)
    return function.emitError("register allocation requires a nonnegative ")
               << kScratchSizeAttrName << " function attribute",
           failure();
  return RegAllocConfig{static_cast<unsigned>(grfCount.getInt()),
                        static_cast<unsigned>(reserved.getInt()),
                        scratch ? scratch.getInt() : 0};
}

static void collectFunctions(Operation *target,
                             SmallVectorImpl<func::FuncOp> &functions) {
  if (auto function = dyn_cast<func::FuncOp>(target)) {
    functions.push_back(function);
    return;
  }
  target->walk([&](func::FuncOp function) { functions.push_back(function); });
}

static unsigned getLoopIteration(func::FuncOp function) {
  IntegerAttr attr = function->getAttrOfType<IntegerAttr>(kLoopIterationAttr);
  return attr ? static_cast<unsigned>(attr.getInt()) : 1;
}

static LogicalResult buildTransformArfState(Operation *target) {
  SmallVector<func::FuncOp> functions;
  collectFunctions(target, functions);
  for (func::FuncOp function : functions) {
    FailureOr<SmallVector<ArfLiveRange>> ranges = buildArfState(function);
    if (failed(ranges))
      return failure();
    Builder builder(function.getContext());
    function->setAttr(kStateAttr, packArfState(builder, *ranges));
  }
  return success();
}

static LogicalResult runTransformArfLinearScan(Operation *target) {
  SmallVector<func::FuncOp> functions;
  collectFunctions(target, functions);
  for (func::FuncOp function : functions) {
    DictionaryAttr packed = function->getAttrOfType<DictionaryAttr>(kStateAttr);
    StringAttr stage = packed ? packed.getAs<StringAttr>(kStageAttr) : nullptr;
    if (!stage || stage.getValue() != kArfBuildStage)
      return function.emitError(
                 "ARF linear scan requires ARF live-range input"),
             failure();
    FailureOr<SmallVector<ArfLiveRange>> ranges = buildArfState(function);
    if (failed(ranges) || failed(tryAllocateArfs(*ranges)))
      return failure();
    commitArfAllocation(*ranges);
    function->removeAttr(kStateAttr);
  }
  return success();
}

static LogicalResult buildTransformState(Operation *target) {
  SmallVector<func::FuncOp> functions;
  collectFunctions(target, functions);
  for (func::FuncOp function : functions) {
    if (failed(validateRegAllocFunction(function)) ||
        failed(prepareRegisterAllocation(function)))
      return failure();
    AllocationState state;
    if (failed(buildState(function, state)))
      return failure();
    Builder builder(function.getContext());
    function->setAttr(kStateAttr, packState(builder, getLoopIteration(function),
                                            kBuildStage, state));
  }
  return success();
}

static LogicalResult runTransformLinearScan(Operation *target) {
  SmallVector<func::FuncOp> functions;
  collectFunctions(target, functions);
  for (func::FuncOp function : functions) {
    FailureOr<RegAllocConfig> config = validateRegAllocFunction(function);
    if (failed(config))
      return failure();
    DictionaryAttr packed = function->getAttrOfType<DictionaryAttr>(kStateAttr);
    if (!packed ||
        packed.getAs<StringAttr>(kStageAttr).getValue() != kBuildStage)
      return function.emitError("linear scan requires alias-state input"),
             failure();
    AllocationState state;
    if (failed(buildState(function, state)))
      return failure();
    std::optional<AllocationFailure> allocationFailure =
        tryAllocate(state, config->grfLimit, config->reservedGrfCount);
    Builder builder(function.getContext());
    if (!allocationFailure) {
      if (repairTupleOwnershipHandoff(function, state)) {
        function->removeAttr(kStateAttr);
        continue;
      }
      commitAllocation(state);
      function->setAttr(
          kStateAttr,
          packState(builder, getLoopIteration(function), kSuccessStage, state));
      continue;
    }
    function->setAttr(kStateAttr,
                      packState(builder, getLoopIteration(function),
                                kFailureStage, state, allocationFailure));
  }
  return success();
}

static FailureOr<std::optional<AllocationFailure>>
readTransformFailure(func::FuncOp function) {
  DictionaryAttr packed = function->getAttrOfType<DictionaryAttr>(kStateAttr);
  if (!packed)
    return std::optional<AllocationFailure>();
  StringAttr stage = packed.getAs<StringAttr>(kStageAttr);
  if (!stage)
    return function.emitError("regalloc state is missing its stage"), failure();
  if (stage.getValue() != kFailureStage)
    return std::optional<AllocationFailure>();
  IntegerAttr component = packed.getAs<IntegerAttr>("failed_component");
  IntegerAttr position = packed.getAs<IntegerAttr>("failure_position");
  if (!component || !position)
    return function.emitError("regalloc failure state is incomplete"),
           failure();
  return std::optional<AllocationFailure>(AllocationFailure{
      static_cast<unsigned>(component.getInt()), position.getInt()});
}

static LogicalResult runTransformRematRelief(Operation *target) {
  SmallVector<func::FuncOp> functions;
  collectFunctions(target, functions);
  for (func::FuncOp function : functions) {
    FailureOr<std::optional<AllocationFailure>> failureRecord =
        readTransformFailure(function);
    if (failed(failureRecord))
      return failure();
    if (!*failureRecord)
      continue;
    AllocationState state;
    if (failed(buildState(function, state)))
      return failure();
    if (rematerialize(state, **failureRecord))
      function->removeAttr(kStateAttr);
  }
  return success();
}

static LogicalResult runTransformScratchRelief(Operation *target) {
  SmallVector<func::FuncOp> functions;
  collectFunctions(target, functions);
  for (func::FuncOp function : functions) {
    FailureOr<std::optional<AllocationFailure>> failureRecord =
        readTransformFailure(function);
    if (failed(failureRecord))
      return failure();
    if (!*failureRecord)
      continue;
    FailureOr<RegAllocConfig> config = validateRegAllocFunction(function);
    if (failed(config))
      return failure();
    AllocationState state;
    if (failed(buildState(function, state)))
      return failure();
    int64_t nextScratchOffset = config->scratchSize;
    if (spillToScratch(function, state, **failureRecord, nextScratchOffset))
      function->removeAttr(kStateAttr);
  }
  return success();
}

static SmallVector<transform::MappedValue>
buildHandleMapping(ArrayRef<Operation *> targets) {
  SmallVector<transform::MappedValue> mapping;
  for (Operation *target : targets)
    mapping.push_back(target);
  return mapping;
}

static DiagnosedSilenceableFailure
runNamedSequence(transform::TransformOpInterface caller,
                 transform::NamedSequenceOp callee,
                 MutableArrayRef<SmallVector<transform::MappedValue>> bindings,
                 transform::TransformState &state,
                 SmallVectorImpl<SmallVector<transform::MappedValue>> &out) {
  if (callee.isExternal())
    return caller.emitDefiniteFailure()
           << "named sequence `" << callee.getSymName()
           << "` is external; cannot invoke";
  Block &block = callee.getBody().front();
  if (block.getNumArguments() != bindings.size())
    return caller.emitDefiniteFailure()
           << "named sequence `" << callee.getSymName() << "` expects "
           << block.getNumArguments() << " arguments, got " << bindings.size();
  auto scope = state.make_region_scope(callee.getBody());
  for (auto [argument, mapping] :
       llvm::zip_equal(block.getArguments(), bindings))
    if (failed(state.mapBlockArgument(argument, mapping)))
      return DiagnosedSilenceableFailure::definiteFailure();
  for (Operation &operation : block.without_terminator()) {
    DiagnosedSilenceableFailure result =
        state.applyTransform(cast<transform::TransformOpInterface>(operation));
    if (!result.succeeded())
      return result;
  }
  transform::detail::prepareValueMappings(
      out, block.getTerminator()->getOperands(), state);
  return DiagnosedSilenceableFailure::success();
}

enum class LoopDecision { Restart, Done, Stalled };

static FailureOr<LoopDecision> classifyLoop(ArrayRef<Operation *> targets) {
  LoopDecision decision = LoopDecision::Done;
  SmallVector<func::FuncOp> functions;
  SmallVector<func::FuncOp> stalledFunctions;
  for (Operation *target : targets)
    collectFunctions(target, functions);
  if (functions.empty())
    return failure();
  for (func::FuncOp function : functions) {
    DictionaryAttr packed = function->getAttrOfType<DictionaryAttr>(kStateAttr);
    if (!packed) {
      decision = LoopDecision::Restart;
      continue;
    }
    StringAttr stage = packed.getAs<StringAttr>(kStageAttr);
    if (!stage)
      return failure();
    if (stage.getValue() == kFailureStage) {
      stalledFunctions.push_back(function);
      if (decision != LoopDecision::Restart)
        decision = LoopDecision::Stalled;
    } else if (stage.getValue() != kSuccessStage)
      return failure();
  }
  if (decision != LoopDecision::Stalled)
    return decision;
  for (func::FuncOp function : stalledFunctions) {
    IntegerAttr grfCount =
        function->getAttrOfType<IntegerAttr>(kGrfCountAttrName);
    function.emitError("register allocation exhausted ")
        << grfCount.getInt()
        << " GRFs and no rematerialization or scratch candidate can relieve "
           "pressure";
  }
  return decision;
}

static void clearLoopState(ArrayRef<Operation *> targets, unsigned iteration) {
  SmallVector<func::FuncOp> functions;
  for (Operation *target : targets)
    collectFunctions(target, functions);
  for (func::FuncOp function : functions) {
    function->removeAttr(kStateAttr);
    function->setAttr(
        kLoopIterationAttr,
        IntegerAttr::get(IntegerType::get(function.getContext(), 32),
                         iteration));
  }
}

static void finishLoop(ArrayRef<Operation *> targets, unsigned iterations) {
  SmallVector<func::FuncOp> functions;
  for (Operation *target : targets)
    collectFunctions(target, functions);
  for (func::FuncOp function : functions) {
    function->removeAttr(kStateAttr);
    function->removeAttr(kLoopIterationAttr);
    function->setAttr(
        kIterationAttr,
        IntegerAttr::get(IntegerType::get(function.getContext(), 32),
                         iterations));
  }
}

template <typename OpTy, typename ApplyFn>
static DiagnosedSilenceableFailure
applyStage(OpTy op, transform::TransformResults &results,
           transform::TransformState &state, ApplyFn apply) {
  SmallVector<Operation *> targets;
  for (Operation *target : state.getPayloadOps(op.getTarget())) {
    targets.push_back(target);
    if (failed(apply(target)))
      return op.emitDefiniteFailure() << "regalloc transform stage failed";
  }
  results.set(cast<OpResult>(op.getResult()), targets);
  return DiagnosedSilenceableFailure::success();
}

template <typename OpTy>
static void
getStageEffects(OpTy op,
                SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  transform::consumesHandle(op.getTargetMutable(), effects);
  transform::producesHandle(op->getOpResults(), effects);
  transform::modifiesPayload(effects);
}

} // namespace

namespace inter::xemachine {

LogicalResult
TransformRegAllocLoopOp::verifySymbolUses(SymbolTableCollection &symbolTable) {
  transform::NamedSequenceOp body =
      symbolTable.lookupNearestSymbolFrom<transform::NamedSequenceOp>(
          getOperation(), getBodyAttr());
  if (!body)
    return emitOpError() << "body symbol `" << getBody()
                         << "` does not resolve to a transform.named_sequence";
  if (getMaxIterations() == 0 ||
      getMaxIterations() > std::numeric_limits<unsigned>::max())
    return emitOpError("requires max_iterations in the unsigned integer range");
  return success();
}

DiagnosedSilenceableFailure
TransformRegAllocLoopOp::apply(transform::TransformRewriter &,
                               transform::TransformResults &results,
                               transform::TransformState &state) {
  transform::NamedSequenceOp body =
      SymbolTable::lookupNearestSymbolFrom<transform::NamedSequenceOp>(
          getOperation(), getBodyAttr());
  SmallVector<Operation *> current;
  llvm::append_range(current, state.getPayloadOps(getTarget()));
  unsigned maxIterations = static_cast<unsigned>(getMaxIterations());
  for (unsigned iteration = 1; iteration <= maxIterations; ++iteration) {
    clearLoopState(current, iteration);
    SmallVector<SmallVector<transform::MappedValue>> bindings;
    bindings.push_back(buildHandleMapping(current));
    SmallVector<SmallVector<transform::MappedValue>> output;
    DiagnosedSilenceableFailure status =
        runNamedSequence(*this, body, bindings, state, output);
    if (!status.succeeded())
      return status;
    if (output.size() != 1)
      return emitDefiniteFailure(
          "regalloc loop body must yield one operation handle");
    SmallVector<Operation *> yielded;
    for (transform::MappedValue mapping : output.front()) {
      Operation *payload = mapping.dyn_cast<Operation *>();
      if (!payload)
        return emitDefiniteFailure(
            "regalloc loop body must yield operation handles");
      yielded.push_back(payload);
    }
    FailureOr<LoopDecision> decision = classifyLoop(yielded);
    if (failed(decision))
      return emitDefiniteFailure("failed to classify regalloc loop state");
    if (*decision == LoopDecision::Restart) {
      current = std::move(yielded);
      continue;
    }
    if (*decision == LoopDecision::Stalled)
      return emitDefiniteFailure(
          "register allocation stalled after all relief providers");
    finishLoop(yielded, iteration);
    results.set(cast<OpResult>(getResult()), yielded);
    return DiagnosedSilenceableFailure::success();
  }
  return emitDefiniteFailure() << "regalloc transform loop exceeded "
                               << getMaxIterations() << " iterations";
}

void TransformRegAllocLoopOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getStageEffects(*this, effects);
}

DiagnosedSilenceableFailure
TransformRegAllocBuildStateOp::apply(transform::TransformRewriter &,
                                     transform::TransformResults &results,
                                     transform::TransformState &state) {
  return applyStage(*this, results, state, buildTransformState);
}

void TransformRegAllocBuildStateOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getStageEffects(*this, effects);
}

DiagnosedSilenceableFailure
TransformRegAllocArfBuildStateOp::apply(transform::TransformRewriter &,
                                        transform::TransformResults &results,
                                        transform::TransformState &state) {
  return applyStage(*this, results, state, buildTransformArfState);
}

void TransformRegAllocArfBuildStateOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getStageEffects(*this, effects);
}

DiagnosedSilenceableFailure
TransformRegAllocArfLinearScanOp::apply(transform::TransformRewriter &,
                                        transform::TransformResults &results,
                                        transform::TransformState &state) {
  return applyStage(*this, results, state, runTransformArfLinearScan);
}

void TransformRegAllocArfLinearScanOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getStageEffects(*this, effects);
}

DiagnosedSilenceableFailure
TransformRegAllocLinearScanOp::apply(transform::TransformRewriter &,
                                     transform::TransformResults &results,
                                     transform::TransformState &state) {
  return applyStage(*this, results, state, runTransformLinearScan);
}

void TransformRegAllocLinearScanOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getStageEffects(*this, effects);
}

DiagnosedSilenceableFailure
TransformRegAllocRematReliefOp::apply(transform::TransformRewriter &,
                                      transform::TransformResults &results,
                                      transform::TransformState &state) {
  return applyStage(*this, results, state, runTransformRematRelief);
}

void TransformRegAllocRematReliefOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getStageEffects(*this, effects);
}

DiagnosedSilenceableFailure
TransformRegAllocScratchReliefOp::apply(transform::TransformRewriter &,
                                        transform::TransformResults &results,
                                        transform::TransformState &state) {
  return applyStage(*this, results, state, runTransformScratchRelief);
}

void TransformRegAllocScratchReliefOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  getStageEffects(*this, effects);
}

} // namespace inter::xemachine
