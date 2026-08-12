#include "inter/Dialect/XeMachine/IR/XeMachineRegAllocPreparation.h"

#include "inter/Dialect/XeMachine/IR/XeMachine.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <optional>

using namespace mlir;
using namespace inter::xemachine;

namespace {

constexpr StringLiteral kRegisterCopyAttr = "xemachine.regalloc_copy";
constexpr StringLiteral kImmediateLegalizationAttr =
    "xemachine.immediate_legalization";

static bool isMarkedCopy(Operation *operation) {
  return operation && operation->hasAttr(kRegisterCopyAttr);
}

static void legalizeWideImmediates(func::FuncOp function) {
  SmallVector<OpOperand *> operands;
  function.walk([&](Operation *operation) {
    TypeAttr elementType = operation->getAttrOfType<TypeAttr>("elemType");
    if (!elementType || !elementType.getValue().isInteger(64) ||
        isa<MovOp>(operation))
      return;
    for (OpOperand &operand : operation->getOpOperands())
      if (operand.get().getDefiningOp<ImmOp>())
        operands.push_back(&operand);
  });
  for (OpOperand *operand : operands) {
    Operation *owner = operand->getOwner();
    OpBuilder builder(owner);
    MovOp move = MovOp::create(
        builder, owner->getLoc(), RegType::get(function.getContext(), 2, -1),
        builder.getI64Type(), /*execSize=*/1, DstRegionAttr::get(
            function.getContext(), 1),
        RegionAttr(), IntegerAttr(), IntegerAttr(), TypeAttr(), /*noMask=*/true,
        /*maskOffset=*/0, operand->get());
    move->setAttr(kImmediateLegalizationAttr, builder.getUnitAttr());
    operand->set(move.getDst());
    constexpr std::array<StringLiteral, 3> regionNames = {
        "src0Region", "src1Region", "src2Region"};
    unsigned operandNumber = operand->getOperandNumber();
    if (operandNumber < regionNames.size() &&
        !owner->getAttr(regionNames[operandNumber]))
      owner->setAttr(regionNames[operandNumber],
                     RegionAttr::get(function.getContext(), 0, 1, 0));
  }
}

static bool isMarkedCopy(Value value, StringRef kind) {
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false;
  StringAttr marker = definition->getAttrOfType<StringAttr>(kRegisterCopyAttr);
  return marker && marker.getValue() == kind;
}

static Value getCopiedSource(Value value) {
  Operation *definition = value.getDefiningOp();
  if (!isMarkedCopy(definition))
    return {};
  if (MovOp move = dyn_cast<MovOp>(definition))
    return move.getSrc();

  TupleFromElementsOp tuple = dyn_cast<TupleFromElementsOp>(definition);
  if (!tuple || tuple.getElements().empty())
    return {};
  Value source;
  for (Value element : tuple.getElements()) {
    MovOp move = element.getDefiningOp<MovOp>();
    if (!move || !isMarkedCopy(move))
      return {};
    if (!source)
      source = move.getSrc();
    else if (source != move.getSrc())
      return {};
  }
  return source;
}

static FailureOr<Value> materializeRegisterCopy(OpBuilder &builder,
                                                Location location, Value source,
                                                RegType destinationType,
                                                StringRef kind) {
  RegType sourceType = dyn_cast<RegType>(source.getType());
  if (!sourceType)
    return emitError(location)
               << "cannot materialize XeMachine register copy from "
               << source.getType(),
           failure();

  uint32_t width = sourceType.getWidthDwords();
  bool singleInstructionWidth = width <= 32 && llvm::isPowerOf2_32(width);
  if (!singleInstructionWidth && (width == 0 || width % 16 != 0))
    return emitError(location)
               << "cannot materialize XeMachine register copy of " << width
               << " dwords; width must be a legal execution size or divisible "
                  "by 16",
           failure();
  if (destinationType.getWidthDwords() != width)
    return emitError(location)
               << "cannot materialize XeMachine register copy from " << width
               << " dwords to " << destinationType.getWidthDwords()
               << " dwords",
           failure();

  SmallVector<Value, 4> pieces;
  MLIRContext *context = builder.getContext();
  Type i32 = builder.getI32Type();
  for (uint32_t offset = 0; offset < width;) {
    uint32_t pieceWidth = std::min<uint32_t>(32, width - offset);
    int32_t destinationBase = destinationType.getBaseGRF();
    if (destinationBase >= 0)
      destinationBase += offset / 16;
    RegType pieceType = RegType::get(context, pieceWidth, destinationBase);
    IntegerAttr sourceSub =
        offset == 0 ? IntegerAttr() : builder.getI32IntegerAttr(offset);
    MovOp move =
        MovOp::create(builder, location, pieceType, i32,
                      /*execSize=*/pieceWidth, DstRegionAttr(), RegionAttr(),
                      IntegerAttr(), sourceSub, TypeAttr(), /*noMask=*/false,
                      /*maskOffset=*/0, source);
    move->setAttr(kRegisterCopyAttr, builder.getStringAttr(kind));
    pieces.push_back(move.getDst());
    offset += pieceWidth;
  }

  if (pieces.size() == 1)
    return pieces.front();
  TupleFromElementsOp tuple = TupleFromElementsOp::create(
      builder, location, destinationType, ValueRange(pieces));
  tuple->setAttr(kRegisterCopyAttr, builder.getStringAttr(kind));
  return tuple.getTuple();
}

static bool isPotentiallyAfter(Operation *candidate, Operation *anchor,
                               DominanceInfo &dominance) {
  if (candidate == anchor)
    return false;

  Block *anchorBlock = anchor->getBlock();
  Operation *ancestor = candidate;
  while (ancestor && ancestor->getBlock() != anchorBlock)
    ancestor = ancestor->getParentOp();
  if (ancestor)
    return ancestor != anchor && anchor->isBeforeInBlock(ancestor);
  if (dominance.properlyDominates(candidate, anchor))
    return false;
  return true;
}

static bool isLiveAfter(Value value, Operation *anchor,
                        DominanceInfo &dominance,
                        Operation *ignoredUser = nullptr) {
  for (OpOperand &use : value.getUses()) {
    Operation *owner = use.getOwner();
    if (owner == anchor || owner == ignoredUser)
      continue;
    if (isPotentiallyAfter(owner, anchor, dominance))
      return true;
  }
  return false;
}

struct AliasEdge {
  Value target;
  int64_t offset;
};

struct AliasValueInfo {
  unsigned component;
  int64_t offset;
};

class WeightedOverlapSummary {
public:
  LogicalResult build(func::FuncOp function, bool includeRegionAliases = true) {
    values.clear();
    graph.clear();
    info.clear();
    componentOrigins.clear();
    componentInconsistent.clear();

    auto remember = [&](Value value) {
      RegType type = dyn_cast<RegType>(value.getType());
      if (!type || graph.count(value))
        return;
      values.push_back(value);
      graph.try_emplace(value);
    };
    function.walk([&](Operation *operation) {
      for (Value result : operation->getResults())
        remember(result);
      for (Region &region : operation->getRegions())
        for (Block &block : region)
          for (BlockArgument argument : block.getArguments())
            remember(argument);
    });

    auto connect = [&](Value storage, Value alias, int64_t offset) {
      remember(storage);
      remember(alias);
      graph[storage].push_back({alias, offset});
      graph[alias].push_back({storage, -offset});
    };
    function.walk([&](Operation *operation) {
      RegisterStorageAliasOpInterface interface =
          dyn_cast<RegisterStorageAliasOpInterface>(operation);
      if (interface) {
        SmallVector<RegisterStorageAlias, 4> aliases;
        interface.getRegisterStorageAliases(aliases);
        for (const RegisterStorageAlias &relation : aliases)
          connect(relation.storage, relation.alias, relation.offset);
      }
      if (!includeRegionAliases)
        return;

      auto connectYields = [&](Region &region) {
        if (region.empty())
          return;
        YieldOp yield = dyn_cast<YieldOp>(region.front().getTerminator());
        if (!yield || yield.getValues().size() != operation->getNumResults())
          return;
        for (auto [result, value] :
             llvm::zip_equal(operation->getResults(), yield.getValues()))
          connect(result, value, 0);
      };
      if (ExecIfOp branch = dyn_cast<ExecIfOp>(operation)) {
        connectYields(branch.getThenRegion());
        connectYields(branch.getElseRegion());
        return;
      }
      if (UniformIfOp branch = dyn_cast<UniformIfOp>(operation)) {
        connectYields(branch.getThenRegion());
        connectYields(branch.getElseRegion());
        return;
      }
      UniformLoopOp loop = dyn_cast<UniformLoopOp>(operation);
      if (!loop || loop.getBody().empty())
        return;
      Block &body = loop.getBody().front();
      ContinueIfOp terminator = dyn_cast<ContinueIfOp>(body.getTerminator());
      if (!terminator || loop.getInits().size() != body.getNumArguments() ||
          terminator.getCarried().size() != body.getNumArguments() ||
          loop.getNumResults() != body.getNumArguments())
        return;
      for (auto [init, result] :
           llvm::zip_equal(loop.getInits(), loop.getResults()))
        connect(result, init, 0);
    });

    SmallVector<Value, 16> pending;
    for (Value root : values) {
      if (info.count(root))
        continue;
      unsigned component = componentOrigins.size();
      componentOrigins.push_back(std::nullopt);
      componentInconsistent.push_back(false);
      info.try_emplace(root, AliasValueInfo{component, 0});
      pending.push_back(root);
      while (!pending.empty()) {
        Value value = pending.pop_back_val();
        AliasValueInfo valueInfo = info.lookup(value);
        RegType type = cast<RegType>(value.getType());
        if (type.getBaseGRF() >= 0) {
          int64_t origin =
              static_cast<int64_t>(type.getBaseGRF()) * 16 - valueInfo.offset;
          std::optional<int64_t> &knownOrigin = componentOrigins[component];
          if (knownOrigin && *knownOrigin != origin) {
            componentInconsistent[component] = true;
            knownOrigin.reset();
          } else if (!componentInconsistent[component]) {
            knownOrigin = origin;
          }
        }
        for (AliasEdge edge : graph.lookup(value)) {
          int64_t expected = valueInfo.offset + edge.offset;
          auto existing = info.find(edge.target);
          if (existing == info.end()) {
            info.try_emplace(edge.target, AliasValueInfo{component, expected});
            pending.push_back(edge.target);
            continue;
          }
          if (existing->second.component != component ||
              existing->second.offset != expected)
            componentInconsistent[component] = true;
        }
      }
    }
    return success();
  }

  bool overlaps(Value lhs, Value rhs) const {
    if (lhs == rhs)
      return true;
    RegType lhsType = dyn_cast<RegType>(lhs.getType());
    RegType rhsType = dyn_cast<RegType>(rhs.getType());
    if (!lhsType || !rhsType || lhsType.getWidthDwords() == 0 ||
        rhsType.getWidthDwords() == 0)
      return false;

    auto lhsIt = info.find(lhs);
    auto rhsIt = info.find(rhs);
    if (lhsIt != info.end() && rhsIt != info.end() &&
        lhsIt->second.component == rhsIt->second.component)
      return componentInconsistent[lhsIt->second.component] ||
             rangesOverlap(lhsIt->second.offset, lhsType.getWidthDwords(),
                           rhsIt->second.offset, rhsType.getWidthDwords());

    std::optional<int64_t> lhsStart = getAbsoluteStart(lhs, lhsType);
    std::optional<int64_t> rhsStart = getAbsoluteStart(rhs, rhsType);
    return lhsStart && rhsStart &&
           rangesOverlap(*lhsStart, lhsType.getWidthDwords(), *rhsStart,
                         rhsType.getWidthDwords());
  }

  SmallVector<Value, 8> getOverlappingValues(Value value) const {
    SmallVector<Value, 8> overlapping;
    for (Value candidate : values)
      if (overlaps(value, candidate))
        overlapping.push_back(candidate);
    return overlapping;
  }

private:
  static bool rangesOverlap(int64_t lhsStart, int64_t lhsWidth,
                            int64_t rhsStart, int64_t rhsWidth) {
    return lhsStart < rhsStart + rhsWidth && rhsStart < lhsStart + lhsWidth;
  }

  std::optional<int64_t> getAbsoluteStart(Value value, RegType type) const {
    auto valueIt = info.find(value);
    if (valueIt != info.end()) {
      if (componentInconsistent[valueIt->second.component])
        return std::nullopt;
      std::optional<int64_t> origin =
          componentOrigins[valueIt->second.component];
      if (origin)
        return *origin + valueIt->second.offset;
    }
    if (type.getBaseGRF() >= 0)
      return static_cast<int64_t>(type.getBaseGRF()) * 16;
    return std::nullopt;
  }

  SmallVector<Value> values;
  DenseMap<Value, SmallVector<AliasEdge, 4>> graph;
  DenseMap<Value, AliasValueInfo> info;
  SmallVector<std::optional<int64_t>> componentOrigins;
  SmallVector<bool> componentInconsistent;
};

static LogicalResult copyTupleElement(TupleFromElementsOp tuple, unsigned index,
                                      StringRef kind) {
  Value source = tuple.getElements()[index];
  RegType sourceType = cast<RegType>(source.getType());
  RegType copyType =
      RegType::get(tuple.getContext(), sourceType.getWidthDwords(), -1);
  OpBuilder builder(tuple);
  FailureOr<Value> copy =
      materializeRegisterCopy(builder, tuple.getLoc(), source, copyType, kind);
  if (failed(copy))
    return failure();
  tuple->setOperand(index, *copy);
  return success();
}

static std::optional<int64_t> getTupleElementOffset(TupleToElementsOp split,
                                                    Value element) {
  int64_t offset = 0;
  for (Value candidate : split.getElements()) {
    if (candidate == element)
      return offset;
    offset += cast<RegType>(candidate.getType()).getWidthDwords();
  }
  return std::nullopt;
}

static std::optional<int64_t> getGRFCount(func::FuncOp function) {
  IntegerAttr count =
      function->getAttrOfType<IntegerAttr>("xemachine.grf_count");
  return count ? std::optional<int64_t>(count.getInt()) : std::nullopt;
}

static LogicalResult repairTupleSplits(func::FuncOp function) {
  std::optional<int64_t> grfCount = getGRFCount(function);
  SmallVector<TupleToElementsOp> splits;
  function.walk([&](TupleToElementsOp split) { splits.push_back(split); });
  for (TupleToElementsOp split : splits) {
    RegType sourceType = cast<RegType>(split.getTuple().getType());
    if (sourceType.getBaseGRF() >= 0)
      continue;
    int64_t offset = 0;
    std::optional<int64_t> inferredSourceOrigin;
    OpBuilder builder(split);
    builder.setInsertionPointAfter(split);
    for (Value element : split.getElements()) {
      RegType elementType = cast<RegType>(element.getType());
      int64_t inferredOrigin =
          static_cast<int64_t>(elementType.getBaseGRF()) * 16 - offset;
      bool invalidOrigin =
          elementType.getBaseGRF() >= 0 &&
          (inferredOrigin < 0 ||
           (grfCount &&
            inferredOrigin + sourceType.getWidthDwords() > *grfCount * 16) ||
           (inferredSourceOrigin && *inferredSourceOrigin != inferredOrigin));
      if (invalidOrigin) {
        element.setType(RegType::get(function.getContext(),
                                     elementType.getWidthDwords(), -1));
        FailureOr<Value> copy = materializeRegisterCopy(
            builder, split.getLoc(), element, elementType, "tuple-element");
        if (failed(copy))
          return failure();
        element.replaceAllUsesExcept(*copy, copy->getDefiningOp());
      } else if (elementType.getBaseGRF() >= 0) {
        inferredSourceOrigin = inferredOrigin;
      }
      offset += elementType.getWidthDwords();
    }
  }
  return success();
}

static bool isAlignedTupleView(TupleFromElementsOp tuple,
                               std::optional<int64_t> grfCount) {
  TupleToElementsOp split;
  std::optional<int64_t> sourceShift;
  int64_t consumerOffset = 0;
  for (Value element : tuple.getElements()) {
    TupleToElementsOp elementSplit = element.getDefiningOp<TupleToElementsOp>();
    if (!elementSplit || (split && split != elementSplit))
      return false;
    split = elementSplit;
    std::optional<int64_t> sourceOffset = getTupleElementOffset(split, element);
    if (!sourceOffset || *sourceOffset < consumerOffset)
      return false;
    int64_t shift = *sourceOffset - consumerOffset;
    if (sourceShift && *sourceShift != shift)
      return false;
    sourceShift = shift;
    consumerOffset += cast<RegType>(element.getType()).getWidthDwords();
  }
  if (!split || !sourceShift || *sourceShift % 16 != 0)
    return false;
  RegType sourceType = cast<RegType>(split.getTuple().getType());
  RegType resultType = cast<RegType>(tuple.getTuple().getType());
  if (*sourceShift + resultType.getWidthDwords() > sourceType.getWidthDwords())
    return false;
  if (sourceType.getBaseGRF() >= 0 && resultType.getBaseGRF() < 0)
    return false;
  if (sourceType.getBaseGRF() < 0 && resultType.getBaseGRF() >= 0) {
    int64_t inferredOrigin =
        static_cast<int64_t>(resultType.getBaseGRF()) * 16 - *sourceShift;
    if (inferredOrigin < 0 ||
        (grfCount &&
         inferredOrigin + sourceType.getWidthDwords() > *grfCount * 16))
      return false;
  }
  return sourceType.getBaseGRF() < 0 || resultType.getBaseGRF() < 0 ||
         resultType.getBaseGRF() == sourceType.getBaseGRF() + *sourceShift / 16;
}

static LogicalResult repairTupleSlots(func::FuncOp function) {
  if (failed(repairTupleSplits(function)))
    return failure();
  std::optional<int64_t> grfCount = getGRFCount(function);
  SmallVector<TupleFromElementsOp> tuples;
  function.walk([&](TupleFromElementsOp tuple) {
    if (!isMarkedCopy(tuple))
      tuples.push_back(tuple);
  });

  DenseMap<Value, int64_t> anchorSlots;
  function.walk([&](TupleToElementsOp split) {
    int64_t offset = 0;
    for (Value element : split.getElements()) {
      anchorSlots.try_emplace(element, offset);
      offset += cast<RegType>(element.getType()).getWidthDwords();
    }
  });
  DenseSet<Value> consumed;
  for (TupleFromElementsOp tuple : tuples) {
    RegType tupleType = cast<RegType>(tuple.getTuple().getType());
    bool preserveView = isAlignedTupleView(tuple, grfCount);
    int64_t offset = 0;
    for (unsigned index = 0; index < tuple.getElements().size(); ++index) {
      Value element = tuple.getElements()[index];
      RegType elementType = cast<RegType>(element.getType());
      bool fixedConflict =
          elementType.getBaseGRF() >= 0 &&
          (tupleType.getBaseGRF() < 0 || offset % 16 != 0 ||
           elementType.getBaseGRF() != tupleType.getBaseGRF() + offset / 16);
      DenseMap<Value, int64_t>::const_iterator anchor =
          anchorSlots.find(element);
      bool slotMismatch =
          anchor != anchorSlots.end() && anchor->second != offset;
      bool sourceDrag = element.getDefiningOp<TupleToElementsOp>() != nullptr;
      bool needsCopy =
          !preserveView && (fixedConflict || slotMismatch || sourceDrag ||
                            consumed.contains(element));
      if (needsCopy && failed(copyTupleElement(tuple, index, "tuple-element")))
        return failure();
      element = tuple.getElements()[index];
      if (!preserveView) {
        anchorSlots[element] = offset;
        consumed.insert(element);
      }
      offset += elementType.getWidthDwords();
    }
  }
  return success();
}

static bool isAvailableBefore(Value value, Operation *operation,
                              DominanceInfo &dominance) {
  Operation *definition = value.getDefiningOp();
  return definition != operation &&
         dominance.properlyDominates(value, operation);
}

static bool storageIsLiveAfter(const WeightedOverlapSummary &summary,
                               Value value, Operation *operation,
                               DominanceInfo &dominance,
                               Operation *ignoredUser = nullptr) {
  return llvm::any_of(summary.getOverlappingValues(value), [&](Value alias) {
    return isAvailableBefore(alias, operation, dominance) &&
           isLiveAfter(alias, operation, dominance, ignoredUser);
  });
}

static LogicalResult repairUpdateTuples(func::FuncOp function) {
  SmallVector<UpdateTupleOp> updates;
  function.walk([&](UpdateTupleOp update) { updates.push_back(update); });
  for (UpdateTupleOp update : updates) {
    Value base = update.getBase();
    if (isMarkedCopy(base, "update-base") &&
        llvm::all_of(update.getUpdates(), [](Value replacement) {
          return isMarkedCopy(replacement, "update-value");
        }))
      continue;
    RegType baseType = cast<RegType>(base.getType());
    RegType copyType =
        RegType::get(function.getContext(), baseType.getWidthDwords(), -1);
    OpBuilder builder(update);
    FailureOr<Value> copy = materializeRegisterCopy(
        builder, update.getLoc(), base, copyType, "update-base");
    if (failed(copy))
      return failure();
    update->setOperand(0, *copy);
    for (unsigned index = 0; index < update.getUpdates().size(); ++index) {
      Value replacement = update.getUpdates()[index];
      RegType replacementType = cast<RegType>(replacement.getType());
      RegType replacementCopyType = RegType::get(
          function.getContext(), replacementType.getWidthDwords(), -1);
      FailureOr<Value> replacementCopy =
          materializeRegisterCopy(builder, update.getLoc(), replacement,
                                  replacementCopyType, "update-value");
      if (failed(replacementCopy))
        return failure();
      update->setOperand(index + 1, *replacementCopy);
    }
  }
  return success();
}

static bool isDefinedInside(Operation *container, Value value) {
  if (Operation *definition = value.getDefiningOp())
    return container->isProperAncestor(definition);
  BlockArgument argument = dyn_cast<BlockArgument>(value);
  if (!argument)
    return false;
  Operation *parent = argument.getOwner()->getParentOp();
  return parent && (parent == container || container->isProperAncestor(parent));
}

static bool isDefinedInside(Region &region, Value value) {
  if (Operation *definition = value.getDefiningOp())
    return region.isAncestor(definition->getParentRegion());
  BlockArgument argument = dyn_cast<BlockArgument>(value);
  return argument && region.isAncestor(argument.getOwner()->getParent());
}

static bool storageIsDefinedInside(const WeightedOverlapSummary &summary,
                                   Region &region, Value value,
                                   Operation *operation,
                                   DominanceInfo &dominance) {
  return llvm::all_of(summary.getOverlappingValues(value), [&](Value alias) {
    return !isAvailableBefore(alias, operation, dominance) ||
           isDefinedInside(region, alias);
  });
}

static bool storageIsDefinedInside(const WeightedOverlapSummary &summary,
                                   Operation *container, Value value,
                                   Operation *operation,
                                   DominanceInfo &dominance) {
  return llvm::all_of(summary.getOverlappingValues(value), [&](Value alias) {
    return !isAvailableBefore(alias, operation, dominance) ||
           isDefinedInside(container, alias);
  });
}

static bool storageHasUseInside(const WeightedOverlapSummary &summary,
                                Value value, Operation *container,
                                Operation *operation,
                                DominanceInfo &dominance) {
  return llvm::any_of(summary.getOverlappingValues(value), [&](Value alias) {
    if (!isAvailableBefore(alias, operation, dominance))
      return false;
    return llvm::any_of(alias.getUses(), [&](OpOperand &use) {
      Operation *owner = use.getOwner();
      return owner != container && container->isProperAncestor(owner);
    });
  });
}

template <typename IfOp>
static LogicalResult repairBranchYields(IfOp branch,
                                        const WeightedOverlapSummary &summary,
                                        DominanceInfo &dominance) {
  for (Region *region : {&branch.getThenRegion(), &branch.getElseRegion()}) {
    for (Block &block : *region) {
      YieldOp yield = dyn_cast<YieldOp>(block.getTerminator());
      if (!yield)
        continue;
      if (yield.getValues().size() != branch.getNumResults())
        return branch.emitError(
            "cannot prepare branch with mismatched yield/result counts");
      bool prepared =
          llvm::all_of(llvm::enumerate(yield.getValues()), [&](auto indexed) {
            return !isa<RegType>(
                       branch->getResult(indexed.index()).getType()) ||
                   isMarkedCopy(indexed.value(), "branch-yield");
          });
      if (prepared)
        continue;

      OpBuilder builder(yield);
      unsigned registerResultCount =
          llvm::count_if(branch->getResults(), [](Value result) {
            return isa<RegType>(result.getType());
          });
      if (registerResultCount == 1) {
        for (unsigned index = 0; index < yield.getValues().size(); ++index) {
          Value value = yield.getValues()[index];
          RegType resultType =
              dyn_cast<RegType>(branch->getResult(index).getType());
          bool localStorage =
              resultType &&
              storageIsDefinedInside(summary, *region, value, yield, dominance);
          if (!resultType ||
              (localStorage && !isa<ExecIfOp>(branch.getOperation())))
            continue;
          FailureOr<Value> destination = materializeRegisterCopy(
              builder, yield.getLoc(), value, resultType, "branch-yield");
          if (failed(destination))
            return failure();
          yield->setOperand(index, *destination);
        }
        continue;
      }

      SmallVector<Value, 4> snapshots(yield.getValues().size());
      for (unsigned index = 0; index < yield.getValues().size(); ++index) {
        Value value = yield.getValues()[index];
        RegType resultType =
            dyn_cast<RegType>(branch->getResult(index).getType());
        if (!resultType)
          continue;
        RegType snapshotType =
            RegType::get(branch.getContext(), resultType.getWidthDwords(), -1);
        FailureOr<Value> snapshot = materializeRegisterCopy(
            builder, yield.getLoc(), value, snapshotType, "branch-snapshot");
        if (failed(snapshot))
          return failure();
        snapshots[index] = *snapshot;
      }
      for (auto [index, snapshot] : llvm::enumerate(snapshots)) {
        if (!snapshot)
          continue;
        RegType resultType = cast<RegType>(branch->getResult(index).getType());
        FailureOr<Value> destination = materializeRegisterCopy(
            builder, yield.getLoc(), snapshot, resultType, "branch-yield");
        if (failed(destination))
          return failure();
        yield->setOperand(index, *destination);
      }
    }
  }
  return success();
}

static LogicalResult repairBranches(func::FuncOp function) {
  WeightedOverlapSummary summary;
  if (failed(summary.build(function)))
    return failure();
  DominanceInfo dominance(function);
  SmallVector<Operation *> branches;
  function.walk([&](Operation *operation) {
    if (isa<ExecIfOp, UniformIfOp>(operation))
      branches.push_back(operation);
  });
  for (Operation *operation : branches) {
    if (ExecIfOp branch = dyn_cast<ExecIfOp>(operation)) {
      if (failed(repairBranchYields(branch, summary, dominance)))
        return failure();
      continue;
    }
    if (failed(repairBranchYields(cast<UniformIfOp>(operation), summary,
                                  dominance)))
      return failure();
  }
  return success();
}

static LogicalResult repairLoopInits(func::FuncOp function,
                                     DominanceInfo &dominance,
                                     const WeightedOverlapSummary &summary) {
  SmallVector<UniformLoopOp> loops;
  function.walk([&](UniformLoopOp loop) { loops.push_back(loop); });
  for (UniformLoopOp loop : loops) {
    if (loop.getBody().empty())
      continue;
    Block &body = loop.getBody().front();
    if (loop.getInits().size() != body.getNumArguments())
      return loop.emitError(
          "cannot prepare loop with mismatched init/argument counts");
    SmallVector<Value, 4> previous;
    for (unsigned index = 0; index < loop.getInits().size(); ++index) {
      Value init = loop.getInits()[index];
      Value originalInit = init;
      RegType argumentType =
          dyn_cast<RegType>(body.getArgument(index).getType());
      if (!argumentType)
        continue;
      bool duplicate = llvm::any_of(previous, [&](Value earlier) {
        return summary.overlaps(earlier, init);
      });
      bool invariantRead =
          storageHasUseInside(summary, init, loop, loop, dominance);
      bool nestedRepetitive = false;
      for (Operation *parent = loop->getParentOp(); parent;
           parent = parent->getParentOp()) {
        UniformLoopOp enclosing = dyn_cast<UniformLoopOp>(parent);
        if (!enclosing)
          continue;
        nestedRepetitive =
            !storageIsDefinedInside(summary, enclosing, init, loop, dominance);
        break;
      }
      bool liveThrough = storageIsLiveAfter(summary, init, loop, dominance) ||
                         invariantRead || nestedRepetitive;
      if (duplicate || liveThrough) {
        OpBuilder builder(loop);
        FailureOr<Value> copy = materializeRegisterCopy(
            builder, loop.getLoc(), init, argumentType, "loop-init");
        if (failed(copy))
          return failure();
        loop->setOperand(index, *copy);
        init = *copy;
      }
      previous.push_back(originalInit);
    }
  }
  return success();
}

static bool isPreparedBackedge(Value carried) {
  Value snapshot = getCopiedSource(carried);
  return snapshot && getCopiedSource(snapshot);
}

static bool hasArgumentUseAfterDefinition(const WeightedOverlapSummary &summary,
                                          UniformLoopOp loop,
                                          BlockArgument argument,
                                          Operation *definition) {
  Block *block = argument.getOwner();
  Operation *top = definition;
  while (top && top->getBlock() != block)
    top = top->getParentOp();
  if (!top)
    return true;
  return llvm::any_of(summary.getOverlappingValues(argument), [&](Value alias) {
    if (alias != argument) {
      if (BlockArgument aliasArgument = dyn_cast<BlockArgument>(alias)) {
        if (aliasArgument.getOwner() != block)
          return false;
      } else {
        Operation *aliasDefinition = alias.getDefiningOp();
        if (!aliasDefinition || aliasDefinition == definition ||
            !loop->isProperAncestor(aliasDefinition))
          return false;
      }
    }
    return llvm::any_of(alias.getUses(), [&](OpOperand &use) {
      Operation *owner = use.getOwner();
      if (owner->hasTrait<OpTrait::IsTerminator>())
        return false;
      Operation *user = owner;
      while (user && user->getBlock() != block)
        user = user->getParentOp();
      return !user || (user != top && top->isBeforeInBlock(user));
    });
  });
}

static LogicalResult
repairLoopBackedges(func::FuncOp function,
                    const WeightedOverlapSummary &summary) {
  SmallVector<UniformLoopOp> loops;
  function.walk([&](UniformLoopOp loop) { loops.push_back(loop); });
  for (UniformLoopOp loop : loops) {
    if (loop.getBody().empty())
      continue;
    Block &body = loop.getBody().front();
    ContinueIfOp terminator = dyn_cast<ContinueIfOp>(body.getTerminator());
    if (!terminator)
      continue;
    if (terminator.getCarried().size() != body.getNumArguments())
      return loop.emitError(
          "cannot prepare loop with mismatched carried/argument counts");

    bool hazardous = false;
    for (auto [index, carried] : llvm::enumerate(terminator.getCarried())) {
      if (!isa<RegType>(carried.getType()))
        continue;
      if (carried != body.getArgument(index) && !isPreparedBackedge(carried)) {
        Operation *definition = carried.getDefiningOp();
        if (!definition || !loop->isProperAncestor(definition) ||
            hasArgumentUseAfterDefinition(summary, loop,
                                          body.getArgument(index), definition))
          hazardous = true;
      }
      for (unsigned other = 0; other < body.getNumArguments(); ++other) {
        if (other != index &&
            summary.overlaps(carried, body.getArgument(other)))
          hazardous = true;
      }
      for (Value earlier : terminator.getCarried().take_front(index))
        if (isa<RegType>(earlier.getType()) &&
            summary.overlaps(carried, earlier))
          hazardous = true;
    }
    if (!hazardous)
      continue;

    bool alreadyPrepared =
        llvm::all_of(terminator.getCarried(), [&](Value carried) {
          return !isa<RegType>(carried.getType()) ||
                 isPreparedBackedge(carried);
        });
    if (alreadyPrepared)
      continue;

    // Complete every read before emitting any write tied to a destination.
    SmallVector<Value, 4> snapshots(terminator.getCarried().size());
    OpBuilder builder(terminator);
    for (auto [index, carried] : llvm::enumerate(terminator.getCarried())) {
      RegType carriedType = dyn_cast<RegType>(carried.getType());
      if (!carriedType)
        continue;
      RegType snapshotType =
          RegType::get(function.getContext(), carriedType.getWidthDwords(), -1);
      FailureOr<Value> snapshot = materializeRegisterCopy(
          builder, terminator.getLoc(), carried, snapshotType, "loop-snapshot");
      if (failed(snapshot))
        return failure();
      snapshots[index] = *snapshot;
    }

    for (auto [index, snapshot] : llvm::enumerate(snapshots)) {
      if (!snapshot)
        continue;
      RegType destinationType =
          cast<RegType>(body.getArgument(index).getType());
      FailureOr<Value> destination =
          materializeRegisterCopy(builder, terminator.getLoc(), snapshot,
                                  destinationType, "loop-backedge");
      if (failed(destination))
        return failure();
      terminator->setOperand(index + 1, *destination);
    }
  }
  return success();
}

} // namespace

LogicalResult
inter::xemachine::prepareRegisterAllocation(func::FuncOp function) {
  if (!function)
    return failure();

  legalizeWideImmediates(function);
  if (failed(repairTupleSlots(function)))
    return failure();
  if (failed(repairUpdateTuples(function)))
    return failure();
  if (failed(repairBranches(function)))
    return failure();

  WeightedOverlapSummary initSummary;
  if (failed(initSummary.build(function)))
    return failure();
  {
    DominanceInfo dominance(function);
    if (failed(repairLoopInits(function, dominance, initSummary)))
      return failure();
  }

  WeightedOverlapSummary backedgeSummary;
  if (failed(backedgeSummary.build(function)))
    return failure();
  return repairLoopBackedges(function, backedgeSummary);
}
