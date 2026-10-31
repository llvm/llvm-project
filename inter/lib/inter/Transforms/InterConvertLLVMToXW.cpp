#include "inter/Dialect/Inter/IR/XW.h"
#include "inter/Dialect/XeMachine/IR/XeMachineABI.h"

#include "inter/Transforms/Passes.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Transforms/DialectConversion.h"
#include "llvm/Support/MathExtras.h"

#include <array>

namespace inter {
#define GEN_PASS_DEF_CONVERTLLVMTOXW
#include "inter/Transforms/Passes.h.inc"
} // namespace inter.

using namespace mlir;

namespace {

static void eraseDebugInfoModuleFlags(ModuleOp moduleOp) {
  SmallVector<LLVM::ModuleFlagsOp> debugFlags;
  for (LLVM::ModuleFlagsOp flags : moduleOp.getOps<LLVM::ModuleFlagsOp>()) {
    if (llvm::all_of(flags.getFlags(), [](Attribute flag) {
          auto moduleFlag = dyn_cast<LLVM::ModuleFlagAttr>(flag);
          return moduleFlag && moduleFlag.getKey() == "Debug Info Version";
        }))
      debugFlags.push_back(flags);
  }
  for (LLVM::ModuleFlagsOp flags : debugFlags)
    flags.erase();
}

static bool containsLLVMType(Type type) {
  if (type.getDialect().getNamespace() ==
      LLVM::LLVMDialect::getDialectNamespace())
    return true;
  if (auto function = dyn_cast<FunctionType>(type))
    return llvm::any_of(function.getInputs(), containsLLVMType) ||
           llvm::any_of(function.getResults(), containsLLVMType);
  if (auto tuple = dyn_cast<TupleType>(type))
    return llvm::any_of(tuple.getTypes(), containsLLVMType);
  return false;
}

static bool containsLLVMType(Attribute attribute) {
  if (attribute.getDialect().getNamespace() ==
      LLVM::LLVMDialect::getDialectNamespace())
    return true;
  if (auto type = dyn_cast<TypeAttr>(attribute))
    return containsLLVMType(type.getValue());
  if (auto array = dyn_cast<ArrayAttr>(attribute))
    return llvm::any_of(
        array, [](Attribute nested) { return containsLLVMType(nested); });
  if (auto dictionary = dyn_cast<DictionaryAttr>(attribute))
    return llvm::any_of(dictionary, [](NamedAttribute attr) {
      return containsLLVMType(attr.getValue());
    });
  return false;
}

static DictionaryAttr getPreservedAttributes(Operation *op, Builder &builder) {
  NamedAttrList imported;
  for (NamedAttribute attr : op->getDiscardableAttrs())
    if (!attr.getName().strref().starts_with("llvm.") &&
        !containsLLVMType(attr.getValue()) &&
        attr.getValue().getDialect().getNamespace() !=
            LLVM::LLVMDialect::getDialectNamespace())
      imported.set(attr.getName(), attr.getValue());
  return builder.getDictionaryAttr(imported);
}

static void preserveAttributes(Operation *source, Operation *target,
                               Builder &builder) {
  for (NamedAttribute attr : getPreservedAttributes(source, builder))
    target->setAttr(attr.getName(), attr.getValue());
}

static Type getPayloadType(Type type) {
  if (auto simd = dyn_cast<xw::SimdType>(type))
    return simd.getElementType();
  return type;
}

static FailureOr<int64_t> getFunctionSimdWidth(Operation *op) {
  FunctionOpInterface function = op->getParentOfType<FunctionOpInterface>();
  if (!function)
    return op->emitOpError("requires an enclosing function"), failure();
  IntegerAttr width = function->getAttrOfType<IntegerAttr>(
      xw::XWDialect::getSimdWidthAttrName());
  if (!width)
    return op->emitOpError("enclosing function is missing xw.simd_width"),
           failure();
  return width.getInt();
}

static std::optional<int64_t> getCardinality(Type type) {
  if (auto simd = dyn_cast<xw::SimdType>(type))
    return simd.getCardinality();
  if (auto mask = dyn_cast<xw::MaskType>(type))
    return mask.getCardinality();
  return std::nullopt;
}

static FailureOr<std::optional<int64_t>>
getExactCardinality(Operation *op, ValueRange values) {
  std::optional<int64_t> cardinality;
  for (Value value : values) {
    std::optional<int64_t> candidate = getCardinality(value.getType());
    if (!candidate)
      continue;
    if (cardinality && cardinality != candidate)
      return op->emitOpError("SIMD cardinalities must match; use xw.expand "
                             "explicitly"),
             failure();
    cardinality = candidate;
  }
  return cardinality;
}

static Value createSplat(OpBuilder &builder, Location loc, Value value,
                         int64_t cardinality) {
  Type type =
      xw::SimdType::get(value.getContext(), value.getType(), cardinality);
  return xw::SplatOp::create(builder, loc, type, value);
}

static Value splatToCardinality(Operation *anchor, Value value,
                                int64_t cardinality) {
  if (getCardinality(value.getType()))
    return value;
  OpBuilder builder(anchor);
  return createSplat(builder, anchor->getLoc(), value, cardinality);
}

static Value unwrapMaterializedShape(Value value) {
  UnrealizedConversionCastOp cast =
      value.getDefiningOp<UnrealizedConversionCastOp>();
  if (cast && cast->getNumOperands() == 1 && cast->getNumResults() == 1 &&
      getPayloadType(cast.getOperand(0).getType()) ==
          getPayloadType(cast.getResult(0).getType()))
    return cast.getOperand(0);
  return value;
}

static SmallVector<Value> unwrapMaterializedShapes(ValueRange values) {
  SmallVector<Value> unwrapped;
  unwrapped.reserve(values.size());
  for (Value value : values)
    unwrapped.push_back(unwrapMaterializedShape(value));
  return unwrapped;
}

static Operation *replaceDivergentIf(scf::IfOp op, Value condition) {
  OpBuilder builder(op);
  xw::WhereOp where =
      xw::WhereOp::create(builder, op.getLoc(), op.getResultTypes(), condition);
  where->setDiscardableAttrs(op->getDiscardableAttrDictionary());
  where.getThenRegion().takeBody(op.getThenRegion());
  where.getElseRegion().takeBody(op.getElseRegion());
  for (Region *region : {&where.getThenRegion(), &where.getElseRegion()}) {
    if (region->empty())
      continue;
    scf::YieldOp yield = cast<scf::YieldOp>(region->front().getTerminator());
    OpBuilder yieldBuilder(yield);
    xw::YieldOp::create(yieldBuilder, yield.getLoc(), yield.getOperands());
    yield.erase();
  }
  op->replaceAllUsesWith(where->getResults());
  op.erase();
  return where.getOperation();
}

static FailureOr<Operation *>
replaceOperationShape(Operation *op, ValueRange operands, Type resultType) {
  if (op->getNumRegions() != 0 || op->getNumSuccessors() != 0)
    return op->emitOpError(
               "shape reconciliation requires a regionless operation"),
           failure();
  Value oldResult = op->getResult(0);
  SmallVector<scf::IfOp> divergentIfs;
  for (OpOperand &use : oldResult.getUses()) {
    if (auto ifOp = dyn_cast<scf::IfOp>(use.getOwner())) {
      if (use.getOperandNumber() == 0) {
        divergentIfs.push_back(ifOp);
        continue;
      }
    }
    if (isa<scf::ConditionOp>(use.getOwner()) && use.getOperandNumber() == 0)
      return use.getOwner()->emitOpError(
                 "lane-varying loop conditions have no XW representation"),
             failure();
  }

  Operation *replacement = op->clone();
  replacement->setOperands(operands);
  replacement->getResult(0).setType(resultType);
  op->getBlock()->getOperations().insert(op->getIterator(), replacement);

  for (OpOperand &use : llvm::make_early_inc_range(oldResult.getUses())) {
    Operation *owner = use.getOwner();
    bool structuredCrossing =
        isa<scf::YieldOp, xw::YieldOp>(owner) ||
        (isa<scf::ConditionOp>(owner) && use.getOperandNumber() != 0) ||
        (isa<scf::ForOp>(owner) && use.getOperandNumber() >= 3) ||
        isa<scf::WhileOp>(owner);
    if (!structuredCrossing)
      continue;
    OpBuilder builder(owner);
    Value materialized = UnrealizedConversionCastOp::create(
                             builder, owner->getLoc(), oldResult.getType(),
                             replacement->getResult(0))
                             .getResult(0);
    use.set(materialized);
  }

  for (scf::IfOp ifOp : divergentIfs)
    replaceDivergentIf(ifOp, replacement->getResult(0));
  op->replaceAllUsesWith(replacement->getResults());
  op->erase();
  return replacement;
}

static FailureOr<Type> joinStructuredShape(Operation *op, TypeRange types,
                                           const Twine &name) {
  if (types.empty())
    return op->emitOpError() << name << " have no boundary types", failure();
  Type payload = getPayloadType(types.front());
  std::optional<int64_t> cardinality = getCardinality(types.front());
  for (Type type : types.drop_front()) {
    if (getPayloadType(type) != payload)
      return op->emitOpError() << name << " have incompatible payload types "
                               << payload << " and " << getPayloadType(type),
             failure();
    std::optional<int64_t> candidate = getCardinality(type);
    if (cardinality && candidate && cardinality != candidate)
      return op->emitOpError() << name << " have differing SIMD cardinalities "
                               << *cardinality << " and " << *candidate,
             failure();
    if (candidate)
      cardinality = candidate;
  }
  if (!cardinality)
    return payload;
  if (isa<xw::MaskType>(payload)) {
    if (!llvm::all_of(types, [&](Type type) { return type == types.front(); }))
      return op->emitOpError() << name << " cannot mix bare and XW mask values",
             failure();
    return types.front();
  }
  return xw::SimdType::get(op->getContext(), payload, *cardinality);
}

static Value adaptStructuredValue(OpBuilder &builder, Location loc, Value value,
                                  Type type) {
  value = unwrapMaterializedShape(value);
  if (value.getType() == type)
    return value;
  if (isa<xw::SimdType>(type))
    return xw::SplatOp::create(builder, loc, type, value);
  return UnrealizedConversionCastOp::create(builder, loc, type, value)
      .getResult(0);
}

static LogicalResult reconcileIfShape(scf::IfOp op, bool &changed) {
  if (op.getNumResults() == 0)
    return success();
  SmallVector<Type> resultTypes(op.getResultTypes());
  SmallVector<Type> joinedTypes;
  joinedTypes.reserve(op.getNumResults());
  scf::YieldOp thenYield = op.thenYield();
  scf::YieldOp elseYield = op.elseYield();
  for (unsigned index : llvm::seq<unsigned>(op.getNumResults())) {
    std::array<Type, 3> types = {
        resultTypes[index],
        unwrapMaterializedShape(thenYield.getOperand(index)).getType(),
        unwrapMaterializedShape(elseYield.getOperand(index)).getType()};
    FailureOr<Type> joined =
        joinStructuredShape(op, types, Twine("if result #") + Twine(index));
    if (failed(joined))
      return failure();
    joinedTypes.push_back(*joined);
  }
  if (joinedTypes == resultTypes &&
      llvm::all_of(llvm::seq<unsigned>(op.getNumResults()), [&](unsigned i) {
        return thenYield.getOperand(i).getType() == joinedTypes[i] &&
               elseYield.getOperand(i).getType() == joinedTypes[i];
      }))
    return success();

  scf::IfOp replacement = cast<scf::IfOp>(op->clone());
  scf::YieldOp replacementThen = replacement.thenYield();
  scf::YieldOp replacementElse = replacement.elseYield();
  for (unsigned index : llvm::seq<unsigned>(op.getNumResults())) {
    replacement.getResult(index).setType(joinedTypes[index]);
    for (scf::YieldOp yield : {replacementThen, replacementElse}) {
      OpBuilder builder(yield);
      yield->setOperand(index, adaptStructuredValue(builder, yield.getLoc(),
                                                    yield.getOperand(index),
                                                    joinedTypes[index]));
    }
  }
  op->getBlock()->getOperations().insert(op->getIterator(), replacement);
  op->replaceAllUsesWith(replacement->getResults());
  op.erase();
  changed = true;
  return success();
}

static LogicalResult reconcileWhereShape(xw::WhereOp op, bool &changed) {
  SmallVector<Type> resultTypes(op.getResultTypes());
  SmallVector<Type> joinedTypes;
  joinedTypes.reserve(op.getNumResults());
  xw::YieldOp thenYield =
      cast<xw::YieldOp>(op.getThenRegion().front().getTerminator());
  xw::YieldOp elseYield =
      cast<xw::YieldOp>(op.getElseRegion().front().getTerminator());
  for (unsigned index : llvm::seq<unsigned>(op.getNumResults())) {
    std::array<Type, 3> types = {
        resultTypes[index],
        unwrapMaterializedShape(thenYield.getOperand(index)).getType(),
        unwrapMaterializedShape(elseYield.getOperand(index)).getType()};
    FailureOr<Type> joined =
        joinStructuredShape(op, types, Twine("where result #") + Twine(index));
    if (failed(joined))
      return failure();
    joinedTypes.push_back(*joined);
  }
  if (joinedTypes == resultTypes &&
      llvm::all_of(llvm::seq<unsigned>(op.getNumResults()), [&](unsigned i) {
        return thenYield.getOperand(i).getType() == joinedTypes[i] &&
               elseYield.getOperand(i).getType() == joinedTypes[i];
      }))
    return success();

  xw::WhereOp replacement = cast<xw::WhereOp>(op->clone());
  xw::YieldOp replacementThen =
      cast<xw::YieldOp>(replacement.getThenRegion().front().getTerminator());
  xw::YieldOp replacementElse =
      cast<xw::YieldOp>(replacement.getElseRegion().front().getTerminator());
  for (unsigned index : llvm::seq<unsigned>(op.getNumResults())) {
    replacement.getResult(index).setType(joinedTypes[index]);
    for (xw::YieldOp yield : {replacementThen, replacementElse}) {
      OpBuilder builder(yield);
      yield->setOperand(index, adaptStructuredValue(builder, yield.getLoc(),
                                                    yield.getOperand(index),
                                                    joinedTypes[index]));
    }
  }
  op->getBlock()->getOperations().insert(op->getIterator(), replacement);
  op->replaceAllUsesWith(replacement->getResults());
  op.erase();
  changed = true;
  return success();
}

static LogicalResult reconcileForShape(scf::ForOp op, bool &changed) {
  SmallVector<Type> joinedTypes;
  joinedTypes.reserve(op.getNumResults());
  scf::YieldOp yield = cast<scf::YieldOp>(op.getBody()->getTerminator());
  for (unsigned index : llvm::seq<unsigned>(op.getNumResults())) {
    std::array<Type, 4> types = {
        unwrapMaterializedShape(op.getInitArgs()[index]).getType(),
        op.getRegionIterArgs()[index].getType(),
        unwrapMaterializedShape(yield.getOperand(index)).getType(),
        op.getResult(index).getType()};
    FailureOr<Type> joined = joinStructuredShape(
        op, types, Twine("for loop-carried value #") + Twine(index));
    if (failed(joined))
      return failure();
    joinedTypes.push_back(*joined);
  }
  bool needsRewrite = llvm::any_of(
      llvm::seq<unsigned>(op.getNumResults()), [&](unsigned index) {
        return op.getInitArgs()[index].getType() != joinedTypes[index] ||
               op.getRegionIterArgs()[index].getType() != joinedTypes[index] ||
               yield.getOperand(index).getType() != joinedTypes[index] ||
               op.getResult(index).getType() != joinedTypes[index];
      });
  if (!needsRewrite)
    return success();

  scf::ForOp replacement = cast<scf::ForOp>(op->clone());
  scf::YieldOp replacementYield =
      cast<scf::YieldOp>(replacement.getBody()->getTerminator());
  OpBuilder outerBuilder(op);
  for (unsigned index : llvm::seq<unsigned>(op.getNumResults())) {
    Value init = adaptStructuredValue(outerBuilder, op.getLoc(),
                                      replacement.getInitArgs()[index],
                                      joinedTypes[index]);
    replacement->setOperand(3 + index, init);
    replacement.getRegionIterArgs()[index].setType(joinedTypes[index]);
    replacement.getResult(index).setType(joinedTypes[index]);
    OpBuilder yieldBuilder(replacementYield);
    replacementYield->setOperand(
        index, adaptStructuredValue(yieldBuilder, replacementYield.getLoc(),
                                    replacementYield.getOperand(index),
                                    joinedTypes[index]));
  }
  op->getBlock()->getOperations().insert(op->getIterator(), replacement);
  op->replaceAllUsesWith(replacement->getResults());
  op.erase();
  changed = true;
  return success();
}

static LogicalResult reconcileWhileShape(scf::WhileOp op, bool &changed) {
  Block &before = op.getBefore().front();
  Block &after = op.getAfter().front();
  scf::ConditionOp condition = cast<scf::ConditionOp>(before.getTerminator());
  scf::YieldOp yield = cast<scf::YieldOp>(after.getTerminator());
  SmallVector<Type> joinedTypes;
  joinedTypes.reserve(op.getNumResults());
  for (unsigned index : llvm::seq<unsigned>(op.getNumResults())) {
    std::array<Type, 6> types = {
        unwrapMaterializedShape(op.getInits()[index]).getType(),
        before.getArgument(index).getType(),
        unwrapMaterializedShape(condition.getArgs()[index]).getType(),
        after.getArgument(index).getType(),
        unwrapMaterializedShape(yield.getOperand(index)).getType(),
        op.getResult(index).getType()};
    FailureOr<Type> joined = joinStructuredShape(
        op, types, Twine("while loop-carried value #") + Twine(index));
    if (failed(joined))
      return failure();
    joinedTypes.push_back(*joined);
  }
  bool needsRewrite = llvm::any_of(
      llvm::seq<unsigned>(op.getNumResults()), [&](unsigned index) {
        return op.getInits()[index].getType() != joinedTypes[index] ||
               before.getArgument(index).getType() != joinedTypes[index] ||
               condition.getArgs()[index].getType() != joinedTypes[index] ||
               after.getArgument(index).getType() != joinedTypes[index] ||
               yield.getOperand(index).getType() != joinedTypes[index] ||
               op.getResult(index).getType() != joinedTypes[index];
      });
  if (!needsRewrite)
    return success();

  scf::WhileOp replacement = cast<scf::WhileOp>(op->clone());
  Block &replacementBefore = replacement.getBefore().front();
  Block &replacementAfter = replacement.getAfter().front();
  scf::ConditionOp replacementCondition =
      cast<scf::ConditionOp>(replacementBefore.getTerminator());
  scf::YieldOp replacementYield =
      cast<scf::YieldOp>(replacementAfter.getTerminator());
  OpBuilder outerBuilder(op);
  for (unsigned index : llvm::seq<unsigned>(op.getNumResults())) {
    Value init =
        adaptStructuredValue(outerBuilder, op.getLoc(),
                             replacement.getInits()[index], joinedTypes[index]);
    replacement->setOperand(index, init);
    replacementBefore.getArgument(index).setType(joinedTypes[index]);
    replacementAfter.getArgument(index).setType(joinedTypes[index]);
    replacement.getResult(index).setType(joinedTypes[index]);
    OpBuilder conditionBuilder(replacementCondition);
    replacementCondition.getArgsMutable().slice(index, 1).assign(
        adaptStructuredValue(conditionBuilder, replacementCondition.getLoc(),
                             replacementCondition.getArgs()[index],
                             joinedTypes[index]));
    OpBuilder yieldBuilder(replacementYield);
    replacementYield->setOperand(
        index, adaptStructuredValue(yieldBuilder, replacementYield.getLoc(),
                                    replacementYield.getOperand(index),
                                    joinedTypes[index]));
  }
  op->getBlock()->getOperations().insert(op->getIterator(), replacement);
  op->replaceAllUsesWith(replacement->getResults());
  op.erase();
  changed = true;
  return success();
}

static LogicalResult reconcileStructuredShapes(ModuleOp module, bool &changed) {
  SmallVector<Operation *> candidates;
  module.walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (isa<scf::IfOp, xw::WhereOp, scf::ForOp, scf::WhileOp>(op))
      candidates.push_back(op);
  });
  for (Operation *candidate : candidates) {
    if (auto ifOp = dyn_cast<scf::IfOp>(candidate)) {
      if (failed(reconcileIfShape(ifOp, changed)))
        return failure();
    } else if (auto where = dyn_cast<xw::WhereOp>(candidate)) {
      if (failed(reconcileWhereShape(where, changed)))
        return failure();
    } else if (auto forOp = dyn_cast<scf::ForOp>(candidate)) {
      if (failed(reconcileForShape(forOp, changed)))
        return failure();
    } else if (failed(reconcileWhileShape(cast<scf::WhileOp>(candidate),
                                          changed))) {
      return failure();
    }
  }
  return success();
}

static LogicalResult reconcileMaterializedShapes(ModuleOp module) {
  module.walk([](scf::IfOp op) { op->removeAttr("xw.boundary_converted"); });

  SmallVector<UnrealizedConversionCastOp> casts;
  module.walk([&](UnrealizedConversionCastOp cast) { casts.push_back(cast); });
  SmallVector<UnrealizedConversionCastOp> remainingCasts;
  reconcileUnrealizedCasts(casts, &remainingCasts);
  casts = std::move(remainingCasts);
  for (UnrealizedConversionCastOp cast : casts) {
    if (cast->getNumOperands() != 1 || cast->getNumResults() != 1 ||
        getPayloadType(cast.getOperand(0).getType()) !=
            getPayloadType(cast.getResult(0).getType()))
      return cast.emitOpError(
          "non-shape unrealized conversion survived the LLVM boundary");
  }

  auto normalizeMixed = [&]() -> LogicalResult {
    SmallVector<Operation *> candidates;
    module.walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (isa<xw::CmpIOp, xw::CmpFOp, xw::PtrCmpOp, xw::SelectOp>(op))
        candidates.push_back(op);
    });
    for (Operation *op : candidates) {
      if (isa<xw::CmpIOp, xw::CmpFOp, xw::PtrCmpOp>(op)) {
        SmallVector<Value> operands =
            unwrapMaterializedShapes(op->getOperands());
        FailureOr<std::optional<int64_t>> cardinality =
            getExactCardinality(op, operands);
        if (failed(cardinality))
          return failure();
        if (!*cardinality)
          continue;
        OpBuilder builder(op);
        for (unsigned index : llvm::seq<unsigned>(op->getNumOperands())) {
          Value operand = operands[index];
          if (getCardinality(operand.getType()))
            continue;
          operands[index] =
              createSplat(builder, op->getLoc(), operand, **cardinality);
        }
        if (failed(replaceOperationShape(
                op, operands,
                xw::MaskType::get(op->getContext(), **cardinality))))
          return failure();
        continue;
      }

      auto select = dyn_cast<xw::SelectOp>(op);
      if (!select)
        continue;
      SmallVector<Value> operands = unwrapMaterializedShapes(op->getOperands());
      FailureOr<std::optional<int64_t>> armCardinality =
          getExactCardinality(op, ValueRange(operands).drop_front());
      if (failed(armCardinality))
        return failure();
      std::optional<int64_t> conditionCardinality =
          getCardinality(operands.front().getType());
      if (conditionCardinality && *armCardinality &&
          conditionCardinality != *armCardinality) {
        op->emitOpError("select mask and arm cardinalities must match; use "
                        "xw.expand explicitly");
        return failure();
      }
      std::optional<int64_t> cardinality =
          conditionCardinality ? conditionCardinality : *armCardinality;
      if (!cardinality)
        continue;
      OpBuilder builder(op);
      for (unsigned index : {1u, 2u}) {
        Value operand = operands[index];
        if (getCardinality(operand.getType()))
          continue;
        operands[index] =
            createSplat(builder, op->getLoc(), operand, *cardinality);
      }
      if (operands[1].getType() != operands[2].getType()) {
        op->emitOpError("select arms must have the same converted type");
        return failure();
      }
      if (failed(replaceOperationShape(op, operands, operands[1].getType())))
        return failure();
    }
    return success();
  };
  if (failed(normalizeMixed()))
    return failure();

  bool changed;
  do {
    changed = false;
    SmallVector<Operation *> candidates;
    module.walk<WalkOrder::PostOrder>([&](Operation *op) {
      if (op->getNumResults() != 0 &&
          (isa<xw::CmpIOp, xw::CmpFOp, xw::PtrCmpOp, xw::PtrAddOp, xw::FreezeOp,
               xw::BitcastOp>(op) ||
           op->hasTrait<OpTrait::Elementwise>()))
        candidates.push_back(op);
    });
    for (Operation *op : candidates) {
      SmallVector<Value> operands = unwrapMaterializedShapes(op->getOperands());
      FailureOr<std::optional<int64_t>> cardinality =
          getExactCardinality(op, operands);
      if (failed(cardinality))
        return failure();
      if (!*cardinality || op->getNumResults() == 0)
        continue;

      Type resultType = op->getResult(0).getType();
      Type replacement;
      if (isa<xw::CmpIOp, xw::CmpFOp, xw::PtrCmpOp>(op))
        replacement = xw::MaskType::get(op->getContext(), **cardinality);
      else if (auto ptradd = dyn_cast<xw::PtrAddOp>(op)) {
        Type pointer = getPayloadType(ptradd.getBase().getType());
        replacement =
            xw::SimdType::get(op->getContext(), pointer, **cardinality);
      } else if (isa<xw::FreezeOp, xw::BitcastOp>(op) ||
                 op->hasTrait<OpTrait::Elementwise>()) {
        replacement = xw::SimdType::get(
            op->getContext(), getPayloadType(resultType), **cardinality);
      }
      if (replacement && replacement != resultType) {
        if (failed(replaceOperationShape(op, operands, replacement)))
          return failure();
        changed = true;
      }
    }
    if (failed(reconcileStructuredShapes(module, changed)))
      return failure();
  } while (changed);

  if (failed(normalizeMixed()))
    return failure();

  casts.clear();
  module.walk([&](UnrealizedConversionCastOp cast) { casts.push_back(cast); });
  for (UnrealizedConversionCastOp cast : casts) {
    Value replacement = cast.getOperand(0);
    if (!getCardinality(replacement.getType()))
      if (auto resultType =
              dyn_cast<xw::SimdType>(cast.getResult(0).getType())) {
        OpBuilder builder(cast);
        replacement = xw::SplatOp::create(builder, cast.getLoc(), resultType,
                                          replacement);
      }
    cast.getResult(0).replaceAllUsesWith(replacement);
    cast.erase();
  }

  SmallVector<xw::SplatOp> redundantSplats;
  module.walk([&](xw::SplatOp splat) {
    if (splat.getSource().getType() == splat.getResult().getType())
      redundantSplats.push_back(splat);
  });
  for (xw::SplatOp splat : redundantSplats) {
    splat.getResult().replaceAllUsesWith(splat.getSource());
    splat.erase();
  }

  module->removeAttr(LLVM::LLVMDialect::getTargetTripleAttrName());
  module->removeAttr("llvm.module_asm");
  module->removeAttr("dlti.dl_spec");

  Operation *illegal = nullptr;
  module.walk([&](Operation *op) {
    auto hasLLVMType = [](Type type) { return containsLLVMType(type); };
    if (isa<UnrealizedConversionCastOp>(op) ||
        op->getName().getDialectNamespace() ==
            LLVM::LLVMDialect::getDialectNamespace() ||
        op->getName().getDialectNamespace() ==
            cf::ControlFlowDialect::getDialectNamespace() ||
        llvm::any_of(op->getOperandTypes(), hasLLVMType) ||
        llvm::any_of(op->getResultTypes(), hasLLVMType) ||
        llvm::any_of(op->getAttrs(), [](NamedAttribute attr) {
          return containsLLVMType(attr.getValue());
        })) {
      illegal = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (illegal)
    return illegal->emitOpError("LLVM type survived the closed XW boundary");
  return success();
}

class LLVMToXWTypeConverter final : public TypeConverter {
public:
  explicit LLVMToXWTypeConverter(MLIRContext *context) {
    addConversion([](Type type) { return type; });
    addConversion([context](LLVM::LLVMPointerType pointer) -> Type {
      std::optional<inter::xemachine::KernelAddressSpace> decoded =
          inter::xemachine::KernelABI::get().decodeAddressSpace(
              pointer.getAddressSpace());
      if (!decoded)
        return {};
      Attribute addressSpace;
      switch (*decoded) {
      case inter::xemachine::KernelAddressSpace::privateSpace:
        addressSpace = xw::PrivateAddressSpaceAttr::get(context);
        break;
      case inter::xemachine::KernelAddressSpace::global:
        addressSpace = xw::GlobalAddressSpaceAttr::get(context);
        break;
      case inter::xemachine::KernelAddressSpace::constant:
        addressSpace = xw::ConstantAddressSpaceAttr::get(context);
        break;
      case inter::xemachine::KernelAddressSpace::local:
        addressSpace = xw::LocalAddressSpaceAttr::get(context);
        break;
      case inter::xemachine::KernelAddressSpace::generic:
        addressSpace = xw::GenericAddressSpaceAttr::get(context);
        break;
      }
      return xw::PtrType::get(context, addressSpace);
    });
    addConversion([&](LLVM::LLVMArrayType array) -> Type {
      Type element = convertType(array.getElementType());
      if (!element)
        return {};
      return VectorType::get({static_cast<int64_t>(array.getNumElements())},
                             element);
    });
    addConversion([&](LLVM::LLVMStructType structure) -> Type {
      if (structure.isOpaque())
        return {};
      SmallVector<Type> elements;
      if (failed(convertTypes(structure.getBody(), elements)))
        return {};
      return TupleType::get(context, elements);
    });
    addConversion([&](LLVM::LLVMFunctionType function) -> Type {
      SmallVector<Type> inputs;
      SmallVector<Type> results;
      if (function.isVarArg() ||
          failed(convertTypes(function.getParams(), inputs)) ||
          failed(convertTypes(function.getReturnTypes(), results)))
        return {};
      return FunctionType::get(context, inputs, results);
    });
    addConversion([context](LLVM::LLVMVoidType) -> Type {
      return NoneType::get(context);
    });
    addConversion([](LLVM::LLVMByteType byte) -> Type {
      return IntegerType::get(byte.getContext(), byte.getBitWidth());
    });
    addConversion([this](VectorType vector) -> Type {
      Type element = convertType(vector.getElementType());
      if (!element)
        return {};
      return VectorType::get(vector.getShape(), element,
                             vector.getScalableDims());
    });

    addSourceMaterialization(materializeCast);
    addTargetMaterialization(materializeCast);
  }

private:
  static Value materializeCast(OpBuilder &builder, Type type, ValueRange inputs,
                               Location loc) {
    if (inputs.size() != 1)
      return {};
    return UnrealizedConversionCastOp::create(builder, loc, type, inputs)
        .getResult(0);
  }
};

enum class BuiltinKind {
  LaneId,
  SubgroupId,
  GlobalId,
  LocalId,
  GroupId,
  GlobalSize,
  LocalSize,
  NumGroups,
  LaunchGridSize,
  LaunchBlockSize,
  Barrier,
  AtomicAdd,
};

static std::optional<BuiltinKind> classifyBuiltin(StringRef symbol) {
  return StringSwitch<std::optional<BuiltinKind>>(symbol)
      .Cases({"_Z22get_sub_group_local_idv", "_Z22get_sub_group_local_id",
              "get_sub_group_local_id"},
             BuiltinKind::LaneId)
      .Cases(
          {"_Z16get_sub_group_idv", "_Z16get_sub_group_id", "get_sub_group_id"},
          BuiltinKind::SubgroupId)
      .Cases({"_Z13get_global_idj", "_Z13get_global_idm", "get_global_id"},
             BuiltinKind::GlobalId)
      .Cases({"_Z12get_local_idj", "_Z12get_local_idm", "get_local_id"},
             BuiltinKind::LocalId)
      .Cases({"_Z12get_group_idj", "_Z12get_group_idm", "get_group_id"},
             BuiltinKind::GroupId)
      .Cases(
          {"_Z15get_global_sizej", "_Z15get_global_sizem", "get_global_size"},
          BuiltinKind::GlobalSize)
      .Cases({"_Z14get_local_sizej", "_Z14get_local_sizem", "get_local_size"},
             BuiltinKind::LocalSize)
      .Cases({"_Z14get_num_groupsj", "_Z14get_num_groupsm", "get_num_groups"},
             BuiltinKind::NumGroups)
      .Cases({"__builtin_IB_get_global_size", "__spirv_BuiltInGlobalSize"},
             BuiltinKind::LaunchGridSize)
      .Cases({"__builtin_IB_get_local_size", "__spirv_BuiltInWorkgroupSize"},
             BuiltinKind::LaunchBlockSize)
      .Cases({"_Z7barrierj", "barrier"}, BuiltinKind::Barrier)
      .Cases(
          {"_Z12atomic_addPVU3AS1ii", "_Z10atomic_addPU3AS1Vjj", "atomic_add"},
          BuiltinKind::AtomicAdd)
      .Default(std::nullopt);
}

static arith::CmpIPredicate convertPredicate(LLVM::ICmpPredicate predicate) {
  switch (predicate) {
  case LLVM::ICmpPredicate::eq:
    return arith::CmpIPredicate::eq;
  case LLVM::ICmpPredicate::ne:
    return arith::CmpIPredicate::ne;
  case LLVM::ICmpPredicate::slt:
    return arith::CmpIPredicate::slt;
  case LLVM::ICmpPredicate::sle:
    return arith::CmpIPredicate::sle;
  case LLVM::ICmpPredicate::sgt:
    return arith::CmpIPredicate::sgt;
  case LLVM::ICmpPredicate::sge:
    return arith::CmpIPredicate::sge;
  case LLVM::ICmpPredicate::ult:
    return arith::CmpIPredicate::ult;
  case LLVM::ICmpPredicate::ule:
    return arith::CmpIPredicate::ule;
  case LLVM::ICmpPredicate::ugt:
    return arith::CmpIPredicate::ugt;
  case LLVM::ICmpPredicate::uge:
    return arith::CmpIPredicate::uge;
  }
  llvm_unreachable("unknown LLVM integer comparison predicate");
}

static arith::CmpFPredicate convertPredicate(LLVM::FCmpPredicate predicate) {
  switch (predicate) {
  case LLVM::FCmpPredicate::_false:
    return arith::CmpFPredicate::AlwaysFalse;
  case LLVM::FCmpPredicate::oeq:
    return arith::CmpFPredicate::OEQ;
  case LLVM::FCmpPredicate::ogt:
    return arith::CmpFPredicate::OGT;
  case LLVM::FCmpPredicate::oge:
    return arith::CmpFPredicate::OGE;
  case LLVM::FCmpPredicate::olt:
    return arith::CmpFPredicate::OLT;
  case LLVM::FCmpPredicate::ole:
    return arith::CmpFPredicate::OLE;
  case LLVM::FCmpPredicate::one:
    return arith::CmpFPredicate::ONE;
  case LLVM::FCmpPredicate::ord:
    return arith::CmpFPredicate::ORD;
  case LLVM::FCmpPredicate::ueq:
    return arith::CmpFPredicate::UEQ;
  case LLVM::FCmpPredicate::ugt:
    return arith::CmpFPredicate::UGT;
  case LLVM::FCmpPredicate::uge:
    return arith::CmpFPredicate::UGE;
  case LLVM::FCmpPredicate::ult:
    return arith::CmpFPredicate::ULT;
  case LLVM::FCmpPredicate::ule:
    return arith::CmpFPredicate::ULE;
  case LLVM::FCmpPredicate::une:
    return arith::CmpFPredicate::UNE;
  case LLVM::FCmpPredicate::uno:
    return arith::CmpFPredicate::UNO;
  case LLVM::FCmpPredicate::_true:
    return arith::CmpFPredicate::AlwaysTrue;
  }
  llvm_unreachable("unknown LLVM floating-point comparison predicate");
}

static arith::IntegerOverflowFlags
convertOverflowFlags(LLVM::IntegerOverflowFlags flags) {
  arith::IntegerOverflowFlags converted = arith::IntegerOverflowFlags::none;
  if (LLVM::bitEnumContainsAny(flags, LLVM::IntegerOverflowFlags::nsw))
    converted = converted | arith::IntegerOverflowFlags::nsw;
  if (LLVM::bitEnumContainsAny(flags, LLVM::IntegerOverflowFlags::nuw))
    converted = converted | arith::IntegerOverflowFlags::nuw;
  return converted;
}

static SmallVector<Value> unwrapOperands(ValueRange operands) {
  SmallVector<Value> unwrapped;
  unwrapped.reserve(operands.size());
  for (Value operand : operands) {
    if (UnrealizedConversionCastOp cast =
            operand.getDefiningOp<UnrealizedConversionCastOp>();
        cast && cast->getNumOperands() == 1)
      operand = cast->getOperand(0);
    unwrapped.push_back(operand);
  }
  return unwrapped;
}

static Type distributeType(Type type, ValueRange operands) {
  for (Value operand : operands)
    if (auto simd = dyn_cast<xw::SimdType>(operand.getType()))
      return xw::SimdType::get(type.getContext(), type, simd.getCardinality());
  return type;
}

static FailureOr<Type> convertResultType(Operation *op,
                                         const TypeConverter &converter,
                                         ConversionPatternRewriter &rewriter) {
  Type type = converter.convertType(op->getResult(0).getType());
  if (!type) {
    (void)rewriter.notifyMatchFailure(op, "result type has no XW conversion");
    return failure();
  }
  return type;
}

static LogicalResult replaceConverted(Operation *source, Operation *target,
                                      ConversionPatternRewriter &rewriter) {
  preserveAttributes(source, target, rewriter);
  rewriter.replaceOp(source,
                     target->getResults().take_front(source->getNumResults()));
  return success();
}

class RejectLLVMUndef final : public OpConversionPattern<LLVM::UndefOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(LLVM::UndefOp op, OpAdaptor,
                                ConversionPatternRewriter &) const override {
    return op.emitOpError("undef has no sound XW representation");
  }
};

class ConvertLLVMPoison final : public OpConversionPattern<LLVM::PoisonOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::PoisonOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type type = getTypeConverter()->convertType(op.getType());
    if (!type)
      return rewriter.notifyMatchFailure(op,
                                         "poison type has no XW conversion");
    rewriter.replaceOpWithNewOp<ub::PoisonOp>(
        op, type, ub::PoisonAttr::get(op.getContext()));
    return success();
  }
};

class ConvertLLVMFreeze final : public OpConversionPattern<LLVM::FreezeOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::FreezeOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands = unwrapOperands(adaptor.getOperands());
    Type type = getTypeConverter()->convertType(op.getType());
    if (operands.size() != 1)
      return rewriter.notifyMatchFailure(op, "freeze requires one operand");
    if (!type || type != operands.front().getType())
      return rewriter.notifyMatchFailure(
          op, "freeze source and result must have the same converted shape");
    xw::FreezeOp converted =
        xw::FreezeOp::create(rewriter, op.getLoc(), type, operands.front());
    return replaceConverted(op, converted, rewriter);
  }
};

class RejectLLVMFence final : public OpConversionPattern<LLVM::FenceOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult matchAndRewrite(LLVM::FenceOp op, OpAdaptor,
                                ConversionPatternRewriter &) const override {
    return op.emitOpError(
        "LLVM fence ordering and scope have no exact XW representation");
  }
};

class ConvertLLVMGlobal final : public OpConversionPattern<LLVM::GlobalOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::GlobalOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getAddrSpace() !=
        static_cast<unsigned>(inter::xemachine::KernelAddressSpace::local))
      return op.emitOpError(
          "only local-address-space LLVM globals are semantic allocations");
    rewriter.eraseOp(op);
    return success();
  }
};

class ConvertLLVMAddressOf final
    : public OpConversionPattern<LLVM::AddressOfOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::AddressOfOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto pointer = dyn_cast<LLVM::LLVMPointerType>(op.getType());
    if (!pointer ||
        pointer.getAddressSpace() !=
            static_cast<unsigned>(inter::xemachine::KernelAddressSpace::local))
      return rewriter.notifyMatchFailure(
          op, "only addresses of local LLVM globals are supported");
    Type resultType = getTypeConverter()->convertType(op.getType());
    IntegerAttr offset = op->getAttrOfType<IntegerAttr>("xw.offset");
    if (!offset)
      return op.emitOpError(
          "referenced local global is missing an assigned SLM offset");
    xw::LocalMemoryBaseOp converted = xw::LocalMemoryBaseOp::create(
        rewriter, op.getLoc(), resultType, offset);
    converted->setAttr("xw.global", op.getGlobalNameAttr());
    if (IntegerAttr bytes = op->getAttrOfType<IntegerAttr>("xw.bytesize"))
      converted->setAttr("xw.bytesize", bytes);
    if (IntegerAttr alignment = op->getAttrOfType<IntegerAttr>("xw.alignment"))
      converted->setAttr("xw.alignment", alignment);
    return replaceConverted(op, converted, rewriter);
  }
};

class ConvertLLVMFunc final : public OpConversionPattern<LLVM::LLVMFuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::LLVMFuncOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op.isExternal())
      return rewriter.notifyMatchFailure(
          op, "defined LLVM function survived import");
    if (!classifyBuiltin(op.getName()))
      return rewriter.notifyMatchFailure(op, "unrecognized LLVM declaration");
    rewriter.eraseOp(op);
    return success();
  }
};

template <typename LLVMOp, xw::BinaryKind Kind>
class ConvertLLVMIntegerBinary final : public OpConversionPattern<LLVMOp> {
public:
  using OpConversionPattern<LLVMOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVMOp op, typename LLVMOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands = unwrapOperands(adaptor.getOperands());
    FailureOr<Type> type =
        convertResultType(op, *this->getTypeConverter(), rewriter);
    if (failed(type))
      return failure();
    Type resultType = distributeType(*type, operands);
    xw::BinaryOp converted = xw::BinaryOp::create(
        rewriter, op.getLoc(), resultType, Kind, operands[0], operands[1]);
    if constexpr (std::is_same_v<LLVMOp, LLVM::AddOp> ||
                  std::is_same_v<LLVMOp, LLVM::SubOp> ||
                  std::is_same_v<LLVMOp, LLVM::MulOp> ||
                  std::is_same_v<LLVMOp, LLVM::ShlOp>)
      converted.setOverflowFlags(convertOverflowFlags(op.getOverflowFlags()));
    return replaceConverted(op, converted, rewriter);
  }
};

template <typename LLVMOp, typename XWOp>
class ConvertLLVMFloatBinary final : public OpConversionPattern<LLVMOp> {
public:
  using OpConversionPattern<LLVMOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVMOp op, typename LLVMOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands = unwrapOperands(adaptor.getOperands());
    FailureOr<int64_t> width = getFunctionSimdWidth(op);
    if (failed(width))
      return failure();
    for (Value &operand : operands)
      if (!isa<xw::SimdType>(operand.getType()))
        operand = createSplat(rewriter, op.getLoc(), operand, *width);
    XWOp converted =
        XWOp::create(rewriter, op.getLoc(), operands.front().getType(),
                     operands[0], operands[1]);
    return replaceConverted(op, converted, rewriter);
  }
};

template <typename LLVMOp>
class RejectLLVMFloatBinary final : public OpConversionPattern<LLVMOp> {
public:
  using OpConversionPattern<LLVMOp>::OpConversionPattern;

  LogicalResult matchAndRewrite(LLVMOp op, typename LLVMOp::Adaptor,
                                ConversionPatternRewriter &) const override {
    return op.emitOpError(
        "floating division and remainder have no exact XW operation");
  }
};

template <typename LLVMOp, xw::CastKind Kind>
class ConvertLLVMCast final : public OpConversionPattern<LLVMOp> {
public:
  using OpConversionPattern<LLVMOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVMOp op, typename LLVMOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands = unwrapOperands(adaptor.getOperands());
    FailureOr<Type> type =
        convertResultType(op, *this->getTypeConverter(), rewriter);
    if (failed(type))
      return failure();
    Type resultType = distributeType(*type, operands);
    if constexpr (std::is_same_v<LLVMOp, LLVM::BitcastOp>)
      if (resultType == operands.front().getType()) {
        rewriter.replaceOp(op, operands.front());
        return success();
      }
    NamedAttrList policy;
    if constexpr (std::is_same_v<LLVMOp, LLVM::SExtOp>)
      policy.set("extension", xw::CastExtensionPolicyAttr::get(
                                  op.getContext(), xw::CastExtension::Sign));
    if constexpr (std::is_same_v<LLVMOp, LLVM::ZExtOp>)
      policy.set("extension", xw::CastExtensionPolicyAttr::get(
                                  op.getContext(), xw::CastExtension::Zero));
    if constexpr (std::is_same_v<LLVMOp, LLVM::SIToFPOp> ||
                  std::is_same_v<LLVMOp, LLVM::FPToSIOp>)
      policy.set("signedness",
                 xw::CastSignednessPolicyAttr::get(op.getContext(),
                                                   xw::CastSignedness::Signed));
    if constexpr (std::is_same_v<LLVMOp, LLVM::UIToFPOp> ||
                  std::is_same_v<LLVMOp, LLVM::FPToUIOp>)
      policy.set("signedness",
                 xw::CastSignednessPolicyAttr::get(
                     op.getContext(), xw::CastSignedness::Unsigned));
    xw::CastOp converted = xw::CastOp::create(
        rewriter, op.getLoc(), resultType, Kind, operands.front(),
        policy.empty() ? DictionaryAttr() : rewriter.getDictionaryAttr(policy));
    if constexpr (std::is_same_v<LLVMOp, LLVM::TruncOp>)
      converted.setOverflowFlags(convertOverflowFlags(op.getOverflowFlags()));
    return replaceConverted(op, converted, rewriter);
  }
};

class ConvertLLVMBitcast final : public OpConversionPattern<LLVM::BitcastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands = unwrapOperands(adaptor.getOperands());
    FailureOr<Type> type = convertResultType(op, *getTypeConverter(), rewriter);
    if (failed(type))
      return failure();
    Type resultType = distributeType(*type, operands);
    if (resultType == operands.front().getType()) {
      rewriter.replaceOp(op, operands.front());
      return success();
    }
    xw::BitcastOp converted = xw::BitcastOp::create(
        rewriter, op.getLoc(), resultType, operands.front());
    return replaceConverted(op, converted, rewriter);
  }
};

class ConvertLLVMICmp final : public OpConversionPattern<LLVM::ICmpOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::ICmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands = unwrapOperands(adaptor.getOperands());
    FailureOr<std::optional<int64_t>> cardinality =
        getExactCardinality(op, operands);
    if (failed(cardinality))
      return failure();
    Type resultType = rewriter.getI1Type();
    if (*cardinality) {
      for (Value &operand : operands)
        operand = splatToCardinality(op, operand, **cardinality);
      resultType = xw::MaskType::get(op.getContext(), **cardinality);
    }
    arith::CmpIPredicate predicate = convertPredicate(op.getPredicate());
    Type operandType = op.getLhs().getType();
    bool pointer = isa<LLVM::LLVMPointerType>(operandType) ||
                   (isa<VectorType>(operandType) &&
                    isa<LLVM::LLVMPointerType>(
                        cast<VectorType>(operandType).getElementType()));
    Operation *converted;
    if (pointer) {
      if (predicate != arith::CmpIPredicate::eq &&
          predicate != arith::CmpIPredicate::ne)
        return op.emitOpError("pointer comparison predicate must be eq or ne");
      converted = xw::PtrCmpOp::create(rewriter, op.getLoc(), resultType,
                                       predicate, operands[0], operands[1]);
    } else {
      converted = xw::CmpIOp::create(rewriter, op.getLoc(), resultType,
                                     predicate, operands[0], operands[1]);
    }
    return replaceConverted(op, converted, rewriter);
  }
};

class ConvertLLVMFCmp final : public OpConversionPattern<LLVM::FCmpOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::FCmpOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands = unwrapOperands(adaptor.getOperands());
    FailureOr<std::optional<int64_t>> cardinality =
        getExactCardinality(op, operands);
    if (failed(cardinality))
      return failure();
    Type resultType = rewriter.getI1Type();
    if (*cardinality) {
      for (Value &operand : operands)
        operand = splatToCardinality(op, operand, **cardinality);
      resultType = xw::MaskType::get(op.getContext(), **cardinality);
    }
    xw::CmpFOp converted = xw::CmpFOp::create(
        rewriter, op.getLoc(), resultType, convertPredicate(op.getPredicate()),
        operands[0], operands[1]);
    return replaceConverted(op, converted, rewriter);
  }
};

class ConvertLLVMSelect final : public OpConversionPattern<LLVM::SelectOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::SelectOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands = unwrapOperands(adaptor.getOperands());
    if (operands.size() != 3)
      return rewriter.notifyMatchFailure(op, "select requires three operands");
    FailureOr<std::optional<int64_t>> armCardinality =
        getExactCardinality(op, ValueRange(operands).drop_front());
    if (failed(armCardinality))
      return failure();
    std::optional<int64_t> conditionCardinality =
        getCardinality(operands.front().getType());
    if (conditionCardinality && *armCardinality &&
        conditionCardinality != *armCardinality)
      return op.emitOpError("select mask and arm cardinalities must match; use "
                            "xw.expand explicitly");
    std::optional<int64_t> cardinality =
        conditionCardinality ? conditionCardinality : *armCardinality;
    if (cardinality) {
      operands[1] = splatToCardinality(op, operands[1], *cardinality);
      operands[2] = splatToCardinality(op, operands[2], *cardinality);
    }
    if (operands[1].getType() != operands[2].getType())
      return op.emitOpError("select arms must have the same converted type");
    xw::SelectOp converted =
        xw::SelectOp::create(rewriter, op.getLoc(), operands[1].getType(),
                             operands[0], operands[1], operands[2]);
    return replaceConverted(op, converted, rewriter);
  }
};

class ConvertLLVMAddrspaceCast final
    : public OpConversionPattern<LLVM::AddrSpaceCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::AddrSpaceCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    LLVM::LLVMPointerType source =
        cast<LLVM::LLVMPointerType>(op.getArg().getType());
    LLVM::LLVMPointerType result = cast<LLVM::LLVMPointerType>(op.getType());
    bool sourceLocal =
        source.getAddressSpace() ==
        static_cast<unsigned>(inter::xemachine::KernelAddressSpace::local);
    bool resultLocal =
        result.getAddressSpace() ==
        static_cast<unsigned>(inter::xemachine::KernelAddressSpace::local);
    bool sourceGeneric =
        source.getAddressSpace() ==
        static_cast<unsigned>(inter::xemachine::KernelAddressSpace::generic);
    bool resultGeneric =
        result.getAddressSpace() ==
        static_cast<unsigned>(inter::xemachine::KernelAddressSpace::generic);
    if ((sourceLocal && resultGeneric) || (sourceGeneric && resultLocal))
      return op.emitOpError("local and generic address-space casts require "
                            "provenance-preserving selection");
    SmallVector<Value> operands = unwrapOperands(adaptor.getOperands());
    FailureOr<Type> type = convertResultType(op, *getTypeConverter(), rewriter);
    if (failed(type))
      return failure();
    xw::AddrspaceCastOp converted = xw::AddrspaceCastOp::create(
        rewriter, op.getLoc(), *type, operands.front());
    return replaceConverted(op, converted, rewriter);
  }
};

template <typename LLVMOp, typename XWOp>
class ConvertLLVMOneSourcePointerCast final
    : public OpConversionPattern<LLVMOp> {
public:
  using OpConversionPattern<LLVMOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVMOp op, typename LLVMOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands = unwrapOperands(adaptor.getOperands());
    FailureOr<Type> type =
        convertResultType(op, *this->getTypeConverter(), rewriter);
    if (failed(type))
      return failure();
    XWOp converted =
        XWOp::create(rewriter, op.getLoc(), *type, operands.front());
    return replaceConverted(op, converted, rewriter);
  }
};

class ConvertLLVMConstant final : public OpConversionPattern<LLVM::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::ConstantOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Type> type = convertResultType(op, *getTypeConverter(), rewriter);
    if (failed(type))
      return failure();
    xw::ConstantOp converted = xw::ConstantOp::create(
        rewriter, op.getLoc(), *type, cast<TypedAttr>(op.getValue()));
    return replaceConverted(op, converted, rewriter);
  }
};

class ConvertLLVMZero final : public OpConversionPattern<LLVM::ZeroOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::ZeroOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    FailureOr<Type> type = convertResultType(op, *getTypeConverter(), rewriter);
    if (failed(type))
      return failure();
    if (!isa<xw::PtrType>(*type))
      return op.emitOpError("only pointer LLVM zero values map to xw.null");
    xw::NullOp converted = xw::NullOp::create(rewriter, op.getLoc(), *type);
    return replaceConverted(op, converted, rewriter);
  }
};

class ConvertLLVMLoad final : public OpConversionPattern<LLVM::LoadOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::LoadOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getVolatile_())
      return op.emitOpError(
          "volatile LLVM load has no exact XW representation");
    if (op.getOrdering() != LLVM::AtomicOrdering::not_atomic)
      return op.emitOpError("atomic LLVM load has no exact XW representation");
    if (op.getSyncscope())
      return op.emitOpError(
          "LLVM load syncscope has no exact XW representation");
    SmallVector<Value> operands = unwrapOperands(adaptor.getOperands());
    FailureOr<Type> type = convertResultType(op, *getTypeConverter(), rewriter);
    FailureOr<int64_t> width = getFunctionSimdWidth(op);
    if (failed(type) || failed(width))
      return failure();
    Type resultType = xw::SimdType::get(op.getContext(), *type, *width);
    xw::LoadOp converted = xw::LoadOp::create(
        rewriter, op.getLoc(), resultType,
        xw::MemTokenType::get(op.getContext()), operands.front(), Value());
    return replaceConverted(op, converted, rewriter);
  }
};

class ConvertLLVMStore final : public OpConversionPattern<LLVM::StoreOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::StoreOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getVolatile_())
      return op.emitOpError(
          "volatile LLVM store has no exact XW representation");
    if (op.getOrdering() != LLVM::AtomicOrdering::not_atomic)
      return op.emitOpError("atomic LLVM store has no exact XW representation");
    if (op.getSyncscope())
      return op.emitOpError(
          "LLVM store syncscope has no exact XW representation");
    SmallVector<Value> operands = unwrapOperands(adaptor.getOperands());
    xw::StoreOp converted = xw::StoreOp::create(
        rewriter, op.getLoc(), xw::MemTokenType::get(op.getContext()),
        operands[0], operands[1], Value());
    return replaceConverted(op, converted, rewriter);
  }
};

class ConvertLLVMAtomicRMW final
    : public OpConversionPattern<LLVM::AtomicRMWOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::AtomicRMWOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getOrdering() != LLVM::AtomicOrdering::monotonic)
      return op.emitOpError(
          "only monotonic LLVM atomic RMW ordering is supported");
    if (op.getSyncscope())
      return op.emitOpError(
          "LLVM atomic RMW syncscope has no exact XW representation");
    if (op.getVolatile_())
      return op.emitOpError(
          "volatile LLVM atomic RMW has no exact XW representation");
    if (op.getBinOp() != LLVM::AtomicBinOp::add)
      return op.emitOpError("only integer add LLVM atomic RMW is supported");
    SmallVector<Value> operands = unwrapOperands(adaptor.getOperands());
    FailureOr<int64_t> width = getFunctionSimdWidth(op);
    if (failed(width))
      return failure();
    Value value = operands[1];
    if (!isa<xw::SimdType>(value.getType()))
      value = createSplat(rewriter, op.getLoc(), value, *width);
    xw::AtomicRMWOp converted = xw::AtomicRMWOp::create(
        rewriter, op.getLoc(), value.getType(),
        xw::MemTokenType::get(op.getContext()), arith::AtomicRMWKind::addi,
        value, operands[0], Value());
    return replaceConverted(op, converted, rewriter);
  }
};

class ConvertLLVMCall final : public OpConversionPattern<LLVM::CallOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::CallOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    std::optional<StringRef> callee = op.getCallee();
    if (!callee)
      return rewriter.notifyMatchFailure(op, "indirect calls are unsupported");
    std::optional<BuiltinKind> builtin = classifyBuiltin(*callee);
    if (!builtin)
      return rewriter.notifyMatchFailure(op, "unsupported LLVM operation");
    SmallVector<Value> operands = unwrapOperands(adaptor.getOperands());
    SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op,
                                         "result type has no XW conversion");
    FailureOr<int64_t> width = getFunctionSimdWidth(op);
    if (failed(width))
      return failure();
    if (!resultTypes.empty() &&
        (*builtin == BuiltinKind::LaneId || *builtin == BuiltinKind::GlobalId ||
         *builtin == BuiltinKind::LocalId))
      resultTypes.front() =
          xw::SimdType::get(op.getContext(), resultTypes.front(), *width);

    std::optional<int64_t> dimension;
    if (*builtin != BuiltinKind::LaneId &&
        *builtin != BuiltinKind::SubgroupId &&
        *builtin != BuiltinKind::Barrier &&
        *builtin != BuiltinKind::AtomicAdd) {
      if (adaptor.getOperands().size() != 1)
        return rewriter.notifyMatchFailure(op,
                                           "dimension query requires one axis");
      dimension = getConstantIntValue(adaptor.getOperands().front());
      if (!dimension || *dimension < 0 || *dimension > 2)
        return rewriter.notifyMatchFailure(
            op, "dimension query axis must be a constant in [0, 2]");
    }

    Type tokenType = xw::MemTokenType::get(op.getContext());
    Operation *converted = nullptr;
    switch (*builtin) {
    case BuiltinKind::LaneId:
      converted =
          xw::LaneIdOp::create(rewriter, op.getLoc(), resultTypes.front());
      break;
    case BuiltinKind::SubgroupId:
      converted =
          xw::SubgroupIdOp::create(rewriter, op.getLoc(), resultTypes.front());
      break;
    case BuiltinKind::GlobalId:
      converted = xw::GlobalIdOp::create(rewriter, op.getLoc(),
                                         resultTypes.front(), *dimension);
      break;
    case BuiltinKind::LocalId:
      converted = xw::LocalIdOp::create(rewriter, op.getLoc(),
                                        resultTypes.front(), *dimension);
      break;
    case BuiltinKind::GroupId:
      converted = xw::GroupIdOp::create(rewriter, op.getLoc(),
                                        resultTypes.front(), *dimension);
      break;
    case BuiltinKind::GlobalSize:
      converted = xw::GlobalSizeOp::create(rewriter, op.getLoc(),
                                           resultTypes.front(), *dimension);
      break;
    case BuiltinKind::LocalSize:
      converted = xw::LocalSizeOp::create(rewriter, op.getLoc(),
                                          resultTypes.front(), *dimension);
      break;
    case BuiltinKind::NumGroups:
      converted = xw::NumGroupsOp::create(rewriter, op.getLoc(),
                                          resultTypes.front(), *dimension);
      break;
    case BuiltinKind::LaunchGridSize:
      converted = xw::LaunchGridSizeOp::create(rewriter, op.getLoc(),
                                               resultTypes.front(), *dimension);
      break;
    case BuiltinKind::LaunchBlockSize:
      converted = xw::LaunchBlockSizeOp::create(
          rewriter, op.getLoc(), resultTypes.front(), *dimension);
      break;
    case BuiltinKind::Barrier:
      converted =
          xw::BarrierOp::create(rewriter, op.getLoc(), tokenType, ValueRange());
      break;
    case BuiltinKind::AtomicAdd: {
      if (operands.size() != 2)
        return rewriter.notifyMatchFailure(
            op, "atomic add builtin requires pointer and value operands");
      Value value = operands[1];
      if (!isa<xw::SimdType>(value.getType()))
        value = createSplat(rewriter, op.getLoc(), value, *width);
      resultTypes.front() = value.getType();
      converted = xw::AtomicRMWOp::create(
          rewriter, op.getLoc(), resultTypes.front(), tokenType,
          arith::AtomicRMWKind::addi, value, operands[0], Value());
      break;
    }
    }
    return replaceConverted(op, converted, rewriter);
  }
};

class ConvertLLVMGEP final : public OpConversionPattern<LLVM::GEPOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LLVM::GEPOp gep, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands = unwrapOperands(adaptor.getOperands());
    return convertGEP(gep, operands, rewriter);
  }

private:
  static FailureOr<uint64_t>
  getTypeStride(LLVM::GEPOp gep, const DataLayout &layout, Type type) {
    llvm::TypeSize size = layout.getTypeSize(type);
    if (size.isScalable())
      return gep.emitOpError("cannot index a scalable type"), failure();
    return llvm::alignTo(size.getFixedValue(),
                         layout.getTypeABIAlignment(type));
  }

  static Value createIntegerConstant(ConversionPatternRewriter &rewriter,
                                     Location loc, IntegerType type,
                                     int64_t value) {
    return xw::ConstantOp::create(rewriter, loc, type,
                                  rewriter.getIntegerAttr(type, value));
  }

  static Value createBinary(ConversionPatternRewriter &rewriter, Location loc,
                            xw::BinaryKind kind, Value lhs, Value rhs) {
    Type resultType = isa<xw::SimdType>(lhs.getType())   ? lhs.getType()
                      : isa<xw::SimdType>(rhs.getType()) ? rhs.getType()
                                                         : lhs.getType();
    return xw::BinaryOp::create(rewriter, loc, resultType, kind, lhs, rhs);
  }

  LogicalResult convertGEP(LLVM::GEPOp gep, ArrayRef<Value> operands,
                           ConversionPatternRewriter &rewriter) const {
    auto pointer = dyn_cast<LLVM::LLVMPointerType>(gep.getBase().getType());
    if (!pointer || operands.empty())
      return rewriter.notifyMatchFailure(gep, "GEP requires a scalar pointer");
    DataLayout layout = DataLayout::closest(gep);
    std::optional<uint64_t> width = layout.getTypeIndexBitwidth(pointer);
    if (!width || !*width)
      return rewriter.notifyMatchFailure(gep, "pointer index width is unknown");
    std::optional<inter::xemachine::KernelAddressSpace> addressSpace =
        inter::xemachine::KernelABI::get().decodeAddressSpace(
            pointer.getAddressSpace());
    if (!addressSpace)
      return rewriter.notifyMatchFailure(
          gep, "pointer address space is unsupported");
    uint64_t indexWidth =
        inter::xemachine::KernelABI::get().getMachinePointerIndexBitWidth(
            *addressSpace);
    IntegerType indexType = IntegerType::get(gep.getContext(), indexWidth);
    Type currentType = gep.getElemType();
    Value offset = createIntegerConstant(rewriter, gep.getLoc(), indexType, 0);
    unsigned dynamicIndex = 1;

    for (auto [position, index] : llvm::enumerate(gep.getIndices())) {
      uint64_t stride = 0;
      if (position == 0) {
        FailureOr<uint64_t> size = getTypeStride(gep, layout, currentType);
        if (failed(size))
          return failure();
        stride = *size;
      } else if (auto array = dyn_cast<LLVM::LLVMArrayType>(currentType)) {
        currentType = array.getElementType();
        FailureOr<uint64_t> size = getTypeStride(gep, layout, currentType);
        if (failed(size))
          return failure();
        stride = *size;
      } else if (auto vector = dyn_cast<VectorType>(currentType)) {
        currentType = vector.getElementType();
        llvm::TypeSize size = layout.getTypeSize(currentType);
        if (size.isScalable())
          return rewriter.notifyMatchFailure(gep, "scalable vector GEP");
        stride = size.getFixedValue();
      } else if (auto structure = dyn_cast<LLVM::LLVMStructType>(currentType)) {
        if (!isa<IntegerAttr>(index))
          return rewriter.notifyMatchFailure(
              gep, "struct GEP requires an in-range constant field");
        IntegerAttr constant = cast<IntegerAttr>(index);
        if (constant.getInt() < 0 || static_cast<uint64_t>(constant.getInt()) >=
                                         structure.getBody().size())
          return rewriter.notifyMatchFailure(
              gep, "struct GEP requires an in-range constant field");
        uint64_t field = constant.getInt();
        uint64_t byteOffset = 0;
        for (unsigned i : llvm::seq<unsigned>(field)) {
          Type element = structure.getBody()[i];
          if (!structure.isPacked())
            byteOffset =
                llvm::alignTo(byteOffset, layout.getTypeABIAlignment(element));
          llvm::TypeSize size = layout.getTypeSize(element);
          if (size.isScalable())
            return rewriter.notifyMatchFailure(gep, "scalable struct field");
          byteOffset += size.getFixedValue();
        }
        currentType = structure.getBody()[field];
        if (!structure.isPacked())
          byteOffset = llvm::alignTo(byteOffset,
                                     layout.getTypeABIAlignment(currentType));
        offset = createBinary(
            rewriter, gep.getLoc(), xw::BinaryKind::AddI, offset,
            createIntegerConstant(rewriter, gep.getLoc(), indexType,
                                  static_cast<int64_t>(byteOffset)));
        continue;
      } else {
        return rewriter.notifyMatchFailure(gep, "unsupported GEP aggregate");
      }

      Value term;
      if (isa<IntegerAttr>(index)) {
        IntegerAttr constant = cast<IntegerAttr>(index);
        term = createIntegerConstant(rewriter, gep.getLoc(), indexType,
                                     constant.getInt());
      } else {
        if (dynamicIndex >= operands.size())
          return rewriter.notifyMatchFailure(gep, "missing dynamic GEP index");
        term = operands[dynamicIndex++];
        Type termElementType = term.getType();
        int64_t cardinality = 0;
        if (auto simd = dyn_cast<xw::SimdType>(term.getType())) {
          termElementType = simd.getElementType();
          cardinality = simd.getCardinality();
        }
        if (termElementType != indexType) {
          Type resultType =
              cardinality ? Type(xw::SimdType::get(gep.getContext(), indexType,
                                                   cardinality))
                          : Type(indexType);
          DictionaryAttr policy;
          IntegerType sourceType = cast<IntegerType>(termElementType);
          if (sourceType.getWidth() < indexType.getWidth()) {
            NamedAttrList fields;
            fields.set("extension",
                       xw::CastExtensionPolicyAttr::get(
                           gep.getContext(), xw::CastExtension::Sign));
            policy = rewriter.getDictionaryAttr(fields);
          }
          term = xw::CastOp::create(rewriter, gep.getLoc(), resultType,
                                    xw::CastKind::IntConvert, term, policy);
        }
      }
      if (stride != 1) {
        if (llvm::isPowerOf2_64(stride)) {
          for (unsigned bit : llvm::seq<unsigned>(0, llvm::Log2_64(stride))) {
            (void)bit;
            term = createBinary(rewriter, gep.getLoc(), xw::BinaryKind::AddI,
                                term, term);
          }
        } else {
          term = createBinary(
              rewriter, gep.getLoc(), xw::BinaryKind::MulI, term,
              createIntegerConstant(rewriter, gep.getLoc(), indexType, stride));
        }
      }
      offset = createBinary(rewriter, gep.getLoc(), xw::BinaryKind::AddI,
                            offset, term);
    }

    Type resultType = getTypeConverter()->convertType(gep.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(gep, "pointer type has no XW mapping");
    if (auto simd = dyn_cast<xw::SimdType>(offset.getType()))
      resultType = xw::SimdType::get(gep.getContext(), resultType,
                                     simd.getCardinality());
    xw::PtrAddOp converted = xw::PtrAddOp::create(
        rewriter, gep.getLoc(), resultType, operands.front(), offset);
    preserveAttributes(gep, converted, rewriter);
    rewriter.replaceOp(gep, converted.getResult());
    return success();
  }
};

class ConvertPoison final : public OpConversionPattern<ub::PoisonOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(ub::PoisonOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Attribute value = op.getValue();
    if (value && !isa<ub::PoisonAttr>(value))
      return op.emitOpError("only full ub.poison is supported");
    Type type = getTypeConverter()->convertType(op.getType());
    if (!type)
      return rewriter.notifyMatchFailure(op,
                                         "poison type has no XW conversion");
    rewriter.replaceOpWithNewOp<ub::PoisonOp>(
        op, type, ub::PoisonAttr::get(op.getContext()));
    return success();
  }
};

class ConvertSCFIf final : public OpConversionPattern<scf::IfOp> {
public:
  ConvertSCFIf(TypeConverter &converter, MLIRContext *context)
      : OpConversionPattern(converter, context, 2) {}

  LogicalResult
  matchAndRewrite(scf::IfOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type conditionType = adaptor.getCondition().getType();
    if (!conditionType.isInteger(1) && !isa<xw::MaskType>(conditionType))
      return op.emitOpError("converted condition must be i1 or an XW mask");

    SmallVector<Type> resultTypes;
    if (failed(
            getTypeConverter()->convertTypes(op.getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op,
                                         "result type has no XW conversion");

    Operation *converted;
    if (conditionType.isInteger(1)) {
      scf::IfOp convertedIf =
          scf::IfOp::create(rewriter, op.getLoc(), resultTypes,
                            adaptor.getCondition(), false, false);
      convertedIf.getThenRegion().takeBody(op.getThenRegion());
      convertedIf.getElseRegion().takeBody(op.getElseRegion());
      preserveAttributes(op, convertedIf, rewriter);
      convertedIf->setAttr("xw.boundary_converted", rewriter.getUnitAttr());
      converted = convertedIf;
    } else {
      xw::WhereOp convertedWhere = xw::WhereOp::create(
          rewriter, op.getLoc(), resultTypes, adaptor.getCondition());
      convertedWhere.getThenRegion().takeBody(op.getThenRegion());
      convertedWhere.getElseRegion().takeBody(op.getElseRegion());
      preserveAttributes(op, convertedWhere, rewriter);
      for (Region *region :
           {&convertedWhere.getThenRegion(), &convertedWhere.getElseRegion()}) {
        if (region->empty())
          continue;
        scf::YieldOp yield =
            cast<scf::YieldOp>(region->front().getTerminator());
        OpBuilder builder(yield);
        xw::YieldOp::create(builder, yield.getLoc(), yield.getOperands());
        yield.erase();
      }
      converted = convertedWhere;
    }
    rewriter.replaceOp(op, converted->getResults());
    return success();
  }
};

class ConvertFuncReturn final : public OpConversionPattern<func::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(func::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

class ConvertArithConstant final
    : public OpConversionPattern<arith::ConstantOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::ConstantOp op, OpAdaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<xw::ConstantOp>(op, op.getType(),
                                                op.getValue());
    return success();
  }
};

class ConvertArithTruncI final : public OpConversionPattern<arith::TruncIOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::TruncIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op,
                                         "result type has no XW conversion");
    if (auto sourceType = dyn_cast<xw::SimdType>(adaptor.getIn().getType()))
      resultType = xw::SimdType::get(op.getContext(), resultType,
                                     sourceType.getCardinality());

    xw::CastOp converted = xw::CastOp::create(
        rewriter, op.getLoc(), resultType, xw::CastKind::IntConvert,
        adaptor.getIn(), DictionaryAttr());
    converted.setOverflowFlags(op.getOverflowFlags());
    rewriter.replaceOp(op, converted.getResult());
    return success();
  }
};

class ConvertArithXOrI final : public OpConversionPattern<arith::XOrIOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::XOrIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    SmallVector<Value> operands = unwrapOperands(adaptor.getOperands());
    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op,
                                         "result type has no XW conversion");
    resultType = distributeType(resultType, operands);
    xw::BinaryOp converted =
        xw::BinaryOp::create(rewriter, op.getLoc(), resultType,
                             xw::BinaryKind::XOrI, operands[0], operands[1]);
    rewriter.replaceOp(op, converted.getResult());
    return success();
  }
};

struct ConvertLLVMToXW final
    : inter::impl::ConvertLLVMToXWBase<ConvertLLVMToXW> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    eraseDebugInfoModuleFlags(getOperation());
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      for (Type type : function.getArgumentTypes()) {
        auto pointer = dyn_cast<LLVM::LLVMPointerType>(type);
        if (pointer && !inter::xemachine::KernelABI::get().decodeAddressSpace(
                           pointer.getAddressSpace())) {
          function.emitOpError()
              << "pointer address space " << pointer.getAddressSpace()
              << " has no XW mapping";
          return signalPassFailure();
        }
      }
    }
    DenseSet<FlatSymbolRefAttr> referencedLocalGlobals;
    getOperation().walk([&](LLVM::AddressOfOp address) {
      referencedLocalGlobals.insert(address.getGlobalNameAttr());
    });
    uint64_t slmOffset = 0;
    for (LLVM::GlobalOp global : getOperation().getOps<LLVM::GlobalOp>()) {
      if (global.getAddrSpace() !=
          static_cast<unsigned>(inter::xemachine::KernelAddressSpace::local))
        continue;
      FlatSymbolRefAttr symbol =
          FlatSymbolRefAttr::get(context, global.getSymName());
      if (!referencedLocalGlobals.contains(symbol))
        continue;
      DataLayout layout = DataLayout::closest(global);
      llvm::TypeSize size = layout.getTypeSize(global.getGlobalType());
      if (size.isScalable()) {
        global.emitOpError("local global has scalable size");
        return signalPassFailure();
      }
      uint64_t alignment =
          std::max<uint64_t>(layout.getTypeABIAlignment(global.getGlobalType()),
                             global.getAlignment().value_or(1));
      slmOffset = llvm::alignTo(slmOffset, alignment);
      getOperation().walk([&](LLVM::AddressOfOp address) {
        if (address.getGlobalName() != global.getSymName())
          return;
        address->setAttr("xw.bytesize",
                         IntegerAttr::get(IntegerType::get(context, 64),
                                          size.getFixedValue()));
        address->setAttr(
            "xw.alignment",
            IntegerAttr::get(IntegerType::get(context, 64), alignment));
        address->setAttr(
            "xw.offset",
            IntegerAttr::get(IntegerType::get(context, 64), slmOffset));
      });
      slmOffset += size.getFixedValue();
    }
    LLVMToXWTypeConverter converter(context);
    RewritePatternSet patterns(context);
    patterns
        .add<RejectLLVMUndef, ConvertLLVMPoison, ConvertLLVMFreeze,
             RejectLLVMFence, ConvertLLVMGlobal, ConvertLLVMAddressOf,
             ConvertLLVMFunc, ConvertLLVMCall,
             ConvertLLVMIntegerBinary<LLVM::AddOp, xw::BinaryKind::AddI>,
             ConvertLLVMIntegerBinary<LLVM::SubOp, xw::BinaryKind::SubI>,
             ConvertLLVMIntegerBinary<LLVM::MulOp, xw::BinaryKind::MulI>,
             ConvertLLVMIntegerBinary<LLVM::UDivOp, xw::BinaryKind::DivUI>,
             ConvertLLVMIntegerBinary<LLVM::SDivOp, xw::BinaryKind::DivSI>,
             ConvertLLVMIntegerBinary<LLVM::URemOp, xw::BinaryKind::RemUI>,
             ConvertLLVMIntegerBinary<LLVM::SRemOp, xw::BinaryKind::RemSI>,
             ConvertLLVMIntegerBinary<LLVM::ShlOp, xw::BinaryKind::ShLI>,
             ConvertLLVMIntegerBinary<LLVM::LShrOp, xw::BinaryKind::ShRUI>,
             ConvertLLVMIntegerBinary<LLVM::AShrOp, xw::BinaryKind::ShRSI>,
             ConvertLLVMIntegerBinary<LLVM::AndOp, xw::BinaryKind::AndI>,
             ConvertLLVMIntegerBinary<LLVM::OrOp, xw::BinaryKind::OrI>,
             ConvertLLVMIntegerBinary<LLVM::XOrOp, xw::BinaryKind::XOrI>,
             ConvertLLVMFloatBinary<LLVM::FAddOp, xw::FAddOp>,
             ConvertLLVMFloatBinary<LLVM::FSubOp, xw::FSubOp>,
             ConvertLLVMFloatBinary<LLVM::FMulOp, xw::FMulOp>,
             RejectLLVMFloatBinary<LLVM::FDivOp>,
             RejectLLVMFloatBinary<LLVM::FRemOp>,
             ConvertLLVMCast<LLVM::SExtOp, xw::CastKind::IntConvert>,
             ConvertLLVMCast<LLVM::ZExtOp, xw::CastKind::IntConvert>,
             ConvertLLVMCast<LLVM::TruncOp, xw::CastKind::IntConvert>,
             ConvertLLVMCast<LLVM::FPExtOp, xw::CastKind::FpConvert>,
             ConvertLLVMCast<LLVM::FPTruncOp, xw::CastKind::FpConvert>,
             ConvertLLVMCast<LLVM::SIToFPOp, xw::CastKind::IntToFp>,
             ConvertLLVMCast<LLVM::UIToFPOp, xw::CastKind::IntToFp>,
             ConvertLLVMCast<LLVM::FPToSIOp, xw::CastKind::FpToInt>,
             ConvertLLVMCast<LLVM::FPToUIOp, xw::CastKind::FpToInt>,
             ConvertLLVMBitcast, ConvertLLVMICmp, ConvertLLVMFCmp,
             ConvertLLVMSelect, ConvertLLVMAddrspaceCast,
             ConvertLLVMOneSourcePointerCast<LLVM::PtrToIntOp, xw::PtrToIntOp>,
             ConvertLLVMOneSourcePointerCast<LLVM::IntToPtrOp, xw::IntToPtrOp>,
             ConvertLLVMConstant, ConvertLLVMZero, ConvertLLVMLoad,
             ConvertLLVMStore, ConvertLLVMAtomicRMW, ConvertLLVMGEP,
             ConvertPoison, ConvertSCFIf, ConvertFuncReturn,
             ConvertArithConstant, ConvertArithTruncI, ConvertArithXOrI>(
            converter, context);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    scf::populateSCFStructuralTypeConversions(converter, patterns);

    ConversionTarget target(*context);
    target.addLegalDialect<xw::XWDialect, func::FuncDialect, scf::SCFDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalDialect<LLVM::LLVMDialect, cf::ControlFlowDialect,
                             ub::UBDialect>();
    target.addDynamicallyLegalOp<ub::PoisonOp>([&](ub::PoisonOp poison) {
      Attribute value = poison.getValue();
      return (!value || isa<ub::PoisonAttr>(value)) &&
             converter.isLegal(poison.getType());
    });
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [&](UnrealizedConversionCastOp cast) {
          if (cast->getNumOperands() != 1 || cast->getNumResults() != 1)
            return false;
          Type converted = converter.convertType(cast.getOperand(0).getType());
          return converted && getPayloadType(converted) ==
                                  getPayloadType(cast.getResult(0).getType());
        });
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      auto legalType = [](Type type) { return !containsLLVMType(type); };
      bool legalBuiltin = op->getName().getDialectNamespace() == "builtin";
      bool legalTerminator =
          isa<func::ReturnOp, scf::YieldOp, scf::ConditionOp>(op);
      return (legalBuiltin || legalTerminator) &&
             llvm::all_of(op->getOperandTypes(), legalType) &&
             llvm::all_of(op->getResultTypes(), legalType);
    });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp function) {
      return converter.isSignatureLegal(function.getFunctionType()) &&
             converter.isLegal(&function.getBody());
    });
    scf::populateSCFStructuralTypeConversionTarget(converter, target);
    target.addDynamicallyLegalOp<scf::IfOp>([&](scf::IfOp op) {
      return op->hasAttr("xw.boundary_converted") &&
             converter.isLegal(op.getCondition().getType()) &&
             converter.isLegal(op.getResultTypes()) &&
             converter.isLegal(&op.getThenRegion()) &&
             converter.isLegal(&op.getElseRegion());
    });

    if (failed(
            applyFullConversion(getOperation(), target, std::move(patterns))) ||
        failed(reconcileMaterializedShapes(getOperation())))
      signalPassFailure();
  }
};

} // namespace.
