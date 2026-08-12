#include "inter/Dialect/Inter/IR/XW.h"

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

namespace inter {
#define GEN_PASS_DEF_CONVERTLLVMTOXW
#include "inter/Transforms/Passes.h.inc"
} // namespace inter.

using namespace mlir;

namespace {

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

static DictionaryAttr getImportedAttributes(Operation *op, Builder &builder) {
  NamedAttrList imported;
  for (NamedAttribute attr : op->getAttrs())
    if (!attr.getName().strref().starts_with("llvm.") &&
        !containsLLVMType(attr.getValue()) &&
        attr.getValue().getDialect().getNamespace() !=
            LLVM::LLVMDialect::getDialectNamespace())
      imported.set(attr.getName(), attr.getValue());
  return builder.getDictionaryAttr(imported);
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
  OperationState state(loc, "xw.splat");
  state.addOperands(value);
  state.addTypes(
      xw::SimdType::get(value.getContext(), value.getType(), cardinality));
  return builder.create(state)->getResult(0);
}

static Value splatToCardinality(Operation *anchor, Value value,
                                int64_t cardinality) {
  if (getCardinality(value.getType()))
    return value;
  OpBuilder builder(anchor);
  return createSplat(builder, anchor->getLoc(), value, cardinality);
}

static Operation *replaceDivergentIf(scf::IfOp op, Value condition) {
  OperationState state(op.getLoc(), "xw.where");
  state.addOperands(condition);
  state.addTypes(op.getResultTypes());
  state.addAttributes(op->getDiscardableAttrDictionary().getValue());
  Region *thenRegion = state.addRegion();
  Region *elseRegion = state.addRegion();
  thenRegion->takeBody(op.getThenRegion());
  elseRegion->takeBody(op.getElseRegion());
  for (Region *region : {thenRegion, elseRegion}) {
    if (region->empty())
      continue;
    scf::YieldOp yield = cast<scf::YieldOp>(region->front().getTerminator());
    OpBuilder builder(yield);
    OperationState yieldState(yield.getLoc(), "xw.yield");
    yieldState.addOperands(yield.getOperands());
    builder.create(yieldState);
    yield.erase();
  }
  OpBuilder builder(op);
  Operation *where = builder.create(state);
  op->replaceAllUsesWith(where->getResults());
  op.erase();
  return where;
}

static FailureOr<Operation *>
replaceOperationShape(Operation *op, ValueRange operands, Type resultType) {
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

  SmallVector<Type> resultTypes(op->getResultTypes());
  resultTypes.front() = resultType;
  OperationState state(op->getLoc(), op->getName());
  state.addOperands(operands);
  state.addTypes(resultTypes);
  state.addAttributes(op->getAttrs());
  OpBuilder builder(op);
  Operation *replacement = builder.create(state);

  for (scf::IfOp ifOp : divergentIfs)
    replaceDivergentIf(ifOp, replacement->getResult(0));
  op->replaceAllUsesWith(replacement->getResults());
  op->erase();
  return replacement;
}

static LogicalResult reconcileMaterializedShapes(ModuleOp module) {
  module.walk([](scf::IfOp op) { op->removeAttr("xw.boundary_converted"); });

  SmallVector<UnrealizedConversionCastOp> casts;
  module.walk([&](UnrealizedConversionCastOp cast) { casts.push_back(cast); });
  for (UnrealizedConversionCastOp cast : casts) {
    if (cast->getNumOperands() != 1 || cast->getNumResults() != 1 ||
        getPayloadType(cast.getOperand(0).getType()) !=
            getPayloadType(cast.getResult(0).getType()))
      return cast.emitOpError(
          "non-shape unrealized conversion survived the LLVM boundary");
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

  auto normalizeMixed = [&]() -> LogicalResult {
    SmallVector<Operation *> candidates;
    module.walk([&](Operation *op) {
      if (isa<xw::CmpIOp, xw::CmpFOp, xw::PtrCmpOp, xw::SelectOp>(op))
        candidates.push_back(op);
    });
    for (Operation *op : candidates) {
      if (isa<xw::CmpIOp, xw::CmpFOp, xw::PtrCmpOp>(op)) {
        FailureOr<std::optional<int64_t>> cardinality =
            getExactCardinality(op, op->getOperands());
        if (failed(cardinality))
          return failure();
        if (!*cardinality)
          continue;
        OpBuilder builder(op);
        SmallVector<Value> operands(op->getOperands());
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
      FailureOr<std::optional<int64_t>> armCardinality = getExactCardinality(
          op, ValueRange{select.getTrueValue(), select.getFalseValue()});
      if (failed(armCardinality))
        return failure();
      std::optional<int64_t> conditionCardinality =
          getCardinality(select.getCondition().getType());
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
      SmallVector<Value> operands(op->getOperands());
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
    module.walk([&](Operation *op) {
      if (op->getNumResults() != 0 &&
          (isa<xw::CmpIOp, xw::CmpFOp, xw::PtrCmpOp, xw::PtrAddOp>(op) ||
           op->hasTrait<OpTrait::Elementwise>()))
        candidates.push_back(op);
    });
    for (Operation *op : candidates) {
      std::optional<int64_t> cardinality;
      for (Type type : op->getOperandTypes())
        if (std::optional<int64_t> candidate = getCardinality(type))
          cardinality =
              cardinality ? std::max(*cardinality, *candidate) : candidate;
      if (!cardinality || op->getNumResults() == 0)
        continue;

      Type resultType = op->getResult(0).getType();
      Type replacement;
      if (isa<xw::CmpIOp, xw::CmpFOp, xw::PtrCmpOp>(op))
        replacement = xw::MaskType::get(op->getContext(), *cardinality);
      else if (auto ptradd = dyn_cast<xw::PtrAddOp>(op)) {
        Type pointer = getPayloadType(ptradd.getBase().getType());
        replacement =
            xw::SimdType::get(op->getContext(), pointer, *cardinality);
      } else if (op->hasTrait<OpTrait::Elementwise>()) {
        replacement = xw::SimdType::get(
            op->getContext(), getPayloadType(resultType), *cardinality);
      }
      if (replacement && replacement != resultType) {
        if (failed(replaceOperationShape(op, op->getOperands(), replacement)))
          return failure();
        changed = true;
      }
    }
  } while (changed);

  if (failed(normalizeMixed()))
    return failure();

  SmallVector<xw::SplatOp> redundantSplats;
  module.walk([&](xw::SplatOp splat) {
    if (splat.getSource().getType() == splat.getResult().getType())
      redundantSplats.push_back(splat);
  });
  for (xw::SplatOp splat : redundantSplats) {
    splat.getResult().replaceAllUsesWith(splat.getSource());
    splat.erase();
  }

  SmallVector<StringAttr> consumedModuleAttrs;
  for (NamedAttribute attr : module->getAttrs()) {
    StringRef name = attr.getName().strref();
    if (name.starts_with("llvm.") || name == "dlti.dl_spec")
      consumedModuleAttrs.push_back(attr.getName());
  }
  for (StringAttr name : consumedModuleAttrs)
    module->removeAttr(name);

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
      Attribute addressSpace;
      switch (pointer.getAddressSpace()) {
      case 0:
        addressSpace = xw::PrivateAddressSpaceAttr::get(context);
        break;
      case 1:
        addressSpace = xw::GlobalAddressSpaceAttr::get(context);
        break;
      case 2:
        addressSpace = xw::ConstantAddressSpaceAttr::get(context);
        break;
      case 3:
        addressSpace = xw::LocalAddressSpaceAttr::get(context);
        break;
      case 4:
        addressSpace = xw::GenericAddressSpaceAttr::get(context);
        break;
      default:
        return {};
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

static StringRef classifyLLVMOperation(Operation *op) {
  StringRef name = op->getName().getStringRef();
  return StringSwitch<StringRef>(name)
      .Cases({"llvm.add", "llvm.sub", "llvm.mul", "llvm.udiv", "llvm.sdiv",
              "llvm.urem", "llvm.srem", "llvm.shl", "llvm.lshr", "llvm.ashr",
              "llvm.and", "llvm.or", "llvm.xor"},
             "xw.binary")
      .Case("llvm.fadd", "xw.fadd")
      .Case("llvm.fsub", "xw.fsub")
      .Case("llvm.fmul", "xw.fmul")
      .Cases({"llvm.sext", "llvm.zext", "llvm.trunc", "llvm.fpext",
              "llvm.fptrunc", "llvm.sitofp", "llvm.uitofp", "llvm.fptosi",
              "llvm.fptoui", "llvm.bitcast"},
             "xw.cast")
      .Case("llvm.icmp", "xw.cmpi")
      .Case("llvm.fcmp", "xw.cmpf")
      .Case("llvm.select", "xw.select")
      .Case("llvm.addrspacecast", "xw.addrspace_cast")
      .Case("llvm.ptrtoint", "xw.ptr_to_int")
      .Case("llvm.inttoptr", "xw.int_to_ptr")
      .Case("llvm.mlir.constant", "xw.constant")
      .Case("llvm.mlir.null", "xw.null")
      .Case("llvm.load", "xw.load")
      .Case("llvm.store", "xw.store")
      .Case("llvm.atomicrmw", "xw.atomic_rmw")
      .Default("");
}

static StringRef classifyBuiltin(StringRef symbol) {
  return StringSwitch<StringRef>(symbol)
      .Cases({"_Z22get_sub_group_local_idv", "_Z22get_sub_group_local_id",
              "get_sub_group_local_id"},
             "xw.lane_id")
      .Cases({"_Z13get_global_idj", "_Z13get_global_idm", "get_global_id"},
             "xw.global_id")
      .Cases({"_Z12get_local_idj", "_Z12get_local_idm", "get_local_id"},
             "xw.local_id")
      .Cases({"_Z12get_group_idj", "_Z12get_group_idm", "get_group_id"},
             "xw.group_id")
      .Cases(
          {"_Z15get_global_sizej", "_Z15get_global_sizem", "get_global_size"},
          "xw.global_size")
      .Cases({"_Z14get_local_sizej", "_Z14get_local_sizem", "get_local_size"},
             "xw.local_size")
      .Cases({"_Z14get_num_groupsj", "_Z14get_num_groupsm", "get_num_groups"},
             "xw.num_groups")
      .Cases({"__builtin_IB_get_global_size", "__spirv_BuiltInGlobalSize"},
             "xw.launch_grid_size")
      .Cases({"__builtin_IB_get_local_size", "__spirv_BuiltInWorkgroupSize"},
             "xw.launch_block_size")
      .Cases({"_Z7barrierj", "barrier"}, "xw.barrier")
      .Cases(
          {"_Z12atomic_addPVU3AS1ii", "_Z10atomic_addPU3AS1Vjj", "atomic_add"},
          "xw.atomic_rmw")
      .Default("");
}

class ConvertLLVMOperation final : public ConversionPattern {
public:
  ConvertLLVMOperation(TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult
  matchAndRewrite(Operation *op, ArrayRef<Value> operands,
                  ConversionPatternRewriter &rewriter) const override {
    if (op->getName().getDialectNamespace() !=
        LLVM::LLVMDialect::getDialectNamespace())
      return failure();

    if (auto gep = dyn_cast<LLVM::GEPOp>(op))
      return convertGEP(gep, operands, rewriter);

    if (isa<LLVM::UndefOp>(op))
      return op->emitOpError("undef has no sound XW representation");

    if (auto poison = dyn_cast<LLVM::PoisonOp>(op)) {
      Type type = getTypeConverter()->convertType(poison.getType());
      if (!type)
        return rewriter.notifyMatchFailure(op,
                                           "poison type has no XW conversion");
      rewriter.replaceOpWithNewOp<ub::PoisonOp>(
          poison, type, ub::PoisonAttr::get(op->getContext()));
      return success();
    }

    if (auto freeze = dyn_cast<LLVM::FreezeOp>(op)) {
      if (operands.size() != 1)
        return rewriter.notifyMatchFailure(op, "freeze requires one operand");
      Type type = getTypeConverter()->convertType(freeze.getType());
      if (!type || type != operands.front().getType())
        return rewriter.notifyMatchFailure(
            op, "freeze source and result must have the same converted shape");
      xw::FreezeOp converted = xw::FreezeOp::create(rewriter, freeze.getLoc(),
                                                    type, operands.front());
      converted->setAttrs(getImportedAttributes(op, rewriter));
      rewriter.replaceOp(freeze, converted.getResult());
      return success();
    }

    if (isa<LLVM::FenceOp>(op))
      return op->emitOpError(
          "LLVM fence ordering and scope have no exact XW representation");

    if (auto global = dyn_cast<LLVM::GlobalOp>(op)) {
      if (global.getAddrSpace() != 3)
        return global.emitOpError(
            "only local-address-space LLVM globals are semantic allocations");
      rewriter.eraseOp(global);
      return success();
    }

    if (auto address = dyn_cast<LLVM::AddressOfOp>(op)) {
      auto pointer = dyn_cast<LLVM::LLVMPointerType>(address.getType());
      if (!pointer || pointer.getAddressSpace() != 3)
        return rewriter.notifyMatchFailure(
            op, "only addresses of local LLVM globals are supported");
      Type resultType = getTypeConverter()->convertType(address.getType());
      OperationState state(address.getLoc(), "xw.local_memory_base");
      state.addTypes(resultType);
      IntegerAttr offset = address->getAttrOfType<IntegerAttr>("xw.offset");
      if (!offset)
        return address.emitOpError(
            "referenced local global is missing an assigned SLM offset");
      state.addAttribute("offset", offset);
      state.addAttribute("xw.global", address.getGlobalNameAttr());
      if (IntegerAttr bytes =
              address->getAttrOfType<IntegerAttr>("xw.bytesize"))
        state.addAttribute("xw.bytesize", bytes);
      if (IntegerAttr alignment =
              address->getAttrOfType<IntegerAttr>("xw.alignment"))
        state.addAttribute("xw.alignment", alignment);
      state.addAttribute("xw.imported",
                         getImportedAttributes(address, rewriter));
      rewriter.replaceOp(address, rewriter.create(state)->getResults());
      return success();
    }

    if (isa<LLVM::BitcastOp>(op) && operands.size() == 1) {
      Type resultType =
          getTypeConverter()->convertType(op->getResult(0).getType());
      if (resultType == operands.front().getType()) {
        rewriter.replaceOp(op, operands.front());
        return success();
      }
    }

    if (auto function = dyn_cast<LLVM::LLVMFuncOp>(op)) {
      if (!function.isExternal())
        return rewriter.notifyMatchFailure(
            op, "defined LLVM function survived import");
      if (!classifyBuiltin(function.getName()).empty()) {
        rewriter.eraseOp(op);
        return success();
      }
      return rewriter.notifyMatchFailure(op, "unrecognized LLVM declaration");
    }

    StringRef replacement = classifyLLVMOperation(op);
    if (auto call = dyn_cast<LLVM::CallOp>(op)) {
      auto callee = call.getCallee();
      if (!callee)
        return rewriter.notifyMatchFailure(op,
                                           "indirect calls are unsupported");
      replacement = classifyBuiltin(*callee);
    }
    if (replacement.empty())
      return isa<LLVM::FDivOp, LLVM::FRemOp>(op)
                 ? op->emitOpError("floating division and remainder have no "
                                   "exact XW operation")
                 : rewriter.notifyMatchFailure(op,
                                               "unsupported LLVM operation");

    if (auto cast = dyn_cast<LLVM::AddrSpaceCastOp>(op)) {
      LLVM::LLVMPointerType source =
          mlir::cast<LLVM::LLVMPointerType>(cast.getArg().getType());
      LLVM::LLVMPointerType result =
          mlir::cast<LLVM::LLVMPointerType>(cast.getType());
      bool sourceLocal = source.getAddressSpace() == 3;
      bool resultLocal = result.getAddressSpace() == 3;
      bool sourceGeneric = source.getAddressSpace() == 4;
      bool resultGeneric = result.getAddressSpace() == 4;
      if ((sourceLocal && resultGeneric) || (sourceGeneric && resultLocal))
        return cast.emitOpError("local and generic address-space casts require "
                                "provenance-preserving selection");
    }

    SmallVector<Value> rewrittenOperands;
    rewrittenOperands.reserve(operands.size());
    for (Value operand : operands) {
      if (UnrealizedConversionCastOp cast =
              operand.getDefiningOp<UnrealizedConversionCastOp>();
          cast && cast->getNumOperands() == 1)
        operand = cast->getOperand(0);
      rewrittenOperands.push_back(operand);
    }

    FailureOr<int64_t> width = getFunctionSimdWidth(op);
    if (failed(width))
      return failure();
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(),
                                                resultTypes)))
      return rewriter.notifyMatchFailure(op,
                                         "result type has no XW conversion");

    auto distributedType = [&](Type elementType) -> Type {
      for (Value operand : rewrittenOperands)
        if (auto simd = dyn_cast<xw::SimdType>(operand.getType()))
          return xw::SimdType::get(op->getContext(), elementType,
                                   simd.getCardinality());
      return elementType;
    };
    auto splatOperand = [&](unsigned index) {
      Value value = rewrittenOperands[index];
      if (isa<xw::SimdType>(value.getType()))
        return;
      OperationState splatState(op->getLoc(), "xw.splat");
      splatState.addOperands(value);
      splatState.addTypes(
          xw::SimdType::get(op->getContext(), value.getType(), *width));
      rewrittenOperands[index] = rewriter.create(splatState)->getResult(0);
    };
    if (replacement == "xw.fadd" || replacement == "xw.fsub" ||
        replacement == "xw.fmul") {
      splatOperand(0);
      splatOperand(1);
      resultTypes.front() = rewrittenOperands.front().getType();
    }
    if (!resultTypes.empty() &&
        (replacement == "xw.binary" || replacement == "xw.cast" ||
         replacement == "xw.select"))
      resultTypes.front() = distributedType(resultTypes.front());
    if (!resultTypes.empty() &&
        (replacement == "xw.cmpi" || replacement == "xw.cmpf")) {
      FailureOr<std::optional<int64_t>> cardinality =
          getExactCardinality(op, rewrittenOperands);
      if (failed(cardinality))
        return failure();
      if (*cardinality) {
        for (unsigned index : llvm::seq<unsigned>(rewrittenOperands.size()))
          rewrittenOperands[index] =
              splatToCardinality(op, rewrittenOperands[index], **cardinality);
        resultTypes.front() =
            xw::MaskType::get(op->getContext(), **cardinality);
      } else {
        resultTypes.front() = rewriter.getI1Type();
      }
    }
    if (!resultTypes.empty() && replacement == "xw.select") {
      if (rewrittenOperands.size() != 3)
        return rewriter.notifyMatchFailure(op,
                                           "select requires three operands");
      FailureOr<std::optional<int64_t>> armCardinality =
          getExactCardinality(op, ValueRange(rewrittenOperands).drop_front());
      if (failed(armCardinality))
        return failure();
      std::optional<int64_t> conditionCardinality =
          getCardinality(rewrittenOperands.front().getType());
      if (conditionCardinality && *armCardinality &&
          conditionCardinality != *armCardinality)
        return op->emitOpError("select mask and arm cardinalities must match; "
                               "use xw.expand explicitly");
      std::optional<int64_t> cardinality =
          conditionCardinality ? conditionCardinality : *armCardinality;
      if (cardinality) {
        rewrittenOperands[1] =
            splatToCardinality(op, rewrittenOperands[1], *cardinality);
        rewrittenOperands[2] =
            splatToCardinality(op, rewrittenOperands[2], *cardinality);
      }
      if (rewrittenOperands[1].getType() != rewrittenOperands[2].getType())
        return op->emitOpError("select arms must have the same converted type");
      resultTypes.front() = rewrittenOperands[1].getType();
    }
    if (!resultTypes.empty() &&
        (replacement == "xw.lane_id" || replacement == "xw.global_id" ||
         replacement == "xw.local_id" || replacement == "xw.load"))
      resultTypes.front() =
          xw::SimdType::get(op->getContext(), resultTypes.front(), *width);

    NamedAttrList attrs;
    attrs.set("xw.imported", getImportedAttributes(op, rewriter));
    if (replacement == "xw.binary") {
      StringRef name = op->getName().getStringRef();
      int32_t kind = StringSwitch<int32_t>(name)
                         .Case("llvm.add", 0)
                         .Case("llvm.sub", 1)
                         .Case("llvm.mul", 2)
                         .Case("llvm.shl", 3)
                         .Case("llvm.lshr", 4)
                         .Case("llvm.ashr", 5)
                         .Case("llvm.and", 6)
                         .Case("llvm.or", 7)
                         .Case("llvm.xor", 8)
                         .Case("llvm.udiv", 9)
                         .Case("llvm.sdiv", 10)
                         .Case("llvm.urem", 11)
                         .Case("llvm.srem", 12)
                         .Default(-1);
      if (kind < 0)
        return rewriter.notifyMatchFailure(op, "unsupported XW binary kind");
      attrs.set("kind", rewriter.getI32IntegerAttr(kind));
    } else if (replacement == "xw.cast") {
      StringRef name = op->getName().getStringRef();
      int32_t kind = name == "llvm.fpext" || name == "llvm.fptrunc"   ? 0
                     : name == "llvm.sitofp" || name == "llvm.uitofp" ? 2
                     : name == "llvm.fptosi" || name == "llvm.fptoui" ? 3
                                                                      : 1;
      attrs.set("kind", rewriter.getI32IntegerAttr(kind));
      NamedAttrList policy;
      if (name == "llvm.sext")
        policy.set("extension", xw::CastExtensionPolicyAttr::get(
                                    op->getContext(), xw::CastExtension::Sign));
      else if (name == "llvm.zext")
        policy.set("extension", xw::CastExtensionPolicyAttr::get(
                                    op->getContext(), xw::CastExtension::Zero));
      if (name == "llvm.sitofp" || name == "llvm.fptosi")
        policy.set("signedness",
                   xw::CastSignednessPolicyAttr::get(
                       op->getContext(), xw::CastSignedness::Signed));
      else if (name == "llvm.uitofp" || name == "llvm.fptoui")
        policy.set("signedness",
                   xw::CastSignednessPolicyAttr::get(
                       op->getContext(), xw::CastSignedness::Unsigned));
      if (!policy.empty())
        attrs.set("policy", rewriter.getDictionaryAttr(policy));
    } else if (replacement == "xw.cmpi" || replacement == "xw.cmpf") {
      if (auto compare = dyn_cast<LLVM::ICmpOp>(op)) {
        Type operandType = compare.getLhs().getType();
        bool pointer = isa<LLVM::LLVMPointerType>(operandType) ||
                       (isa<VectorType>(operandType) &&
                        isa<LLVM::LLVMPointerType>(
                            cast<VectorType>(operandType).getElementType()));
        if (pointer) {
          arith::CmpIPredicate converted = static_cast<arith::CmpIPredicate>(
              static_cast<uint64_t>(compare.getPredicate()));
          if (converted != arith::CmpIPredicate::eq &&
              converted != arith::CmpIPredicate::ne)
            return op->emitOpError(
                "pointer comparison predicate must be eq or ne");
          replacement = "xw.ptr_cmp";
        }
        attrs.set("predicate",
                  arith::CmpIPredicateAttr::get(
                      op->getContext(),
                      static_cast<arith::CmpIPredicate>(
                          static_cast<uint64_t>(compare.getPredicate()))));
      } else if (auto compare = dyn_cast<LLVM::FCmpOp>(op)) {
        attrs.set("predicate",
                  arith::CmpFPredicateAttr::get(
                      op->getContext(),
                      static_cast<arith::CmpFPredicate>(
                          static_cast<uint64_t>(compare.getPredicate()))));
      } else {
        return rewriter.notifyMatchFailure(op,
                                           "comparison predicate is missing");
      }
    } else if (auto constant = dyn_cast<LLVM::ConstantOp>(op)) {
      attrs.set("value", constant.getValue());
    }

    if (auto call = dyn_cast<LLVM::CallOp>(op)) {
      if (replacement == "xw.global_id" || replacement == "xw.local_id" ||
          replacement == "xw.group_id" || replacement == "xw.global_size" ||
          replacement == "xw.local_size" || replacement == "xw.num_groups" ||
          replacement == "xw.launch_grid_size" ||
          replacement == "xw.launch_block_size") {
        if (operands.size() != 1)
          return rewriter.notifyMatchFailure(
              op, "dimension query requires one axis");
        std::optional<int64_t> dimension =
            getConstantIntValue(operands.front());
        if (!dimension || *dimension < 0 || *dimension > 2)
          return rewriter.notifyMatchFailure(
              op, "dimension query axis must be a constant in [0, 2]");
        attrs.set("dim", rewriter.getI64IntegerAttr(*dimension));
        rewrittenOperands.clear();
      } else if (replacement == "xw.barrier") {
        rewrittenOperands.clear();
      }
    }

    if (auto atomic = dyn_cast<LLVM::AtomicRMWOp>(op)) {
      if (atomic.getOrdering() != LLVM::AtomicOrdering::monotonic)
        return atomic.emitOpError(
            "only monotonic LLVM atomic RMW ordering is supported");
      if (atomic.getSyncscope())
        return atomic.emitOpError(
            "LLVM atomic RMW syncscope has no exact XW representation");
      if (atomic.getVolatile_())
        return atomic.emitOpError(
            "volatile LLVM atomic RMW has no exact XW representation");
      arith::AtomicRMWKind kind;
      switch (atomic.getBinOp()) {
      case LLVM::AtomicBinOp::add:
        kind = arith::AtomicRMWKind::addi;
        break;
      default:
        return atomic.emitOpError(
            "only integer add LLVM atomic RMW is supported");
      }
      attrs.set("xw.imported", rewriter.getDictionaryAttr({}));
      attrs.set("kind", arith::AtomicRMWKindAttr::get(op->getContext(), kind));
      rewrittenOperands = {rewrittenOperands[1], rewrittenOperands[0]};
    } else if (isa<LLVM::CallOp>(op) && replacement == "xw.atomic_rmw") {
      attrs.set("kind", arith::AtomicRMWKindAttr::get(
                            op->getContext(), arith::AtomicRMWKind::addi));
      if (operands.size() != 2)
        return rewriter.notifyMatchFailure(
            op, "atomic add builtin requires pointer and value operands");
      rewrittenOperands = {rewrittenOperands[1], rewrittenOperands[0]};
    }
    if (replacement == "xw.atomic_rmw") {
      splatOperand(0);
      resultTypes.front() = rewrittenOperands.front().getType();
    }
    if (replacement == "xw.load" || replacement == "xw.atomic_rmw") {
      Type token = xw::MemTokenType::get(op->getContext());
      resultTypes.push_back(token);
    } else if (replacement == "xw.store" || replacement == "xw.barrier") {
      Type token = xw::MemTokenType::get(op->getContext());
      resultTypes.push_back(token);
    }

    OperationState state(op->getLoc(), replacement);
    state.addOperands(rewrittenOperands);
    state.addTypes(resultTypes);
    state.addAttributes(attrs);
    Operation *converted = rewriter.create(state);
    rewriter.replaceOp(op,
                       converted->getResults().take_front(op->getNumResults()));
    return success();
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
    OperationState state(loc, "xw.constant");
    state.addTypes(type);
    state.addAttribute("value", rewriter.getIntegerAttr(type, value));
    return rewriter.create(state)->getResult(0);
  }

  static Value createBinary(ConversionPatternRewriter &rewriter, Location loc,
                            StringRef kind, Value lhs, Value rhs) {
    OperationState state(loc, "xw.binary");
    state.addOperands({lhs, rhs});
    Type resultType = isa<xw::SimdType>(lhs.getType())   ? lhs.getType()
                      : isa<xw::SimdType>(rhs.getType()) ? rhs.getType()
                                                         : lhs.getType();
    state.addTypes(resultType);
    int32_t value = StringSwitch<int32_t>(kind)
                        .Case("add", 0)
                        .Case("mul", 2)
                        .Case("shl", 3)
                        .Default(-1);
    assert(value >= 0 && "unsupported generated GEP binary kind");
    state.addAttribute("kind", rewriter.getI32IntegerAttr(value));
    return rewriter.create(state)->getResult(0);
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
    IntegerType indexType = IntegerType::get(gep.getContext(), *width);
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
            rewriter, gep.getLoc(), "add", offset,
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
          OperationState castState(gep.getLoc(), "xw.cast");
          castState.addOperands(term);
          castState.addTypes(
              cardinality ? Type(xw::SimdType::get(gep.getContext(), indexType,
                                                   cardinality))
                          : Type(indexType));
          castState.addAttribute("kind", rewriter.getI32IntegerAttr(1));
          IntegerType sourceType = cast<IntegerType>(termElementType);
          if (sourceType.getWidth() < indexType.getWidth()) {
            NamedAttrList policy;
            policy.set("extension",
                       xw::CastExtensionPolicyAttr::get(
                           gep.getContext(), xw::CastExtension::Sign));
            castState.addAttribute("policy",
                                   rewriter.getDictionaryAttr(policy));
          }
          term = rewriter.create(castState)->getResult(0);
        }
      }
      if (stride != 1) {
        if (llvm::isPowerOf2_64(stride)) {
          for (unsigned bit : llvm::seq<unsigned>(0, llvm::Log2_64(stride))) {
            (void)bit;
            term = createBinary(rewriter, gep.getLoc(), "add", term, term);
          }
        } else {
          term = createBinary(
              rewriter, gep.getLoc(), "mul", term,
              createIntegerConstant(rewriter, gep.getLoc(), indexType, stride));
        }
      }
      offset = createBinary(rewriter, gep.getLoc(), "add", offset, term);
    }

    Type resultType = getTypeConverter()->convertType(gep.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(gep, "pointer type has no XW mapping");
    if (auto simd = dyn_cast<xw::SimdType>(offset.getType()))
      resultType = xw::SimdType::get(gep.getContext(), resultType,
                                     simd.getCardinality());
    OperationState state(gep.getLoc(), "xw.ptradd");
    state.addOperands({operands.front(), offset});
    state.addTypes(resultType);
    state.addAttribute("gep_flags",
                       rewriter.getI32IntegerAttr(
                           static_cast<uint32_t>(gep.getNoWrapFlags())));
    state.addAttribute("xw.imported", getImportedAttributes(gep, rewriter));
    Operation *converted = rewriter.create(state);
    rewriter.replaceOp(gep, converted->getResults());
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

    OperationState state(op.getLoc(),
                         conditionType.isInteger(1) ? "scf.if" : "xw.where");
    state.addOperands(adaptor.getCondition());
    state.addTypes(resultTypes);
    state.addAttributes(getImportedAttributes(op, rewriter).getValue());
    if (conditionType.isInteger(1))
      state.addAttribute("xw.boundary_converted", rewriter.getUnitAttr());
    Region *thenRegion = state.addRegion();
    Region *elseRegion = state.addRegion();
    thenRegion->takeBody(op.getThenRegion());
    elseRegion->takeBody(op.getElseRegion());

    if (isa<xw::MaskType>(conditionType)) {
      for (Region *region : {thenRegion, elseRegion}) {
        if (region->empty())
          continue;
        scf::YieldOp yield =
            cast<scf::YieldOp>(region->front().getTerminator());
        OpBuilder builder(yield);
        xw::YieldOp::create(builder, yield.getLoc(), yield.getOperands());
        yield.erase();
      }
    }

    Operation *converted = rewriter.create(state);
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
    OperationState state(op.getLoc(), "xw.constant");
    state.addTypes(op.getType());
    state.addAttribute("value", op.getValue());
    rewriter.replaceOp(op, rewriter.create(state)->getResults());
    return success();
  }
};

class ConvertArithTruncI final : public OpConversionPattern<arith::TruncIOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(arith::TruncIOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (op.getOverflowFlags() != arith::IntegerOverflowFlags::none)
      return op.emitOpError(
          "integer truncation overflow flags have no exact XW representation");

    Type resultType = getTypeConverter()->convertType(op.getType());
    if (!resultType)
      return rewriter.notifyMatchFailure(op,
                                         "result type has no XW conversion");
    if (auto sourceType = dyn_cast<xw::SimdType>(adaptor.getIn().getType()))
      resultType = xw::SimdType::get(op.getContext(), resultType,
                                     sourceType.getCardinality());

    OperationState state(op.getLoc(), "xw.cast");
    state.addOperands(adaptor.getIn());
    state.addTypes(resultType);
    state.addAttribute(
        "kind", rewriter.getI32IntegerAttr(
                    static_cast<int32_t>(xw::CastKind::IntConvert)));
    state.addAttribute("xw.imported", getImportedAttributes(op, rewriter));
    rewriter.replaceOp(op, rewriter.create(state)->getResults());
    return success();
  }
};

struct ConvertLLVMToXW final
    : inter::impl::ConvertLLVMToXWBase<ConvertLLVMToXW> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    for (func::FuncOp function : getOperation().getOps<func::FuncOp>()) {
      for (Type type : function.getArgumentTypes()) {
        auto pointer = dyn_cast<LLVM::LLVMPointerType>(type);
        if (pointer && pointer.getAddressSpace() > 4) {
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
      if (global.getAddrSpace() != 3)
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
    patterns.add<ConvertLLVMOperation, ConvertPoison, ConvertSCFIf,
                 ConvertFuncReturn, ConvertArithConstant, ConvertArithTruncI>(
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
