#include "inter/Dialect/Inter/IR/XW.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Func/Transforms/FuncConversions.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Patterns.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "inter/Transforms/Passes.h"
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
  if (type.getDialect().getNamespace() == LLVM::LLVMDialect::getDialectNamespace())
    return true;
  if (auto function = dyn_cast<FunctionType>(type))
    return llvm::any_of(function.getInputs(), containsLLVMType) ||
           llvm::any_of(function.getResults(), containsLLVMType);
  if (auto tuple = dyn_cast<TupleType>(type))
    return llvm::any_of(tuple.getTypes(), containsLLVMType);
  return false;
}

static bool containsLLVMType(Attribute attribute) {
  if (auto type = dyn_cast<TypeAttr>(attribute))
    return containsLLVMType(type.getValue());
  if (auto array = dyn_cast<ArrayAttr>(attribute))
    return llvm::any_of(array, [](Attribute nested) {
      return containsLLVMType(nested);
    });
  if (auto dictionary = dyn_cast<DictionaryAttr>(attribute))
    return llvm::any_of(dictionary, [](NamedAttribute attr) {
      return containsLLVMType(attr.getValue());
    });
  return false;
}

static DictionaryAttr getImportedAttributes(Operation *op,
                                             Builder &builder) {
  NamedAttrList imported;
  for (NamedAttribute attr : op->getAttrs())
    if (!containsLLVMType(attr.getValue()) &&
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

static std::optional<int64_t> getCardinality(Type type) {
  if (auto simd = dyn_cast<xw::SimdType>(type))
    return simd.getCardinality();
  if (auto mask = dyn_cast<xw::MaskType>(type))
    return mask.getCardinality();
  return std::nullopt;
}

static LogicalResult reconcileMaterializedShapes(ModuleOp module) {
  SmallVector<UnrealizedConversionCastOp> casts;
  module.walk([&](UnrealizedConversionCastOp cast) { casts.push_back(cast); });
  for (UnrealizedConversionCastOp cast : casts) {
    if (cast->getNumOperands() != 1 || cast->getNumResults() != 1 ||
        getPayloadType(cast.getOperand(0).getType()) !=
            getPayloadType(cast.getResult(0).getType()))
      return cast.emitOpError(
          "non-shape unrealized conversion survived the LLVM boundary");
    cast.getResult(0).replaceAllUsesWith(cast.getOperand(0));
    cast.erase();
  }

  bool changed;
  do {
    changed = false;
    module.walk([&](Operation *op) {
      std::optional<int64_t> cardinality;
      for (Type type : op->getOperandTypes())
        if (std::optional<int64_t> candidate = getCardinality(type))
          cardinality = cardinality
                            ? std::max(*cardinality, *candidate)
                            : candidate;
      if (!cardinality || op->getNumResults() == 0)
        return;

      Type resultType = op->getResult(0).getType();
      Type replacement;
      if (isa<xw::CmpIOp, xw::CmpFOp>(op))
        replacement = xw::MaskType::get(op->getContext(), *cardinality);
      else if (auto ptradd = dyn_cast<xw::PtrAddOp>(op)) {
        Type pointer = getPayloadType(ptradd.getBase().getType());
        replacement = xw::SimdType::get(op->getContext(), pointer,
                                        *cardinality);
      } else if (op->hasTrait<OpTrait::Elementwise>()) {
        replacement = xw::SimdType::get(op->getContext(),
                                        getPayloadType(resultType),
                                        *cardinality);
      }
      if (replacement && replacement != resultType) {
        op->getResult(0).setType(replacement);
        changed = true;
      }
    });
  } while (changed);

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
    if (llvm::any_of(op->getOperandTypes(), hasLLVMType) ||
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
      if (function.isVarArg() || failed(convertTypes(function.getParams(), inputs)) ||
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
              "llvm.and", "llvm.or", "llvm.xor"}, "xw.binary")
      .Case("llvm.fadd", "arith.addf")
      .Case("llvm.fsub", "arith.subf")
      .Case("llvm.fmul", "arith.mulf")
      .Case("llvm.fdiv", "arith.divf")
      .Case("llvm.frem", "arith.remf")
      .Cases({"llvm.sext", "llvm.zext", "llvm.trunc", "llvm.fpext",
              "llvm.fptrunc", "llvm.sitofp", "llvm.uitofp", "llvm.fptosi",
              "llvm.fptoui", "llvm.bitcast"}, "xw.cast")
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
      .Case("llvm.fence", "xw.barrier")
      .Default("");
}

static StringRef classifyBuiltin(StringRef symbol) {
  return StringSwitch<StringRef>(symbol)
      .Cases({"_Z13get_global_idj", "get_global_id"}, "xw.global_id")
      .Cases({"_Z12get_local_idj", "get_local_id"}, "xw.local_id")
      .Cases({"_Z12get_group_idj", "get_group_id"}, "xw.group_id")
      .Cases({"_Z15get_global_sizej", "get_global_size"}, "xw.global_size")
      .Cases({"_Z14get_local_sizej", "get_local_size"}, "xw.local_size")
      .Cases({"_Z7barrierj", "barrier"}, "xw.barrier")
      .Cases({"_Z12atomic_addPVU3AS1ii", "_Z10atomic_addPU3AS1Vjj",
              "atomic_add"},
             "xw.atomic_rmw")
      .Default("");
}

class ConvertLLVMOperation final : public ConversionPattern {
public:
  ConvertLLVMOperation(TypeConverter &converter, MLIRContext *context)
      : ConversionPattern(converter, MatchAnyOpTypeTag(), 1, context) {}

  LogicalResult matchAndRewrite(
      Operation *op, ArrayRef<Value> operands,
      ConversionPatternRewriter &rewriter) const override {
    if (op->getName().getDialectNamespace() !=
        LLVM::LLVMDialect::getDialectNamespace())
      return failure();

    if (auto gep = dyn_cast<LLVM::GEPOp>(op))
      return convertGEP(gep, operands, rewriter);

    if (auto global = dyn_cast<LLVM::GlobalOp>(op)) {
      if (global.getAddrSpace() != 3)
        return rewriter.notifyMatchFailure(
            op, "only local-address-space LLVM globals are semantic allocations");
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
      state.addAttribute("offset", rewriter.getI64IntegerAttr(0));
      state.addAttribute("xw.global", address.getGlobalNameAttr());
      if (IntegerAttr bytes = address->getAttrOfType<IntegerAttr>("xw.bytesize"))
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
      Type resultType = getTypeConverter()->convertType(op->getResult(0).getType());
      if (resultType == operands.front().getType()) {
        rewriter.replaceOp(op, operands.front());
        return success();
      }
    }

    if (auto function = dyn_cast<LLVM::LLVMFuncOp>(op)) {
      if (!function.isExternal())
        return rewriter.notifyMatchFailure(op, "defined LLVM function survived import");
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
        return rewriter.notifyMatchFailure(op, "indirect calls are unsupported");
      replacement = classifyBuiltin(*callee);
    }
    if (replacement.empty())
      return rewriter.notifyMatchFailure(op, "unsupported LLVM operation");

    SmallVector<Value> rewrittenOperands;
    rewrittenOperands.reserve(operands.size());
    for (Value operand : operands) {
      if (UnrealizedConversionCastOp cast =
              operand.getDefiningOp<UnrealizedConversionCastOp>();
          cast && cast->getNumOperands() == 1)
        operand = cast->getOperand(0);
      rewrittenOperands.push_back(operand);
    }
    if (llvm::any_of(rewrittenOperands, [](Value value) {
          return isa<xw::SimdType>(value.getType());
        })) {
      replacement = StringSwitch<StringRef>(replacement)
                        .Case("arith.addf", "xw.fadd")
                        .Case("arith.subf", "xw.fsub")
                        .Case("arith.mulf", "xw.fmul")
                        .Default(replacement);
    }
    SmallVector<Type> resultTypes;
    if (failed(getTypeConverter()->convertTypes(op->getResultTypes(), resultTypes)))
      return rewriter.notifyMatchFailure(op, "result type has no XW conversion");

    auto distributedType = [&](Type elementType) -> Type {
      for (Value operand : rewrittenOperands)
        if (auto simd = dyn_cast<xw::SimdType>(operand.getType()))
          return xw::SimdType::get(op->getContext(), elementType,
                                   simd.getCardinality());
      return elementType;
    };
    if (!resultTypes.empty() &&
        (replacement == "xw.binary" || replacement == "xw.cast" ||
         replacement == "xw.select" || replacement == "xw.fadd" ||
         replacement == "xw.fsub" || replacement == "xw.fmul"))
      resultTypes.front() = distributedType(resultTypes.front());
    if (!resultTypes.empty() &&
        (replacement == "xw.cmpi" || replacement == "xw.cmpf")) {
      Type distributed = distributedType(rewriter.getI1Type());
      if (auto simd = dyn_cast<xw::SimdType>(distributed))
        resultTypes.front() =
            xw::MaskType::get(op->getContext(), simd.getCardinality());
    }
    if (!resultTypes.empty() &&
        (replacement == "xw.global_id" || replacement == "xw.local_id" ||
         replacement == "xw.load" || replacement == "xw.atomic_rmw"))
      resultTypes.front() =
          xw::SimdType::get(op->getContext(), resultTypes.front(), 16);

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
        policy.set("signedness", xw::CastSignednessPolicyAttr::get(
                                     op->getContext(),
                                     xw::CastSignedness::Signed));
      else if (name == "llvm.uitofp" || name == "llvm.fptoui")
        policy.set("signedness", xw::CastSignednessPolicyAttr::get(
                                     op->getContext(),
                                     xw::CastSignedness::Unsigned));
      if (!policy.empty())
        attrs.set("policy", rewriter.getDictionaryAttr(policy));
    } else if (replacement == "xw.cmpi" || replacement == "xw.cmpf") {
      Attribute predicate = op->getAttr("predicate");
      if (!predicate)
        return rewriter.notifyMatchFailure(op, "comparison predicate is missing");
      attrs.set("predicate", predicate);
    } else if (auto constant = dyn_cast<LLVM::ConstantOp>(op)) {
      attrs.set("value", constant.getValue());
    }

    if (auto call = dyn_cast<LLVM::CallOp>(op)) {
      if (replacement == "xw.global_id" || replacement == "xw.local_id" ||
          replacement == "xw.group_id" || replacement == "xw.global_size" ||
          replacement == "xw.local_size") {
        if (operands.size() != 1)
          return rewriter.notifyMatchFailure(op, "dimension query requires one axis");
        std::optional<int64_t> dimension = getConstantIntValue(operands.front());
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
      arith::AtomicRMWKind kind;
      switch (atomic.getBinOp()) {
      case LLVM::AtomicBinOp::xchg:
        kind = arith::AtomicRMWKind::assign;
        break;
      case LLVM::AtomicBinOp::add:
        kind = arith::AtomicRMWKind::addi;
        break;
      case LLVM::AtomicBinOp::_and:
        kind = arith::AtomicRMWKind::andi;
        break;
      case LLVM::AtomicBinOp::_or:
        kind = arith::AtomicRMWKind::ori;
        break;
      case LLVM::AtomicBinOp::_xor:
        kind = arith::AtomicRMWKind::xori;
        break;
      case LLVM::AtomicBinOp::max:
        kind = arith::AtomicRMWKind::maxs;
        break;
      case LLVM::AtomicBinOp::min:
        kind = arith::AtomicRMWKind::mins;
        break;
      case LLVM::AtomicBinOp::umax:
        kind = arith::AtomicRMWKind::maxu;
        break;
      case LLVM::AtomicBinOp::umin:
        kind = arith::AtomicRMWKind::minu;
        break;
      case LLVM::AtomicBinOp::fadd:
        kind = arith::AtomicRMWKind::addf;
        break;
      default:
        return rewriter.notifyMatchFailure(op, "unsupported atomic RMW kind");
      }
      attrs.set("kind", arith::AtomicRMWKindAttr::get(op->getContext(), kind));
      rewrittenOperands = {operands[1], operands[0]};
    } else if (isa<LLVM::CallOp>(op) && replacement == "xw.atomic_rmw") {
      attrs.set("kind", arith::AtomicRMWKindAttr::get(
                            op->getContext(), arith::AtomicRMWKind::addi));
      if (operands.size() != 2)
        return rewriter.notifyMatchFailure(
            op, "atomic add builtin requires pointer and value operands");
      rewrittenOperands = {operands[1], operands[0]};
      Type valueType = rewrittenOperands.front().getType();
      resultTypes.front() = xw::SimdType::get(
          op->getContext(), getPayloadType(valueType), 16);
      if (!isa<xw::SimdType>(valueType)) {
        OperationState splatState(op->getLoc(), "xw.splat");
        splatState.addOperands(rewrittenOperands.front());
        splatState.addTypes(resultTypes.front());
        rewrittenOperands.front() = rewriter.create(splatState)->getResult(0);
      }
    }
    if (replacement == "xw.load" || replacement == "xw.atomic_rmw") {
      Type token = xw::MemTokenType::get(op->getContext());
      resultTypes.push_back(token);
      attrs.set("xw.provisional_cardinality", rewriter.getI32IntegerAttr(16));
    } else if (replacement == "xw.store" || replacement == "xw.barrier") {
      Type token = xw::MemTokenType::get(op->getContext());
      resultTypes.push_back(token);
    } else if (replacement == "xw.global_id" || replacement == "xw.local_id") {
      attrs.set("xw.provisional_cardinality", rewriter.getI32IntegerAttr(16));
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
  static FailureOr<uint64_t> getTypeStride(LLVM::GEPOp gep,
                                           const DataLayout &layout,
                                           Type type) {
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
    Type resultType = isa<xw::SimdType>(lhs.getType()) ? lhs.getType()
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
        if (constant.getInt() < 0 ||
            static_cast<uint64_t>(constant.getInt()) >= structure.getBody().size())
          return rewriter.notifyMatchFailure(
              gep, "struct GEP requires an in-range constant field");
        uint64_t field = constant.getInt();
        uint64_t byteOffset = 0;
        for (unsigned i : llvm::seq<unsigned>(field)) {
          Type element = structure.getBody()[i];
          if (!structure.isPacked())
            byteOffset = llvm::alignTo(byteOffset,
                                       layout.getTypeABIAlignment(element));
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
          castState.addTypes(cardinality ? Type(xw::SimdType::get(
                                                   gep.getContext(), indexType,
                                                   cardinality))
                                         : Type(indexType));
          castState.addAttribute("kind", rewriter.getI32IntegerAttr(1));
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

struct ConvertLLVMToXW final
    : inter::impl::ConvertLLVMToXWBase<ConvertLLVMToXW> {
  void runOnOperation() override {
    MLIRContext *context = &getContext();
    for (LLVM::GlobalOp global : getOperation().getOps<LLVM::GlobalOp>()) {
      if (global.getAddrSpace() != 3)
        continue;
      DataLayout layout = DataLayout::closest(global);
      llvm::TypeSize size = layout.getTypeSize(global.getGlobalType());
      if (size.isScalable()) {
        global.emitOpError("local global has scalable size");
        return signalPassFailure();
      }
      uint64_t alignment = layout.getTypeABIAlignment(global.getGlobalType());
      getOperation().walk([&](LLVM::AddressOfOp address) {
        if (address.getGlobalName() != global.getSymName())
          return;
        address->setAttr("xw.bytesize",
                         IntegerAttr::get(IntegerType::get(context, 64),
                                          size.getFixedValue()));
        address->setAttr("xw.alignment",
                         IntegerAttr::get(IntegerType::get(context, 64),
                                          alignment));
      });
    }
    LLVMToXWTypeConverter converter(context);
    RewritePatternSet patterns(context);
    patterns.add<ConvertLLVMOperation, ConvertFuncReturn>(converter, context);
    populateFunctionOpInterfaceTypeConversionPattern<func::FuncOp>(patterns,
                                                                   converter);
    scf::populateSCFStructuralTypeConversions(converter, patterns);

    ConversionTarget target(*context);
    target.addLegalDialect<xw::XWDialect, arith::ArithDialect,
                           func::FuncDialect, scf::SCFDialect>();
    target.addLegalOp<ModuleOp>();
    target.addIllegalDialect<LLVM::LLVMDialect, cf::ControlFlowDialect>();
    target.markUnknownOpDynamicallyLegal([](Operation *op) {
      auto legalType = [](Type type) { return !containsLLVMType(type); };
      bool legalBuiltin = op->getName().getDialectNamespace() == "builtin";
      bool legalTerminator = isa<func::ReturnOp, scf::YieldOp,
                                 scf::ConditionOp>(op);
      return (legalBuiltin || legalTerminator) &&
             llvm::all_of(op->getOperandTypes(), legalType) &&
             llvm::all_of(op->getResultTypes(), legalType);
    });
    target.addDynamicallyLegalOp<func::FuncOp>([&](func::FuncOp function) {
      return converter.isSignatureLegal(function.getFunctionType()) &&
             converter.isLegal(&function.getBody());
    });
    scf::populateSCFStructuralTypeConversionTarget(converter, target);

    if (failed(applyFullConversion(getOperation(), target, std::move(patterns))) ||
        failed(reconcileMaterializedShapes(getOperation())))
      signalPassFailure();
  }
};

} // namespace.
