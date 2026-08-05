//===- CallConvLoweringPass.cpp - Lower CIR to ABI calling convention ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass walks every cir.func and cir.call in the module, computes a
// FunctionClassification for it (via either an ABI target or a pre-built
// classification injected as a function attribute), and dispatches to
// CIRABIRewriteContext to perform the actual IR rewriting.
//
// Two driver modes (mutually exclusive):
//
//   target=test
//     Use the MLIR test ABI target (mlir/lib/ABI/Targets/Test/) to classify
//     each function.  Predictable rules that approximate x86_64 SysV.  Real
//     targets (x86_64, AArch64) will be added once the LLVM ABI library
//     ships them.
//
//   classification-attr=<name>
//     Read a DictionaryAttr named <name> from each cir.func and parse it via
//     mlir::abi::test::parseClassificationAttr.  Used by tests to inject any
//     classification (including shapes the test target itself does not
//     produce) without depending on a real ABI target.
//
// The pass requires a `dlti.dl_spec` attribute on the module so the
// classifier can query type sizes and alignments.
//
//===----------------------------------------------------------------------===//

#include "PassDetail.h"
#include "TargetLowering/CIRABIRewriteContext.h"

#include "mlir/ABI/ABIRewriteContext.h"
#include "mlir/ABI/ABITypeMapper.h"
#include "mlir/ABI/Targets/Test/TestTarget.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ABI/FunctionInfo.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ABI/Types.h"
#include "llvm/ADT/TypeSwitch.h"
#include "llvm/IR/CallingConv.h"

using namespace mlir;
using namespace mlir::abi;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_CALLCONVLOWERING
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

namespace {

//===----------------------------------------------------------------------===//
// x86_64 System V classifier bridge (scalar and struct/array types)
//
// Maps CIR types to llvm::abi::Type, runs the LLVM ABI Lowering Library's
// SysV x86_64 classifier, and converts the result back into the
// dialect-agnostic mlir::abi::FunctionClassification that CIRABIRewriteContext
// consumes.  Integer (including `_BitInt` up to 128 bits) / pointer / bool /
// f32 / f64 scalars and struct / array aggregates are handled.  Unions,
// `_Complex`, vectors, wider floats, and packed or padded records are reported
// NYI by classifyX86_64Function so an unsupported signature fails the pass
// instead of being misclassified.
//===----------------------------------------------------------------------===//

/// Whether a struct's declared argument-passing kind (from the module's
/// record-layout metadata) allows it to be passed in registers.  A record with
/// no layout entry (e.g. an anonymous struct) has no C++ non-trivial reason to
/// be forced to memory, so it defaults to can-pass-in-registers.
static bool recordCanPassInRegs(ModuleOp modOp, cir::RecordType recTy) {
  mlir::StringAttr name = recTy.getName();
  if (!name)
    return true;
  auto dict = modOp->getAttrOfType<DictionaryAttr>(
      cir::CIRDialect::getRecordLayoutsAttrName());
  if (!dict)
    return true;
  auto layout = dict.getAs<cir::RecordLayoutAttr>(name);
  if (!layout)
    return true;
  return layout.getArgPassingKind() == cir::ArgPassingKind::CanPassInRegs;
}

/// The CIR types the x86_64 bridge handles.  Scalars: an integer up to 128
/// bits (including `_BitInt` and `__int128`), pointer, bool, void, f32, or f64.
/// Aggregates: a complete struct whose fields are all themselves supported, or
/// an array of a supported element type.  A `_BitInt` wider than 128 bits,
/// unions, `_Complex`, vectors, wider floats, and packed or padded records are
/// not handled and are reported NYI at the reject() choke point in
/// classifyX86_64Function.
static bool isSupportedType(mlir::Type ty) {
  // A pointer is only handled in the default address space (null) or an
  // already-lowered target address space.  A LangAddressSpaceAttr must be
  // lowered before this pass, so reject it rather than silently dropping it.
  if (auto ptrTy = dyn_cast<cir::PointerType>(ty))
    return !ptrTy.getAddrSpace() ||
           mlir::isa<cir::TargetAddressSpaceAttr>(ptrTy.getAddrSpace());
  if (isa<cir::VoidType, cir::BoolType, cir::SingleType, cir::DoubleType>(ty))
    return true;
  if (auto intTy = dyn_cast<cir::IntType>(ty)) {
    // Integers up to 64 bits, __int128, and _BitInt up to 128 bits are
    // handled: the classifier extends a width below 32, widens 33 through 63
    // to i64, coerces 65 through 127 to a {i64, i64} pair, and passes 32, 64,
    // and 128 in the natural type.  A wider _BitInt classifies Indirect,
    // where at a multiple of 8 the byval attributes the rewriter appends
    // duplicate the llvm.noundef CIRGen already emitted and trip the
    // uniqueness assertion on the merged dictionary.  The bound is a blanket
    // 128 because the widths that do not collide reach that same untested
    // Indirect path.  Non-_BitInt intermediate widths (65..127) do not arise
    // from C.  Both stay rejected.
    if (intTy.getIsBitInt())
      return intTy.getWidth() <= 128;
    return intTy.getWidth() <= 64 || intTy.getWidth() == 128;
  }
  if (auto arrTy = dyn_cast<cir::ArrayType>(ty))
    return isSupportedType(arrTy.getElementType());
  if (auto recTy = dyn_cast<cir::RecordType>(ty)) {
    // Unions and packed / padded records each need classification this bridge
    // does not implement (a union widen fixup and pad-aware eightbyte
    // classification), so reject them here and report NYI rather than
    // misclassify.  A zero-field record (a C empty struct) classifies as
    // Ignore and is dropped from the lowered signature.  CIRGen lays out an
    // empty C++ class as a single padded byte, which the padded check rejects.
    // A real one-byte struct such as `{char[1]}` has a field and is not
    // padded, so it is classified normally.
    if (recTy.isUnion() || !recTy.isComplete() || recTy.getPacked() ||
        recTy.getPadded())
      return false;
    return llvm::all_of(recTy.getMembers(),
                        [](mlir::Type m) { return isSupportedType(m); });
  }
  return false;
}

/// Convert an llvm::abi::Type coercion type back to a scalar CIR type.
static mlir::Type abiTypeToCIR(const llvm::abi::Type *ty, MLIRContext *ctx) {
  if (!ty)
    return nullptr;
  return llvm::TypeSwitch<const llvm::abi::Type *, mlir::Type>(ty)
      .Case(
          [&](const llvm::abi::VoidType *) { return cir::VoidType::get(ctx); })
      .Case([&](const llvm::abi::IntegerType *intTy) {
        return cir::IntType::get(ctx, intTy->getSizeInBits().getFixedValue(),
                                 intTy->isSigned());
      })
      .Case([&](const llvm::abi::FloatType *fltTy) {
        return cir::getFloatingPointType(*fltTy->getSemantics(), ctx);
      })
      .Case([&](const llvm::abi::PointerType *) {
        return cir::PointerType::get(cir::VoidType::get(ctx));
      })
      .Case([&](const llvm::abi::RecordType *recTy) -> mlir::Type {
        SmallVector<mlir::Type> fieldTypes;
        fieldTypes.reserve(recTy->getFields().size());
        for (const auto &field : recTy->getFields()) {
          mlir::Type fieldCIR = abiTypeToCIR(field.FieldType, ctx);
          if (!fieldCIR)
            return nullptr;
          fieldTypes.push_back(fieldCIR);
        }
        // Coercion types are plain register tuples, not the source record.
        return cir::StructType::get(ctx, fieldTypes, /*packed=*/false,
                                    /*padded=*/false, /*is_class=*/false);
      })
      .Default([](const llvm::abi::Type *) -> mlir::Type { return nullptr; });
}

/// Map a CIR type to an llvm::abi::Type.  classifyX86_64Function pre-filters
/// the signature, so only the scalar and struct/array types handled here can
/// reach this function.
static const llvm::abi::Type *mapCIRType(mlir::Type type,
                                         mlir::abi::ABITypeMapper &typeMapper,
                                         const DataLayout &dl, ModuleOp modOp) {
  llvm::abi::TypeBuilder &tb = typeMapper.getTypeBuilder();
  return llvm::TypeSwitch<mlir::Type, const llvm::abi::Type *>(type)
      .Case([&](cir::IntType intTy) {
        return tb.getIntegerType(intTy.getWidth(),
                                 llvm::Align(dl.getTypeABIAlignment(type)),
                                 intTy.isSigned(), intTy.getIsBitInt());
      })
      .Case([&](cir::PointerType ptrTy) {
        unsigned addrSpace = 0;
        if (auto targetAsAttr =
                dyn_cast_if_present<cir::TargetAddressSpaceAttr>(
                    ptrTy.getAddrSpace()))
          addrSpace = targetAsAttr.getValue();
        return tb.getPointerType(dl.getTypeSizeInBits(type),
                                 llvm::Align(dl.getTypeABIAlignment(type)),
                                 addrSpace);
      })
      .Case([&](cir::BoolType) {
        return tb.getIntegerType(dl.getTypeSizeInBits(type),
                                 llvm::Align(dl.getTypeABIAlignment(type)),
                                 /*Signed=*/false);
      })
      .Case([&](cir::VoidType) { return tb.getVoidType(); })
      .Case([&](cir::SingleType) {
        return tb.getFloatType(llvm::APFloat::IEEEsingle(),
                               llvm::Align(dl.getTypeABIAlignment(type)));
      })
      .Case([&](cir::DoubleType) {
        return tb.getFloatType(llvm::APFloat::IEEEdouble(),
                               llvm::Align(dl.getTypeABIAlignment(type)));
      })
      .Case([&](cir::ArrayType arrTy) {
        const llvm::abi::Type *elemAbi =
            mapCIRType(arrTy.getElementType(), typeMapper, dl, modOp);
        return tb.getArrayType(elemAbi, arrTy.getSize(),
                               dl.getTypeSizeInBits(type).getFixedValue());
      })
      .Case([&](cir::RecordType recTy) -> const llvm::abi::Type * {
        // isSupportedType rejects unions, packed / padded, and empty-for-ABI
        // records, so this handles a plain struct: map each field at its
        // naturally-aligned offset.
        SmallVector<llvm::abi::FieldInfo> fields;
        fields.reserve(recTy.getMembers().size());
        uint64_t offsetBits = 0;
        for (mlir::Type fieldTy : recTy.getMembers()) {
          const llvm::abi::Type *mappedField =
              mapCIRType(fieldTy, typeMapper, dl, modOp);
          offsetBits =
              llvm::alignTo(offsetBits, dl.getTypeABIAlignment(fieldTy) * 8);
          fields.push_back(llvm::abi::FieldInfo(mappedField, offsetBits));
          offsetBits += dl.getTypeSizeInBits(fieldTy).getFixedValue();
        }
        llvm::abi::RecordFlags flags = llvm::abi::RecordFlags::None;
        if (recordCanPassInRegs(modOp, recTy))
          flags = flags | llvm::abi::RecordFlags::CanPassInRegisters;
        return tb.getRecordType(fields,
                                llvm::TypeSize::getFixed(
                                    dl.getTypeSizeInBits(type).getFixedValue()),
                                llvm::Align(dl.getTypeABIAlignment(type)),
                                llvm::abi::StructPacking::Default,
                                /*BaseClasses=*/{}, /*VirtualBaseClasses=*/{},
                                flags);
      })
      .Default([](mlir::Type) -> const llvm::abi::Type * {
        llvm_unreachable(
            "mapCIRType: type not pre-filtered by classifyX86_64Function");
      });
}

/// Convert an llvm::abi::ArgInfo into the ArgClassification consumed by
/// CIRABIRewriteContext.
///
/// Direct: the value passes in register(s).  A coercion is forwarded in the
/// three cases where the value has to be rebuilt on the wire: an aggregate
/// unpacked into the register(s) holding it, a scalar too wide for one register
/// split into a tuple of them, and a scalar the classifier widens to fill its
/// eightbyte.  getDirect keeps canFlatten set so the rewriter can split a
/// multi-field coerced struct into individual wire arguments.  Any other scalar
/// passes in its natural CIR type, which a null coercion denotes.  A coercion
/// this bridge cannot represent (an SSE <2 x float>, say) yields std::nullopt
/// so the caller reports NYI rather than silently passing the value unchanged.
///
/// Extend: bool or a sub-register integer needs a signext/zeroext attribute.
/// The x86_64 classifier (llvm/lib/ABI/Targets/X86.cpp) only returns Extend
/// for an integer or bool operand, so any other origTy is asserted rather
/// than silently handled.
///
/// Indirect: an aggregate that does not fit in registers is passed via a
/// pointer (sret for returns, byval for arguments).
///
/// Ignore: a void return, or a zero-field record dropped from the signature.
static std::optional<ArgClassification>
convertABIArgInfo(const llvm::abi::ArgInfo &info, MLIRContext *ctx,
                  mlir::Type origTy) {
  if (info.isDirect()) {
    // The classifier names a coerce type even where it matches the natural
    // type, so a non-null coerce does not by itself mean a rewrite is needed.
    // Leaving a scalar alone also preserves its ABI alignment: abiTypeToCIR
    // drops the bit-precise flag, so a _BitInt(128) routed through it would
    // come back as !cir.int<s, 128> with __int128's 16-byte alignment instead
    // of 8.
    const llvm::abi::Type *coerceAbi = info.getCoerceToType();
    bool isAggregate = isa_and_present<cir::RecordType, cir::ArrayType>(origTy);
    bool coerceIsRegisterTuple =
        isa_and_present<llvm::abi::RecordType>(coerceAbi);
    // Compare widths rather than identity: a coerce no wider than the natural
    // type carries the same value and needs no rewrite.
    auto origInt = dyn_cast_if_present<cir::IntType>(origTy);
    const auto *coerceInt =
        dyn_cast_if_present<llvm::abi::IntegerType>(coerceAbi);
    bool coerceWidensScalar =
        origInt && coerceInt &&
        coerceInt->getSizeInBits().getFixedValue() > origInt.getWidth();
    if (!isAggregate && !coerceIsRegisterTuple && !coerceWidensScalar)
      return ArgClassification::getDirect(nullptr);
    // The coerce must be a type this bridge can represent.  One it cannot map
    // (an SSE vector, or a nested type it does not handle) yields a null type.
    // Report that as NYI instead of leaving the value as an unchanged by-value
    // record.
    mlir::Type coerced = abiTypeToCIR(coerceAbi, ctx);
    if (!coerced)
      return std::nullopt;
    return ArgClassification::getDirect(coerced);
  }
  if (info.isExtend()) {
    if (isa_and_present<cir::BoolType>(origTy))
      return ArgClassification::getExtend(nullptr, info.isSignExt());
    assert((!origTy || isa<cir::IntType>(origTy)) &&
           "the x86_64 classifier only returns Extend for integers and bool");
    mlir::Type extendedTy = abiTypeToCIR(info.getCoerceToType(), ctx);
    return ArgClassification::getExtend(extendedTy, info.isSignExt());
  }
  if (info.isIndirect())
    return ArgClassification::getIndirect(info.getIndirectAlign(),
                                          info.getIndirectByVal());
  assert(info.isIgnore() && "Unexpected classification");
  return ArgClassification::getIgnore();
}

/// Whether a signature accepts arguments beyond its declared parameters, and
/// where the declared ones end when it does.  An argument past the ellipsis is
/// unnamed, and the x86_64 rules pass some unnamed types differently.
///
/// llvm::abi::FunctionInfo::create takes this boundary as an optional count,
/// where an absent value means the signature accepts no optional arguments.
/// Any value there makes FunctionInfo::isVariadic() answer true, so the
/// non-variadic case has to be spelled as an absent one.  That encoding is
/// applied where classifyX86_64Signature builds the FunctionInfo.
///
/// Mirrors classic CodeGen's `RequiredArgs` in `CGFunctionInfo.h`, and CIRGen's
/// copy in `CIRGenFunctionInfo.h`.
class RequiredArgs {
  /// The number of leading arguments that are declared parameters, or ~0U if
  /// the signature accepts no optional arguments.
  unsigned numRequired;

public:
  enum All_t { All };

  /// A signature with no ellipsis, where every argument is declared.
  RequiredArgs(All_t) : numRequired(~0U) {}

  /// A signature whose leading \p n arguments are declared and whose remaining
  /// arguments pass through an ellipsis.
  explicit RequiredArgs(unsigned n) : numRequired(n) { assert(n != ~0U); }

  bool allowsOptionalArgs() const { return numRequired != ~0U; }

  unsigned getNumRequiredArgs() const {
    assert(allowsOptionalArgs() && "signature accepts no optional arguments");
    return numRequired;
  }
};

/// Where \p fnTy's declared parameters end and its ellipsis arguments begin.
///
/// The only x86_64 rule that reads this boundary today sends an unnamed vector
/// wider than 128 bits to memory, and isSupportedType rejects every vector, so
/// no input that currently reaches classification can observe the difference.
static RequiredArgs requiredArgs(cir::FuncType fnTy) {
  if (!fnTy.isVarArg())
    return RequiredArgs::All;
  return RequiredArgs(fnTy.getNumInputs());
}

/// Classify an x86_64 SysV signature (return type + argument types) using the
/// LLVM ABI library.  Shared by the cir.func path, the variadic-call path and
/// the indirect-call path (the latter classifies from the callee function
/// pointer's pointee FuncType).  \p required marks how many leading entries in
/// \p inputs are declared parameters, the classifier treating the rest as
/// arguments passed through an ellipsis.  Returns std::nullopt and emits an NYI
/// error via \p emitError if the signature uses a type the bridge does not
/// handle yet.
static std::optional<FunctionClassification> classifyX86_64Signature(
    mlir::Type retCIR, mlir::TypeRange inputs, RequiredArgs required,
    MLIRContext *ctx, const DataLayout &dl,
    mlir::abi::ABITypeMapper &typeMapper,
    const llvm::abi::TargetInfo &targetInfo, ModuleOp modOp,
    llvm::function_ref<mlir::InFlightDiagnostic()> emitError) {
  assert(retCIR && "signature return type must be non-null");
  assert((!required.allowsOptionalArgs() ||
          required.getNumRequiredArgs() <= inputs.size()) &&
         "declared parameters cannot outnumber the classified arguments");
  bool voidRet = isa<cir::VoidType>(retCIR);

  auto reject = [&](mlir::Type t) -> bool {
    if (isSupportedType(t))
      return false;
    emitError()
        << "x86_64 calling-convention lowering not yet implemented for type "
        << t;
    return true;
  };
  if (!voidRet && reject(retCIR))
    return std::nullopt;
  for (mlir::Type a : inputs)
    if (reject(a))
      return std::nullopt;

  const llvm::abi::Type *retAbi =
      voidRet ? typeMapper.getTypeBuilder().getVoidType()
              : mapCIRType(retCIR, typeMapper, dl, modOp);
  SmallVector<const llvm::abi::Type *> argAbi;
  for (mlir::Type a : inputs)
    argAbi.push_back(mapCIRType(a, typeMapper, dl, modOp));

  std::optional<unsigned> numRequired;
  if (required.allowsOptionalArgs())
    numRequired = required.getNumRequiredArgs();

  std::unique_ptr<llvm::abi::FunctionInfo> fi = llvm::abi::FunctionInfo::create(
      llvm::CallingConv::C, retAbi, argAbi, numRequired);
  targetInfo.computeInfo(*fi);

  // convertABIArgInfo returns nullopt when the classifier picks a coercion
  // this bridge cannot represent (e.g. an SSE vector coerce for an all-float
  // aggregate).  Report it as NYI rather than emitting a wrong signature.
  auto nyiCoercion = [&](mlir::Type t) {
    emitError() << "x86_64 calling-convention lowering not yet "
                   "implemented for the ABI coercion of type "
                << t;
  };

  FunctionClassification fc;
  fc.returnsVoid = voidRet;
  mlir::Type origRet = voidRet ? mlir::Type() : retCIR;
  std::optional<ArgClassification> retAc =
      convertABIArgInfo(fi->getReturnInfo(), ctx, origRet);
  if (!retAc) {
    nyiCoercion(retCIR);
    return std::nullopt;
  }
  fc.returnInfo = *retAc;
  for (unsigned i = 0, e = fi->arg_size(); i < e; ++i) {
    mlir::Type origArg = i < inputs.size() ? inputs[i] : mlir::Type();
    std::optional<ArgClassification> ac =
        convertABIArgInfo(fi->getArgInfo(i).Info, ctx, origArg);
    if (!ac) {
      nyiCoercion(origArg);
      return std::nullopt;
    }
    fc.argInfos.push_back(*ac);
  }
  return fc;
}

/// Classify a cir.func for x86_64 SysV using the LLVM ABI library.  Returns
/// std::nullopt and emits an NYI error if the signature uses a type the bridge
/// does not handle yet.
static std::optional<FunctionClassification>
classifyX86_64Function(cir::FuncOp func, const DataLayout &dl,
                       mlir::abi::ABITypeMapper &typeMapper,
                       const llvm::abi::TargetInfo &targetInfo,
                       ModuleOp modOp) {
  cir::FuncType fnTy = func.getFunctionType();
  return classifyX86_64Signature(fnTy.getReturnType(), fnTy.getInputs(),
                                 requiredArgs(fnTy), func->getContext(), dl,
                                 typeMapper, targetInfo, modOp,
                                 [&]() { return func.emitOpError(); });
}

/// Classify a call that passes arguments through an ellipsis.  The callee's
/// own classification covers only its declared parameters, but an ellipsis
/// argument competes for the same argument registers as a declared one, so
/// what the ABI does with it depends on the whole argument list: the same
/// small struct is passed in registers early in the list and in memory once
/// the integer registers are gone.  Classifying from the call's operands
/// rather than the callee's signature is what makes that accounting right.
static std::optional<FunctionClassification> classifyX86_64VariadicCall(
    cir::CIRCallOpInterface call, cir::FuncType calleeTy, const DataLayout &dl,
    mlir::abi::ABITypeMapper &typeMapper,
    const llvm::abi::TargetInfo &targetInfo, ModuleOp modOp) {
  assert(calleeTy.isVarArg() &&
         "only a variadic callee can take more operands than it declares");
  Operation *op = call.getOperation();
  return classifyX86_64Signature(
      calleeTy.getReturnType(), call.getArgOperands().getTypes(),
      requiredArgs(calleeTy), op->getContext(), dl, typeMapper, targetInfo,
      modOp, [&]() { return op->emitOpError(); });
}

#ifndef NDEBUG
/// Whether \p callFc classifies a call's leading arguments and its return
/// exactly as \p calleeFc classifies the callee's declared parameters and
/// return.  A function definition and its call sites are rewritten from
/// separate classifications, so the two would silently disagree on the wire
/// format if an ellipsis argument could ever change how a declared parameter
/// is passed.
static bool classifiesSamePrefix(const FunctionClassification &calleeFc,
                                 const FunctionClassification &callFc) {
  if (callFc.argInfos.size() < calleeFc.argInfos.size())
    return false;
  return calleeFc.returnInfo == callFc.returnInfo &&
         std::equal(calleeFc.argInfos.begin(), calleeFc.argInfos.end(),
                    callFc.argInfos.begin());
}
#endif

struct CallConvLoweringPass
    : public impl::CallConvLoweringBase<CallConvLoweringPass> {
  using CallConvLoweringBase::CallConvLoweringBase;
  void runOnOperation() override;
};

/// Record on \p fc whether \p returnType is CIR's void.  The x86_64 classifier
/// answers this itself, but the other two drivers cannot: the test target is
/// dialect-neutral and has no notion of CIR's void, and the
/// classification-attr schema carries no return type at all.  Both route
/// through here so a classification always reaches needsRewrite paired with
/// the return type it was built from.
static std::optional<FunctionClassification>
withReturnVoidness(std::optional<FunctionClassification> fc,
                   mlir::Type returnType) {
  if (fc)
    fc->returnsVoid = mlir::isa<cir::VoidType>(returnType);
  return fc;
}

/// Classify \p func using whichever driver mode is configured.  Returns
/// std::nullopt and emits an error on the function if classification fails
/// (e.g. injection-driver mode but the function is missing the attribute,
/// or the attribute is malformed).
std::optional<FunctionClassification>
classifyFunction(cir::FuncOp func, const DataLayout &dl,
                 cir::CallConvTarget target, StringRef classificationAttrName) {
  ArrayRef<Type> argTypes = func.getFunctionType().getInputs();
  Type returnType = func.getFunctionType().getReturnType();

  if (!classificationAttrName.empty()) {
    auto attr = func->getAttrOfType<DictionaryAttr>(classificationAttrName);
    if (!attr) {
      func.emitOpError()
          << "missing classification attribute '" << classificationAttrName
          << "' (CallConvLowering driver mode 'classification-attr')";
      return std::nullopt;
    }
    return withReturnVoidness(mlir::abi::test::parseClassificationAttr(
                                  attr, [&]() { return func.emitOpError(); }),
                              returnType);
  }

  // The x86_64 target is handled directly in runOnOperation (it needs a shared
  // ABITypeMapper and TargetInfo), so only the test target reaches here.
  assert(target == cir::CallConvTarget::Test &&
         "classifyFunction only handles the test target");
  return withReturnVoidness(mlir::abi::test::classify(argTypes, returnType, dl),
                            returnType);
}

/// Find the cir.func declaration matching a direct cir.call / cir.try_call
/// callee, if any.  Returns nullptr if the callee is indirect or the symbol
/// cannot be resolved.  Takes a SymbolTable instead of a ModuleOp so the
/// symbol lookup is amortized across all the call sites the driver walks
/// (ModuleOp::lookupSymbol is linear per call).
cir::FuncOp lookupCallee(Operation *callOp, SymbolTable &symbolTable) {
  FlatSymbolRefAttr callee;
  if (auto call = dyn_cast<cir::CallOp>(callOp))
    callee = call.getCalleeAttr();
  else if (auto tryCall = dyn_cast<cir::TryCallOp>(callOp))
    callee = tryCall.getCalleeAttr();
  else
    return nullptr;
  if (!callee)
    return nullptr;
  return symbolTable.lookup<cir::FuncOp>(callee.getValue());
}

/// The signature an indirect call reaches its callee through, or a null type
/// for a direct call.  The callee's pointer-to-function shape is asserted
/// rather than verified: the dialect checks operand types against the callee
/// only for a direct call, so IR that breaks it fails here instead of in the
/// verifier.
cir::FuncType indirectCalleeType(cir::CIRCallOpInterface call) {
  if (!call.isIndirect())
    return {};
  return cast<cir::FuncType>(
      cast<cir::PointerType>(call.getIndirectCall().getType()).getPointee());
}

void CallConvLoweringPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();

  bool haveTarget = target != cir::CallConvTarget::None;
  bool haveAttr = !classificationAttr.empty();
  if (haveTarget == haveAttr) {
    moduleOp.emitOpError() << "CallConvLowering requires exactly one of "
                              "'target' or 'classification-attr' pass options";
    signalPassFailure();
    return;
  }

  if (!moduleOp->hasAttr(DLTIDialect::kDataLayoutAttrName)) {
    moduleOp.emitOpError()
        << "CallConvLowering requires a DataLayout (dlti.dl_spec attribute "
           "on the module)";
    signalPassFailure();
    return;
  }

  DataLayout dl(moduleOp);
  CIRABIRewriteContext rewriteCtx(moduleOp, dl);
  SymbolTable symbolTable(moduleOp);

  // For the x86_64 target, build the LLVM ABI library classifier once and
  // reuse it (and its type mapper) across every function.
  std::optional<mlir::abi::ABITypeMapper> x86TypeMapper;
  std::unique_ptr<llvm::abi::TargetInfo> x86Target;
  if (target == cir::CallConvTarget::X86_64) {
    x86TypeMapper.emplace(dl);
    x86Target = llvm::abi::createX86_64TargetInfo(
        x86TypeMapper->getTypeBuilder(), x86AvxAbiLevel.getValue(),
        /*Has64BitPointers=*/true, llvm::abi::ABICompatInfo());
  }

  // Classify every cir.func up front.  No IR mutation happens here, so
  // later walks can consult any function's classification regardless of
  // visitation order.
  llvm::MapVector<cir::FuncOp, FunctionClassification> classifications;
  bool anyFailed = false;
  moduleOp.walk([&](cir::FuncOp f) {
    std::optional<FunctionClassification> fc;
    if (x86Target)
      fc = classifyX86_64Function(f, dl, *x86TypeMapper, *x86Target, moduleOp);
    else
      fc = classifyFunction(f, dl, target, classificationAttr);
    if (!fc) {
      anyFailed = true;
      return;
    }
    classifications.insert({f, std::move(*fc)});
  });
  if (anyFailed) {
    signalPassFailure();
    return;
  }

  // Build a callee-to-callers index.  One module walk collects every direct
  // cir.call / cir.try_call to each cir.func; the loop below rewrites a
  // function and all of its call sites together.  Indirect or unresolved
  // callees are skipped here; rewriteCallSite errors on those at the end.
  //
  // A call that passes arguments through an ellipsis gets its own
  // classification, recorded here while every signature is still in its
  // original form.  The callee's classification covers only its declared
  // parameters and cannot describe those extra arguments.
  llvm::DenseMap<cir::FuncOp, SmallVector<Operation *>> callers;
  // Keyed on the call op collected below, looked up once when that same op is
  // rewritten.  A key must never come from an op created during the rewrite:
  // a recycled address could match an unrelated entry.
  llvm::DenseMap<Operation *, FunctionClassification> variadicCallSites;
  moduleOp.walk([&](Operation *op) {
    auto call = dyn_cast<cir::CIRCallOpInterface>(op);
    if (!call)
      return;
    cir::FuncOp callee = lookupCallee(op, symbolTable);
    if (!callee)
      return;
    callers[callee].push_back(op);

    // Only the x86_64 driver classifies per call site.  Under the other
    // drivers the classification comes from a fixed per-function source, so
    // such a call stays short a classification and rewriteCallSite reports it.
    cir::FuncType calleeTy = callee.getFunctionType();
    if (!x86Target || call.getNumArgOperands() <= calleeTy.getNumInputs())
      return;
    // A callee declared without a prototype also takes more operands than it
    // declares, and the verifier allows it.  Those extra arguments are named
    // rather than passed through an ellipsis, so the accounting below does not
    // describe them.
    if (!calleeTy.isVarArg()) {
      op->emitOpError() << "extra arguments to a callee without a prototype "
                           "not yet implemented in CallConvLowering";
      anyFailed = true;
      return;
    }
    std::optional<FunctionClassification> fc = classifyX86_64VariadicCall(
        call, calleeTy, dl, *x86TypeMapper, *x86Target, moduleOp);
    if (!fc) {
      anyFailed = true;
      return;
    }
    variadicCallSites.insert({op, std::move(*fc)});
  });
  if (anyFailed) {
    signalPassFailure();
    return;
  }

  // Rewrite each function together with every direct call to it.  By the
  // time we move on to function F+1, F's signature and every direct call to
  // F have already been brought into alignment, and F+1..FN are still in
  // their original (mutually consistent) form, so the IR is verifier-clean
  // at every outer-iteration boundary.
  //
  // There is still a brief inner window where F's signature has been
  // rewritten but its callers have not yet caught up -- we have no way to
  // mutate both sides of a call atomically.  No verifier runs inside the
  // pass, and at pass exit the module is verifier-clean.  Fusing the inner
  // loop here keeps the invalid window per-function rather than module-wide.
  OpBuilder builder(ctx);
  for (auto &kv : classifications) {
    cir::FuncOp func = kv.first;
    const FunctionClassification &fc = kv.second;
    if (failed(rewriteCtx.rewriteFunctionDefinition(func, fc, builder))) {
      signalPassFailure();
      return;
    }
    for (Operation *callOp : callers.lookup(func)) {
      const FunctionClassification *callFc = &fc;
      if (auto it = variadicCallSites.find(callOp);
          it != variadicCallSites.end()) {
        callFc = &it->second;
        assert(classifiesSamePrefix(fc, *callFc) &&
               "a call site's declared parameters must be classified the same "
               "way as the callee's");
      }
      if (failed(rewriteCtx.rewriteCallSite(callOp, *callFc, builder))) {
        signalPassFailure();
        return;
      }
    }
  }

  // Rewrite indirect call sites.  The callee is opaque, so classify from the
  // function pointer's pointee FuncType and let rewriteCallSite retype the
  // callee pointer to match the coerced signature.  Collect the calls first:
  // when an sret rewrite reuses a single-use store's destination as the return
  // slot it erases that store, which is the operation a live walk has already
  // cached as the next one to visit.
  SmallVector<cir::CIRCallOpInterface> indirectCalls;
  moduleOp.walk([&](cir::CIRCallOpInterface c) {
    cir::FuncType calleeTy = indirectCalleeType(c);
    if (!calleeTy)
      return;
    // A cir.try_call is in this walk so that a variadic one reaches the
    // ellipsis accounting below.  CIRABIRewriteContext cannot rebuild a
    // cir.try_call at all, so a non-variadic one has never been rewritten
    // here.  Keep it out rather than start reporting a gap that has nothing
    // to do with the ellipsis.
    if (!calleeTy.isVarArg() && isa<cir::TryCallOp>(c.getOperation()))
      return;
    indirectCalls.push_back(c);
  });
  for (cir::CIRCallOpInterface c : indirectCalls) {
    // classification-attr mode injects a per-function classification, which
    // cannot describe a callee resolved at run time.  Report it rather than
    // leave the indirect call unrewritten while direct calls are coerced.
    if (!classificationAttr.empty()) {
      c->emitOpError() << "indirect call cannot be classified in the "
                          "'classification-attr' driver mode";
      signalPassFailure();
      return;
    }
    cir::FuncType funcTy = indirectCalleeType(c);
    auto classifySignature =
        [&](mlir::TypeRange argTypes) -> std::optional<FunctionClassification> {
      if (x86Target)
        return classifyX86_64Signature(funcTy.getReturnType(), argTypes,
                                       requiredArgs(funcTy), ctx, dl,
                                       *x86TypeMapper, *x86Target, moduleOp,
                                       [&]() { return c->emitOpError(); });
      return withReturnVoidness(
          mlir::abi::test::classify(argTypes, funcTy.getReturnType(), dl),
          funcTy.getReturnType());
    };

    // An argument passed through an ellipsis has no counterpart in the
    // pointee's parameter list, so classify the call's own operands to learn
    // what the ABI does with it.  If nothing in the full list needs a rewrite
    // the call already carries its wire form and can stand as written.
    // Anything else needs a rewrite the pointee's signature cannot describe,
    // since it has no entry for the arguments past the ellipsis.
    if (c.getNumArgOperands() > funcTy.getNumInputs()) {
      std::optional<FunctionClassification> callFc =
          classifySignature(c.getArgOperands().getTypes());
      if (!callFc) {
        signalPassFailure();
        return;
      }
      if (!callFc->needsRewrite())
        continue;
      c->emitOpError() << "variadic arguments to an indirect call not yet "
                          "implemented in CallConvLowering";
      signalPassFailure();
      return;
    }

    std::optional<FunctionClassification> fc =
        classifySignature(funcTy.getInputs());
    if (!fc) {
      signalPassFailure();
      return;
    }
    if (failed(rewriteCtx.rewriteCallSite(c.getOperation(), *fc, builder))) {
      signalPassFailure();
      return;
    }
  }
}

} // namespace

std::unique_ptr<Pass> mlir::createCallConvLoweringPass() {
  return std::make_unique<CallConvLoweringPass>();
}

std::unique_ptr<Pass>
mlir::createCallConvLoweringPass(cir::CallConvTarget target,
                                 llvm::abi::X86AVXABILevel x86AvxAbiLevel) {
  CallConvLoweringOptions options;
  options.target = target;
  options.x86AvxAbiLevel = x86AvxAbiLevel;
  return std::make_unique<CallConvLoweringPass>(options);
}
