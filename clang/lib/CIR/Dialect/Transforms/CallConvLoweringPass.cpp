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
// x86_64 System V classifier bridge (scalar types)
//
// Maps scalar CIR types to llvm::abi::Type, runs the LLVM ABI Lowering
// Library's SysV x86_64 classifier, and converts the result back into the
// dialect-agnostic mlir::abi::FunctionClassification that CIRABIRewriteContext
// consumes.  Only integer / pointer / bool / f32 / f64 signatures are handled;
// aggregates and other leaf types are reported NYI by classifyX86_64Function
// so an unsupported signature fails the pass instead of being misclassified.
//===----------------------------------------------------------------------===//

/// llvm::Align requires a power of two; DataLayout can report non-power-of-two
/// alignments for unusual types.
static llvm::Align safeAlign(uint64_t a) {
  return llvm::Align(llvm::PowerOf2Ceil(std::max<uint64_t>(a, 1)));
}

/// The scalar CIR types the x86_64 bridge handles.  A regular integer up to
/// 64 bits, pointer, bool, void, f32, or f64 is a single-register Direct or
/// Extend argument.  `_BitInt`, `__int128`, and wider/other types need coercion
/// or indirect passing, which this scalar bridge does not do.
static bool isSupportedScalarType(mlir::Type ty) {
  if (isa<cir::VoidType, cir::BoolType, cir::PointerType, cir::SingleType,
          cir::DoubleType>(ty))
    return true;
  if (auto intTy = dyn_cast<cir::IntType>(ty))
    return !intTy.getIsBitInt() && intTy.getWidth() <= 64;
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
      .Default([](const llvm::abi::Type *) -> mlir::Type { return nullptr; });
}

/// Map a scalar CIR type to an llvm::abi::Type.  classifyX86_64Function
/// pre-filters the signature, so only the scalar types handled here can
/// reach this function.
static const llvm::abi::Type *mapCIRType(mlir::Type type,
                                         mlir::abi::ABITypeMapper &typeMapper,
                                         const DataLayout &dl) {
  llvm::abi::TypeBuilder &tb = typeMapper.getTypeBuilder();
  return llvm::TypeSwitch<mlir::Type, const llvm::abi::Type *>(type)
      .Case([&](cir::IntType intTy) {
        return tb.getIntegerType(intTy.getWidth(),
                                 safeAlign(dl.getTypeABIAlignment(type)),
                                 intTy.isSigned());
      })
      .Case([&](cir::PointerType ptrTy) {
        unsigned addrSpace = 0;
        if (auto targetAsAttr =
                dyn_cast_if_present<cir::TargetAddressSpaceAttr>(
                    ptrTy.getAddrSpace()))
          addrSpace = targetAsAttr.getValue();
        return tb.getPointerType(dl.getTypeSizeInBits(type),
                                 safeAlign(dl.getTypeABIAlignment(type)),
                                 addrSpace);
      })
      .Case([&](cir::BoolType) {
        return tb.getIntegerType(dl.getTypeSizeInBits(type),
                                 safeAlign(dl.getTypeABIAlignment(type)),
                                 /*Signed=*/false);
      })
      .Case([&](cir::VoidType) { return tb.getVoidType(); })
      .Case([&](cir::SingleType) {
        return tb.getFloatType(llvm::APFloat::IEEEsingle(),
                               safeAlign(dl.getTypeABIAlignment(type)));
      })
      .Case([&](cir::DoubleType) {
        return tb.getFloatType(llvm::APFloat::IEEEdouble(),
                               safeAlign(dl.getTypeABIAlignment(type)));
      })
      .Default([](mlir::Type) -> const llvm::abi::Type * {
        llvm_unreachable(
            "mapCIRType: type not pre-filtered by classifyX86_64Function");
      });
}

/// Convert an llvm::abi::ArgInfo for a scalar type into the ArgClassification
/// consumed by CIRABIRewriteContext.
///
/// Direct: every scalar this bridge maps passes as-is, so no coercion type
/// is needed (nullptr means "same as the original CIR type").
///
/// Extend: bool or a sub-register integer needs a signext/zeroext attribute.
/// Every ArgInfo::getExtend() call site in the x86_64 classifier
/// (llvm/lib/ABI/Targets/X86.cpp) is gated on the operand being an integer,
/// so a non-integer, non-bool origTy here would mean the classifier
/// disagreed with its own source -- asserted rather than silently handled.
///
/// Indirect: needed once records/_BitInt/vectors are supported (sret,
/// byval, and large _BitInt all classify Indirect), but
/// X86_64TargetInfo::getIndirectResult()/getIndirectReturnResult() only
/// return Indirect for aggregates or _BitInt, neither of which the
/// scalar-only type set this bridge accepts can produce.  Unreachable until
/// a later PR adds those types and the record/vector/complex/int type gate
/// that belongs here.
///
/// Ignore: a void return has no register or stack slot.
static ArgClassification convertABIArgInfo(const llvm::abi::ArgInfo &info,
                                           MLIRContext *ctx,
                                           mlir::Type origTy) {
  if (info.isDirect())
    return ArgClassification::getDirect(nullptr);
  if (info.isExtend()) {
    if (origTy && isa<cir::BoolType>(origTy))
      return ArgClassification::getExtend(nullptr, info.isSignExt());
    assert((!origTy || isa<cir::IntType>(origTy)) &&
           "the x86_64 classifier only returns Extend for integers and bool");
    mlir::Type extendedTy = abiTypeToCIR(info.getCoerceToType(), ctx);
    return ArgClassification::getExtend(extendedTy, info.isSignExt());
  }
  if (info.isIndirect())
    llvm_unreachable("Indirect classification is impossible for the "
                     "scalar-only type set this bridge accepts");
  return ArgClassification::getIgnore();
}

/// Classify a cir.func for x86_64 SysV using the LLVM ABI library.  Returns
/// std::nullopt and emits an NYI error if the signature uses a type the scalar
/// bridge does not handle yet.
static std::optional<FunctionClassification>
classifyX86_64Function(cir::FuncOp func, const DataLayout &dl,
                       mlir::abi::ABITypeMapper &typeMapper,
                       const llvm::abi::TargetInfo &targetInfo) {
  MLIRContext *ctx = func->getContext();
  cir::FuncType fnTy = func.getFunctionType();
  mlir::Type retCIR = fnTy.getReturnType();
  assert(retCIR && "FuncType::getReturnType() never returns null");
  bool voidRet = isa<cir::VoidType>(retCIR);

  auto reject = [&](mlir::Type t) -> bool {
    if (isSupportedScalarType(t))
      return false;
    func.emitOpError()
        << "x86_64 calling-convention lowering not yet implemented for type "
        << t;
    return true;
  };
  if (!voidRet && reject(retCIR))
    return std::nullopt;
  for (mlir::Type a : fnTy.getInputs())
    if (reject(a))
      return std::nullopt;

  const llvm::abi::Type *retAbi =
      voidRet ? typeMapper.getTypeBuilder().getVoidType()
              : mapCIRType(retCIR, typeMapper, dl);
  SmallVector<const llvm::abi::Type *> argAbi;
  for (mlir::Type a : fnTy.getInputs())
    argAbi.push_back(mapCIRType(a, typeMapper, dl));

  std::unique_ptr<llvm::abi::FunctionInfo> fi =
      llvm::abi::FunctionInfo::create(llvm::CallingConv::C, retAbi, argAbi);
  targetInfo.computeInfo(*fi);

  FunctionClassification fc;
  mlir::Type origRet = voidRet ? mlir::Type() : retCIR;
  fc.returnInfo = convertABIArgInfo(fi->getReturnInfo(), ctx, origRet);
  auto inputs = fnTy.getInputs();
  for (unsigned i = 0, e = fi->arg_size(); i < e; ++i) {
    mlir::Type origArg = i < inputs.size() ? inputs[i] : mlir::Type();
    fc.argInfos.push_back(
        convertABIArgInfo(fi->getArgInfo(i).Info, ctx, origArg));
  }
  return fc;
}

/// The classifier targets this pass drives internally via a shared
/// ABITypeMapper/TargetInfo.  (The classification-attr injection mode is
/// orthogonal -- it names an arbitrary attribute, not a fixed target -- and
/// stays string-keyed.)  kCallConvTargetNames is the single source of truth
/// for both parsing the `target` pass option and the unknown-target
/// diagnostic in classifyFunction; it is indexed by the enum value below.
enum class CallConvTarget { Test, X86_64, Last = X86_64 };

constexpr llvm::StringRef kCallConvTargetNames[] = {"test", "x86_64"};
static_assert(std::size(kCallConvTargetNames) ==
                  static_cast<size_t>(CallConvTarget::Last) + 1,
              "kCallConvTargetNames must have one entry per CallConvTarget");

std::optional<CallConvTarget> parseCallConvTarget(StringRef target) {
  for (size_t i = 0; i < std::size(kCallConvTargetNames); ++i)
    if (target == kCallConvTargetNames[i])
      return static_cast<CallConvTarget>(i);
  return std::nullopt;
}

std::string supportedCallConvTargets() {
  std::string result;
  for (llvm::StringRef name : kCallConvTargetNames) {
    if (!result.empty())
      result += ", ";
    result += name.str();
  }
  return result;
}

bool needsRewrite(const FunctionClassification &fc) {
  if ((fc.returnInfo.kind != ArgKind::Direct) || fc.returnInfo.coercedType)
    return true;
  for (const ArgClassification &ac : fc.argInfos)
    if ((ac.kind != ArgKind::Direct) || ac.coercedType)
      return true;
  return false;
}

struct CallConvLoweringPass
    : public impl::CallConvLoweringBase<CallConvLoweringPass> {
  using CallConvLoweringBase::CallConvLoweringBase;
  void runOnOperation() override;
};

/// Classify \p func using whichever driver mode is configured.  Returns
/// std::nullopt and emits an error on the function if classification fails
/// (e.g. injection-driver mode but the function is missing the attribute,
/// or the attribute is malformed).
std::optional<FunctionClassification>
classifyFunction(cir::FuncOp func, const DataLayout &dl, StringRef target,
                 StringRef classificationAttrName) {
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
    return mlir::abi::test::parseClassificationAttr(
        attr, [&]() { return func.emitOpError(); });
  }

  if (parseCallConvTarget(target) == CallConvTarget::Test)
    return mlir::abi::test::classify(argTypes, returnType, dl);

  // Note: the "x86_64" target is handled directly in runOnOperation (it needs
  // a shared ABITypeMapper and TargetInfo), so it never reaches here.
  func.emitOpError() << "unknown target '" << target
                     << "' (supported: " << supportedCallConvTargets() << ")";
  return std::nullopt;
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

void CallConvLoweringPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();
  MLIRContext *ctx = &getContext();

  if (target.empty() == classificationAttr.empty()) {
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
  if (parseCallConvTarget(target) == CallConvTarget::X86_64) {
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
      fc = classifyX86_64Function(f, dl, *x86TypeMapper, *x86Target);
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
  llvm::DenseMap<cir::FuncOp, SmallVector<Operation *>> callers;
  moduleOp.walk([&](Operation *op) {
    if (!isa<cir::CallOp, cir::TryCallOp>(op))
      return;
    if (cir::FuncOp callee = lookupCallee(op, symbolTable))
      callers[callee].push_back(op);
  });

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
      if (failed(rewriteCtx.rewriteCallSite(callOp, fc, builder))) {
        signalPassFailure();
        return;
      }
    }
  }

  // Reject indirect calls when the module contains any ABI rewrite that
  // would need call-site lowering.  We cannot strip or coerce operands
  // without a resolved callee symbol.
  const FunctionClassification *rewriteFc = nullptr;
  for (auto &kv : classifications) {
    if (needsRewrite(kv.second)) {
      rewriteFc = &kv.second;
      break;
    }
  }
  if (rewriteFc) {
    moduleOp.walk([&](cir::CallOp c) {
      if (!c.isIndirect())
        return;
      if (failed(rewriteCtx.rewriteCallSite(c, *rewriteFc, builder)))
        anyFailed = true;
    });
    if (anyFailed) {
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
mlir::createCallConvLoweringPass(llvm::StringRef target,
                                 llvm::abi::X86AVXABILevel x86AvxAbiLevel) {
  CallConvLoweringOptions options;
  options.target = target.str();
  options.x86AvxAbiLevel = x86AvxAbiLevel;
  return std::make_unique<CallConvLoweringPass>(options);
}
