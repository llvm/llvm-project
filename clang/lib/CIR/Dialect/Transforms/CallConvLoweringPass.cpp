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

#include "CIRABIClassifier.h"
#include "PassDetail.h"
#include "TargetLowering/CIRABIRewriteContext.h"

#include "mlir/ABI/ABIRewriteContext.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/Passes.h"
#include "llvm/ABI/TargetInfo.h"
#include "llvm/ADT/ScopeExit.h"

#include <algorithm>

using namespace mlir;
using namespace mlir::abi;
using namespace cir;

namespace mlir {
#define GEN_PASS_DEF_CALLCONVLOWERING
#include "clang/CIR/Dialect/Passes.h.inc"
} // namespace mlir

/// Whether \p ty, or an aggregate member/element reached by value (never
/// through a pointer), is an incomplete record.  Such a record has no known
/// layout, so no eightbyte classification can be built for it.
static bool hasIncompleteRecordByValue(mlir::Type ty) {
  if (auto recTy = dyn_cast<cir::RecordType>(ty))
    return !recTy.isComplete() ||
           llvm::any_of(recTy.getMembers(), hasIncompleteRecordByValue);
  if (auto arrTy = dyn_cast<cir::ArrayType>(ty))
    return hasIncompleteRecordByValue(arrTy.getElementType());
  return false;
}

/// Whether \p fc gives the callee access to memory through a pointer the ABI
/// introduced: an sret slot for an indirect return, or an indirect argument.
static bool hasIndirectSlot(const FunctionClassification &fc) {
  if (fc.returnInfo.kind == ArgKind::Indirect)
    return true;
  return llvm::any_of(fc.argInfos, [](const ArgClassification &ac) {
    return ac.kind == ArgKind::Indirect;
  });
}

/// Record on \p op that the memory its pointer arguments address may be read
/// and written.  Absent effects already allow that, since an absent attribute
/// means the effects are unknown, so they are left absent.
template <typename OpTy> static void widenForArgMemory(OpTy op) {
  cir::MemoryEffectsAttr effects = op.getMemoryEffectsAttr();
  if (!effects)
    return;
  op.setMemoryEffectsAttr(cir::MemoryEffectsAttr::get(
      effects.getContext(), effects.getOther(), cir::ModRefInfo::ModRef,
      effects.getInaccessibleMem(), effects.getErrnoMem(),
      effects.getTargetMem0(), effects.getTargetMem1()));
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

namespace {

struct CallConvLoweringPass
    : public impl::CallConvLoweringBase<CallConvLoweringPass> {
  using CallConvLoweringBase::CallConvLoweringBase;

  CallConvLoweringPass(const CallConvLoweringOptions &options,
                       const llvm::abi::X86ABICompatInfo &x86AbiCompat)
      : CallConvLoweringBase(options), x86AbiCompat(x86AbiCompat) {}

  void runOnOperation() override;

  /// The x86_64 flags whose value depends on the target and the requested ABI
  /// compatibility version.  Carried outside the pass options because the
  /// struct has no command-line parser, so a cir-opt run gets the library
  /// defaults rather than a target's values.
  llvm::abi::X86ABICompatInfo x86AbiCompat;
};

} // namespace

/// Build the classifier the pass options select.  Exactly one of \p target
/// and \p classificationAttrName is set, which runOnOperation checks before
/// calling.
static std::unique_ptr<CIRABIClassifier>
createABIClassifier(cir::CallConvTarget target,
                    StringRef classificationAttrName, const DataLayout &dl,
                    ModuleOp modOp, const ABIClassifierOptions &options) {
  if (!classificationAttrName.empty())
    return createInjectedABIClassifier(dl, classificationAttrName);

  switch (target) {
  case cir::CallConvTarget::Test:
    return createTestTargetABIClassifier(dl);
  case cir::CallConvTarget::X86_64:
    return createX86_64ABIClassifier(dl, modOp, options);
  case cir::CallConvTarget::AMDGPU:
    return createAMDGPUABIClassifier(dl, modOp);
  case cir::CallConvTarget::None:
    break;
  }
  llvm_unreachable("an unset driver mode is rejected before this point");
}

namespace {

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
  // A non-byval indirect parameter's slot outlives the rewrite that retypes
  // the parameter, so that a call forwarding the parameter can still recognize
  // it.  Draining on scope exit collapses those slots whichever way this
  // function returns.
  llvm::scope_exit drainParamSlots(
      [&] { rewriteCtx.finalizeParameterSlots(); });
  SymbolTable symbolTable(moduleOp);

  cir::ABIClassifierOptions classifierOptions;
  classifierOptions.x86AvxAbiLevel = x86AvxAbiLevel;
  classifierOptions.allowsX86TargetAttrAvx = allowsX86TargetAttrAvx;
  classifierOptions.x86AbiCompat = x86AbiCompat;
  std::unique_ptr<cir::CIRABIClassifier> classifier = createABIClassifier(
      target, classificationAttr, dl, moduleOp, classifierOptions);

  // Classify every cir.func up front.  No IR mutation happens here, so
  // later walks can consult any function's classification regardless of
  // visitation order.
  llvm::MapVector<cir::FuncOp, FunctionClassification> classifications;
  bool anyFailed = false;
  moduleOp.walk([&](cir::FuncOp f) {
    // A complete type is required at any call or definition, so only a
    // declaration can carry an incomplete-by-value parameter or return type,
    // and no translation unit can ever call or define it with real argument
    // data.  Leave it unclassified.
    cir::FuncType fnTy = f.getFunctionType();
    if (f.isDeclaration() &&
        (hasIncompleteRecordByValue(fnTy.getReturnType()) ||
         llvm::any_of(fnTy.getInputs(), hasIncompleteRecordByValue)))
      return;
    std::optional<FunctionClassification> fc = classifier->classifyFunction(f);
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

    cir::FuncType calleeTy = callee.getFunctionType();
    if (call.getNumArgOperands() <= calleeTy.getNumInputs())
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
    std::optional<FunctionClassification> fc =
        classifier->classifyVariadicCall(call, callee);
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

  // A cir.get_global holding a function's address carries the signature the
  // source wrote, which the verifier ties to the callee, so it goes stale when
  // that callee is rewritten.  A function address in a global initializer is a
  // GlobalViewAttr instead, whose recorded type the verifier does not tie to
  // the callee and which is dropped at opaque-pointer lowering, so it needs no
  // counterpart.
  llvm::DenseMap<cir::FuncOp, SmallVector<cir::GetGlobalOp>> addressTakers;
  moduleOp.walk([&](cir::GetGlobalOp getGlobal) {
    auto ptrTy = cast<cir::PointerType>(getGlobal.getAddr().getType());
    if (!isa<cir::FuncType>(ptrTy.getPointee()))
      return;
    // A get_global's pointee must equal the named symbol's type, and the
    // GlobalOp verifier rejects a function type there, so this names a
    // cir.func.
    auto callee = cast<cir::FuncOp>(symbolTable.lookup(getGlobal.getName()));
    addressTakers[callee].push_back(getGlobal);
  });

  // Restate every non-byval indirect parameter's slot alignment and route any
  // use of such a parameter as a call argument through a load of that slot,
  // before any definition or call site is rewritten.  Doing this up front
  // makes the forwarding decision independent of the order the callee and its
  // caller were declared in.
  for (auto &kv : classifications) {
    if (failed(rewriteCtx.prepareNonByvalParameters(kv.first, kv.second))) {
      signalPassFailure();
      return;
    }
  }

  // An sret slot or an indirect argument is a pointer the source never
  // wrote, and an ellipsis can carry one the declared parameters do not
  // describe.  The callee reads and writes that memory whatever const or
  // pure recorded about it, so the recorded effects have to widen.
  for (auto &kv : classifications) {
    cir::FuncOp func = kv.first;
    if (hasIndirectSlot(kv.second) || func.getFunctionType().isVarArg())
      widenForArgMemory(func);
  }

  // Rewrite each function together with every direct call to it and every op
  // holding its address.  By the time we move on to function F+1, F's
  // signature and every reference to F have already been brought into
  // alignment, and F+1..FN are still in their original (mutually consistent)
  // form, so the IR is verifier-clean at every outer-iteration boundary.
  //
  // There is still a brief inner window where F's signature has been
  // rewritten but its references have not yet caught up -- we have no way to
  // mutate both sides of a call atomically.  No verifier runs inside the
  // pass, and at pass exit the module is verifier-clean.  Fusing the inner
  // loops here keeps the invalid window per-function rather than module-wide.
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
      if (hasIndirectSlot(*callFc))
        widenForArgMemory(cast<cir::CIRCallOpInterface>(callOp));
      if (failed(rewriteCtx.rewriteCallSite(callOp, *callFc, builder))) {
        signalPassFailure();
        return;
      }
    }
    for (cir::GetGlobalOp addrOp : addressTakers.lookup(func))
      rewriteCtx.rewriteFunctionAddress(addrOp, func, builder);
  }

  // Rewrite indirect call sites.  The callee is opaque, so classify from the
  // function pointer's pointee FuncType and let rewriteCallSite retype the
  // callee pointer to match the coerced signature.  Collect the calls first:
  // when an sret rewrite reuses a single-use store's destination as the return
  // slot it erases that store, which is the operation a live walk has already
  // cached as the next one to visit.
  SmallVector<cir::CIRCallOpInterface> indirectCalls;
  moduleOp.walk([&](cir::CIRCallOpInterface c) {
    if (c.isIndirect())
      indirectCalls.push_back(c);
  });
  for (cir::CIRCallOpInterface c : indirectCalls) {
    std::optional<FunctionClassification> fc =
        classifier->classifyIndirectCall(c);
    if (!fc) {
      signalPassFailure();
      return;
    }
    if (hasIndirectSlot(*fc))
      widenForArgMemory(c);
    if (failed(rewriteCtx.rewriteCallSite(c.getOperation(), *fc, builder))) {
      signalPassFailure();
      return;
    }
  }

  // Collect the fetches before rewriting any, because rewriting one erases
  // it, and a live walk holds the op it is about to visit next.
  SmallVector<cir::VAArgOp> vaArgs;
  moduleOp.walk([&](cir::VAArgOp v) { vaArgs.push_back(v); });
  for (cir::VAArgOp v : vaArgs) {
    std::optional<ArgClassification> ac = classifier->classifyVAArg(v);
    if (!ac) {
      signalPassFailure();
      return;
    }
    if (failed(rewriteCtx.rewriteVAArg(v.getOperation(), *ac, builder))) {
      signalPassFailure();
      return;
    }
  }
}

} // namespace

std::unique_ptr<Pass> mlir::createCallConvLoweringPass() {
  return std::make_unique<CallConvLoweringPass>();
}

std::unique_ptr<Pass> mlir::createCallConvLoweringPass(
    cir::CallConvTarget target, llvm::abi::X86AVXABILevel x86AvxAbiLevel,
    bool allowsX86TargetAttrAvx,
    const llvm::abi::X86ABICompatInfo &x86AbiCompat) {
  CallConvLoweringOptions options;
  options.target = target;
  options.x86AvxAbiLevel = x86AvxAbiLevel;
  options.allowsX86TargetAttrAvx = allowsX86TargetAttrAvx;
  return std::make_unique<CallConvLoweringPass>(options, x86AbiCompat);
}
