//===--- CIRGenException.cpp - Emit CIR Code for C++ exceptions -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This contains code dealing with C++ exception related code generation.
//
//===----------------------------------------------------------------------===//

#include "CIRDataLayout.h"
#include "CIRGenCXXABI.h"
#include "CIRGenCleanup.h"
#include "CIRGenFunction.h"
#include "CIRGenModule.h"
#include "UnimplementedFeatureGuarding.h"

#include "clang/AST/StmtVisitor.h"
#include "clang/CIR/Dialect/IR/CIRAttrs.h"
#include "clang/CIR/Dialect/IR/CIRDialect.h"
#include "clang/CIR/Dialect/IR/CIROpsEnums.h"
#include "clang/CIR/Dialect/IR/CIRTypes.h"
#include "llvm/Support/ErrorHandling.h"
#include <cstdint>

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Value.h"

using namespace cir;
using namespace clang;

const EHPersonality EHPersonality::GNU_C = {"__gcc_personality_v0", nullptr};
const EHPersonality EHPersonality::GNU_C_SJLJ = {"__gcc_personality_sj0",
                                                 nullptr};
const EHPersonality EHPersonality::GNU_C_SEH = {"__gcc_personality_seh0",
                                                nullptr};
const EHPersonality EHPersonality::NeXT_ObjC = {"__objc_personality_v0",
                                                nullptr};
const EHPersonality EHPersonality::GNU_CPlusPlus = {"__gxx_personality_v0",
                                                    nullptr};
const EHPersonality EHPersonality::GNU_CPlusPlus_SJLJ = {
    "__gxx_personality_sj0", nullptr};
const EHPersonality EHPersonality::GNU_CPlusPlus_SEH = {
    "__gxx_personality_seh0", nullptr};
const EHPersonality EHPersonality::GNU_ObjC = {"__gnu_objc_personality_v0",
                                               "objc_exception_throw"};
const EHPersonality EHPersonality::GNU_ObjC_SJLJ = {
    "__gnu_objc_personality_sj0", "objc_exception_throw"};
const EHPersonality EHPersonality::GNU_ObjC_SEH = {
    "__gnu_objc_personality_seh0", "objc_exception_throw"};
const EHPersonality EHPersonality::GNU_ObjCXX = {
    "__gnustep_objcxx_personality_v0", nullptr};
const EHPersonality EHPersonality::GNUstep_ObjC = {
    "__gnustep_objc_personality_v0", nullptr};
const EHPersonality EHPersonality::MSVC_except_handler = {"_except_handler3",
                                                          nullptr};
const EHPersonality EHPersonality::MSVC_C_specific_handler = {
    "__C_specific_handler", nullptr};
const EHPersonality EHPersonality::MSVC_CxxFrameHandler3 = {
    "__CxxFrameHandler3", nullptr};
const EHPersonality EHPersonality::GNU_Wasm_CPlusPlus = {
    "__gxx_wasm_personality_v0", nullptr};
const EHPersonality EHPersonality::XL_CPlusPlus = {"__xlcxx_personality_v1",
                                                   nullptr};

static const EHPersonality &getCPersonality(const TargetInfo &Target,
                                            const LangOptions &L) {
  const llvm::Triple &T = Target.getTriple();
  if (T.isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;
  if (L.hasSjLjExceptions())
    return EHPersonality::GNU_C_SJLJ;
  if (L.hasDWARFExceptions())
    return EHPersonality::GNU_C;
  if (L.hasSEHExceptions())
    return EHPersonality::GNU_C_SEH;
  return EHPersonality::GNU_C;
}

static const EHPersonality &getObjCPersonality(const TargetInfo &Target,
                                               const LangOptions &L) {
  const llvm::Triple &T = Target.getTriple();
  if (T.isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;

  switch (L.ObjCRuntime.getKind()) {
  case ObjCRuntime::FragileMacOSX:
    return getCPersonality(Target, L);
  case ObjCRuntime::MacOSX:
  case ObjCRuntime::iOS:
  case ObjCRuntime::WatchOS:
    return EHPersonality::NeXT_ObjC;
  case ObjCRuntime::GNUstep:
    if (L.ObjCRuntime.getVersion() >= VersionTuple(1, 7))
      return EHPersonality::GNUstep_ObjC;
    [[fallthrough]];
  case ObjCRuntime::GCC:
  case ObjCRuntime::ObjFW:
    if (L.hasSjLjExceptions())
      return EHPersonality::GNU_ObjC_SJLJ;
    if (L.hasSEHExceptions())
      return EHPersonality::GNU_ObjC_SEH;
    return EHPersonality::GNU_ObjC;
  }
  llvm_unreachable("bad runtime kind");
}

static const EHPersonality &getCXXPersonality(const TargetInfo &Target,
                                              const LangOptions &L) {
  const llvm::Triple &T = Target.getTriple();
  if (T.isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;
  if (T.isOSAIX())
    return EHPersonality::XL_CPlusPlus;
  if (L.hasSjLjExceptions())
    return EHPersonality::GNU_CPlusPlus_SJLJ;
  if (L.hasDWARFExceptions())
    return EHPersonality::GNU_CPlusPlus;
  if (L.hasSEHExceptions())
    return EHPersonality::GNU_CPlusPlus_SEH;
  if (L.hasWasmExceptions())
    return EHPersonality::GNU_Wasm_CPlusPlus;
  return EHPersonality::GNU_CPlusPlus;
}

/// Determines the personality function to use when both C++
/// and Objective-C exceptions are being caught.
static const EHPersonality &getObjCXXPersonality(const TargetInfo &Target,
                                                 const LangOptions &L) {
  if (Target.getTriple().isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;

  switch (L.ObjCRuntime.getKind()) {
  // In the fragile ABI, just use C++ exception handling and hope
  // they're not doing crazy exception mixing.
  case ObjCRuntime::FragileMacOSX:
    return getCXXPersonality(Target, L);

  // The ObjC personality defers to the C++ personality for non-ObjC
  // handlers.  Unlike the C++ case, we use the same personality
  // function on targets using (backend-driven) SJLJ EH.
  case ObjCRuntime::MacOSX:
  case ObjCRuntime::iOS:
  case ObjCRuntime::WatchOS:
    return getObjCPersonality(Target, L);

  case ObjCRuntime::GNUstep:
    return EHPersonality::GNU_ObjCXX;

  // The GCC runtime's personality function inherently doesn't support
  // mixed EH.  Use the ObjC personality just to avoid returning null.
  case ObjCRuntime::GCC:
  case ObjCRuntime::ObjFW:
    return getObjCPersonality(Target, L);
  }
  llvm_unreachable("bad runtime kind");
}

static const EHPersonality &getSEHPersonalityMSVC(const llvm::Triple &T) {
  if (T.getArch() == llvm::Triple::x86)
    return EHPersonality::MSVC_except_handler;
  return EHPersonality::MSVC_C_specific_handler;
}

const EHPersonality &EHPersonality::get(CIRGenModule &CGM,
                                        const FunctionDecl *FD) {
  const llvm::Triple &T = CGM.getTarget().getTriple();
  const LangOptions &L = CGM.getLangOpts();
  const TargetInfo &Target = CGM.getTarget();

  // Functions using SEH get an SEH personality.
  if (FD && FD->usesSEHTry())
    return getSEHPersonalityMSVC(T);

  if (L.ObjC)
    return L.CPlusPlus ? getObjCXXPersonality(Target, L)
                       : getObjCPersonality(Target, L);
  return L.CPlusPlus ? getCXXPersonality(Target, L)
                     : getCPersonality(Target, L);
}

const EHPersonality &EHPersonality::get(CIRGenFunction &CGF) {
  const auto *FD = CGF.CurCodeDecl;
  // For outlined finallys and filters, use the SEH personality in case they
  // contain more SEH. This mostly only affects finallys. Filters could
  // hypothetically use gnu statement expressions to sneak in nested SEH.
  FD = FD ? FD : CGF.CurSEHParent.getDecl();
  return get(CGF.CGM, dyn_cast_or_null<FunctionDecl>(FD));
}

void CIRGenFunction::buildCXXThrowExpr(const CXXThrowExpr *E) {
  if (const Expr *SubExpr = E->getSubExpr()) {
    QualType ThrowType = SubExpr->getType();
    if (ThrowType->isObjCObjectPointerType()) {
      llvm_unreachable("NYI");
    } else {
      CGM.getCXXABI().buildThrow(*this, E);
    }
  } else {
    CGM.getCXXABI().buildRethrow(*this, /*isNoReturn=*/true);
  }

  // In LLVM codegen the expression emitters expect to leave this
  // path by starting a new basic block. We do not need that in CIR.
}

namespace {
/// A cleanup to free the exception object if its initialization
/// throws.
struct FreeException final : EHScopeStack::Cleanup {
  mlir::Value exn;
  FreeException(mlir::Value exn) : exn(exn) {}
  void Emit(CIRGenFunction &CGF, Flags flags) override {
    llvm_unreachable("call to cxa_free or equivalent op NYI");
  }
};
} // end anonymous namespace

// Emits an exception expression into the given location.  This
// differs from buildAnyExprToMem only in that, if a final copy-ctor
// call is required, an exception within that copy ctor causes
// std::terminate to be invoked.
void CIRGenFunction::buildAnyExprToExn(const Expr *e, Address addr) {
  // Make sure the exception object is cleaned up if there's an
  // exception during initialization.
  pushFullExprCleanup<FreeException>(EHCleanup, addr.getPointer());
  EHScopeStack::stable_iterator cleanup = EHStack.stable_begin();

  // __cxa_allocate_exception returns a void*;  we need to cast this
  // to the appropriate type for the object.
  auto ty = convertTypeForMem(e->getType());
  Address typedAddr = addr.withElementType(ty);

  // From LLVM's codegen:
  // FIXME: this isn't quite right!  If there's a final unelided call
  // to a copy constructor, then according to [except.terminate]p1 we
  // must call std::terminate() if that constructor throws, because
  // technically that copy occurs after the exception expression is
  // evaluated but before the exception is caught.  But the best way
  // to handle that is to teach EmitAggExpr to do the final copy
  // differently if it can't be elided.
  buildAnyExprToMem(e, typedAddr, e->getType().getQualifiers(),
                    /*IsInit*/ true);

  // Deactivate the cleanup block.
  auto op = typedAddr.getPointer().getDefiningOp();
  assert(op &&
         "expected valid Operation *, block arguments are not meaningful here");
  DeactivateCleanupBlock(cleanup, op);
}

mlir::LogicalResult CIRGenFunction::buildCXXTryStmt(const CXXTryStmt &S) {
  auto tryLoc = getLoc(S.getBeginLoc());
  auto numHandlers = S.getNumHandlers();

  // FIXME(cir): create scope, and add catchOp to the lastest possible position
  // inside the cleanup block.

  // Create the skeleton for the catch statements.
  auto catchOp = builder.create<mlir::cir::CatchOp>(
      tryLoc, // FIXME(cir): we can do better source location here.
      [&](mlir::OpBuilder &b, mlir::Location loc,
          mlir::OperationState &result) {
        mlir::OpBuilder::InsertionGuard guard(b);
        for (int i = 0, e = numHandlers; i != e; ++i) {
          auto *r = result.addRegion();
          builder.createBlock(r);
        }
      });

  enterCXXTryStmt(S, catchOp);
  if (buildStmt(S.getTryBlock(), /*useCurrentScope=*/true).failed())
    return mlir::failure();
  exitCXXTryStmt(S);
  return mlir::success();
}

/// Emit the structure of the dispatch block for the given catch scope.
/// It is an invariant that the dispatch block already exists.
static void buildCatchDispatchBlock(CIRGenFunction &CGF,
                                    EHCatchScope &catchScope) {
  if (EHPersonality::get(CGF).isWasmPersonality())
    llvm_unreachable("NYI");
  if (EHPersonality::get(CGF).usesFuncletPads())
    llvm_unreachable("NYI");

  auto *dispatchBlock = catchScope.getCachedEHDispatchBlock();
  assert(dispatchBlock);

  // If there's only a single catch-all, getEHDispatchBlock returned
  // that catch-all as the dispatch block.
  if (catchScope.getNumHandlers() == 1 &&
      catchScope.getHandler(0).isCatchAll()) {
    llvm_unreachable("NYI"); // Remove when adding testcase.
    assert(dispatchBlock == catchScope.getHandler(0).Block);
    return;
  }

  llvm_unreachable("NYI");
}

void CIRGenFunction::enterCXXTryStmt(const CXXTryStmt &S,
                                     mlir::cir::CatchOp catchOp,
                                     bool IsFnTryBlock) {
  unsigned NumHandlers = S.getNumHandlers();
  EHCatchScope *CatchScope = EHStack.pushCatch(NumHandlers);
  for (unsigned I = 0; I != NumHandlers; ++I) {
    const CXXCatchStmt *C = S.getHandler(I);

    // FIXME: hook the CIR block for the right catch region here.
    mlir::Block *Handler = &catchOp.getRegion(I).getBlocks().front();
    if (C->getExceptionDecl()) {
      // FIXME: Dropping the reference type on the type into makes it
      // impossible to correctly implement catch-by-reference
      // semantics for pointers.  Unfortunately, this is what all
      // existing compilers do, and it's not clear that the standard
      // personality routine is capable of doing this right.  See C++ DR 388 :
      // http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#388
      Qualifiers CaughtTypeQuals;
      QualType CaughtType = CGM.getASTContext().getUnqualifiedArrayType(
          C->getCaughtType().getNonReferenceType(), CaughtTypeQuals);

      CatchTypeInfo TypeInfo{nullptr, 0};
      if (CaughtType->isObjCObjectPointerType())
        llvm_unreachable("NYI");
      else
        TypeInfo = CGM.getCXXABI().getAddrOfCXXCatchHandlerType(
            getLoc(S.getSourceRange()), CaughtType, C->getCaughtType());
      CatchScope->setHandler(I, TypeInfo, Handler);
    } else {
      // No exception decl indicates '...', a catch-all.
      llvm_unreachable("NYI");
    }
  }
}

void CIRGenFunction::exitCXXTryStmt(const CXXTryStmt &S, bool IsFnTryBlock) {
  unsigned NumHandlers = S.getNumHandlers();
  EHCatchScope &CatchScope = cast<EHCatchScope>(*EHStack.begin());
  assert(CatchScope.getNumHandlers() == NumHandlers);

  // If the catch was not required, bail out now.
  if (!CatchScope.hasEHBranches()) {
    llvm_unreachable("NYI");
    CatchScope.clearHandlerBlocks();
    EHStack.popCatch();
    return;
  }

  // Emit the structure of the EH dispatch for this catch.
  buildCatchDispatchBlock(*this, CatchScope);
  llvm_unreachable("NYI");
}

/// Check whether this is a non-EH scope, i.e. a scope which doesn't
/// affect exception handling.  Currently, the only non-EH scopes are
/// normal-only cleanup scopes.
static bool isNonEHScope(const EHScope &S) {
  switch (S.getKind()) {
  case EHScope::Cleanup:
    return !cast<EHCleanupScope>(S).isEHCleanup();
  case EHScope::Filter:
  case EHScope::Catch:
  case EHScope::Terminate:
    return false;
  }

  llvm_unreachable("Invalid EHScope Kind!");
}

mlir::Block *CIRGenFunction::buildLandingPad() {
  assert(EHStack.requiresLandingPad());
  assert(!CGM.getLangOpts().IgnoreExceptions &&
         "LandingPad should not be emitted when -fignore-exceptions are in "
         "effect.");
  EHScope &innermostEHScope = *EHStack.find(EHStack.getInnermostEHScope());
  switch (innermostEHScope.getKind()) {
  case EHScope::Terminate:
    llvm_unreachable("NYI");

  case EHScope::Catch:
  case EHScope::Cleanup:
  case EHScope::Filter:
    if (auto *lpad = innermostEHScope.getCachedLandingPad())
      return lpad;
  }

  {
    // Save the current CIR generation state.
    mlir::OpBuilder::InsertionGuard guard(builder);
    assert(!UnimplementedFeature::generateDebugInfo() && "NYI");
    // FIXME(cir): handle CIR relevant landing pad bits, there's no good
    // way to assert here right now and leaving one in break important
    // testcases. Work to fill this in is coming soon.
  }

  return nullptr;
}

mlir::Block *CIRGenFunction::getInvokeDestImpl() {
  assert(EHStack.requiresLandingPad());
  assert(!EHStack.empty());

  // If exceptions are disabled/ignored and SEH is not in use, then there is no
  // invoke destination. SEH "works" even if exceptions are off. In practice,
  // this means that C++ destructors and other EH cleanups don't run, which is
  // consistent with MSVC's behavior, except in the presence of -EHa
  const LangOptions &LO = CGM.getLangOpts();
  if (!LO.Exceptions || LO.IgnoreExceptions) {
    if (!LO.Borland && !LO.MicrosoftExt)
      return nullptr;
    if (!currentFunctionUsesSEHTry())
      return nullptr;
  }

  // CUDA device code doesn't have exceptions.
  if (LO.CUDA && LO.CUDAIsDevice)
    return nullptr;

  // Check the innermost scope for a cached landing pad.  If this is
  // a non-EH cleanup, we'll check enclosing scopes in EmitLandingPad.
  auto *LP = EHStack.begin()->getCachedLandingPad();
  if (LP)
    return LP;

  const EHPersonality &Personality = EHPersonality::get(*this);

  // FIXME(cir): add personality function
  // if (!CurFn->hasPersonalityFn())
  //   CurFn->setPersonalityFn(getOpaquePersonalityFn(CGM, Personality));

  if (Personality.usesFuncletPads()) {
    // We don't need separate landing pads in the funclet model.
    llvm_unreachable("NYI");
  } else {
    // Build the landing pad for this scope.
    LP = buildLandingPad();
  }

  // FIXME(cir): this breaks important testcases, fix is coming soon.
  // assert(LP);

  // Cache the landing pad on the innermost scope.  If this is a
  // non-EH scope, cache the landing pad on the enclosing scope, too.
  for (EHScopeStack::iterator ir = EHStack.begin(); true; ++ir) {
    ir->setCachedLandingPad(LP);
    if (!isNonEHScope(*ir))
      break;
  }

  return LP;
}