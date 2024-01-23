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
#include "llvm/Support/SaveAndRestore.h"

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

mlir::Block *CIRGenFunction::getEHResumeBlock(bool isCleanup) {
  // Just like some other try/catch related logic: return the basic block
  // pointer but only use it to denote we're tracking things, but there
  // shouldn't be any changes to that block after work done in this function.
  auto catchOp = currExceptionInfo.catchOp;
  assert(catchOp.getNumRegions() && "expected at least one region");
  auto &fallbackRegion = catchOp.getRegion(catchOp.getNumRegions() - 1);

  auto *resumeBlock = &fallbackRegion.getBlocks().back();
  if (!resumeBlock->empty())
    return resumeBlock;

  auto ip = getBuilder().saveInsertionPoint();
  getBuilder().setInsertionPointToStart(resumeBlock);

  const EHPersonality &Personality = EHPersonality::get(*this);

  // This can always be a call because we necessarily didn't find
  // anything on the EH stack which needs our help.
  const char *RethrowName = Personality.CatchallRethrowFn;
  if (RethrowName != nullptr && !isCleanup) {
    llvm_unreachable("NYI");
  }

  getBuilder().create<mlir::cir::ResumeOp>(catchOp.getLoc());
  getBuilder().restoreInsertionPoint(ip);
  return resumeBlock;
}

mlir::LogicalResult CIRGenFunction::buildCXXTryStmt(const CXXTryStmt &S) {
  auto loc = getLoc(S.getSourceRange());
  mlir::OpBuilder::InsertPoint scopeIP;

  // Create a scope to hold try local storage for catch params.
  [[maybe_unused]] auto s = builder.create<mlir::cir::ScopeOp>(
      loc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        scopeIP = getBuilder().saveInsertionPoint();
      });

  auto r = mlir::success();
  {
    mlir::OpBuilder::InsertionGuard guard(getBuilder());
    getBuilder().restoreInsertionPoint(scopeIP);
    r = buildCXXTryStmtUnderScope(S);
    getBuilder().create<mlir::cir::YieldOp>(loc);
  }
  return r;
}

mlir::LogicalResult
CIRGenFunction::buildCXXTryStmtUnderScope(const CXXTryStmt &S) {
  const llvm::Triple &T = getTarget().getTriple();
  // If we encounter a try statement on in an OpenMP target region offloaded to
  // a GPU, we treat it as a basic block.
  const bool IsTargetDevice =
      (CGM.getLangOpts().OpenMPIsTargetDevice && (T.isNVPTX() || T.isAMDGCN()));
  assert(!IsTargetDevice && "NYI");

  auto numHandlers = S.getNumHandlers();
  auto tryLoc = getLoc(S.getBeginLoc());
  auto scopeLoc = getLoc(S.getSourceRange());

  mlir::OpBuilder::InsertPoint beginInsertTryBody;
  auto ehPtrTy = mlir::cir::PointerType::get(
      getBuilder().getContext(),
      getBuilder().getType<::mlir::cir::ExceptionInfoType>());
  mlir::Value exceptionInfoInsideTry;

  // Create the scope to represent only the C/C++ `try {}` part. However, don't
  // populate right away. Reserve some space to store the exception info but
  // don't emit the bulk right away, for now only make sure the scope returns
  // the exception information.
  auto tryScope = builder.create<mlir::cir::ScopeOp>(
      scopeLoc, /*scopeBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Type &yieldTy, mlir::Location loc) {
        // Allocate space for our exception info that might be passed down
        // to `cir.try_call` everytime a call happens.
        yieldTy = ehPtrTy;
        exceptionInfoInsideTry = b.create<mlir::cir::AllocaOp>(
            loc, /*addr type*/ getBuilder().getPointerTo(yieldTy),
            /*var type*/ yieldTy, "__exception_ptr",
            CGM.getSize(CharUnits::One()), nullptr);

        beginInsertTryBody = getBuilder().saveInsertionPoint();
      });

  // The catch {} parts consume the exception information provided by a
  // try scope. Also don't emit the code right away for catch clauses, for
  // now create the regions and consume the try scope result.
  // Note that clauses are later populated in CIRGenFunction::buildLandingPad.
  auto catchOp = builder.create<mlir::cir::CatchOp>(
      tryLoc,
      tryScope->getResult(
          0), // FIXME(cir): we can do better source location here.
      [&](mlir::OpBuilder &b, mlir::Location loc,
          mlir::OperationState &result) {
        mlir::OpBuilder::InsertionGuard guard(b);
        // Once for each handler and one for fallback (which could be a
        // resume or rethrow).
        for (int i = 0, e = numHandlers + 1; i != e; ++i) {
          auto *r = result.addRegion();
          builder.createBlock(r);
        }
      });

  // Finally emit the body for try/catch.
  auto emitTryCatchBody = [&]() -> mlir::LogicalResult {
    auto loc = catchOp.getLoc();
    mlir::OpBuilder::InsertionGuard guard(getBuilder());
    getBuilder().restoreInsertionPoint(beginInsertTryBody);
    CIRGenFunction::LexicalScope lexScope{*this, loc,
                                          getBuilder().getInsertionBlock()};

    {
      ExceptionInfoRAIIObject ehx{*this, {exceptionInfoInsideTry, catchOp}};
      // Attach the basic blocks for the catchOp regions into ScopeCatch info.
      enterCXXTryStmt(S, catchOp);
      // Emit the body for the `try {}` part.
      if (buildStmt(S.getTryBlock(), /*useCurrentScope=*/true).failed())
        return mlir::failure();

      auto v = getBuilder().create<mlir::cir::LoadOp>(loc, ehPtrTy,
                                                      exceptionInfoInsideTry);
      getBuilder().create<mlir::cir::YieldOp>(loc, v.getResult());
    }

    {
      ExceptionInfoRAIIObject ehx{*this, {tryScope->getResult(0), catchOp}};
      // Emit catch clauses.
      exitCXXTryStmt(S);
    }

    return mlir::success();
  };

  return emitTryCatchBody();
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

  // In traditional LLVM codegen, the right handler is selected (with calls to
  // eh_typeid_for) and the selector value is loaded. After that, blocks get
  // connected for later codegen. In CIR, these are all implicit behaviors of
  // cir.catch - not a lot of work to do.
  //
  // Test against each of the exception types we claim to catch.
  for (unsigned i = 0, e = catchScope.getNumHandlers();; ++i) {
    assert(i < e && "ran off end of handlers!");
    const EHCatchScope::Handler &handler = catchScope.getHandler(i);

    auto typeValue = handler.Type.RTTI;
    assert(handler.Type.Flags == 0 && "catch handler flags not supported");
    assert(typeValue && "fell into catch-all case!");
    // Check for address space mismatch: if (typeValue->getType() != argTy)
    assert(!UnimplementedFeature::addressSpace());

    bool nextIsEnd = false;
    // If this is the last handler, we're at the end, and the next
    // block is the block for the enclosing EH scope. Make sure to call
    // getEHDispatchBlock for caching it.
    if (i + 1 == e) {
      (void)CGF.getEHDispatchBlock(catchScope.getEnclosingEHScope());
      nextIsEnd = true;

      // If the next handler is a catch-all, we're at the end, and the
      // next block is that handler.
    } else if (catchScope.getHandler(i + 1).isCatchAll()) {
      // Block already created when creating CatchOp, just mark this
      // is the end.
      nextIsEnd = true;
    }

    // If the next handler is a catch-all, we're completely done.
    if (nextIsEnd)
      return;
  }
}

void CIRGenFunction::enterCXXTryStmt(const CXXTryStmt &S,
                                     mlir::cir::CatchOp catchOp,
                                     bool IsFnTryBlock) {
  unsigned NumHandlers = S.getNumHandlers();
  EHCatchScope *CatchScope = EHStack.pushCatch(NumHandlers);
  for (unsigned I = 0; I != NumHandlers; ++I) {
    const CXXCatchStmt *C = S.getHandler(I);

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

  // Copy the handler blocks off before we pop the EH stack.  Emitting
  // the handlers might scribble on this memory.
  SmallVector<EHCatchScope::Handler, 8> Handlers(
      CatchScope.begin(), CatchScope.begin() + NumHandlers);

  EHStack.popCatch();

  // Determine if we need an implicit rethrow for all these catch handlers;
  // see the comment below.
  bool doImplicitRethrow = false;
  if (IsFnTryBlock)
    doImplicitRethrow = isa<CXXDestructorDecl>(CurCodeDecl) ||
                        isa<CXXConstructorDecl>(CurCodeDecl);

  // Wasm uses Windows-style EH instructions, but merges all catch clauses into
  // one big catchpad. So we save the old funclet pad here before we traverse
  // each catch handler.
  SaveAndRestore RestoreCurrentFuncletPad(CurrentFuncletPad);
  mlir::Block *WasmCatchStartBlock = nullptr;
  if (EHPersonality::get(*this).isWasmPersonality()) {
    llvm_unreachable("NYI");
  }

  bool HasCatchAll = false;
  for (unsigned I = NumHandlers; I != 0; --I) {
    HasCatchAll |= Handlers[I - 1].isCatchAll();
    mlir::Block *CatchBlock = Handlers[I - 1].Block;
    mlir::OpBuilder::InsertionGuard guard(getBuilder());
    getBuilder().setInsertionPointToStart(CatchBlock);

    // Catch the exception if this isn't a catch-all.
    const CXXCatchStmt *C = S.getHandler(I - 1);

    // Enter a cleanup scope, including the catch variable and the
    // end-catch.
    RunCleanupsScope CatchScope(*this);

    // Initialize the catch variable and set up the cleanups.
    SaveAndRestore RestoreCurrentFuncletPad(CurrentFuncletPad);
    CGM.getCXXABI().emitBeginCatch(*this, C);

    // Emit the PGO counter increment.
    assert(!UnimplementedFeature::incrementProfileCounter());

    // Perform the body of the catch.
    (void)buildStmt(C->getHandlerBlock(), /*useCurrentScope=*/true);

    // [except.handle]p11:
    //   The currently handled exception is rethrown if control
    //   reaches the end of a handler of the function-try-block of a
    //   constructor or destructor.

    // It is important that we only do this on fallthrough and not on
    // return.  Note that it's illegal to put a return in a
    // constructor function-try-block's catch handler (p14), so this
    // really only applies to destructors.
    if (doImplicitRethrow && HaveInsertPoint()) {
      llvm_unreachable("NYI");
    }

    // Fall out through the catch cleanups.
    CatchScope.ForceCleanup();
  }

  // Because in wasm we merge all catch clauses into one big catchpad, in case
  // none of the types in catch handlers matches after we test against each   of
  // them, we should unwind to the next EH enclosing scope. We generate a   call
  // to rethrow function here to do that.
  if (EHPersonality::get(*this).isWasmPersonality() && !HasCatchAll) {
    assert(WasmCatchStartBlock);
    // Navigate for the "rethrow" block we created in emitWasmCatchPadBlock().
    // Wasm uses landingpad-style conditional branches to compare selectors, so
    // we follow the false destination for each of the cond branches to reach
    // the rethrow block.
    llvm_unreachable("NYI");
  }

  assert(!UnimplementedFeature::incrementProfileCounter());
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

mlir::Operation *CIRGenFunction::buildLandingPad() {
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

  auto catchOp = currExceptionInfo.catchOp;
  assert(catchOp && "Should be valid");
  {
    // Save the current CIR generation state.
    mlir::OpBuilder::InsertionGuard guard(builder);
    assert(!UnimplementedFeature::generateDebugInfo() && "NYI");

    // Traditional LLVM codegen creates the lpad basic block, extract
    // values, landing pad instructions, etc.

    // Accumulate all the handlers in scope.
    bool hasCatchAll = false;
    bool hasCleanup = false;
    bool hasFilter = false;
    SmallVector<mlir::Value, 4> filterTypes;
    llvm::SmallPtrSet<mlir::Attribute, 4> catchTypes;
    SmallVector<mlir::Attribute, 4> clauses;

    for (EHScopeStack::iterator I = EHStack.begin(), E = EHStack.end(); I != E;
         ++I) {

      switch (I->getKind()) {
      case EHScope::Cleanup:
        // If we have a cleanup, remember that.
        llvm_unreachable("NYI");
        continue;

      case EHScope::Filter: {
        llvm_unreachable("NYI");
      }

      case EHScope::Terminate:
        // Terminate scopes are basically catch-alls.
        // assert(!hasCatchAll);
        // hasCatchAll = true;
        // goto done;
        llvm_unreachable("NYI");

      case EHScope::Catch:
        break;
      }

      EHCatchScope &catchScope = cast<EHCatchScope>(*I);
      for (unsigned hi = 0, he = catchScope.getNumHandlers(); hi != he; ++hi) {
        EHCatchScope::Handler handler = catchScope.getHandler(hi);
        assert(handler.Type.Flags == 0 &&
               "landingpads do not support catch handler flags");

        // If this is a catch-all, register that and abort.
        if (!handler.Type.RTTI) {
          assert(!hasCatchAll);
          hasCatchAll = true;
          goto done;
        }

        // Check whether we already have a handler for this type.
        if (catchTypes.insert(handler.Type.RTTI).second) {
          // If not, keep track to later add to catch op.
          clauses.push_back(handler.Type.RTTI);
        }
      }
    }

  done:
    // If we have a catch-all, add null to the landingpad.
    assert(!(hasCatchAll && hasFilter));
    if (hasCatchAll) {
      llvm_unreachable("NYI");

      // If we have an EH filter, we need to add those handlers in the
      // right place in the landingpad, which is to say, at the end.
    } else if (hasFilter) {
      // Create a filter expression: a constant array indicating which filter
      // types there are. The personality routine only lands here if the filter
      // doesn't match.
      llvm_unreachable("NYI");

      // Otherwise, signal that we at least have cleanups.
    } else if (hasCleanup) {
      llvm_unreachable("NYI");
    }

    assert((clauses.size() > 0 || hasCleanup) && "CatchOp has no clauses!");

    // Add final array of clauses into catchOp.
    catchOp.setCatchersAttr(
        mlir::ArrayAttr::get(builder.getContext(), clauses));

    // In traditional LLVM codegen. this tells the backend how to generate the
    // landing pad by generating a branch to the dispatch block. In CIR the same
    // function is called to gather some state, but this block info it's not
    // useful per-se.
    (void)getEHDispatchBlock(EHStack.getInnermostEHScope());
  }

  return catchOp;
}

// Differently from LLVM traditional codegen, there are no dispatch blocks
// to look at given cir.try_call does not jump to blocks like invoke does.
// However, we keep this around since other parts of CIRGen use
// getCachedEHDispatchBlock to infer state.
mlir::Block *
CIRGenFunction::getEHDispatchBlock(EHScopeStack::stable_iterator si) {
  if (EHPersonality::get(*this).usesFuncletPads())
    llvm_unreachable("NYI");

  // The dispatch block for the end of the scope chain is a block that
  // just resumes unwinding.
  if (si == EHStack.stable_end())
    return getEHResumeBlock(true);

  // Otherwise, we should look at the actual scope.
  EHScope &scope = *EHStack.find(si);

  auto *dispatchBlock = scope.getCachedEHDispatchBlock();
  if (!dispatchBlock) {
    switch (scope.getKind()) {
    case EHScope::Catch: {
      // Apply a special case to a single catch-all.
      EHCatchScope &catchScope = cast<EHCatchScope>(scope);
      if (catchScope.getNumHandlers() == 1 &&
          catchScope.getHandler(0).isCatchAll()) {
        dispatchBlock = catchScope.getHandler(0).Block;

        // Otherwise, make a dispatch block.
      } else {
        // As said in the function comment, just signal back we
        // have something - even though the block value doesn't
        // have any real meaning.
        dispatchBlock = catchScope.getHandler(0).Block;
        assert(dispatchBlock && "find another approach to signal");
      }
      break;
    }

    case EHScope::Cleanup:
      llvm_unreachable("NYI");
      break;

    case EHScope::Filter:
      llvm_unreachable("NYI");
      break;

    case EHScope::Terminate:
      llvm_unreachable("NYI");
      break;
    }
    scope.setCachedEHDispatchBlock(dispatchBlock);
  }
  return dispatchBlock;
}

mlir::Operation *CIRGenFunction::getInvokeDestImpl() {
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

  assert(LP);

  // Cache the landing pad on the innermost scope.  If this is a
  // non-EH scope, cache the landing pad on the enclosing scope, too.
  for (EHScopeStack::iterator ir = EHStack.begin(); true; ++ir) {
    ir->setCachedLandingPad(LP);
    if (!isNonEHScope(*ir))
      break;
  }

  return LP;
}
