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

#include "CIRGenCXXABI.h"
#include "CIRGenFunction.h"

#include "clang/CIR/MissingFeatures.h"
#include "llvm/Support/SaveAndRestore.h"

using namespace clang;
using namespace clang::CIRGen;

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
const EHPersonality EHPersonality::ZOS_CPlusPlus = {"__zos_cxx_personality_v2",
                                                    nullptr};

static const EHPersonality &getCPersonality(const TargetInfo &target,
                                            const CodeGenOptions &cgOpts) {
  const llvm::Triple &triple = target.getTriple();
  if (triple.isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;
  if (cgOpts.hasSjLjExceptions())
    return EHPersonality::GNU_C_SJLJ;
  if (cgOpts.hasDWARFExceptions())
    return EHPersonality::GNU_C;
  if (cgOpts.hasSEHExceptions())
    return EHPersonality::GNU_C_SEH;
  return EHPersonality::GNU_C;
}

static const EHPersonality &getObjCPersonality(const TargetInfo &target,
                                               const LangOptions &langOpts,
                                               const CodeGenOptions &cgOpts) {
  const llvm::Triple &triple = target.getTriple();
  if (triple.isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;

  switch (langOpts.ObjCRuntime.getKind()) {
  case ObjCRuntime::FragileMacOSX:
    return getCPersonality(target, cgOpts);
  case ObjCRuntime::MacOSX:
  case ObjCRuntime::iOS:
  case ObjCRuntime::WatchOS:
    return EHPersonality::NeXT_ObjC;
  case ObjCRuntime::GNUstep:
    if (langOpts.ObjCRuntime.getVersion() >= VersionTuple(1, 7))
      return EHPersonality::GNUstep_ObjC;
    [[fallthrough]];
  case ObjCRuntime::GCC:
  case ObjCRuntime::ObjFW:
    if (cgOpts.hasSjLjExceptions())
      return EHPersonality::GNU_ObjC_SJLJ;
    if (cgOpts.hasSEHExceptions())
      return EHPersonality::GNU_ObjC_SEH;
    return EHPersonality::GNU_ObjC;
  }
  llvm_unreachable("bad runtime kind");
}

static const EHPersonality &getCXXPersonality(const TargetInfo &target,
                                              const CodeGenOptions &cgOpts) {
  const llvm::Triple &triple = target.getTriple();
  if (triple.isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;
  if (triple.isOSAIX())
    return EHPersonality::XL_CPlusPlus;
  if (cgOpts.hasSjLjExceptions())
    return EHPersonality::GNU_CPlusPlus_SJLJ;
  if (cgOpts.hasDWARFExceptions())
    return EHPersonality::GNU_CPlusPlus;
  if (cgOpts.hasSEHExceptions())
    return EHPersonality::GNU_CPlusPlus_SEH;
  if (cgOpts.hasWasmExceptions())
    return EHPersonality::GNU_Wasm_CPlusPlus;
  return EHPersonality::GNU_CPlusPlus;
}

/// Determines the personality function to use when both C++
/// and Objective-C exceptions are being caught.
static const EHPersonality &getObjCXXPersonality(const TargetInfo &target,
                                                 const LangOptions &langOpts,
                                                 const CodeGenOptions &cgOpts) {
  if (target.getTriple().isWindowsMSVCEnvironment())
    return EHPersonality::MSVC_CxxFrameHandler3;

  switch (langOpts.ObjCRuntime.getKind()) {
  // In the fragile ABI, just use C++ exception handling and hope
  // they're not doing crazy exception mixing.
  case ObjCRuntime::FragileMacOSX:
    return getCXXPersonality(target, cgOpts);

  // The ObjC personality defers to the C++ personality for non-ObjC
  // handlers.  Unlike the C++ case, we use the same personality
  // function on targets using (backend-driven) SJLJ EH.
  case ObjCRuntime::MacOSX:
  case ObjCRuntime::iOS:
  case ObjCRuntime::WatchOS:
    return getObjCPersonality(target, langOpts, cgOpts);

  case ObjCRuntime::GNUstep:
    return EHPersonality::GNU_ObjCXX;

  // The GCC runtime's personality function inherently doesn't support
  // mixed EH.  Use the ObjC personality just to avoid returning null.
  case ObjCRuntime::GCC:
  case ObjCRuntime::ObjFW:
    return getObjCPersonality(target, langOpts, cgOpts);
  }
  llvm_unreachable("bad runtime kind");
}

static const EHPersonality &getSEHPersonalityMSVC(const llvm::Triple &triple) {
  return triple.getArch() == llvm::Triple::x86
             ? EHPersonality::MSVC_except_handler
             : EHPersonality::MSVC_C_specific_handler;
}

const EHPersonality &EHPersonality::get(CIRGenModule &cgm,
                                        const FunctionDecl *fd) {
  const llvm::Triple &triple = cgm.getTarget().getTriple();
  const LangOptions &langOpts = cgm.getLangOpts();
  const CodeGenOptions &cgOpts = cgm.getCodeGenOpts();
  const TargetInfo &target = cgm.getTarget();

  // Functions using SEH get an SEH personality.
  if (fd && fd->usesSEHTry())
    return getSEHPersonalityMSVC(triple);

  if (langOpts.ObjC) {
    return langOpts.CPlusPlus ? getObjCXXPersonality(target, langOpts, cgOpts)
                              : getObjCPersonality(target, langOpts, cgOpts);
  }
  return langOpts.CPlusPlus ? getCXXPersonality(target, cgOpts)
                            : getCPersonality(target, cgOpts);
}

const EHPersonality &EHPersonality::get(CIRGenFunction &cgf) {
  const auto *fg = cgf.curCodeDecl;
  // For outlined finallys and filters, use the SEH personality in case they
  // contain more SEH. This mostly only affects finallys. Filters could
  // hypothetically use gnu statement expressions to sneak in nested SEH.
  fg = fg ? fg : cgf.curSEHParent.getDecl();
  return get(cgf.cgm, dyn_cast_or_null<FunctionDecl>(fg));
}

static llvm::StringRef getPersonalityFn(CIRGenModule &cgm,
                                        const EHPersonality &personality) {
  // Create the personality function type: i32 (...)
  mlir::Type i32Ty = cgm.getBuilder().getI32Type();
  auto funcTy = cir::FuncType::get({}, i32Ty, /*isVarArg=*/true);

  cir::FuncOp personalityFn = cgm.createRuntimeFunction(
      funcTy, personality.personalityFn, mlir::ArrayAttr(), /*isLocal=*/true);

  return personalityFn.getSymName();
}

void CIRGenFunction::emitCXXThrowExpr(const CXXThrowExpr *e) {
  const llvm::Triple &triple = getTarget().getTriple();
  if (cgm.getLangOpts().OpenMPIsTargetDevice &&
      (triple.isNVPTX() || triple.isAMDGCN())) {
    cgm.errorNYI("emitCXXThrowExpr OpenMP with NVPTX or AMDGCN Triples");
    return;
  }

  if (const Expr *subExpr = e->getSubExpr()) {
    QualType throwType = subExpr->getType();
    if (throwType->isObjCObjectPointerType()) {
      cgm.errorNYI("emitCXXThrowExpr ObjCObjectPointerType");
      return;
    }

    cgm.getCXXABI().emitThrow(*this, e);
    return;
  }

  cgm.getCXXABI().emitRethrow(*this, /*isNoReturn=*/true);
}

void CIRGenFunction::emitAnyExprToExn(const Expr *e, Address addr) {
  // Make sure the exception object is cleaned up if there's an
  // exception during initialization.
  assert(!cir::MissingFeatures::ehCleanupScope());

  // __cxa_allocate_exception returns a void*;  we need to cast this
  // to the appropriate type for the object.
  mlir::Type ty = convertTypeForMem(e->getType());
  Address typedAddr = addr.withElementType(builder, ty);

  // From LLVM's codegen:
  // FIXME: this isn't quite right!  If there's a final unelided call
  // to a copy constructor, then according to [except.terminate]p1 we
  // must call std::terminate() if that constructor throws, because
  // technically that copy occurs after the exception expression is
  // evaluated but before the exception is caught.  But the best way
  // to handle that is to teach EmitAggExpr to do the final copy
  // differently if it can't be elided.
  emitAnyExprToMem(e, typedAddr, e->getType().getQualifiers(),
                   /*isInitializer=*/true);

  // Deactivate the cleanup block.
  assert(!cir::MissingFeatures::ehCleanupScope());
}

void CIRGenFunction::populateUnwindResumeBlock(bool isCleanup,
                                               cir::TryOp tryOp) {
  const EHPersonality &personality = EHPersonality::get(*this);
  // This can always be a call because we necessarily didn't find
  // anything on the EH stack which needs our help.
  const char *rethrowName = personality.catchallRethrowFn;
  if (rethrowName != nullptr && !isCleanup) {
    cgm.errorNYI("populateUnwindResumeBlock CatchallRethrowFn");
    return;
  }

  unsigned regionsNum = tryOp->getNumRegions();
  mlir::Region *unwindRegion = &tryOp->getRegion(regionsNum - 1);
  mlir::Block *unwindResumeBlock = &unwindRegion->front();
  if (!unwindResumeBlock->empty())
    return;

  // Emit cir.resume into the unwind region last block
  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(unwindResumeBlock);
  cir::ResumeOp::create(builder, tryOp.getLoc());
}

mlir::LogicalResult CIRGenFunction::emitCXXTryStmt(const CXXTryStmt &s) {
  if (s.getTryBlock()->body_empty())
    return mlir::LogicalResult::success();

  mlir::Location loc = getLoc(s.getSourceRange());
  // Create a scope to hold try local storage for catch params.

  mlir::OpBuilder::InsertPoint scopeIP;
  cir::ScopeOp::create(
      builder, loc,
      /*scopeBuilder=*/[&](mlir::OpBuilder &b, mlir::Location loc) {
        scopeIP = builder.saveInsertionPoint();
      });

  mlir::OpBuilder::InsertionGuard guard(builder);
  builder.restoreInsertionPoint(scopeIP);
  mlir::LogicalResult result = emitCXXTryStmtUnderScope(s);
  cir::YieldOp::create(builder, loc);
  return result;
}

mlir::LogicalResult
CIRGenFunction::emitCXXTryStmtUnderScope(const CXXTryStmt &s) {
  const llvm::Triple &t = getTarget().getTriple();
  // If we encounter a try statement on in an OpenMP target region offloaded to
  // a GPU, we treat it as a basic block.
  const bool isTargetDevice =
      (cgm.getLangOpts().OpenMPIsTargetDevice && (t.isNVPTX() || t.isAMDGCN()));
  if (isTargetDevice) {
    cgm.errorNYI(
        "emitCXXTryStmtUnderScope: OpenMP target region offloaded to GPU");
    return mlir::success();
  }

  unsigned numHandlers = s.getNumHandlers();
  mlir::Location tryLoc = getLoc(s.getBeginLoc());
  mlir::OpBuilder::InsertPoint beginInsertTryBody;

  bool hasCatchAll = false;
  for (unsigned i = 0; i != numHandlers; ++i) {
    hasCatchAll |= s.getHandler(i)->getExceptionDecl() == nullptr;
    if (hasCatchAll)
      break;
  }

  // Create the scope to represent only the C/C++ `try {}` part. However,
  // don't populate right away. Create regions for the catch handlers,
  // but don't emit the handler bodies yet. For now, only make sure the
  // scope returns the exception information.
  auto tryOp = cir::TryOp::create(
      builder, tryLoc,
      /*tryBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc) {
        beginInsertTryBody = builder.saveInsertionPoint();
      },
      /*handlersBuilder=*/
      [&](mlir::OpBuilder &b, mlir::Location loc,
          mlir::OperationState &result) {
        mlir::OpBuilder::InsertionGuard guard(b);

        // We create an extra region for an unwind catch handler in case the
        // catch-all handler doesn't exists
        unsigned numRegionsToCreate =
            hasCatchAll ? numHandlers : numHandlers + 1;

        for (unsigned i = 0; i != numRegionsToCreate; ++i) {
          mlir::Region *region = result.addRegion();
          builder.createBlock(region);
        }
      });

  // Finally emit the body for try/catch.
  {
    mlir::Location loc = tryOp.getLoc();
    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.restoreInsertionPoint(beginInsertTryBody);
    CIRGenFunction::LexicalScope tryScope{*this, loc,
                                          builder.getInsertionBlock()};

    tryScope.setAsTry(tryOp);

    // Attach the basic blocks for the catch regions.
    enterCXXTryStmt(s, tryOp);

    // Emit the body for the `try {}` part.
    {
      mlir::OpBuilder::InsertionGuard guard(builder);
      CIRGenFunction::LexicalScope tryBodyScope{*this, loc,
                                                builder.getInsertionBlock()};
      if (emitStmt(s.getTryBlock(), /*useCurrentScope=*/true).failed())
        return mlir::failure();
    }

    // Emit catch clauses.
    exitCXXTryStmt(s);
  }

  return mlir::success();
}

/// Emit the structure of the dispatch block for the given catch scope.
/// It is an invariant that the dispatch block already exists.
static void emitCatchDispatchBlock(CIRGenFunction &cgf,
                                   EHCatchScope &catchScope, cir::TryOp tryOp) {
  if (EHPersonality::get(cgf).isWasmPersonality()) {
    cgf.cgm.errorNYI("emitCatchDispatchBlock: WasmPersonality");
    return;
  }

  if (EHPersonality::get(cgf).usesFuncletPads()) {
    cgf.cgm.errorNYI("emitCatchDispatchBlock: usesFuncletPads");
    return;
  }

  assert(std::find_if(catchScope.begin(), catchScope.end(),
                      [](const auto &handler) {
                        return !handler.type.rtti && handler.type.flags != 0;
                      }) == catchScope.end() &&
         "catch handler without type value or with not supported flags");

  // There was no catch all handler, populate th EH regions for the
  // enclosing scope.
  if (!std::prev(catchScope.end())->isCatchAll())
    cgf.populateEHCatchRegions(catchScope.getEnclosingEHScope(), tryOp);
}

void CIRGenFunction::enterCXXTryStmt(const CXXTryStmt &s, cir::TryOp tryOp,
                                     bool isFnTryBlock) {
  unsigned numHandlers = s.getNumHandlers();
  EHCatchScope *catchScope = ehStack.pushCatch(numHandlers);
  for (unsigned i = 0; i != numHandlers; ++i) {
    const CXXCatchStmt *catchStmt = s.getHandler(i);
    mlir::Region *handler = &tryOp.getHandlerRegions()[i];
    if (catchStmt->getExceptionDecl()) {
      // FIXME: Dropping the reference type on the type into makes it
      // impossible to correctly implement catch-by-reference
      // semantics for pointers.  Unfortunately, this is what all
      // existing compilers do, and it's not clear that the standard
      // personality routine is capable of doing this right.  See C++ DR 388:
      //   http://www.open-std.org/jtc1/sc22/wg21/docs/cwg_active.html#388
      Qualifiers caughtTypeQuals;
      QualType caughtType = cgm.getASTContext().getUnqualifiedArrayType(
          catchStmt->getCaughtType().getNonReferenceType(), caughtTypeQuals);
      if (caughtType->isObjCObjectPointerType()) {
        cgm.errorNYI("enterCXXTryStmt: caughtType ObjCObjectPointerType");
        return;
      }

      CatchTypeInfo typeInfo = cgm.getCXXABI().getAddrOfCXXCatchHandlerType(
          getLoc(catchStmt->getSourceRange()), caughtType,
          catchStmt->getCaughtType());
      catchScope->setHandler(i, typeInfo, handler, catchStmt);
    } else {
      // No exception decl indicates '...', a catch-all.
      catchScope->setHandler(i, cgm.getCXXABI().getCatchAllTypeInfo(), handler,
                             s.getHandler(i));
    }

    // Under async exceptions, catch(...) needs to catch HW exception too
    // Mark scope with SehTryBegin as a SEH __try scope
    if (getLangOpts().EHAsynch) {
      cgm.errorNYI("enterCXXTryStmt: EHAsynch");
      return;
    }
  }
}

void CIRGenFunction::exitCXXTryStmt(const CXXTryStmt &s, bool isFnTryBlock) {
  unsigned numHandlers = s.getNumHandlers();
  EHCatchScope &catchScope = cast<EHCatchScope>(*ehStack.begin());
  assert(catchScope.getNumHandlers() == numHandlers);
  cir::TryOp tryOp = curLexScope->getTry();

  // If the catch was not required, bail out now.
  if (!catchScope.mayThrow()) {
    catchScope.clearHandlerBlocks();
    ehStack.popCatch();

    // Drop all basic block from all catch regions.
    SmallVector<mlir::Block *> eraseBlocks;
    for (mlir::Region &handlerRegion : tryOp.getHandlerRegions()) {
      if (handlerRegion.empty())
        continue;

      for (mlir::Block &b : handlerRegion.getBlocks())
        eraseBlocks.push_back(&b);
    }

    for (mlir::Block *b : eraseBlocks)
      b->erase();

    tryOp.setHandlerTypesAttr({});
    return;
  }

  // Emit the structure of the EH dispatch for this catch.
  emitCatchDispatchBlock(*this, catchScope, tryOp);

  // Copy the handler blocks off before we pop the EH stack.  Emitting
  // the handlers might scribble on this memory.
  SmallVector<EHCatchScope::Handler> handlers(catchScope.begin(),
                                              catchScope.begin() + numHandlers);

  ehStack.popCatch();

  // Determine if we need an implicit rethrow for all these catch handlers;
  // see the comment below.
  bool doImplicitRethrow =
      isFnTryBlock && isa<CXXDestructorDecl, CXXConstructorDecl>(curCodeDecl);

  // Wasm uses Windows-style EH instructions, but merges all catch clauses into
  // one big catchpad. So we save the old funclet pad here before we traverse
  // each catch handler.
  if (EHPersonality::get(*this).isWasmPersonality()) {
    cgm.errorNYI("exitCXXTryStmt: WASM personality");
    return;
  }

  bool hasCatchAll = false;
  for (auto &handler : llvm::reverse(handlers)) {
    hasCatchAll |= handler.isCatchAll();
    mlir::Region *catchRegion = handler.region;
    const CXXCatchStmt *catchStmt = handler.stmt;

    mlir::OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointToStart(&catchRegion->front());

    // Enter a cleanup scope, including the catch variable and the
    // end-catch.
    RunCleanupsScope catchScope(*this);

    // Initialize the catch variable and set up the cleanups.
    assert(!cir::MissingFeatures::currentFuncletPad());
    cgm.getCXXABI().emitBeginCatch(*this, catchStmt);

    // Emit the PGO counter increment.
    assert(!cir::MissingFeatures::incrementProfileCounter());

    // Perform the body of the catch.
    [[maybe_unused]] mlir::LogicalResult emitResult =
        emitStmt(catchStmt->getHandlerBlock(), /*useCurrentScope=*/true);
    assert(emitResult.succeeded() && "failed to emit catch handler block");

    // [except.handle]p11:
    //   The currently handled exception is rethrown if control
    //   reaches the end of a handler of the function-try-block of a
    //   constructor or destructor.

    // It is important that we only do this on fallthrough and not on
    // return.  Note that it's illegal to put a return in a
    // constructor function-try-block's catch handler (p14), so this
    // really only applies to destructors.
    if (doImplicitRethrow) {
      cgm.errorNYI("exitCXXTryStmt: doImplicitRethrow");
      return;
    }

    // Fall out through the catch cleanups.
    catchScope.forceCleanup();
  }

  // Because in wasm we merge all catch clauses into one big catchpad, in case
  // none of the types in catch handlers matches after we test against each of
  // them, we should unwind to the next EH enclosing scope. We generate a call
  // to rethrow function here to do that.
  if (EHPersonality::get(*this).isWasmPersonality() && !hasCatchAll) {
    cgm.errorNYI("exitCXXTryStmt: WASM personality without catch all");
  }

  assert(!cir::MissingFeatures::incrementProfileCounter());
}

void CIRGenFunction::populateCatchHandlers(cir::TryOp tryOp) {
  assert(ehStack.requiresCatchOrCleanup());
  assert(!cgm.getLangOpts().IgnoreExceptions &&
         "LandingPad should not be emitted when -fignore-exceptions are in "
         "effect.");

  EHScope &innermostEHScope = *ehStack.find(ehStack.getInnermostEHScope());
  switch (innermostEHScope.getKind()) {
  case EHScope::Terminate:
    cgm.errorNYI("populateCatchHandlers: terminate");
    return;

  case EHScope::Catch:
  case EHScope::Cleanup:
  case EHScope::Filter:
    // CIR does not cache landing pads.
    break;
  }

  // If there's an existing TryOp, it means we got a `cir.try` scope
  // that leads to this "landing pad" creation site. Otherwise, exceptions
  // are enabled but a throwing function is called anyways (common pattern
  // with function local static initializers).
  mlir::ArrayAttr handlerTypesAttr = tryOp.getHandlerTypesAttr();
  if (!handlerTypesAttr || handlerTypesAttr.empty()) {
    // Accumulate all the handlers in scope.
    bool hasCatchAll = false;
    llvm::SmallPtrSet<mlir::Attribute, 4> catchTypes;
    llvm::SmallVector<mlir::Attribute> handlerAttrs;
    for (EHScopeStack::iterator i = ehStack.begin(), e = ehStack.end(); i != e;
         ++i) {
      switch (i->getKind()) {
      case EHScope::Cleanup:
        cgm.errorNYI("emitLandingPad: Cleanup");
        return;

      case EHScope::Filter:
        cgm.errorNYI("emitLandingPad: Filter");
        return;

      case EHScope::Terminate:
        cgm.errorNYI("emitLandingPad: Terminate");
        return;

      case EHScope::Catch:
        break;
      } // end switch

      EHCatchScope &catchScope = cast<EHCatchScope>(*i);
      for (const EHCatchScope::Handler &handler :
           llvm::make_range(catchScope.begin(), catchScope.end())) {
        assert(handler.type.flags == 0 &&
               "landingpads do not support catch handler flags");

        // If this is a catch-all, register that and abort.
        if (handler.isCatchAll()) {
          assert(!hasCatchAll);
          hasCatchAll = true;
          break;
        }

        // Check whether we already have a handler for this type.
        // If not, keep track to later add to catch op.
        if (catchTypes.insert(handler.type.rtti).second)
          handlerAttrs.push_back(handler.type.rtti);
      }

      if (hasCatchAll)
        break;
    }

    if (hasCatchAll)
      handlerAttrs.push_back(cir::CatchAllAttr::get(&getMLIRContext()));

    assert(!cir::MissingFeatures::ehScopeFilter());

    // If there's no catch_all, attach the unwind region. This needs to be the
    // last region in the TryOp catch list.
    if (!hasCatchAll)
      handlerAttrs.push_back(cir::UnwindAttr::get(&getMLIRContext()));

    // Add final array of clauses into TryOp.
    tryOp.setHandlerTypesAttr(
        mlir::ArrayAttr::get(&getMLIRContext(), handlerAttrs));
  }

  // In traditional LLVM codegen. this tells the backend how to generate the
  // landing pad by generating a branch to the dispatch block. In CIR,
  // this is used to populate blocks for later filing during
  // cleanup handling.
  populateEHCatchRegions(ehStack.getInnermostEHScope(), tryOp);
}

// Differently from LLVM traditional codegen, there are no dispatch blocks
// to look at given cir.try_call does not jump to blocks like invoke does.
// However.
void CIRGenFunction::populateEHCatchRegions(EHScopeStack::stable_iterator scope,
                                            cir::TryOp tryOp) {
  if (EHPersonality::get(*this).usesFuncletPads()) {
    cgm.errorNYI("getEHDispatchBlock: usesFuncletPads");
    return;
  }

  // The dispatch block for the end of the scope chain is a block that
  // just resumes unwinding.
  if (scope == ehStack.stable_end()) {
    populateUnwindResumeBlock(/*isCleanup=*/true, tryOp);
    return;
  }

  // Otherwise, we should look at the actual scope.
  EHScope &ehScope = *ehStack.find(scope);
  bool mayThrow = ehScope.mayThrow();

  mlir::Block *originalBlock = nullptr;
  if (mayThrow && tryOp) {
    // If the dispatch is cached but comes from a different tryOp, make sure:
    // - Populate current `tryOp` with a new dispatch block regardless.
    // - Update the map to enqueue new dispatchBlock to also get a cleanup. See
    // code at the end of the function.
    cgm.errorNYI("getEHDispatchBlock: mayThrow & tryOp");
    return;
  }

  if (!mayThrow) {
    switch (ehScope.getKind()) {
    case EHScope::Catch: {
      mayThrow = true;

      // LLVM does some optimization with branches here, CIR just keep track of
      // the corresponding calls.
      EHCatchScope &catchScope = cast<EHCatchScope>(ehScope);
      if (catchScope.getNumHandlers() == 1 &&
          catchScope.getHandler(0).isCatchAll()) {
        break;
      }

      // TODO(cir): In the incubator we create a new basic block with YieldOp
      // inside the attached cleanup region, but this part will be redesigned
      break;
    }
    case EHScope::Cleanup: {
      cgm.errorNYI("getEHDispatchBlock: mayThrow & cleanup");
      return;
    }
    case EHScope::Filter: {
      cgm.errorNYI("getEHDispatchBlock: mayThrow & Filter");
      return;
    }
    case EHScope::Terminate: {
      cgm.errorNYI("getEHDispatchBlock: mayThrow & Terminate");
      return;
    }
    }
  }

  if (originalBlock) {
    cgm.errorNYI("getEHDispatchBlock: originalBlock");
    return;
  }

  ehScope.setMayThrow(mayThrow);
}

// in classic codegen this function is mapping to `isInvokeDest` previously and
// currently it's mapping to the conditions that performs early returns in
// `getInvokeDestImpl`, in CIR we need the condition to know if the EH scope may
// throw exception or now.
bool CIRGenFunction::isCatchOrCleanupRequired() {
  // If exceptions are disabled/ignored and SEH is not in use, then there is no
  // invoke destination. SEH "works" even if exceptions are off. In practice,
  // this means that C++ destructors and other EH cleanups don't run, which is
  // consistent with MSVC's behavior, except in the presence of -EHa
  const LangOptions &lo = cgm.getLangOpts();
  if (!lo.Exceptions || lo.IgnoreExceptions) {
    if (!lo.Borland && !lo.MicrosoftExt)
      return false;
    cgm.errorNYI("isInvokeDest: no exceptions or ignore exception");
    return false;
  }

  // CUDA device code doesn't have exceptions.
  if (lo.CUDA && lo.CUDAIsDevice)
    return false;

  return ehStack.requiresCatchOrCleanup();
}

// In classic codegen this function is equivalent to `getInvokeDestImpl`, in
// ClangIR we don't need to return to return any landing pad, we just need to
// populate the catch handlers if they are required
void CIRGenFunction::populateCatchHandlersIfRequired(cir::TryOp tryOp) {
  assert(ehStack.requiresCatchOrCleanup());
  assert(!ehStack.empty());

  const EHPersonality &personality = EHPersonality::get(*this);

  // Set personality function if not already set
  auto funcOp = mlir::cast<cir::FuncOp>(curFn);
  if (!funcOp.getPersonality())
    funcOp.setPersonality(getPersonalityFn(cgm, personality));

  // CIR does not cache landing pads.
  if (personality.usesFuncletPads()) {
    cgm.errorNYI("getInvokeDestImpl: usesFuncletPads");
  } else {
    populateCatchHandlers(tryOp);
  }
}
