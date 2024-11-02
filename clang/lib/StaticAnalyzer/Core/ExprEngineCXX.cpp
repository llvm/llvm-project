//===- ExprEngineCXX.cpp - ExprEngine support for C++ -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the C++ expression evaluation engine.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/ExprEngine.h"
#include "clang/Analysis/ConstructionContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/AST/ParentMap.h"
#include "clang/Basic/PrettyStackTrace.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/AnalysisManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"

using namespace clang;
using namespace ento;

void ExprEngine::CreateCXXTemporaryObject(const MaterializeTemporaryExpr *ME,
                                          ExplodedNode *Pred,
                                          ExplodedNodeSet &Dst) {
  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);
  const Expr *tempExpr = ME->getSubExpr()->IgnoreParens();
  ProgramStateRef state = Pred->getState();
  const LocationContext *LCtx = Pred->getLocationContext();

  state = createTemporaryRegionIfNeeded(state, LCtx, tempExpr, ME);
  Bldr.generateNode(ME, Pred, state);
}

// FIXME: This is the sort of code that should eventually live in a Core
// checker rather than as a special case in ExprEngine.
void ExprEngine::performTrivialCopy(NodeBuilder &Bldr, ExplodedNode *Pred,
                                    const CallEvent &Call) {
  SVal ThisVal;
  bool AlwaysReturnsLValue;
  const CXXRecordDecl *ThisRD = nullptr;
  if (const CXXConstructorCall *Ctor = dyn_cast<CXXConstructorCall>(&Call)) {
    assert(Ctor->getDecl()->isTrivial());
    assert(Ctor->getDecl()->isCopyOrMoveConstructor());
    ThisVal = Ctor->getCXXThisVal();
    ThisRD = Ctor->getDecl()->getParent();
    AlwaysReturnsLValue = false;
  } else {
    assert(cast<CXXMethodDecl>(Call.getDecl())->isTrivial());
    assert(cast<CXXMethodDecl>(Call.getDecl())->getOverloadedOperator() ==
           OO_Equal);
    ThisVal = cast<CXXInstanceCall>(Call).getCXXThisVal();
    ThisRD = cast<CXXMethodDecl>(Call.getDecl())->getParent();
    AlwaysReturnsLValue = true;
  }

  assert(ThisRD);
  if (ThisRD->isEmpty()) {
    // Do nothing for empty classes. Otherwise it'd retrieve an UnknownVal
    // and bind it and RegionStore would think that the actual value
    // in this region at this offset is unknown.
    return;
  }

  const LocationContext *LCtx = Pred->getLocationContext();

  ExplodedNodeSet Dst;
  Bldr.takeNodes(Pred);

  SVal V = Call.getArgSVal(0);

  // If the value being copied is not unknown, load from its location to get
  // an aggregate rvalue.
  if (Optional<Loc> L = V.getAs<Loc>())
    V = Pred->getState()->getSVal(*L);
  else
    assert(V.isUnknownOrUndef());

  const Expr *CallExpr = Call.getOriginExpr();
  evalBind(Dst, CallExpr, Pred, ThisVal, V, true);

  PostStmt PS(CallExpr, LCtx);
  for (ExplodedNodeSet::iterator I = Dst.begin(), E = Dst.end();
       I != E; ++I) {
    ProgramStateRef State = (*I)->getState();
    if (AlwaysReturnsLValue)
      State = State->BindExpr(CallExpr, LCtx, ThisVal);
    else
      State = bindReturnValue(Call, LCtx, State);
    Bldr.generateNode(PS, State, *I);
  }
}


SVal ExprEngine::makeZeroElementRegion(ProgramStateRef State, SVal LValue,
                                       QualType &Ty, bool &IsArray) {
  SValBuilder &SVB = State->getStateManager().getSValBuilder();
  ASTContext &Ctx = SVB.getContext();

  while (const ArrayType *AT = Ctx.getAsArrayType(Ty)) {
    Ty = AT->getElementType();
    LValue = State->getLValue(Ty, SVB.makeZeroArrayIndex(), LValue);
    IsArray = true;
  }

  return LValue;
}

SVal ExprEngine::computeObjectUnderConstruction(
    const Expr *E, ProgramStateRef State, const LocationContext *LCtx,
    const ConstructionContext *CC, EvalCallOptions &CallOpts) {
  SValBuilder &SVB = getSValBuilder();
  MemRegionManager &MRMgr = SVB.getRegionManager();
  ASTContext &ACtx = SVB.getContext();

  // Compute the target region by exploring the construction context.
  if (CC) {
    switch (CC->getKind()) {
    case ConstructionContext::CXX17ElidedCopyVariableKind:
    case ConstructionContext::SimpleVariableKind: {
      const auto *DSCC = cast<VariableConstructionContext>(CC);
      const auto *DS = DSCC->getDeclStmt();
      const auto *Var = cast<VarDecl>(DS->getSingleDecl());
      QualType Ty = Var->getType();
      return makeZeroElementRegion(State, State->getLValue(Var, LCtx), Ty,
                                   CallOpts.IsArrayCtorOrDtor);
    }
    case ConstructionContext::CXX17ElidedCopyConstructorInitializerKind:
    case ConstructionContext::SimpleConstructorInitializerKind: {
      const auto *ICC = cast<ConstructorInitializerConstructionContext>(CC);
      const auto *Init = ICC->getCXXCtorInitializer();
      const CXXMethodDecl *CurCtor = cast<CXXMethodDecl>(LCtx->getDecl());
      Loc ThisPtr = SVB.getCXXThis(CurCtor, LCtx->getStackFrame());
      SVal ThisVal = State->getSVal(ThisPtr);
      if (Init->isBaseInitializer()) {
        const auto *ThisReg = cast<SubRegion>(ThisVal.getAsRegion());
        const CXXRecordDecl *BaseClass =
          Init->getBaseClass()->getAsCXXRecordDecl();
        const auto *BaseReg =
          MRMgr.getCXXBaseObjectRegion(BaseClass, ThisReg,
                                       Init->isBaseVirtual());
        return SVB.makeLoc(BaseReg);
      }
      if (Init->isDelegatingInitializer())
        return ThisVal;

      const ValueDecl *Field;
      SVal FieldVal;
      if (Init->isIndirectMemberInitializer()) {
        Field = Init->getIndirectMember();
        FieldVal = State->getLValue(Init->getIndirectMember(), ThisVal);
      } else {
        Field = Init->getMember();
        FieldVal = State->getLValue(Init->getMember(), ThisVal);
      }

      QualType Ty = Field->getType();
      return makeZeroElementRegion(State, FieldVal, Ty,
                                   CallOpts.IsArrayCtorOrDtor);
    }
    case ConstructionContext::NewAllocatedObjectKind: {
      if (AMgr.getAnalyzerOptions().MayInlineCXXAllocator) {
        const auto *NECC = cast<NewAllocatedObjectConstructionContext>(CC);
        const auto *NE = NECC->getCXXNewExpr();
        SVal V = *getObjectUnderConstruction(State, NE, LCtx);
        if (const SubRegion *MR =
                dyn_cast_or_null<SubRegion>(V.getAsRegion())) {
          if (NE->isArray()) {
            // TODO: In fact, we need to call the constructor for every
            // allocated element, not just the first one!
            CallOpts.IsArrayCtorOrDtor = true;
            return loc::MemRegionVal(getStoreManager().GetElementZeroRegion(
                MR, NE->getType()->getPointeeType()));
          }
          return  V;
        }
        // TODO: Detect when the allocator returns a null pointer.
        // Constructor shall not be called in this case.
      }
      break;
    }
    case ConstructionContext::SimpleReturnedValueKind:
    case ConstructionContext::CXX17ElidedCopyReturnedValueKind: {
      // The temporary is to be managed by the parent stack frame.
      // So build it in the parent stack frame if we're not in the
      // top frame of the analysis.
      const StackFrameContext *SFC = LCtx->getStackFrame();
      if (const LocationContext *CallerLCtx = SFC->getParent()) {
        auto RTC = (*SFC->getCallSiteBlock())[SFC->getIndex()]
                       .getAs<CFGCXXRecordTypedCall>();
        if (!RTC) {
          // We were unable to find the correct construction context for the
          // call in the parent stack frame. This is equivalent to not being
          // able to find construction context at all.
          break;
        }
        if (isa<BlockInvocationContext>(CallerLCtx)) {
          // Unwrap block invocation contexts. They're mostly part of
          // the current stack frame.
          CallerLCtx = CallerLCtx->getParent();
          assert(!isa<BlockInvocationContext>(CallerLCtx));
        }
        return computeObjectUnderConstruction(
            cast<Expr>(SFC->getCallSite()), State, CallerLCtx,
            RTC->getConstructionContext(), CallOpts);
      } else {
        // We are on the top frame of the analysis. We do not know where is the
        // object returned to. Conjure a symbolic region for the return value.
        // TODO: We probably need a new MemRegion kind to represent the storage
        // of that SymbolicRegion, so that we cound produce a fancy symbol
        // instead of an anonymous conjured symbol.
        // TODO: Do we need to track the region to avoid having it dead
        // too early? It does die too early, at least in C++17, but because
        // putting anything into a SymbolicRegion causes an immediate escape,
        // it doesn't cause any leak false positives.
        const auto *RCC = cast<ReturnedValueConstructionContext>(CC);
        // Make sure that this doesn't coincide with any other symbol
        // conjured for the returned expression.
        static const int TopLevelSymRegionTag = 0;
        const Expr *RetE = RCC->getReturnStmt()->getRetValue();
        assert(RetE && "Void returns should not have a construction context");
        QualType ReturnTy = RetE->getType();
        QualType RegionTy = ACtx.getPointerType(ReturnTy);
        return SVB.conjureSymbolVal(&TopLevelSymRegionTag, RetE, SFC, RegionTy,
                                    currBldrCtx->blockCount());
      }
      llvm_unreachable("Unhandled return value construction context!");
    }
    case ConstructionContext::ElidedTemporaryObjectKind: {
      assert(AMgr.getAnalyzerOptions().ShouldElideConstructors);
      const auto *TCC = cast<ElidedTemporaryObjectConstructionContext>(CC);

      // Support pre-C++17 copy elision. We'll have the elidable copy
      // constructor in the AST and in the CFG, but we'll skip it
      // and construct directly into the final object. This call
      // also sets the CallOpts flags for us.
      // If the elided copy/move constructor is not supported, there's still
      // benefit in trying to model the non-elided constructor.
      // Stash our state before trying to elide, as it'll get overwritten.
      ProgramStateRef PreElideState = State;
      EvalCallOptions PreElideCallOpts = CallOpts;

      SVal V = computeObjectUnderConstruction(
          TCC->getConstructorAfterElision(), State, LCtx,
          TCC->getConstructionContextAfterElision(), CallOpts);

      // FIXME: This definition of "copy elision has not failed" is unreliable.
      // It doesn't indicate that the constructor will actually be inlined
      // later; this is still up to evalCall() to decide.
      if (!CallOpts.IsCtorOrDtorWithImproperlyModeledTargetRegion)
        return V;

      // Copy elision failed. Revert the changes and proceed as if we have
      // a simple temporary.
      CallOpts = PreElideCallOpts;
      CallOpts.IsElidableCtorThatHasNotBeenElided = true;
      LLVM_FALLTHROUGH;
    }
    case ConstructionContext::SimpleTemporaryObjectKind: {
      const auto *TCC = cast<TemporaryObjectConstructionContext>(CC);
      const MaterializeTemporaryExpr *MTE = TCC->getMaterializedTemporaryExpr();

      CallOpts.IsTemporaryCtorOrDtor = true;
      if (MTE) {
        if (const ValueDecl *VD = MTE->getExtendingDecl()) {
          assert(MTE->getStorageDuration() != SD_FullExpression);
          if (!VD->getType()->isReferenceType()) {
            // We're lifetime-extended by a surrounding aggregate.
            // Automatic destructors aren't quite working in this case
            // on the CFG side. We should warn the caller about that.
            // FIXME: Is there a better way to retrieve this information from
            // the MaterializeTemporaryExpr?
            CallOpts.IsTemporaryLifetimeExtendedViaAggregate = true;
          }
        }

        if (MTE->getStorageDuration() == SD_Static ||
            MTE->getStorageDuration() == SD_Thread)
          return loc::MemRegionVal(MRMgr.getCXXStaticTempObjectRegion(E));
      }

      return loc::MemRegionVal(MRMgr.getCXXTempObjectRegion(E, LCtx));
    }
    case ConstructionContext::ArgumentKind: {
      // Arguments are technically temporaries.
      CallOpts.IsTemporaryCtorOrDtor = true;

      const auto *ACC = cast<ArgumentConstructionContext>(CC);
      const Expr *E = ACC->getCallLikeExpr();
      unsigned Idx = ACC->getIndex();

      CallEventManager &CEMgr = getStateManager().getCallEventManager();
      auto getArgLoc = [&](CallEventRef<> Caller) -> Optional<SVal> {
        const LocationContext *FutureSFC =
            Caller->getCalleeStackFrame(currBldrCtx->blockCount());
        // Return early if we are unable to reliably foresee
        // the future stack frame.
        if (!FutureSFC)
          return None;

        // This should be equivalent to Caller->getDecl() for now, but
        // FutureSFC->getDecl() is likely to support better stuff (like
        // virtual functions) earlier.
        const Decl *CalleeD = FutureSFC->getDecl();

        // FIXME: Support for variadic arguments is not implemented here yet.
        if (CallEvent::isVariadic(CalleeD))
          return None;

        // Operator arguments do not correspond to operator parameters
        // because this-argument is implemented as a normal argument in
        // operator call expressions but not in operator declarations.
        const TypedValueRegion *TVR = Caller->getParameterLocation(
            *Caller->getAdjustedParameterIndex(Idx), currBldrCtx->blockCount());
        if (!TVR)
          return None;

        return loc::MemRegionVal(TVR);
      };

      if (const auto *CE = dyn_cast<CallExpr>(E)) {
        CallEventRef<> Caller = CEMgr.getSimpleCall(CE, State, LCtx);
        if (Optional<SVal> V = getArgLoc(Caller))
          return *V;
        else
          break;
      } else if (const auto *CCE = dyn_cast<CXXConstructExpr>(E)) {
        // Don't bother figuring out the target region for the future
        // constructor because we won't need it.
        CallEventRef<> Caller =
            CEMgr.getCXXConstructorCall(CCE, /*Target=*/nullptr, State, LCtx);
        if (Optional<SVal> V = getArgLoc(Caller))
          return *V;
        else
          break;
      } else if (const auto *ME = dyn_cast<ObjCMessageExpr>(E)) {
        CallEventRef<> Caller = CEMgr.getObjCMethodCall(ME, State, LCtx);
        if (Optional<SVal> V = getArgLoc(Caller))
          return *V;
        else
          break;
      }
    }
    } // switch (CC->getKind())
  }

  // If we couldn't find an existing region to construct into, assume we're
  // constructing a temporary. Notify the caller of our failure.
  CallOpts.IsCtorOrDtorWithImproperlyModeledTargetRegion = true;
  return loc::MemRegionVal(MRMgr.getCXXTempObjectRegion(E, LCtx));
}

ProgramStateRef ExprEngine::updateObjectsUnderConstruction(
    SVal V, const Expr *E, ProgramStateRef State, const LocationContext *LCtx,
    const ConstructionContext *CC, const EvalCallOptions &CallOpts) {
  if (CallOpts.IsCtorOrDtorWithImproperlyModeledTargetRegion) {
    // Sounds like we failed to find the target region and therefore
    // copy elision failed. There's nothing we can do about it here.
    return State;
  }

  // See if we're constructing an existing region by looking at the
  // current construction context.
  assert(CC && "Computed target region without construction context?");
  switch (CC->getKind()) {
  case ConstructionContext::CXX17ElidedCopyVariableKind:
  case ConstructionContext::SimpleVariableKind: {
    const auto *DSCC = cast<VariableConstructionContext>(CC);
    return addObjectUnderConstruction(State, DSCC->getDeclStmt(), LCtx, V);
    }
    case ConstructionContext::CXX17ElidedCopyConstructorInitializerKind:
    case ConstructionContext::SimpleConstructorInitializerKind: {
      const auto *ICC = cast<ConstructorInitializerConstructionContext>(CC);
      const auto *Init = ICC->getCXXCtorInitializer();
      // Base and delegating initializers handled above
      assert(Init->isAnyMemberInitializer() &&
             "Base and delegating initializers should have been handled by"
             "computeObjectUnderConstruction()");
      return addObjectUnderConstruction(State, Init, LCtx, V);
    }
    case ConstructionContext::NewAllocatedObjectKind: {
      return State;
    }
    case ConstructionContext::SimpleReturnedValueKind:
    case ConstructionContext::CXX17ElidedCopyReturnedValueKind: {
      const StackFrameContext *SFC = LCtx->getStackFrame();
      const LocationContext *CallerLCtx = SFC->getParent();
      if (!CallerLCtx) {
        // No extra work is necessary in top frame.
        return State;
      }

      auto RTC = (*SFC->getCallSiteBlock())[SFC->getIndex()]
                     .getAs<CFGCXXRecordTypedCall>();
      assert(RTC && "Could not have had a target region without it");
      if (isa<BlockInvocationContext>(CallerLCtx)) {
        // Unwrap block invocation contexts. They're mostly part of
        // the current stack frame.
        CallerLCtx = CallerLCtx->getParent();
        assert(!isa<BlockInvocationContext>(CallerLCtx));
      }

      return updateObjectsUnderConstruction(V,
          cast<Expr>(SFC->getCallSite()), State, CallerLCtx,
          RTC->getConstructionContext(), CallOpts);
    }
    case ConstructionContext::ElidedTemporaryObjectKind: {
      assert(AMgr.getAnalyzerOptions().ShouldElideConstructors);
      if (!CallOpts.IsElidableCtorThatHasNotBeenElided) {
        const auto *TCC = cast<ElidedTemporaryObjectConstructionContext>(CC);
        State = updateObjectsUnderConstruction(
            V, TCC->getConstructorAfterElision(), State, LCtx,
            TCC->getConstructionContextAfterElision(), CallOpts);

        // Remember that we've elided the constructor.
        State = addObjectUnderConstruction(
            State, TCC->getConstructorAfterElision(), LCtx, V);

        // Remember that we've elided the destructor.
        if (const auto *BTE = TCC->getCXXBindTemporaryExpr())
          State = elideDestructor(State, BTE, LCtx);

        // Instead of materialization, shamelessly return
        // the final object destination.
        if (const auto *MTE = TCC->getMaterializedTemporaryExpr())
          State = addObjectUnderConstruction(State, MTE, LCtx, V);

        return State;
      }
      // If we decided not to elide the constructor, proceed as if
      // it's a simple temporary.
      LLVM_FALLTHROUGH;
    }
    case ConstructionContext::SimpleTemporaryObjectKind: {
      const auto *TCC = cast<TemporaryObjectConstructionContext>(CC);
      if (const auto *BTE = TCC->getCXXBindTemporaryExpr())
        State = addObjectUnderConstruction(State, BTE, LCtx, V);

      if (const auto *MTE = TCC->getMaterializedTemporaryExpr())
        State = addObjectUnderConstruction(State, MTE, LCtx, V);

      return State;
    }
    case ConstructionContext::ArgumentKind: {
      const auto *ACC = cast<ArgumentConstructionContext>(CC);
      if (const auto *BTE = ACC->getCXXBindTemporaryExpr())
        State = addObjectUnderConstruction(State, BTE, LCtx, V);

      return addObjectUnderConstruction(
          State, {ACC->getCallLikeExpr(), ACC->getIndex()}, LCtx, V);
    }
  }
  llvm_unreachable("Unhandled construction context!");
}

void ExprEngine::handleConstructor(const Expr *E,
                                   ExplodedNode *Pred,
                                   ExplodedNodeSet &destNodes) {
  const auto *CE = dyn_cast<CXXConstructExpr>(E);
  const auto *CIE = dyn_cast<CXXInheritedCtorInitExpr>(E);
  assert(CE || CIE);

  const LocationContext *LCtx = Pred->getLocationContext();
  ProgramStateRef State = Pred->getState();

  SVal Target = UnknownVal();

  if (CE) {
    if (Optional<SVal> ElidedTarget =
            getObjectUnderConstruction(State, CE, LCtx)) {
      // We've previously modeled an elidable constructor by pretending that it
      // in fact constructs into the correct target. This constructor can
      // therefore be skipped.
      Target = *ElidedTarget;
      StmtNodeBuilder Bldr(Pred, destNodes, *currBldrCtx);
      State = finishObjectConstruction(State, CE, LCtx);
      if (auto L = Target.getAs<Loc>())
        State = State->BindExpr(CE, LCtx, State->getSVal(*L, CE->getType()));
      Bldr.generateNode(CE, Pred, State);
      return;
    }
  }

  // FIXME: Handle arrays, which run the same constructor for every element.
  // For now, we just run the first constructor (which should still invalidate
  // the entire array).

  EvalCallOptions CallOpts;
  auto C = getCurrentCFGElement().getAs<CFGConstructor>();
  assert(C || getCurrentCFGElement().getAs<CFGStmt>());
  const ConstructionContext *CC = C ? C->getConstructionContext() : nullptr;

  const CXXConstructExpr::ConstructionKind CK =
      CE ? CE->getConstructionKind() : CIE->getConstructionKind();
  switch (CK) {
  case CXXConstructExpr::CK_Complete: {
    // Inherited constructors are always base class constructors.
    assert(CE && !CIE && "A complete constructor is inherited?!");

    // The target region is found from construction context.
    std::tie(State, Target) =
        handleConstructionContext(CE, State, LCtx, CC, CallOpts);
    break;
  }
  case CXXConstructExpr::CK_VirtualBase: {
    // Make sure we are not calling virtual base class initializers twice.
    // Only the most-derived object should initialize virtual base classes.
    const auto *OuterCtor = dyn_cast_or_null<CXXConstructExpr>(
        LCtx->getStackFrame()->getCallSite());
    assert(
        (!OuterCtor ||
         OuterCtor->getConstructionKind() == CXXConstructExpr::CK_Complete ||
         OuterCtor->getConstructionKind() == CXXConstructExpr::CK_Delegating) &&
        ("This virtual base should have already been initialized by "
         "the most derived class!"));
    (void)OuterCtor;
    LLVM_FALLTHROUGH;
  }
  case CXXConstructExpr::CK_NonVirtualBase:
    // In C++17, classes with non-virtual bases may be aggregates, so they would
    // be initialized as aggregates without a constructor call, so we may have
    // a base class constructed directly into an initializer list without
    // having the derived-class constructor call on the previous stack frame.
    // Initializer lists may be nested into more initializer lists that
    // correspond to surrounding aggregate initializations.
    // FIXME: For now this code essentially bails out. We need to find the
    // correct target region and set it.
    // FIXME: Instead of relying on the ParentMap, we should have the
    // trigger-statement (InitListExpr in this case) passed down from CFG or
    // otherwise always available during construction.
    if (isa_and_nonnull<InitListExpr>(LCtx->getParentMap().getParent(E))) {
      MemRegionManager &MRMgr = getSValBuilder().getRegionManager();
      Target = loc::MemRegionVal(MRMgr.getCXXTempObjectRegion(E, LCtx));
      CallOpts.IsCtorOrDtorWithImproperlyModeledTargetRegion = true;
      break;
    }
    LLVM_FALLTHROUGH;
  case CXXConstructExpr::CK_Delegating: {
    const CXXMethodDecl *CurCtor = cast<CXXMethodDecl>(LCtx->getDecl());
    Loc ThisPtr = getSValBuilder().getCXXThis(CurCtor,
                                              LCtx->getStackFrame());
    SVal ThisVal = State->getSVal(ThisPtr);

    if (CK == CXXConstructExpr::CK_Delegating) {
      Target = ThisVal;
    } else {
      // Cast to the base type.
      bool IsVirtual = (CK == CXXConstructExpr::CK_VirtualBase);
      SVal BaseVal =
          getStoreManager().evalDerivedToBase(ThisVal, E->getType(), IsVirtual);
      Target = BaseVal;
    }
    break;
  }
  }

  if (State != Pred->getState()) {
    static SimpleProgramPointTag T("ExprEngine",
                                   "Prepare for object construction");
    ExplodedNodeSet DstPrepare;
    StmtNodeBuilder BldrPrepare(Pred, DstPrepare, *currBldrCtx);
    BldrPrepare.generateNode(E, Pred, State, &T, ProgramPoint::PreStmtKind);
    assert(DstPrepare.size() <= 1);
    if (DstPrepare.size() == 0)
      return;
    Pred = *BldrPrepare.begin();
  }

  const MemRegion *TargetRegion = Target.getAsRegion();
  CallEventManager &CEMgr = getStateManager().getCallEventManager();
  CallEventRef<> Call =
      CIE ? (CallEventRef<>)CEMgr.getCXXInheritedConstructorCall(
                CIE, TargetRegion, State, LCtx)
          : (CallEventRef<>)CEMgr.getCXXConstructorCall(
                CE, TargetRegion, State, LCtx);

  ExplodedNodeSet DstPreVisit;
  getCheckerManager().runCheckersForPreStmt(DstPreVisit, Pred, E, *this);

  ExplodedNodeSet PreInitialized;
  if (CE) {
    // FIXME: Is it possible and/or useful to do this before PreStmt?
    StmtNodeBuilder Bldr(DstPreVisit, PreInitialized, *currBldrCtx);
    for (ExplodedNodeSet::iterator I = DstPreVisit.begin(),
                                   E = DstPreVisit.end();
         I != E; ++I) {
      ProgramStateRef State = (*I)->getState();
      if (CE->requiresZeroInitialization()) {
        // FIXME: Once we properly handle constructors in new-expressions, we'll
        // need to invalidate the region before setting a default value, to make
        // sure there aren't any lingering bindings around. This probably needs
        // to happen regardless of whether or not the object is zero-initialized
        // to handle random fields of a placement-initialized object picking up
        // old bindings. We might only want to do it when we need to, though.
        // FIXME: This isn't actually correct for arrays -- we need to zero-
        // initialize the entire array, not just the first element -- but our
        // handling of arrays everywhere else is weak as well, so this shouldn't
        // actually make things worse. Placement new makes this tricky as well,
        // since it's then possible to be initializing one part of a multi-
        // dimensional array.
        State = State->bindDefaultZero(Target, LCtx);
      }

      Bldr.generateNode(CE, *I, State, /*tag=*/nullptr,
                        ProgramPoint::PreStmtKind);
    }
  } else {
    PreInitialized = DstPreVisit;
  }

  ExplodedNodeSet DstPreCall;
  getCheckerManager().runCheckersForPreCall(DstPreCall, PreInitialized,
                                            *Call, *this);

  ExplodedNodeSet DstEvaluated;

  if (CE && CE->getConstructor()->isTrivial() &&
      CE->getConstructor()->isCopyOrMoveConstructor() &&
      !CallOpts.IsArrayCtorOrDtor) {
    StmtNodeBuilder Bldr(DstPreCall, DstEvaluated, *currBldrCtx);
    // FIXME: Handle other kinds of trivial constructors as well.
    for (ExplodedNodeSet::iterator I = DstPreCall.begin(), E = DstPreCall.end();
         I != E; ++I)
      performTrivialCopy(Bldr, *I, *Call);

  } else {
    for (ExplodedNodeSet::iterator I = DstPreCall.begin(), E = DstPreCall.end();
         I != E; ++I)
      getCheckerManager().runCheckersForEvalCall(DstEvaluated, *I, *Call, *this,
                                                 CallOpts);
  }

  // If the CFG was constructed without elements for temporary destructors
  // and the just-called constructor created a temporary object then
  // stop exploration if the temporary object has a noreturn constructor.
  // This can lose coverage because the destructor, if it were present
  // in the CFG, would be called at the end of the full expression or
  // later (for life-time extended temporaries) -- but avoids infeasible
  // paths when no-return temporary destructors are used for assertions.
  ExplodedNodeSet DstEvaluatedPostProcessed;
  StmtNodeBuilder Bldr(DstEvaluated, DstEvaluatedPostProcessed, *currBldrCtx);
  const AnalysisDeclContext *ADC = LCtx->getAnalysisDeclContext();
  if (!ADC->getCFGBuildOptions().AddTemporaryDtors) {
    if (llvm::isa_and_nonnull<CXXTempObjectRegion>(TargetRegion) &&
        cast<CXXConstructorDecl>(Call->getDecl())
            ->getParent()
            ->isAnyDestructorNoReturn()) {

      // If we've inlined the constructor, then DstEvaluated would be empty.
      // In this case we still want a sink, which could be implemented
      // in processCallExit. But we don't have that implemented at the moment,
      // so if you hit this assertion, see if you can avoid inlining
      // the respective constructor when analyzer-config cfg-temporary-dtors
      // is set to false.
      // Otherwise there's nothing wrong with inlining such constructor.
      assert(!DstEvaluated.empty() &&
             "We should not have inlined this constructor!");

      for (ExplodedNode *N : DstEvaluated) {
        Bldr.generateSink(E, N, N->getState());
      }

      // There is no need to run the PostCall and PostStmt checker
      // callbacks because we just generated sinks on all nodes in th
      // frontier.
      return;
    }
  }

  ExplodedNodeSet DstPostArgumentCleanup;
  for (ExplodedNode *I : DstEvaluatedPostProcessed)
    finishArgumentConstruction(DstPostArgumentCleanup, I, *Call);

  // If there were other constructors called for object-type arguments
  // of this constructor, clean them up.
  ExplodedNodeSet DstPostCall;
  getCheckerManager().runCheckersForPostCall(DstPostCall,
                                             DstPostArgumentCleanup,
                                             *Call, *this);
  getCheckerManager().runCheckersForPostStmt(destNodes, DstPostCall, E, *this);
}

void ExprEngine::VisitCXXConstructExpr(const CXXConstructExpr *CE,
                                       ExplodedNode *Pred,
                                       ExplodedNodeSet &Dst) {
  handleConstructor(CE, Pred, Dst);
}

void ExprEngine::VisitCXXInheritedCtorInitExpr(
    const CXXInheritedCtorInitExpr *CE, ExplodedNode *Pred,
    ExplodedNodeSet &Dst) {
  handleConstructor(CE, Pred, Dst);
}

void ExprEngine::VisitCXXDestructor(QualType ObjectType,
                                    const MemRegion *Dest,
                                    const Stmt *S,
                                    bool IsBaseDtor,
                                    ExplodedNode *Pred,
                                    ExplodedNodeSet &Dst,
                                    EvalCallOptions &CallOpts) {
  assert(S && "A destructor without a trigger!");
  const LocationContext *LCtx = Pred->getLocationContext();
  ProgramStateRef State = Pred->getState();

  const CXXRecordDecl *RecordDecl = ObjectType->getAsCXXRecordDecl();
  assert(RecordDecl && "Only CXXRecordDecls should have destructors");
  const CXXDestructorDecl *DtorDecl = RecordDecl->getDestructor();
  // FIXME: There should always be a Decl, otherwise the destructor call
  // shouldn't have been added to the CFG in the first place.
  if (!DtorDecl) {
    // Skip the invalid destructor. We cannot simply return because
    // it would interrupt the analysis instead.
    static SimpleProgramPointTag T("ExprEngine", "SkipInvalidDestructor");
    // FIXME: PostImplicitCall with a null decl may crash elsewhere anyway.
    PostImplicitCall PP(/*Decl=*/nullptr, S->getEndLoc(), LCtx, &T);
    NodeBuilder Bldr(Pred, Dst, *currBldrCtx);
    Bldr.generateNode(PP, Pred->getState(), Pred);
    return;
  }

  if (!Dest) {
    // We're trying to destroy something that is not a region. This may happen
    // for a variety of reasons (unknown target region, concrete integer instead
    // of target region, etc.). The current code makes an attempt to recover.
    // FIXME: We probably don't really need to recover when we're dealing
    // with concrete integers specifically.
    CallOpts.IsCtorOrDtorWithImproperlyModeledTargetRegion = true;
    if (const Expr *E = dyn_cast_or_null<Expr>(S)) {
      Dest = MRMgr.getCXXTempObjectRegion(E, Pred->getLocationContext());
    } else {
      static SimpleProgramPointTag T("ExprEngine", "SkipInvalidDestructor");
      NodeBuilder Bldr(Pred, Dst, *currBldrCtx);
      Bldr.generateSink(Pred->getLocation().withTag(&T),
                        Pred->getState(), Pred);
      return;
    }
  }

  CallEventManager &CEMgr = getStateManager().getCallEventManager();
  CallEventRef<CXXDestructorCall> Call =
      CEMgr.getCXXDestructorCall(DtorDecl, S, Dest, IsBaseDtor, State, LCtx);

  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),
                                Call->getSourceRange().getBegin(),
                                "Error evaluating destructor");

  ExplodedNodeSet DstPreCall;
  getCheckerManager().runCheckersForPreCall(DstPreCall, Pred,
                                            *Call, *this);

  ExplodedNodeSet DstInvalidated;
  StmtNodeBuilder Bldr(DstPreCall, DstInvalidated, *currBldrCtx);
  for (ExplodedNodeSet::iterator I = DstPreCall.begin(), E = DstPreCall.end();
       I != E; ++I)
    defaultEvalCall(Bldr, *I, *Call, CallOpts);

  getCheckerManager().runCheckersForPostCall(Dst, DstInvalidated,
                                             *Call, *this);
}

void ExprEngine::VisitCXXNewAllocatorCall(const CXXNewExpr *CNE,
                                          ExplodedNode *Pred,
                                          ExplodedNodeSet &Dst) {
  ProgramStateRef State = Pred->getState();
  const LocationContext *LCtx = Pred->getLocationContext();
  PrettyStackTraceLoc CrashInfo(getContext().getSourceManager(),
                                CNE->getBeginLoc(),
                                "Error evaluating New Allocator Call");
  CallEventManager &CEMgr = getStateManager().getCallEventManager();
  CallEventRef<CXXAllocatorCall> Call =
    CEMgr.getCXXAllocatorCall(CNE, State, LCtx);

  ExplodedNodeSet DstPreCall;
  getCheckerManager().runCheckersForPreCall(DstPreCall, Pred,
                                            *Call, *this);

  ExplodedNodeSet DstPostCall;
  StmtNodeBuilder CallBldr(DstPreCall, DstPostCall, *currBldrCtx);
  for (ExplodedNode *I : DstPreCall) {
    // FIXME: Provide evalCall for checkers?
    defaultEvalCall(CallBldr, I, *Call);
  }
  // If the call is inlined, DstPostCall will be empty and we bail out now.

  // Store return value of operator new() for future use, until the actual
  // CXXNewExpr gets processed.
  ExplodedNodeSet DstPostValue;
  StmtNodeBuilder ValueBldr(DstPostCall, DstPostValue, *currBldrCtx);
  for (ExplodedNode *I : DstPostCall) {
    // FIXME: Because CNE serves as the "call site" for the allocator (due to
    // lack of a better expression in the AST), the conjured return value symbol
    // is going to be of the same type (C++ object pointer type). Technically
    // this is not correct because the operator new's prototype always says that
    // it returns a 'void *'. So we should change the type of the symbol,
    // and then evaluate the cast over the symbolic pointer from 'void *' to
    // the object pointer type. But without changing the symbol's type it
    // is breaking too much to evaluate the no-op symbolic cast over it, so we
    // skip it for now.
    ProgramStateRef State = I->getState();
    SVal RetVal = State->getSVal(CNE, LCtx);

    // If this allocation function is not declared as non-throwing, failures
    // /must/ be signalled by exceptions, and thus the return value will never
    // be NULL. -fno-exceptions does not influence this semantics.
    // FIXME: GCC has a -fcheck-new option, which forces it to consider the case
    // where new can return NULL. If we end up supporting that option, we can
    // consider adding a check for it here.
    // C++11 [basic.stc.dynamic.allocation]p3.
    if (const FunctionDecl *FD = CNE->getOperatorNew()) {
      QualType Ty = FD->getType();
      if (const auto *ProtoType = Ty->getAs<FunctionProtoType>())
        if (!ProtoType->isNothrow())
          State = State->assume(RetVal.castAs<DefinedOrUnknownSVal>(), true);
    }

    ValueBldr.generateNode(
        CNE, I, addObjectUnderConstruction(State, CNE, LCtx, RetVal));
  }

  ExplodedNodeSet DstPostPostCallCallback;
  getCheckerManager().runCheckersForPostCall(DstPostPostCallCallback,
                                             DstPostValue, *Call, *this);
  for (ExplodedNode *I : DstPostPostCallCallback) {
    getCheckerManager().runCheckersForNewAllocator(*Call, Dst, I, *this);
  }
}

void ExprEngine::VisitCXXNewExpr(const CXXNewExpr *CNE, ExplodedNode *Pred,
                                   ExplodedNodeSet &Dst) {
  // FIXME: Much of this should eventually migrate to CXXAllocatorCall.
  // Also, we need to decide how allocators actually work -- they're not
  // really part of the CXXNewExpr because they happen BEFORE the
  // CXXConstructExpr subexpression. See PR12014 for some discussion.

  unsigned blockCount = currBldrCtx->blockCount();
  const LocationContext *LCtx = Pred->getLocationContext();
  SVal symVal = UnknownVal();
  FunctionDecl *FD = CNE->getOperatorNew();

  bool IsStandardGlobalOpNewFunction =
      FD->isReplaceableGlobalAllocationFunction();

  ProgramStateRef State = Pred->getState();

  // Retrieve the stored operator new() return value.
  if (AMgr.getAnalyzerOptions().MayInlineCXXAllocator) {
    symVal = *getObjectUnderConstruction(State, CNE, LCtx);
    State = finishObjectConstruction(State, CNE, LCtx);
  }

  // We assume all standard global 'operator new' functions allocate memory in
  // heap. We realize this is an approximation that might not correctly model
  // a custom global allocator.
  if (symVal.isUnknown()) {
    if (IsStandardGlobalOpNewFunction)
      symVal = svalBuilder.getConjuredHeapSymbolVal(CNE, LCtx, blockCount);
    else
      symVal = svalBuilder.conjureSymbolVal(nullptr, CNE, LCtx, CNE->getType(),
                                            blockCount);
  }

  CallEventManager &CEMgr = getStateManager().getCallEventManager();
  CallEventRef<CXXAllocatorCall> Call =
    CEMgr.getCXXAllocatorCall(CNE, State, LCtx);

  if (!AMgr.getAnalyzerOptions().MayInlineCXXAllocator) {
    // Invalidate placement args.
    // FIXME: Once we figure out how we want allocators to work,
    // we should be using the usual pre-/(default-)eval-/post-call checkers
    // here.
    State = Call->invalidateRegions(blockCount);
    if (!State)
      return;

    // If this allocation function is not declared as non-throwing, failures
    // /must/ be signalled by exceptions, and thus the return value will never
    // be NULL. -fno-exceptions does not influence this semantics.
    // FIXME: GCC has a -fcheck-new option, which forces it to consider the case
    // where new can return NULL. If we end up supporting that option, we can
    // consider adding a check for it here.
    // C++11 [basic.stc.dynamic.allocation]p3.
    if (const auto *ProtoType = FD->getType()->getAs<FunctionProtoType>())
      if (!ProtoType->isNothrow())
        if (auto dSymVal = symVal.getAs<DefinedOrUnknownSVal>())
          State = State->assume(*dSymVal, true);
  }

  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);

  SVal Result = symVal;

  if (CNE->isArray()) {
    // FIXME: allocating an array requires simulating the constructors.
    // For now, just return a symbolicated region.
    if (const auto *NewReg = cast_or_null<SubRegion>(symVal.getAsRegion())) {
      QualType ObjTy = CNE->getType()->getPointeeType();
      const ElementRegion *EleReg =
          getStoreManager().GetElementZeroRegion(NewReg, ObjTy);
      Result = loc::MemRegionVal(EleReg);
    }
    State = State->BindExpr(CNE, Pred->getLocationContext(), Result);
    Bldr.generateNode(CNE, Pred, State);
    return;
  }

  // FIXME: Once we have proper support for CXXConstructExprs inside
  // CXXNewExpr, we need to make sure that the constructed object is not
  // immediately invalidated here. (The placement call should happen before
  // the constructor call anyway.)
  if (FD->isReservedGlobalPlacementOperator()) {
    // Non-array placement new should always return the placement location.
    SVal PlacementLoc = State->getSVal(CNE->getPlacementArg(0), LCtx);
    Result = svalBuilder.evalCast(PlacementLoc, CNE->getType(),
                                  CNE->getPlacementArg(0)->getType());
  }

  // Bind the address of the object, then check to see if we cached out.
  State = State->BindExpr(CNE, LCtx, Result);
  ExplodedNode *NewN = Bldr.generateNode(CNE, Pred, State);
  if (!NewN)
    return;

  // If the type is not a record, we won't have a CXXConstructExpr as an
  // initializer. Copy the value over.
  if (const Expr *Init = CNE->getInitializer()) {
    if (!isa<CXXConstructExpr>(Init)) {
      assert(Bldr.getResults().size() == 1);
      Bldr.takeNodes(NewN);
      evalBind(Dst, CNE, NewN, Result, State->getSVal(Init, LCtx),
               /*FirstInit=*/IsStandardGlobalOpNewFunction);
    }
  }
}

void ExprEngine::VisitCXXDeleteExpr(const CXXDeleteExpr *CDE,
                                    ExplodedNode *Pred, ExplodedNodeSet &Dst) {

  CallEventManager &CEMgr = getStateManager().getCallEventManager();
  CallEventRef<CXXDeallocatorCall> Call = CEMgr.getCXXDeallocatorCall(
      CDE, Pred->getState(), Pred->getLocationContext());

  ExplodedNodeSet DstPreCall;
  getCheckerManager().runCheckersForPreCall(DstPreCall, Pred, *Call, *this);
  ExplodedNodeSet DstPostCall;

  if (AMgr.getAnalyzerOptions().MayInlineCXXAllocator) {
    StmtNodeBuilder Bldr(DstPreCall, DstPostCall, *currBldrCtx);
    for (ExplodedNode *I : DstPreCall) {
      defaultEvalCall(Bldr, I, *Call);
    }
  } else {
    DstPostCall = DstPreCall;
  }
  getCheckerManager().runCheckersForPostCall(Dst, DstPostCall, *Call, *this);
}

void ExprEngine::VisitCXXCatchStmt(const CXXCatchStmt *CS, ExplodedNode *Pred,
                                   ExplodedNodeSet &Dst) {
  const VarDecl *VD = CS->getExceptionDecl();
  if (!VD) {
    Dst.Add(Pred);
    return;
  }

  const LocationContext *LCtx = Pred->getLocationContext();
  SVal V = svalBuilder.conjureSymbolVal(CS, LCtx, VD->getType(),
                                        currBldrCtx->blockCount());
  ProgramStateRef state = Pred->getState();
  state = state->bindLoc(state->getLValue(VD, LCtx), V, LCtx);

  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);
  Bldr.generateNode(CS, Pred, state);
}

void ExprEngine::VisitCXXThisExpr(const CXXThisExpr *TE, ExplodedNode *Pred,
                                    ExplodedNodeSet &Dst) {
  StmtNodeBuilder Bldr(Pred, Dst, *currBldrCtx);

  // Get the this object region from StoreManager.
  const LocationContext *LCtx = Pred->getLocationContext();
  const MemRegion *R =
    svalBuilder.getRegionManager().getCXXThisRegion(
                                  getContext().getCanonicalType(TE->getType()),
                                                    LCtx);

  ProgramStateRef state = Pred->getState();
  SVal V = state->getSVal(loc::MemRegionVal(R));
  Bldr.generateNode(TE, Pred, state->BindExpr(TE, LCtx, V));
}

void ExprEngine::VisitLambdaExpr(const LambdaExpr *LE, ExplodedNode *Pred,
                                 ExplodedNodeSet &Dst) {
  const LocationContext *LocCtxt = Pred->getLocationContext();

  // Get the region of the lambda itself.
  const MemRegion *R = svalBuilder.getRegionManager().getCXXTempObjectRegion(
      LE, LocCtxt);
  SVal V = loc::MemRegionVal(R);

  ProgramStateRef State = Pred->getState();

  // If we created a new MemRegion for the lambda, we should explicitly bind
  // the captures.
  CXXRecordDecl::field_iterator CurField = LE->getLambdaClass()->field_begin();
  for (LambdaExpr::const_capture_init_iterator i = LE->capture_init_begin(),
                                               e = LE->capture_init_end();
       i != e; ++i, ++CurField) {
    FieldDecl *FieldForCapture = *CurField;
    SVal FieldLoc = State->getLValue(FieldForCapture, V);

    SVal InitVal;
    if (!FieldForCapture->hasCapturedVLAType()) {
      Expr *InitExpr = *i;
      assert(InitExpr && "Capture missing initialization expression");
      InitVal = State->getSVal(InitExpr, LocCtxt);
    } else {
      // The field stores the length of a captured variable-length array.
      // These captures don't have initialization expressions; instead we
      // get the length from the VLAType size expression.
      Expr *SizeExpr = FieldForCapture->getCapturedVLAType()->getSizeExpr();
      InitVal = State->getSVal(SizeExpr, LocCtxt);
    }

    State = State->bindLoc(FieldLoc, InitVal, LocCtxt);
  }

  // Decay the Loc into an RValue, because there might be a
  // MaterializeTemporaryExpr node above this one which expects the bound value
  // to be an RValue.
  SVal LambdaRVal = State->getSVal(R);

  ExplodedNodeSet Tmp;
  StmtNodeBuilder Bldr(Pred, Tmp, *currBldrCtx);
  // FIXME: is this the right program point kind?
  Bldr.generateNode(LE, Pred,
                    State->BindExpr(LE, LocCtxt, LambdaRVal),
                    nullptr, ProgramPoint::PostLValueKind);

  // FIXME: Move all post/pre visits to ::Visit().
  getCheckerManager().runCheckersForPostStmt(Dst, Tmp, LE, *this);
}
