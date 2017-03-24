//===--- SemaCoroutines.cpp - Semantic Analysis for Coroutines ------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file implements semantic analysis for C++ Coroutines.
//
//===----------------------------------------------------------------------===//

#include "clang/Sema/SemaInternal.h"
#include "clang/AST/Decl.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/Initialization.h"
#include "clang/Sema/Overload.h"
using namespace clang;
using namespace sema;

static bool lookupMember(Sema &S, const char *Name, CXXRecordDecl *RD,
                         SourceLocation Loc) {
  DeclarationName DN = S.PP.getIdentifierInfo(Name);
  LookupResult LR(S, DN, Loc, Sema::LookupMemberName);
  // Suppress diagnostics when a private member is selected. The same warnings
  // will be produced again when building the call.
  LR.suppressDiagnostics();
  return S.LookupQualifiedName(LR, RD);
}

/// Look up the std::coroutine_traits<...>::promise_type for the given
/// function type.
static QualType lookupPromiseType(Sema &S, const FunctionProtoType *FnType,
                                  SourceLocation KwLoc,
                                  SourceLocation FuncLoc) {
  // FIXME: Cache std::coroutine_traits once we've found it.
  NamespaceDecl *StdExp = S.lookupStdExperimentalNamespace();
  if (!StdExp) {
    S.Diag(KwLoc, diag::err_implied_coroutine_type_not_found)
        << "std::experimental::coroutine_traits";
    return QualType();
  }

  LookupResult Result(S, &S.PP.getIdentifierTable().get("coroutine_traits"),
                      FuncLoc, Sema::LookupOrdinaryName);
  if (!S.LookupQualifiedName(Result, StdExp)) {
    S.Diag(KwLoc, diag::err_implied_coroutine_type_not_found)
        << "std::experimental::coroutine_traits";
    return QualType();
  }

  ClassTemplateDecl *CoroTraits = Result.getAsSingle<ClassTemplateDecl>();
  if (!CoroTraits) {
    Result.suppressDiagnostics();
    // We found something weird. Complain about the first thing we found.
    NamedDecl *Found = *Result.begin();
    S.Diag(Found->getLocation(), diag::err_malformed_std_coroutine_traits);
    return QualType();
  }

  // Form template argument list for coroutine_traits<R, P1, P2, ...>.
  TemplateArgumentListInfo Args(KwLoc, KwLoc);
  Args.addArgument(TemplateArgumentLoc(
      TemplateArgument(FnType->getReturnType()),
      S.Context.getTrivialTypeSourceInfo(FnType->getReturnType(), KwLoc)));
  // FIXME: If the function is a non-static member function, add the type
  // of the implicit object parameter before the formal parameters.
  for (QualType T : FnType->getParamTypes())
    Args.addArgument(TemplateArgumentLoc(
        TemplateArgument(T), S.Context.getTrivialTypeSourceInfo(T, KwLoc)));

  // Build the template-id.
  QualType CoroTrait =
      S.CheckTemplateIdType(TemplateName(CoroTraits), KwLoc, Args);
  if (CoroTrait.isNull())
    return QualType();
  if (S.RequireCompleteType(KwLoc, CoroTrait,
                            diag::err_coroutine_type_missing_specialization))
    return QualType();

  auto *RD = CoroTrait->getAsCXXRecordDecl();
  assert(RD && "specialization of class template is not a class?");

  // Look up the ::promise_type member.
  LookupResult R(S, &S.PP.getIdentifierTable().get("promise_type"), KwLoc,
                 Sema::LookupOrdinaryName);
  S.LookupQualifiedName(R, RD);
  auto *Promise = R.getAsSingle<TypeDecl>();
  if (!Promise) {
    S.Diag(FuncLoc,
           diag::err_implied_std_coroutine_traits_promise_type_not_found)
        << RD;
    return QualType();
  }
  // The promise type is required to be a class type.
  QualType PromiseType = S.Context.getTypeDeclType(Promise);

  auto buildElaboratedType = [&]() {
    auto *NNS = NestedNameSpecifier::Create(S.Context, nullptr, StdExp);
    NNS = NestedNameSpecifier::Create(S.Context, NNS, false,
                                      CoroTrait.getTypePtr());
    return S.Context.getElaboratedType(ETK_None, NNS, PromiseType);
  };

  if (!PromiseType->getAsCXXRecordDecl()) {
    S.Diag(FuncLoc,
           diag::err_implied_std_coroutine_traits_promise_type_not_class)
        << buildElaboratedType();
    return QualType();
  }
  if (S.RequireCompleteType(FuncLoc, buildElaboratedType(),
                            diag::err_coroutine_promise_type_incomplete))
    return QualType();

  return PromiseType;
}

/// Look up the std::coroutine_traits<...>::promise_type for the given
/// function type.
static QualType lookupCoroutineHandleType(Sema &S, QualType PromiseType,
                                          SourceLocation Loc) {
  if (PromiseType.isNull())
    return QualType();

  NamespaceDecl *StdExp = S.lookupStdExperimentalNamespace();
  assert(StdExp && "Should already be diagnosed");

  LookupResult Result(S, &S.PP.getIdentifierTable().get("coroutine_handle"),
                      Loc, Sema::LookupOrdinaryName);
  if (!S.LookupQualifiedName(Result, StdExp)) {
    S.Diag(Loc, diag::err_implied_coroutine_type_not_found)
        << "std::experimental::coroutine_handle";
    return QualType();
  }

  ClassTemplateDecl *CoroHandle = Result.getAsSingle<ClassTemplateDecl>();
  if (!CoroHandle) {
    Result.suppressDiagnostics();
    // We found something weird. Complain about the first thing we found.
    NamedDecl *Found = *Result.begin();
    S.Diag(Found->getLocation(), diag::err_malformed_std_coroutine_handle);
    return QualType();
  }

  // Form template argument list for coroutine_handle<Promise>.
  TemplateArgumentListInfo Args(Loc, Loc);
  Args.addArgument(TemplateArgumentLoc(
      TemplateArgument(PromiseType),
      S.Context.getTrivialTypeSourceInfo(PromiseType, Loc)));

  // Build the template-id.
  QualType CoroHandleType =
      S.CheckTemplateIdType(TemplateName(CoroHandle), Loc, Args);
  if (CoroHandleType.isNull())
    return QualType();
  if (S.RequireCompleteType(Loc, CoroHandleType,
                            diag::err_coroutine_type_missing_specialization))
    return QualType();

  return CoroHandleType;
}

static bool isValidCoroutineContext(Sema &S, SourceLocation Loc,
                                    StringRef Keyword) {
  // 'co_await' and 'co_yield' are not permitted in unevaluated operands.
  if (S.isUnevaluatedContext()) {
    S.Diag(Loc, diag::err_coroutine_unevaluated_context) << Keyword;
    return false;
  }

  // Any other usage must be within a function.
  auto *FD = dyn_cast<FunctionDecl>(S.CurContext);
  if (!FD) {
    S.Diag(Loc, isa<ObjCMethodDecl>(S.CurContext)
                    ? diag::err_coroutine_objc_method
                    : diag::err_coroutine_outside_function) << Keyword;
    return false;
  }

  // An enumeration for mapping the diagnostic type to the correct diagnostic
  // selection index.
  enum InvalidFuncDiag {
    DiagCtor = 0,
    DiagDtor,
    DiagCopyAssign,
    DiagMoveAssign,
    DiagMain,
    DiagConstexpr,
    DiagAutoRet,
    DiagVarargs,
  };
  bool Diagnosed = false;
  auto DiagInvalid = [&](InvalidFuncDiag ID) {
    S.Diag(Loc, diag::err_coroutine_invalid_func_context) << ID << Keyword;
    Diagnosed = true;
    return false;
  };

  // Diagnose when a constructor, destructor, copy/move assignment operator,
  // or the function 'main' are declared as a coroutine.
  auto *MD = dyn_cast<CXXMethodDecl>(FD);
  if (MD && isa<CXXConstructorDecl>(MD))
    return DiagInvalid(DiagCtor);
  else if (MD && isa<CXXDestructorDecl>(MD))
    return DiagInvalid(DiagDtor);
  else if (MD && MD->isCopyAssignmentOperator())
    return DiagInvalid(DiagCopyAssign);
  else if (MD && MD->isMoveAssignmentOperator())
    return DiagInvalid(DiagMoveAssign);
  else if (FD->isMain())
    return DiagInvalid(DiagMain);

  // Emit a diagnostics for each of the following conditions which is not met.
  if (FD->isConstexpr())
    DiagInvalid(DiagConstexpr);
  if (FD->getReturnType()->isUndeducedType())
    DiagInvalid(DiagAutoRet);
  if (FD->isVariadic())
    DiagInvalid(DiagVarargs);

  return !Diagnosed;
}

static ExprResult buildOperatorCoawaitLookupExpr(Sema &SemaRef, Scope *S,
                                                 SourceLocation Loc) {
  DeclarationName OpName =
      SemaRef.Context.DeclarationNames.getCXXOperatorName(OO_Coawait);
  LookupResult Operators(SemaRef, OpName, SourceLocation(),
                         Sema::LookupOperatorName);
  SemaRef.LookupName(Operators, S);

  assert(!Operators.isAmbiguous() && "Operator lookup cannot be ambiguous");
  const auto &Functions = Operators.asUnresolvedSet();
  bool IsOverloaded =
      Functions.size() > 1 ||
      (Functions.size() == 1 && isa<FunctionTemplateDecl>(*Functions.begin()));
  Expr *CoawaitOp = UnresolvedLookupExpr::Create(
      SemaRef.Context, /*NamingClass*/ nullptr, NestedNameSpecifierLoc(),
      DeclarationNameInfo(OpName, Loc), /*RequiresADL*/ true, IsOverloaded,
      Functions.begin(), Functions.end());
  assert(CoawaitOp);
  return CoawaitOp;
}

/// Build a call to 'operator co_await' if there is a suitable operator for
/// the given expression.
static ExprResult buildOperatorCoawaitCall(Sema &SemaRef, SourceLocation Loc,
                                           Expr *E,
                                           UnresolvedLookupExpr *Lookup) {
  UnresolvedSet<16> Functions;
  Functions.append(Lookup->decls_begin(), Lookup->decls_end());
  return SemaRef.CreateOverloadedUnaryOp(Loc, UO_Coawait, Functions, E);
}

static ExprResult buildOperatorCoawaitCall(Sema &SemaRef, Scope *S,
                                           SourceLocation Loc, Expr *E) {
  ExprResult R = buildOperatorCoawaitLookupExpr(SemaRef, S, Loc);
  if (R.isInvalid())
    return ExprError();
  return buildOperatorCoawaitCall(SemaRef, Loc, E,
                                  cast<UnresolvedLookupExpr>(R.get()));
}

static Expr *buildBuiltinCall(Sema &S, SourceLocation Loc, Builtin::ID Id,
                              MultiExprArg CallArgs) {
  StringRef Name = S.Context.BuiltinInfo.getName(Id);
  LookupResult R(S, &S.Context.Idents.get(Name), Loc, Sema::LookupOrdinaryName);
  S.LookupName(R, S.TUScope, /*AllowBuiltinCreation=*/true);

  auto *BuiltInDecl = R.getAsSingle<FunctionDecl>();
  assert(BuiltInDecl && "failed to find builtin declaration");

  ExprResult DeclRef =
      S.BuildDeclRefExpr(BuiltInDecl, BuiltInDecl->getType(), VK_LValue, Loc);
  assert(DeclRef.isUsable() && "Builtin reference cannot fail");

  ExprResult Call =
      S.ActOnCallExpr(/*Scope=*/nullptr, DeclRef.get(), Loc, CallArgs, Loc);

  assert(!Call.isInvalid() && "Call to builtin cannot fail!");
  return Call.get();
}

static ExprResult buildCoroutineHandle(Sema &S, QualType PromiseType,
                                       SourceLocation Loc) {
  QualType CoroHandleType = lookupCoroutineHandleType(S, PromiseType, Loc);
  if (CoroHandleType.isNull())
    return ExprError();

  DeclContext *LookupCtx = S.computeDeclContext(CoroHandleType);
  LookupResult Found(S, &S.PP.getIdentifierTable().get("from_address"), Loc,
                     Sema::LookupOrdinaryName);
  if (!S.LookupQualifiedName(Found, LookupCtx)) {
    S.Diag(Loc, diag::err_coroutine_handle_missing_member)
        << "from_address";
    return ExprError();
  }

  Expr *FramePtr =
      buildBuiltinCall(S, Loc, Builtin::BI__builtin_coro_frame, {});

  CXXScopeSpec SS;
  ExprResult FromAddr =
      S.BuildDeclarationNameExpr(SS, Found, /*NeedsADL=*/false);
  if (FromAddr.isInvalid())
    return ExprError();

  return S.ActOnCallExpr(nullptr, FromAddr.get(), Loc, FramePtr, Loc);
}

struct ReadySuspendResumeResult {
  Expr *Results[3];
  OpaqueValueExpr *OpaqueValue;
  bool IsInvalid;
};

static ExprResult buildMemberCall(Sema &S, Expr *Base, SourceLocation Loc,
                                  StringRef Name, MultiExprArg Args) {
  DeclarationNameInfo NameInfo(&S.PP.getIdentifierTable().get(Name), Loc);

  // FIXME: Fix BuildMemberReferenceExpr to take a const CXXScopeSpec&.
  CXXScopeSpec SS;
  ExprResult Result = S.BuildMemberReferenceExpr(
      Base, Base->getType(), Loc, /*IsPtr=*/false, SS,
      SourceLocation(), nullptr, NameInfo, /*TemplateArgs=*/nullptr,
      /*Scope=*/nullptr);
  if (Result.isInvalid())
    return ExprError();

  return S.ActOnCallExpr(nullptr, Result.get(), Loc, Args, Loc, nullptr);
}

/// Build calls to await_ready, await_suspend, and await_resume for a co_await
/// expression.
static ReadySuspendResumeResult buildCoawaitCalls(Sema &S, VarDecl *CoroPromise,
                                                  SourceLocation Loc, Expr *E) {
  OpaqueValueExpr *Operand = new (S.Context)
      OpaqueValueExpr(Loc, E->getType(), VK_LValue, E->getObjectKind(), E);

  // Assume invalid until we see otherwise.
  ReadySuspendResumeResult Calls = {{}, Operand, /*IsInvalid=*/true};

  ExprResult CoroHandleRes = buildCoroutineHandle(S, CoroPromise->getType(), Loc);
  if (CoroHandleRes.isInvalid())
    return Calls;
  Expr *CoroHandle = CoroHandleRes.get();

  const StringRef Funcs[] = {"await_ready", "await_suspend", "await_resume"};
  MultiExprArg Args[] = {None, CoroHandle, None};
  for (size_t I = 0, N = llvm::array_lengthof(Funcs); I != N; ++I) {
    ExprResult Result = buildMemberCall(S, Operand, Loc, Funcs[I], Args[I]);
    if (Result.isInvalid())
      return Calls;
    Calls.Results[I] = Result.get();
  }

  Calls.IsInvalid = false;
  return Calls;
}

static ExprResult buildPromiseCall(Sema &S, VarDecl *Promise,
                                   SourceLocation Loc, StringRef Name,
                                   MultiExprArg Args) {

  // Form a reference to the promise.
  ExprResult PromiseRef = S.BuildDeclRefExpr(
      Promise, Promise->getType().getNonReferenceType(), VK_LValue, Loc);
  if (PromiseRef.isInvalid())
    return ExprError();

  // Call 'yield_value', passing in E.
  return buildMemberCall(S, PromiseRef.get(), Loc, Name, Args);
}

VarDecl *Sema::buildCoroutinePromise(SourceLocation Loc) {
  assert(isa<FunctionDecl>(CurContext) && "not in a function scope");
  auto *FD = cast<FunctionDecl>(CurContext);

  QualType T =
      FD->getType()->isDependentType()
          ? Context.DependentTy
          : lookupPromiseType(*this, FD->getType()->castAs<FunctionProtoType>(),
                              Loc, FD->getLocation());
  if (T.isNull())
    return nullptr;

  auto *VD = VarDecl::Create(Context, FD, FD->getLocation(), FD->getLocation(),
                             &PP.getIdentifierTable().get("__promise"), T,
                             Context.getTrivialTypeSourceInfo(T, Loc), SC_None);
  CheckVariableDeclarationType(VD);
  if (VD->isInvalidDecl())
    return nullptr;
  ActOnUninitializedDecl(VD);
  assert(!VD->isInvalidDecl());
  return VD;
}

/// Check that this is a context in which a coroutine suspension can appear.
static FunctionScopeInfo *checkCoroutineContext(Sema &S, SourceLocation Loc,
                                                StringRef Keyword,
                                                bool IsImplicit = false) {
  if (!isValidCoroutineContext(S, Loc, Keyword))
    return nullptr;

  assert(isa<FunctionDecl>(S.CurContext) && "not in a function scope");

  auto *ScopeInfo = S.getCurFunction();
  assert(ScopeInfo && "missing function scope for function");

  if (ScopeInfo->FirstCoroutineStmtLoc.isInvalid() && !IsImplicit)
    ScopeInfo->setFirstCoroutineStmt(Loc, Keyword);

  if (ScopeInfo->CoroutinePromise)
    return ScopeInfo;

  ScopeInfo->CoroutinePromise = S.buildCoroutinePromise(Loc);
  if (!ScopeInfo->CoroutinePromise)
    return nullptr;

  return ScopeInfo;
}

static bool actOnCoroutineBodyStart(Sema &S, Scope *SC, SourceLocation KWLoc,
                                    StringRef Keyword) {
  if (!checkCoroutineContext(S, KWLoc, Keyword))
    return false;
  auto *ScopeInfo = S.getCurFunction();
  assert(ScopeInfo->CoroutinePromise);

  // If we have existing coroutine statements then we have already built
  // the initial and final suspend points.
  if (!ScopeInfo->NeedsCoroutineSuspends)
    return true;

  ScopeInfo->setNeedsCoroutineSuspends(false);

  auto *Fn = cast<FunctionDecl>(S.CurContext);
  SourceLocation Loc = Fn->getLocation();
  // Build the initial suspend point
  auto buildSuspends = [&](StringRef Name) mutable -> StmtResult {
    ExprResult Suspend =
        buildPromiseCall(S, ScopeInfo->CoroutinePromise, Loc, Name, None);
    if (Suspend.isInvalid())
      return StmtError();
    Suspend = buildOperatorCoawaitCall(S, SC, Loc, Suspend.get());
    if (Suspend.isInvalid())
      return StmtError();
    Suspend = S.BuildResolvedCoawaitExpr(Loc, Suspend.get(),
                                         /*IsImplicit*/ true);
    Suspend = S.ActOnFinishFullExpr(Suspend.get());
    if (Suspend.isInvalid()) {
      S.Diag(Loc, diag::note_coroutine_promise_call_implicitly_required)
          << ((Name == "initial_suspend") ? 0 : 1);
      S.Diag(KWLoc, diag::note_declared_coroutine_here) << Keyword;
      return StmtError();
    }
    return cast<Stmt>(Suspend.get());
  };

  StmtResult InitSuspend = buildSuspends("initial_suspend");
  if (InitSuspend.isInvalid())
    return true;

  StmtResult FinalSuspend = buildSuspends("final_suspend");
  if (FinalSuspend.isInvalid())
    return true;

  ScopeInfo->setCoroutineSuspends(InitSuspend.get(), FinalSuspend.get());

  return true;
}

ExprResult Sema::ActOnCoawaitExpr(Scope *S, SourceLocation Loc, Expr *E) {
  if (!actOnCoroutineBodyStart(*this, S, Loc, "co_await")) {
    CorrectDelayedTyposInExpr(E);
    return ExprError();
  }

  if (E->getType()->isPlaceholderType()) {
    ExprResult R = CheckPlaceholderExpr(E);
    if (R.isInvalid()) return ExprError();
    E = R.get();
  }
  ExprResult Lookup = buildOperatorCoawaitLookupExpr(*this, S, Loc);
  if (Lookup.isInvalid())
    return ExprError();
  return BuildUnresolvedCoawaitExpr(Loc, E,
                                   cast<UnresolvedLookupExpr>(Lookup.get()));
}

ExprResult Sema::BuildUnresolvedCoawaitExpr(SourceLocation Loc, Expr *E,
                                            UnresolvedLookupExpr *Lookup) {
  auto *FSI = checkCoroutineContext(*this, Loc, "co_await");
  if (!FSI)
    return ExprError();

  if (E->getType()->isPlaceholderType()) {
    ExprResult R = CheckPlaceholderExpr(E);
    if (R.isInvalid())
      return ExprError();
    E = R.get();
  }

  auto *Promise = FSI->CoroutinePromise;
  if (Promise->getType()->isDependentType()) {
    Expr *Res =
        new (Context) DependentCoawaitExpr(Loc, Context.DependentTy, E, Lookup);
    return Res;
  }

  auto *RD = Promise->getType()->getAsCXXRecordDecl();
  if (lookupMember(*this, "await_transform", RD, Loc)) {
    ExprResult R = buildPromiseCall(*this, Promise, Loc, "await_transform", E);
    if (R.isInvalid()) {
      Diag(Loc,
           diag::note_coroutine_promise_implicit_await_transform_required_here)
          << E->getSourceRange();
      return ExprError();
    }
    E = R.get();
  }
  ExprResult Awaitable = buildOperatorCoawaitCall(*this, Loc, E, Lookup);
  if (Awaitable.isInvalid())
    return ExprError();

  return BuildResolvedCoawaitExpr(Loc, Awaitable.get());
}

ExprResult Sema::BuildResolvedCoawaitExpr(SourceLocation Loc, Expr *E,
                                  bool IsImplicit) {
  auto *Coroutine = checkCoroutineContext(*this, Loc, "co_await", IsImplicit);
  if (!Coroutine)
    return ExprError();

  if (E->getType()->isPlaceholderType()) {
    ExprResult R = CheckPlaceholderExpr(E);
    if (R.isInvalid()) return ExprError();
    E = R.get();
  }

  if (E->getType()->isDependentType()) {
    Expr *Res = new (Context)
        CoawaitExpr(Loc, Context.DependentTy, E, IsImplicit);
    return Res;
  }

  // If the expression is a temporary, materialize it as an lvalue so that we
  // can use it multiple times.
  if (E->getValueKind() == VK_RValue)
    E = CreateMaterializeTemporaryExpr(E->getType(), E, true);

  // Build the await_ready, await_suspend, await_resume calls.
  ReadySuspendResumeResult RSS =
      buildCoawaitCalls(*this, Coroutine->CoroutinePromise, Loc, E);
  if (RSS.IsInvalid)
    return ExprError();

  Expr *Res =
      new (Context) CoawaitExpr(Loc, E, RSS.Results[0], RSS.Results[1],
                                RSS.Results[2], RSS.OpaqueValue, IsImplicit);

  return Res;
}

ExprResult Sema::ActOnCoyieldExpr(Scope *S, SourceLocation Loc, Expr *E) {
  if (!actOnCoroutineBodyStart(*this, S, Loc, "co_yield")) {
    CorrectDelayedTyposInExpr(E);
    return ExprError();
  }

  // Build yield_value call.
  ExprResult Awaitable = buildPromiseCall(
      *this, getCurFunction()->CoroutinePromise, Loc, "yield_value", E);
  if (Awaitable.isInvalid())
    return ExprError();

  // Build 'operator co_await' call.
  Awaitable = buildOperatorCoawaitCall(*this, S, Loc, Awaitable.get());
  if (Awaitable.isInvalid())
    return ExprError();

  return BuildCoyieldExpr(Loc, Awaitable.get());
}
ExprResult Sema::BuildCoyieldExpr(SourceLocation Loc, Expr *E) {
  auto *Coroutine = checkCoroutineContext(*this, Loc, "co_yield");
  if (!Coroutine)
    return ExprError();

  if (E->getType()->isPlaceholderType()) {
    ExprResult R = CheckPlaceholderExpr(E);
    if (R.isInvalid()) return ExprError();
    E = R.get();
  }

  if (E->getType()->isDependentType()) {
    Expr *Res = new (Context) CoyieldExpr(Loc, Context.DependentTy, E);
    return Res;
  }

  // If the expression is a temporary, materialize it as an lvalue so that we
  // can use it multiple times.
  if (E->getValueKind() == VK_RValue)
    E = CreateMaterializeTemporaryExpr(E->getType(), E, true);

  // Build the await_ready, await_suspend, await_resume calls.
  ReadySuspendResumeResult RSS =
      buildCoawaitCalls(*this, Coroutine->CoroutinePromise, Loc, E);
  if (RSS.IsInvalid)
    return ExprError();

  Expr *Res = new (Context) CoyieldExpr(Loc, E, RSS.Results[0], RSS.Results[1],
                                        RSS.Results[2], RSS.OpaqueValue);

  return Res;
}

StmtResult Sema::ActOnCoreturnStmt(Scope *S, SourceLocation Loc, Expr *E) {
  if (!actOnCoroutineBodyStart(*this, S, Loc, "co_return")) {
    CorrectDelayedTyposInExpr(E);
    return StmtError();
  }
  return BuildCoreturnStmt(Loc, E);
}

StmtResult Sema::BuildCoreturnStmt(SourceLocation Loc, Expr *E,
                                   bool IsImplicit) {
  auto *FSI = checkCoroutineContext(*this, Loc, "co_return", IsImplicit);
  if (!FSI)
    return StmtError();

  if (E && E->getType()->isPlaceholderType() &&
      !E->getType()->isSpecificPlaceholderType(BuiltinType::Overload)) {
    ExprResult R = CheckPlaceholderExpr(E);
    if (R.isInvalid()) return StmtError();
    E = R.get();
  }

  // FIXME: If the operand is a reference to a variable that's about to go out
  // of scope, we should treat the operand as an xvalue for this overload
  // resolution.
  VarDecl *Promise = FSI->CoroutinePromise;
  ExprResult PC;
  if (E && (isa<InitListExpr>(E) || !E->getType()->isVoidType())) {
    PC = buildPromiseCall(*this, Promise, Loc, "return_value", E);
  } else {
    E = MakeFullDiscardedValueExpr(E).get();
    PC = buildPromiseCall(*this, Promise, Loc, "return_void", None);
  }
  if (PC.isInvalid())
    return StmtError();

  Expr *PCE = ActOnFinishFullExpr(PC.get()).get();

  Stmt *Res = new (Context) CoreturnStmt(Loc, E, PCE, IsImplicit);
  return Res;
}

// Find an appropriate delete for the promise.
static FunctionDecl *findDeleteForPromise(Sema &S, SourceLocation Loc,
                                          QualType PromiseType) {
  FunctionDecl *OperatorDelete = nullptr;

  DeclarationName DeleteName =
      S.Context.DeclarationNames.getCXXOperatorName(OO_Delete);

  auto *PointeeRD = PromiseType->getAsCXXRecordDecl();
  assert(PointeeRD && "PromiseType must be a CxxRecordDecl type");

  if (S.FindDeallocationFunction(Loc, PointeeRD, DeleteName, OperatorDelete))
    return nullptr;

  if (!OperatorDelete) {
    // Look for a global declaration.
    const bool CanProvideSize = S.isCompleteType(Loc, PromiseType);
    const bool Overaligned = false;
    OperatorDelete = S.FindUsualDeallocationFunction(Loc, CanProvideSize,
                                                     Overaligned, DeleteName);
  }
  S.MarkFunctionReferenced(Loc, OperatorDelete);
  return OperatorDelete;
}

namespace {
class SubStmtBuilder : public CoroutineBodyStmt::CtorArgs {
  Sema &S;
  FunctionDecl &FD;
  FunctionScopeInfo &Fn;
  bool IsValid;
  SourceLocation Loc;
  QualType RetType;
  SmallVector<Stmt *, 4> ParamMovesVector;
  const bool IsPromiseDependentType;
  CXXRecordDecl *PromiseRecordDecl = nullptr;

public:
  SubStmtBuilder(Sema &S, FunctionDecl &FD, FunctionScopeInfo &Fn, Stmt *Body)
      : S(S), FD(FD), Fn(Fn), Loc(FD.getLocation()),
        IsPromiseDependentType(
            !Fn.CoroutinePromise ||
            Fn.CoroutinePromise->getType()->isDependentType()) {
    this->Body = Body;
    if (!IsPromiseDependentType) {
      PromiseRecordDecl = Fn.CoroutinePromise->getType()->getAsCXXRecordDecl();
      assert(PromiseRecordDecl && "Type should have already been checked");
    }
    this->IsValid = makePromiseStmt() && makeInitialAndFinalSuspend() &&
                    makeOnException() && makeOnFallthrough() &&
                    makeNewAndDeleteExpr() && makeReturnObject() &&
                    makeParamMoves();
  }

  bool isInvalid() const { return !this->IsValid; }

  bool makePromiseStmt();
  bool makeInitialAndFinalSuspend();
  bool makeNewAndDeleteExpr();
  bool makeOnFallthrough();
  bool makeOnException();
  bool makeReturnObject();
  bool makeParamMoves();
};
}

void Sema::CheckCompletedCoroutineBody(FunctionDecl *FD, Stmt *&Body) {
  FunctionScopeInfo *Fn = getCurFunction();
  assert(Fn && Fn->CoroutinePromise && "not a coroutine");

  if (!Body) {
    assert(FD->isInvalidDecl() &&
           "a null body is only allowed for invalid declarations");
    return;
  }

  if (isa<CoroutineBodyStmt>(Body)) {
    // FIXME(EricWF): Nothing todo. the body is already a transformed coroutine
    // body statement.
    return;
  }

  // Coroutines [stmt.return]p1:
  //   A return statement shall not appear in a coroutine.
  if (Fn->FirstReturnLoc.isValid()) {
    assert(Fn->FirstCoroutineStmtLoc.isValid() &&
                   "first coroutine location not set");
    Diag(Fn->FirstReturnLoc, diag::err_return_in_coroutine);
    Diag(Fn->FirstCoroutineStmtLoc, diag::note_declared_coroutine_here)
            << Fn->getFirstCoroutineStmtKeyword();
  }
  SubStmtBuilder Builder(*this, *FD, *Fn, Body);
  if (Builder.isInvalid())
    return FD->setInvalidDecl();

  // Build body for the coroutine wrapper statement.
  Body = CoroutineBodyStmt::Create(Context, Builder);
}

bool SubStmtBuilder::makePromiseStmt() {
  // Form a declaration statement for the promise declaration, so that AST
  // visitors can more easily find it.
  StmtResult PromiseStmt =
      S.ActOnDeclStmt(S.ConvertDeclToDeclGroup(Fn.CoroutinePromise), Loc, Loc);
  if (PromiseStmt.isInvalid())
    return false;

  this->Promise = PromiseStmt.get();
  return true;
}

bool SubStmtBuilder::makeInitialAndFinalSuspend() {
  if (Fn.hasInvalidCoroutineSuspends())
    return false;
  this->InitialSuspend = cast<Expr>(Fn.CoroutineSuspends.first);
  this->FinalSuspend = cast<Expr>(Fn.CoroutineSuspends.second);
  return true;
}

bool SubStmtBuilder::makeNewAndDeleteExpr() {
  // Form and check allocation and deallocation calls.
  QualType PromiseType = Fn.CoroutinePromise->getType();
  if (PromiseType->isDependentType())
    return true;

  if (S.RequireCompleteType(Loc, PromiseType, diag::err_incomplete_type))
    return false;

  // FIXME: Add support for get_return_object_on_allocation failure.
  // FIXME: Add support for stateful allocators.

  FunctionDecl *OperatorNew = nullptr;
  FunctionDecl *OperatorDelete = nullptr;
  FunctionDecl *UnusedResult = nullptr;
  bool PassAlignment = false;

  S.FindAllocationFunctions(Loc, SourceRange(),
                            /*UseGlobal*/ false, PromiseType,
                            /*isArray*/ false, PassAlignment,
                            /*PlacementArgs*/ None, OperatorNew, UnusedResult);

  OperatorDelete = findDeleteForPromise(S, Loc, PromiseType);

  if (!OperatorDelete || !OperatorNew)
    return false;

  Expr *FramePtr =
      buildBuiltinCall(S, Loc, Builtin::BI__builtin_coro_frame, {});

  Expr *FrameSize =
      buildBuiltinCall(S, Loc, Builtin::BI__builtin_coro_size, {});

  // Make new call.

  ExprResult NewRef =
      S.BuildDeclRefExpr(OperatorNew, OperatorNew->getType(), VK_LValue, Loc);
  if (NewRef.isInvalid())
    return false;

  ExprResult NewExpr =
      S.ActOnCallExpr(S.getCurScope(), NewRef.get(), Loc, FrameSize, Loc);
  if (NewExpr.isInvalid())
    return false;

  // Make delete call.

  QualType OpDeleteQualType = OperatorDelete->getType();

  ExprResult DeleteRef =
      S.BuildDeclRefExpr(OperatorDelete, OpDeleteQualType, VK_LValue, Loc);
  if (DeleteRef.isInvalid())
    return false;

  Expr *CoroFree =
      buildBuiltinCall(S, Loc, Builtin::BI__builtin_coro_free, {FramePtr});

  SmallVector<Expr *, 2> DeleteArgs{CoroFree};

  // Check if we need to pass the size.
  const auto *OpDeleteType =
      OpDeleteQualType.getTypePtr()->getAs<FunctionProtoType>();
  if (OpDeleteType->getNumParams() > 1)
    DeleteArgs.push_back(FrameSize);

  ExprResult DeleteExpr =
      S.ActOnCallExpr(S.getCurScope(), DeleteRef.get(), Loc, DeleteArgs, Loc);
  if (DeleteExpr.isInvalid())
    return false;

  this->Allocate = NewExpr.get();
  this->Deallocate = DeleteExpr.get();

  return true;
}

bool SubStmtBuilder::makeOnFallthrough() {
  if (!PromiseRecordDecl)
    return true;

  // [dcl.fct.def.coroutine]/4
  // The unqualified-ids 'return_void' and 'return_value' are looked up in
  // the scope of class P. If both are found, the program is ill-formed.
  const bool HasRVoid = lookupMember(S, "return_void", PromiseRecordDecl, Loc);
  const bool HasRValue = lookupMember(S, "return_value", PromiseRecordDecl, Loc);

  StmtResult Fallthrough;
  if (HasRVoid && HasRValue) {
    // FIXME Improve this diagnostic
    S.Diag(FD.getLocation(), diag::err_coroutine_promise_return_ill_formed)
        << PromiseRecordDecl;
    return false;
  } else if (HasRVoid) {
    // If the unqualified-id return_void is found, flowing off the end of a
    // coroutine is equivalent to a co_return with no operand. Otherwise,
    // flowing off the end of a coroutine results in undefined behavior.
    Fallthrough = S.BuildCoreturnStmt(FD.getLocation(), nullptr,
                                      /*IsImplicit*/false);
    Fallthrough = S.ActOnFinishFullStmt(Fallthrough.get());
    if (Fallthrough.isInvalid())
      return false;
  }

  this->OnFallthrough = Fallthrough.get();
  return true;
}

bool SubStmtBuilder::makeOnException() {
  // Try to form 'p.unhandled_exception();'

  if (!PromiseRecordDecl)
    return true;

  const bool RequireUnhandledException = S.getLangOpts().CXXExceptions;

  if (!lookupMember(S, "unhandled_exception", PromiseRecordDecl, Loc)) {
    auto DiagID =
        RequireUnhandledException
            ? diag::err_coroutine_promise_unhandled_exception_required
            : diag::
                  warn_coroutine_promise_unhandled_exception_required_with_exceptions;
    S.Diag(Loc, DiagID) << PromiseRecordDecl;
    return !RequireUnhandledException;
  }

  // If exceptions are disabled, don't try to build OnException.
  if (!S.getLangOpts().CXXExceptions)
    return true;

  ExprResult UnhandledException = buildPromiseCall(S, Fn.CoroutinePromise, Loc,
                                                   "unhandled_exception", None);
  UnhandledException = S.ActOnFinishFullExpr(UnhandledException.get(), Loc);
  if (UnhandledException.isInvalid())
    return false;

  this->OnException = UnhandledException.get();
  return true;
}

bool SubStmtBuilder::makeReturnObject() {

  // Build implicit 'p.get_return_object()' expression and form initialization
  // of return type from it.
  ExprResult ReturnObject =
      buildPromiseCall(S, Fn.CoroutinePromise, Loc, "get_return_object", None);
  if (ReturnObject.isInvalid())
    return false;
  QualType RetType = FD.getReturnType();
  if (!RetType->isDependentType()) {
    InitializedEntity Entity =
        InitializedEntity::InitializeResult(Loc, RetType, false);
    ReturnObject = S.PerformMoveOrCopyInitialization(Entity, nullptr, RetType,
                                                   ReturnObject.get());
    if (ReturnObject.isInvalid())
      return false;
  }
  ReturnObject = S.ActOnFinishFullExpr(ReturnObject.get(), Loc);
  if (ReturnObject.isInvalid())
    return false;

  this->ReturnValue = ReturnObject.get();
  return true;
}

bool SubStmtBuilder::makeParamMoves() {
  // FIXME: Perform move-initialization of parameters into frame-local copies.
  return true;
}

StmtResult Sema::BuildCoroutineBodyStmt(CoroutineBodyStmt::CtorArgs Args) {
  CoroutineBodyStmt *Res = CoroutineBodyStmt::Create(Context, Args);
  if (!Res)
    return StmtError();
  return Res;
}
