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

/// Look up the std::coroutine_traits<...>::promise_type for the given
/// function type.
static QualType lookupPromiseType(Sema &S, const FunctionProtoType *FnType,
                                  SourceLocation Loc) {
  // FIXME: Cache std::coroutine_traits once we've found it.
  NamespaceDecl *StdExp = S.lookupStdExperimentalNamespace();
  if (!StdExp) {
    S.Diag(Loc, diag::err_implied_std_coroutine_traits_not_found);
    return QualType();
  }

  LookupResult Result(S, &S.PP.getIdentifierTable().get("coroutine_traits"),
                      Loc, Sema::LookupOrdinaryName);
  if (!S.LookupQualifiedName(Result, StdExp)) {
    S.Diag(Loc, diag::err_implied_std_coroutine_traits_not_found);
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
  TemplateArgumentListInfo Args(Loc, Loc);
  Args.addArgument(TemplateArgumentLoc(
      TemplateArgument(FnType->getReturnType()),
      S.Context.getTrivialTypeSourceInfo(FnType->getReturnType(), Loc)));
  // FIXME: If the function is a non-static member function, add the type
  // of the implicit object parameter before the formal parameters.
  for (QualType T : FnType->getParamTypes())
    Args.addArgument(TemplateArgumentLoc(
        TemplateArgument(T), S.Context.getTrivialTypeSourceInfo(T, Loc)));

  // Build the template-id.
  QualType CoroTrait =
      S.CheckTemplateIdType(TemplateName(CoroTraits), Loc, Args);
  if (CoroTrait.isNull())
    return QualType();
  if (S.RequireCompleteType(Loc, CoroTrait,
                            diag::err_coroutine_traits_missing_specialization))
    return QualType();

  CXXRecordDecl *RD = CoroTrait->getAsCXXRecordDecl();
  assert(RD && "specialization of class template is not a class?");

  // Look up the ::promise_type member.
  LookupResult R(S, &S.PP.getIdentifierTable().get("promise_type"), Loc,
                 Sema::LookupOrdinaryName);
  S.LookupQualifiedName(R, RD);
  auto *Promise = R.getAsSingle<TypeDecl>();
  if (!Promise) {
    S.Diag(Loc, diag::err_implied_std_coroutine_traits_promise_type_not_found)
        << RD;
    return QualType();
  }

  // The promise type is required to be a class type.
  QualType PromiseType = S.Context.getTypeDeclType(Promise);
  if (!PromiseType->getAsCXXRecordDecl()) {
    // Use the fully-qualified name of the type.
    auto *NNS = NestedNameSpecifier::Create(S.Context, nullptr, StdExp);
    NNS = NestedNameSpecifier::Create(S.Context, NNS, false,
                                      CoroTrait.getTypePtr());
    PromiseType = S.Context.getElaboratedType(ETK_None, NNS, PromiseType);

    S.Diag(Loc, diag::err_implied_std_coroutine_traits_promise_type_not_class)
        << PromiseType;
    return QualType();
  }

  return PromiseType;
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

/// Check that this is a context in which a coroutine suspension can appear.
static FunctionScopeInfo *checkCoroutineContext(Sema &S, SourceLocation Loc,
                                                StringRef Keyword) {
  if (!isValidCoroutineContext(S, Loc, Keyword))
    return nullptr;

  assert(isa<FunctionDecl>(S.CurContext) && "not in a function scope");
  auto *FD = cast<FunctionDecl>(S.CurContext);
  auto *ScopeInfo = S.getCurFunction();
  assert(ScopeInfo && "missing function scope for function");

  // If we don't have a promise variable, build one now.
  if (!ScopeInfo->CoroutinePromise) {
    QualType T = FD->getType()->isDependentType()
                     ? S.Context.DependentTy
                     : lookupPromiseType(
                           S, FD->getType()->castAs<FunctionProtoType>(), Loc);
    if (T.isNull())
      return nullptr;

    // Create and default-initialize the promise.
    ScopeInfo->CoroutinePromise =
        VarDecl::Create(S.Context, FD, FD->getLocation(), FD->getLocation(),
                        &S.PP.getIdentifierTable().get("__promise"), T,
                        S.Context.getTrivialTypeSourceInfo(T, Loc), SC_None);
    S.CheckVariableDeclarationType(ScopeInfo->CoroutinePromise);
    if (!ScopeInfo->CoroutinePromise->isInvalidDecl())
      S.ActOnUninitializedDecl(ScopeInfo->CoroutinePromise);
  }

  return ScopeInfo;
}

static Expr *buildBuiltinCall(Sema &S, SourceLocation Loc, Builtin::ID Id,
                              MutableArrayRef<Expr *> CallArgs) {
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

/// Build a call to 'operator co_await' if there is a suitable operator for
/// the given expression.
static ExprResult buildOperatorCoawaitCall(Sema &SemaRef, Scope *S,
                                           SourceLocation Loc, Expr *E) {
  UnresolvedSet<16> Functions;
  SemaRef.LookupOverloadedOperatorName(OO_Coawait, S, E->getType(), QualType(),
                                       Functions);
  return SemaRef.CreateOverloadedUnaryOp(Loc, UO_Coawait, Functions, E);
}

struct ReadySuspendResumeResult {
  bool IsInvalid;
  Expr *Results[3];
};

static ExprResult buildMemberCall(Sema &S, Expr *Base, SourceLocation Loc,
                                  StringRef Name,
                                  MutableArrayRef<Expr *> Args) {
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
static ReadySuspendResumeResult buildCoawaitCalls(Sema &S, SourceLocation Loc,
                                                  Expr *E) {
  // Assume invalid until we see otherwise.
  ReadySuspendResumeResult Calls = {true, {}};

  const StringRef Funcs[] = {"await_ready", "await_suspend", "await_resume"};
  for (size_t I = 0, N = llvm::array_lengthof(Funcs); I != N; ++I) {
    Expr *Operand = new (S.Context) OpaqueValueExpr(
      Loc, E->getType(), VK_LValue, E->getObjectKind(), E);

    // FIXME: Pass coroutine handle to await_suspend.
    ExprResult Result = buildMemberCall(S, Operand, Loc, Funcs[I], None);
    if (Result.isInvalid())
      return Calls;
    Calls.Results[I] = Result.get();
  }

  Calls.IsInvalid = false;
  return Calls;
}

ExprResult Sema::ActOnCoawaitExpr(Scope *S, SourceLocation Loc, Expr *E) {
  auto *Coroutine = checkCoroutineContext(*this, Loc, "co_await");
  if (!Coroutine) {
    CorrectDelayedTyposInExpr(E);
    return ExprError();
  }
  if (E->getType()->isPlaceholderType()) {
    ExprResult R = CheckPlaceholderExpr(E);
    if (R.isInvalid()) return ExprError();
    E = R.get();
  }

  ExprResult Awaitable = buildOperatorCoawaitCall(*this, S, Loc, E);
  if (Awaitable.isInvalid())
    return ExprError();

  return BuildCoawaitExpr(Loc, Awaitable.get());
}
ExprResult Sema::BuildCoawaitExpr(SourceLocation Loc, Expr *E) {
  auto *Coroutine = checkCoroutineContext(*this, Loc, "co_await");
  if (!Coroutine)
    return ExprError();

  if (E->getType()->isPlaceholderType()) {
    ExprResult R = CheckPlaceholderExpr(E);
    if (R.isInvalid()) return ExprError();
    E = R.get();
  }

  if (E->getType()->isDependentType()) {
    Expr *Res = new (Context) CoawaitExpr(Loc, Context.DependentTy, E);
    Coroutine->CoroutineStmts.push_back(Res);
    return Res;
  }

  // If the expression is a temporary, materialize it as an lvalue so that we
  // can use it multiple times.
  if (E->getValueKind() == VK_RValue)
    E = CreateMaterializeTemporaryExpr(E->getType(), E, true);

  // Build the await_ready, await_suspend, await_resume calls.
  ReadySuspendResumeResult RSS = buildCoawaitCalls(*this, Loc, E);
  if (RSS.IsInvalid)
    return ExprError();

  Expr *Res = new (Context) CoawaitExpr(Loc, E, RSS.Results[0], RSS.Results[1],
                                        RSS.Results[2]);
  Coroutine->CoroutineStmts.push_back(Res);
  return Res;
}

static ExprResult buildPromiseCall(Sema &S, FunctionScopeInfo *Coroutine,
                                   SourceLocation Loc, StringRef Name,
                                   MutableArrayRef<Expr *> Args) {
  assert(Coroutine->CoroutinePromise && "no promise for coroutine");

  // Form a reference to the promise.
  auto *Promise = Coroutine->CoroutinePromise;
  ExprResult PromiseRef = S.BuildDeclRefExpr(
      Promise, Promise->getType().getNonReferenceType(), VK_LValue, Loc);
  if (PromiseRef.isInvalid())
    return ExprError();

  // Call 'yield_value', passing in E.
  return buildMemberCall(S, PromiseRef.get(), Loc, Name, Args);
}

ExprResult Sema::ActOnCoyieldExpr(Scope *S, SourceLocation Loc, Expr *E) {
  auto *Coroutine = checkCoroutineContext(*this, Loc, "co_yield");
  if (!Coroutine) {
    CorrectDelayedTyposInExpr(E);
    return ExprError();
  }

  // Build yield_value call.
  ExprResult Awaitable =
      buildPromiseCall(*this, Coroutine, Loc, "yield_value", E);
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
    Coroutine->CoroutineStmts.push_back(Res);
    return Res;
  }

  // If the expression is a temporary, materialize it as an lvalue so that we
  // can use it multiple times.
  if (E->getValueKind() == VK_RValue)
    E = CreateMaterializeTemporaryExpr(E->getType(), E, true);

  // Build the await_ready, await_suspend, await_resume calls.
  ReadySuspendResumeResult RSS = buildCoawaitCalls(*this, Loc, E);
  if (RSS.IsInvalid)
    return ExprError();

  Expr *Res = new (Context) CoyieldExpr(Loc, E, RSS.Results[0], RSS.Results[1],
                                        RSS.Results[2]);
  Coroutine->CoroutineStmts.push_back(Res);
  return Res;
}

StmtResult Sema::ActOnCoreturnStmt(SourceLocation Loc, Expr *E) {
  auto *Coroutine = checkCoroutineContext(*this, Loc, "co_return");
  if (!Coroutine) {
    CorrectDelayedTyposInExpr(E);
    return StmtError();
  }
  return BuildCoreturnStmt(Loc, E);
}

StmtResult Sema::BuildCoreturnStmt(SourceLocation Loc, Expr *E) {
  auto *Coroutine = checkCoroutineContext(*this, Loc, "co_return");
  if (!Coroutine)
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
  ExprResult PC;
  if (E && (isa<InitListExpr>(E) || !E->getType()->isVoidType())) {
    PC = buildPromiseCall(*this, Coroutine, Loc, "return_value", E);
  } else {
    E = MakeFullDiscardedValueExpr(E).get();
    PC = buildPromiseCall(*this, Coroutine, Loc, "return_void", None);
  }
  if (PC.isInvalid())
    return StmtError();

  Expr *PCE = ActOnFinishFullExpr(PC.get()).get();

  Stmt *Res = new (Context) CoreturnStmt(Loc, E, PCE);
  Coroutine->CoroutineStmts.push_back(Res);
  return Res;
}

static ExprResult buildStdCurrentExceptionCall(Sema &S, SourceLocation Loc) {
  NamespaceDecl *Std = S.getStdNamespace();
  if (!Std) {
    S.Diag(Loc, diag::err_implied_std_current_exception_not_found);
    return ExprError();
  }
  LookupResult Result(S, &S.PP.getIdentifierTable().get("current_exception"),
                      Loc, Sema::LookupOrdinaryName);
  if (!S.LookupQualifiedName(Result, Std)) {
    S.Diag(Loc, diag::err_implied_std_current_exception_not_found);
    return ExprError();
  }

  // FIXME The STL is free to provide more than one overload.
  FunctionDecl *FD = Result.getAsSingle<FunctionDecl>();
  if (!FD) {
    S.Diag(Loc, diag::err_malformed_std_current_exception);
    return ExprError();
  }
  ExprResult Res = S.BuildDeclRefExpr(FD, FD->getType(), VK_LValue, Loc);
  Res = S.ActOnCallExpr(/*Scope*/ nullptr, Res.get(), Loc, None, Loc);
  if (Res.isInvalid()) {
    S.Diag(Loc, diag::err_malformed_std_current_exception);
    return ExprError();
  }
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

// Builds allocation and deallocation for the coroutine. Returns false on
// failure.
static bool buildAllocationAndDeallocation(Sema &S, SourceLocation Loc,
                                           FunctionScopeInfo *Fn,
                                           Expr *&Allocation,
                                           Expr *&Deallocation) {
  TypeSourceInfo *TInfo = Fn->CoroutinePromise->getTypeSourceInfo();
  QualType PromiseType = TInfo->getType();
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

  Allocation = NewExpr.get();

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

  Deallocation = DeleteExpr.get();

  return true;
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
    this->IsValid = makePromiseStmt() && makeInitialSuspend() &&
                    makeFinalSuspend() && makeOnException() &&
                    makeOnFallthrough() && makeNewAndDeleteExpr() &&
                    makeReturnObject() && makeParamMoves();
  }

  bool isInvalid() const { return !this->IsValid; }

  bool makePromiseStmt();
  bool makeInitialSuspend();
  bool makeFinalSuspend();
  bool makeNewAndDeleteExpr();
  bool makeOnFallthrough();
  bool makeOnException();
  bool makeReturnObject();
  bool makeParamMoves();
};
}

void Sema::CheckCompletedCoroutineBody(FunctionDecl *FD, Stmt *&Body) {
  FunctionScopeInfo *Fn = getCurFunction();
  assert(Fn && !Fn->CoroutineStmts.empty() && "not a coroutine");

  // Coroutines [stmt.return]p1:
  //   A return statement shall not appear in a coroutine.
  if (Fn->FirstReturnLoc.isValid()) {
    Diag(Fn->FirstReturnLoc, diag::err_return_in_coroutine);
    auto *First = Fn->CoroutineStmts[0];
    Diag(First->getLocStart(), diag::note_declared_coroutine_here)
        << (isa<CoawaitExpr>(First) ? 0 :
            isa<CoyieldExpr>(First) ? 1 : 2);
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

bool SubStmtBuilder::makeInitialSuspend() {
  // Form and check implicit 'co_await p.initial_suspend();' statement.
  ExprResult InitialSuspend =
      buildPromiseCall(S, &Fn, Loc, "initial_suspend", None);
  // FIXME: Support operator co_await here.
  if (!InitialSuspend.isInvalid())
    InitialSuspend = S.BuildCoawaitExpr(Loc, InitialSuspend.get());
  InitialSuspend = S.ActOnFinishFullExpr(InitialSuspend.get());
  if (InitialSuspend.isInvalid())
    return false;

  this->InitialSuspend = InitialSuspend.get();
  return true;
}

bool SubStmtBuilder::makeFinalSuspend() {
  // Form and check implicit 'co_await p.final_suspend();' statement.
  ExprResult FinalSuspend =
      buildPromiseCall(S, &Fn, Loc, "final_suspend", None);
  // FIXME: Support operator co_await here.
  if (!FinalSuspend.isInvalid())
    FinalSuspend = S.BuildCoawaitExpr(Loc, FinalSuspend.get());
  FinalSuspend = S.ActOnFinishFullExpr(FinalSuspend.get());
  if (FinalSuspend.isInvalid())
    return false;

  this->FinalSuspend = FinalSuspend.get();
  return true;
}

bool SubStmtBuilder::makeNewAndDeleteExpr() {
  // Form and check allocation and deallocation calls.
  return buildAllocationAndDeallocation(S, Loc, &Fn, this->Allocate,
                                        this->Deallocate);
}

bool SubStmtBuilder::makeOnFallthrough() {
  if (!PromiseRecordDecl)
    return true;

  // [dcl.fct.def.coroutine]/4
  // The unqualified-ids 'return_void' and 'return_value' are looked up in
  // the scope of class P. If both are found, the program is ill-formed.
  DeclarationName RVoidDN = S.PP.getIdentifierInfo("return_void");
  LookupResult RVoidResult(S, RVoidDN, Loc, Sema::LookupMemberName);
  const bool HasRVoid = S.LookupQualifiedName(RVoidResult, PromiseRecordDecl);

  DeclarationName RValueDN = S.PP.getIdentifierInfo("return_value");
  LookupResult RValueResult(S, RValueDN, Loc, Sema::LookupMemberName);
  const bool HasRValue = S.LookupQualifiedName(RValueResult, PromiseRecordDecl);

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
    Fallthrough = S.BuildCoreturnStmt(FD.getLocation(), nullptr);
    Fallthrough = S.ActOnFinishFullStmt(Fallthrough.get());
    if (Fallthrough.isInvalid())
      return false;
  }

  this->OnFallthrough = Fallthrough.get();
  return true;
}

bool SubStmtBuilder::makeOnException() {
  // Try to form 'p.set_exception(std::current_exception());' to handle
  // uncaught exceptions.
  // TODO: Post WG21 Issaquah 2016 renamed set_exception to unhandled_exception
  // TODO: and dropped exception_ptr parameter. Make it so.

  if (!PromiseRecordDecl)
    return true;

  // If exceptions are disabled, don't try to build OnException.
  if (!S.getLangOpts().CXXExceptions)
    return true;

  ExprResult SetException;

  // [dcl.fct.def.coroutine]/3
  // The unqualified-id set_exception is found in the scope of P by class
  // member access lookup (3.4.5).
  DeclarationName SetExDN = S.PP.getIdentifierInfo("set_exception");
  LookupResult SetExResult(S, SetExDN, Loc, Sema::LookupMemberName);
  if (S.LookupQualifiedName(SetExResult, PromiseRecordDecl)) {
    // Form the call 'p.set_exception(std::current_exception())'
    SetException = buildStdCurrentExceptionCall(S, Loc);
    if (SetException.isInvalid())
      return false;
    Expr *E = SetException.get();
    SetException = buildPromiseCall(S, &Fn, Loc, "set_exception", E);
    SetException = S.ActOnFinishFullExpr(SetException.get(), Loc);
    if (SetException.isInvalid())
      return false;
  }

  this->OnException = SetException.get();
  return true;
}

bool SubStmtBuilder::makeReturnObject() {

  // Build implicit 'p.get_return_object()' expression and form initialization
  // of return type from it.
  ExprResult ReturnObject =
      buildPromiseCall(S, &Fn, Loc, "get_return_object", None);
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
