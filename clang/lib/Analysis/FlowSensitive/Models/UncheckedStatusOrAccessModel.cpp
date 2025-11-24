//===- UncheckedStatusOrAccessModel.cpp -----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Models/UncheckedStatusOrAccessModel.h"

#include <cassert>
#include <utility>

#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/TypeBase.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/ASTMatchers/ASTMatchersInternal.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/CFGMatchSwitch.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/MatchSwitch.h"
#include "clang/Analysis/FlowSensitive/RecordOps.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Basic/LLVM.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringMap.h"

namespace clang::dataflow::statusor_model {
namespace {

using ::clang::ast_matchers::MatchFinder;
using ::clang::ast_matchers::StatementMatcher;

} // namespace

static bool namespaceEquals(const NamespaceDecl *NS,
                            clang::ArrayRef<clang::StringRef> NamespaceNames) {
  while (!NamespaceNames.empty() && NS) {
    if (NS->getName() != NamespaceNames.consume_back())
      return false;
    NS = dyn_cast_or_null<NamespaceDecl>(NS->getParent());
  }
  return NamespaceNames.empty() && !NS;
}

// TODO: move this to a proper place to share with the rest of clang
static bool isTypeNamed(QualType Type, clang::ArrayRef<clang::StringRef> NS,
                        StringRef Name) {
  if (Type.isNull())
    return false;
  if (auto *RD = Type->getAsRecordDecl())
    if (RD->getName() == Name)
      if (const auto *N = dyn_cast_or_null<NamespaceDecl>(RD->getDeclContext()))
        return namespaceEquals(N, NS);
  return false;
}

static bool isStatusOrOperatorBaseType(QualType Type) {
  return isTypeNamed(Type, {"absl", "internal_statusor"}, "OperatorBase");
}

static bool isSafeUnwrap(RecordStorageLocation *StatusOrLoc,
                         const Environment &Env) {
  if (!StatusOrLoc)
    return false;
  auto &StatusLoc = locForStatus(*StatusOrLoc);
  auto *OkVal = Env.get<BoolValue>(locForOk(StatusLoc));
  return OkVal != nullptr && Env.proves(OkVal->formula());
}

static ClassTemplateSpecializationDecl *
getStatusOrBaseClass(const QualType &Ty) {
  auto *RD = Ty->getAsCXXRecordDecl();
  if (RD == nullptr)
    return nullptr;
  if (isStatusOrType(Ty) ||
      // In case we are analyzing code under OperatorBase itself that uses
      // operator* (e.g. to implement operator->).
      isStatusOrOperatorBaseType(Ty))
    return cast<ClassTemplateSpecializationDecl>(RD);
  if (!RD->hasDefinition())
    return nullptr;
  for (const auto &Base : RD->bases())
    if (auto *QT = getStatusOrBaseClass(Base.getType()))
      return QT;
  return nullptr;
}

static QualType getStatusOrValueType(ClassTemplateSpecializationDecl *TRD) {
  return TRD->getTemplateArgs().get(0).getAsType();
}

static auto ofClassStatus() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return ofClass(hasName("::absl::Status"));
}

static auto isStatusMemberCallWithName(llvm::StringRef member_name) {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return cxxMemberCallExpr(
      on(expr(unless(cxxThisExpr()))),
      callee(cxxMethodDecl(hasName(member_name), ofClassStatus())));
}

static auto isStatusOrMemberCallWithName(llvm::StringRef member_name) {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return cxxMemberCallExpr(
      on(expr(unless(cxxThisExpr()))),
      callee(cxxMethodDecl(
          hasName(member_name),
          ofClass(anyOf(statusOrClass(), statusOrOperatorBaseClass())))));
}

static auto isStatusOrOperatorCallWithName(llvm::StringRef operator_name) {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return cxxOperatorCallExpr(
      hasOverloadedOperatorName(operator_name),
      callee(cxxMethodDecl(
          ofClass(anyOf(statusOrClass(), statusOrOperatorBaseClass())))));
}

static auto valueCall() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return anyOf(isStatusOrMemberCallWithName("value"),
               isStatusOrMemberCallWithName("ValueOrDie"));
}

static auto valueOperatorCall() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return expr(anyOf(isStatusOrOperatorCallWithName("*"),
                    isStatusOrOperatorCallWithName("->")));
}

static clang::ast_matchers::TypeMatcher statusType() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return hasCanonicalType(qualType(hasDeclaration(statusClass())));
}

static auto isComparisonOperatorCall(llvm::StringRef operator_name) {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return cxxOperatorCallExpr(
      hasOverloadedOperatorName(operator_name), argumentCountIs(2),
      hasArgument(0, anyOf(hasType(statusType()), hasType(statusOrType()))),
      hasArgument(1, anyOf(hasType(statusType()), hasType(statusOrType()))));
}

static auto isOkStatusCall() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return callExpr(callee(functionDecl(hasName("::absl::OkStatus"))));
}

static auto isNotOkStatusCall() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return callExpr(callee(functionDecl(hasAnyName(
      "::absl::AbortedError", "::absl::AlreadyExistsError",
      "::absl::CancelledError", "::absl::DataLossError",
      "::absl::DeadlineExceededError", "::absl::FailedPreconditionError",
      "::absl::InternalError", "::absl::InvalidArgumentError",
      "::absl::NotFoundError", "::absl::OutOfRangeError",
      "::absl::PermissionDeniedError", "::absl::ResourceExhaustedError",
      "::absl::UnauthenticatedError", "::absl::UnavailableError",
      "::absl::UnimplementedError", "::absl::UnknownError"))));
}

static auto isPointerComparisonOperatorCall(std::string operator_name) {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return binaryOperator(hasOperatorName(operator_name),
                        hasLHS(hasType(hasCanonicalType(pointerType(
                            pointee(anyOf(statusOrType(), statusType())))))),
                        hasRHS(hasType(hasCanonicalType(pointerType(
                            pointee(anyOf(statusOrType(), statusType())))))));
}

// The nullPointerConstant in the two matchers below is to support
// absl::StatusOr<void*> X = nullptr.
// nullptr does not match the bound type.
// TODO: be less restrictive around convertible types in general.
static auto isStatusOrValueAssignmentCall() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return cxxOperatorCallExpr(
      hasOverloadedOperatorName("="),
      callee(cxxMethodDecl(ofClass(statusOrClass()))),
      hasArgument(1, anyOf(hasType(hasUnqualifiedDesugaredType(
                               type(equalsBoundNode("T")))),
                           nullPointerConstant())));
}

static auto isStatusOrValueConstructor() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return cxxConstructExpr(
      hasType(statusOrType()),
      hasArgument(0,
                  anyOf(hasType(hasCanonicalType(type(equalsBoundNode("T")))),
                        nullPointerConstant(),
                        hasType(namedDecl(hasAnyName("absl::in_place_t",
                                                     "std::in_place_t"))))));
}

static auto isStatusOrConstructor() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return cxxConstructExpr(hasType(statusOrType()));
}

static auto isStatusConstructor() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return cxxConstructExpr(hasType(statusType()));
}

static auto
buildDiagnoseMatchSwitch(const UncheckedStatusOrAccessModelOptions &Options) {
  return CFGMatchSwitchBuilder<const Environment,
                               llvm::SmallVector<SourceLocation>>()
      // StatusOr::value, StatusOr::ValueOrDie
      .CaseOfCFGStmt<CXXMemberCallExpr>(
          valueCall(),
          [](const CXXMemberCallExpr *E,
             const ast_matchers::MatchFinder::MatchResult &,
             const Environment &Env) {
            if (!isSafeUnwrap(getImplicitObjectLocation(*E, Env), Env))
              return llvm::SmallVector<SourceLocation>({E->getExprLoc()});
            return llvm::SmallVector<SourceLocation>();
          })

      // StatusOr::operator*, StatusOr::operator->
      .CaseOfCFGStmt<CXXOperatorCallExpr>(
          valueOperatorCall(),
          [](const CXXOperatorCallExpr *E,
             const ast_matchers::MatchFinder::MatchResult &,
             const Environment &Env) {
            RecordStorageLocation *StatusOrLoc =
                Env.get<RecordStorageLocation>(*E->getArg(0));
            if (!isSafeUnwrap(StatusOrLoc, Env))
              return llvm::SmallVector<SourceLocation>({E->getOperatorLoc()});
            return llvm::SmallVector<SourceLocation>();
          })
      .Build();
}

UncheckedStatusOrAccessDiagnoser::UncheckedStatusOrAccessDiagnoser(
    UncheckedStatusOrAccessModelOptions Options)
    : DiagnoseMatchSwitch(buildDiagnoseMatchSwitch(Options)) {}

llvm::SmallVector<SourceLocation> UncheckedStatusOrAccessDiagnoser::operator()(
    const CFGElement &Elt, ASTContext &Ctx,
    const TransferStateForDiagnostics<UncheckedStatusOrAccessModel::Lattice>
        &State) {
  return DiagnoseMatchSwitch(Elt, Ctx, State.Env);
}

BoolValue &initializeStatus(RecordStorageLocation &StatusLoc,
                            Environment &Env) {
  auto &OkVal = Env.makeAtomicBoolValue();
  Env.setValue(locForOk(StatusLoc), OkVal);
  return OkVal;
}

BoolValue &initializeStatusOr(RecordStorageLocation &StatusOrLoc,
                              Environment &Env) {
  return initializeStatus(locForStatus(StatusOrLoc), Env);
}

clang::ast_matchers::DeclarationMatcher statusOrClass() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return classTemplateSpecializationDecl(
      hasName("absl::StatusOr"),
      hasTemplateArgument(0, refersToType(type().bind("T"))));
}

clang::ast_matchers::DeclarationMatcher statusClass() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return cxxRecordDecl(hasName("absl::Status"));
}

clang::ast_matchers::DeclarationMatcher statusOrOperatorBaseClass() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return classTemplateSpecializationDecl(
      hasName("absl::internal_statusor::OperatorBase"));
}

clang::ast_matchers::TypeMatcher statusOrType() {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return hasCanonicalType(qualType(hasDeclaration(statusOrClass())));
}

bool isStatusOrType(QualType Type) {
  return isTypeNamed(Type, {"absl"}, "StatusOr");
}

bool isStatusType(QualType Type) {
  return isTypeNamed(Type, {"absl"}, "Status");
}

llvm::StringMap<QualType> getSyntheticFields(QualType Ty, QualType StatusType,
                                             const CXXRecordDecl &RD) {
  if (auto *TRD = getStatusOrBaseClass(Ty))
    return {{"status", StatusType}, {"value", getStatusOrValueType(TRD)}};
  if (isStatusType(Ty) || (RD.hasDefinition() &&
                           RD.isDerivedFrom(StatusType->getAsCXXRecordDecl())))
    return {{"ok", RD.getASTContext().BoolTy}};
  return {};
}

RecordStorageLocation &locForStatus(RecordStorageLocation &StatusOrLoc) {
  return cast<RecordStorageLocation>(StatusOrLoc.getSyntheticField("status"));
}

StorageLocation &locForOk(RecordStorageLocation &StatusLoc) {
  return StatusLoc.getSyntheticField("ok");
}

BoolValue &valForOk(RecordStorageLocation &StatusLoc, Environment &Env) {
  if (auto *Val = Env.get<BoolValue>(locForOk(StatusLoc)))
    return *Val;
  return initializeStatus(StatusLoc, Env);
}

static void transferStatusOrOkCall(const CXXMemberCallExpr *Expr,
                                   const MatchFinder::MatchResult &,
                                   LatticeTransferState &State) {
  RecordStorageLocation *StatusOrLoc =
      getImplicitObjectLocation(*Expr, State.Env);
  if (StatusOrLoc == nullptr)
    return;

  auto &OkVal = valForOk(locForStatus(*StatusOrLoc), State.Env);
  State.Env.setValue(*Expr, OkVal);
}

static void transferStatusCall(const CXXMemberCallExpr *Expr,
                               const MatchFinder::MatchResult &,
                               LatticeTransferState &State) {
  RecordStorageLocation *StatusOrLoc =
      getImplicitObjectLocation(*Expr, State.Env);
  if (StatusOrLoc == nullptr)
    return;

  RecordStorageLocation &StatusLoc = locForStatus(*StatusOrLoc);

  if (State.Env.getValue(locForOk(StatusLoc)) == nullptr)
    initializeStatusOr(*StatusOrLoc, State.Env);

  if (Expr->isPRValue())
    copyRecord(StatusLoc, State.Env.getResultObjectLocation(*Expr), State.Env);
  else
    State.Env.setStorageLocation(*Expr, StatusLoc);
}

static void transferStatusOkCall(const CXXMemberCallExpr *Expr,
                                 const MatchFinder::MatchResult &,
                                 LatticeTransferState &State) {
  RecordStorageLocation *StatusLoc =
      getImplicitObjectLocation(*Expr, State.Env);
  if (StatusLoc == nullptr)
    return;

  if (Value *Val = State.Env.getValue(locForOk(*StatusLoc)))
    State.Env.setValue(*Expr, *Val);
}

static void transferStatusUpdateCall(const CXXMemberCallExpr *Expr,
                                     const MatchFinder::MatchResult &,
                                     LatticeTransferState &State) {
  // S.Update(OtherS) sets S to the error code of OtherS if it is OK,
  // otherwise does nothing.
  assert(Expr->getNumArgs() == 1);
  auto *Arg = Expr->getArg(0);
  RecordStorageLocation *ArgRecord =
      Arg->isPRValue() ? &State.Env.getResultObjectLocation(*Arg)
                       : State.Env.get<RecordStorageLocation>(*Arg);
  RecordStorageLocation *ThisLoc = getImplicitObjectLocation(*Expr, State.Env);
  if (ThisLoc == nullptr || ArgRecord == nullptr)
    return;

  auto &ThisOkVal = valForOk(*ThisLoc, State.Env);
  auto &ArgOkVal = valForOk(*ArgRecord, State.Env);
  auto &A = State.Env.arena();
  auto &NewVal = State.Env.makeAtomicBoolValue();
  State.Env.assume(A.makeImplies(A.makeNot(ThisOkVal.formula()),
                                 A.makeNot(NewVal.formula())));
  State.Env.assume(A.makeImplies(NewVal.formula(), ArgOkVal.formula()));
  State.Env.setValue(locForOk(*ThisLoc), NewVal);
}

static BoolValue *evaluateStatusEquality(RecordStorageLocation &LhsStatusLoc,
                                         RecordStorageLocation &RhsStatusLoc,
                                         Environment &Env) {
  auto &A = Env.arena();
  // Logically, a Status object is composed of an error code that could take one
  // of multiple possible values, including the "ok" value. We track whether a
  // Status object has an "ok" value and represent this as an `ok` bit. Equality
  // of Status objects compares their error codes. Therefore, merely comparing
  // the `ok` bits isn't sufficient: when two Status objects are assigned non-ok
  // error codes the equality of their respective error codes matters. Since we
  // only track the `ok` bits, we can't make any conclusions about equality when
  // we know that two Status objects have non-ok values.

  auto &LhsOkVal = valForOk(LhsStatusLoc, Env);
  auto &RhsOkVal = valForOk(RhsStatusLoc, Env);

  auto &Res = Env.makeAtomicBoolValue();

  // lhs && rhs => res (a.k.a. !res => !lhs || !rhs)
  Env.assume(A.makeImplies(A.makeAnd(LhsOkVal.formula(), RhsOkVal.formula()),
                           Res.formula()));
  // res => (lhs == rhs)
  Env.assume(A.makeImplies(
      Res.formula(), A.makeEquals(LhsOkVal.formula(), RhsOkVal.formula())));

  return &Res;
}

static BoolValue *
evaluateStatusOrEquality(RecordStorageLocation &LhsStatusOrLoc,
                         RecordStorageLocation &RhsStatusOrLoc,
                         Environment &Env) {
  auto &A = Env.arena();
  // Logically, a StatusOr<T> object is composed of two values - a Status and a
  // value of type T. Equality of StatusOr objects compares both values.
  // Therefore, merely comparing the `ok` bits of the Status values isn't
  // sufficient. When two StatusOr objects are engaged, the equality of their
  // respective values of type T matters. Similarly, when two StatusOr objects
  // have Status values that have non-ok error codes, the equality of the error
  // codes matters. Since we only track the `ok` bits of the Status values, we
  // can't make any conclusions about equality when we know that two StatusOr
  // objects are engaged or when their Status values contain non-ok error codes.
  auto &LhsOkVal = valForOk(locForStatus(LhsStatusOrLoc), Env);
  auto &RhsOkVal = valForOk(locForStatus(RhsStatusOrLoc), Env);
  auto &res = Env.makeAtomicBoolValue();

  // res => (lhs == rhs)
  Env.assume(A.makeImplies(
      res.formula(), A.makeEquals(LhsOkVal.formula(), RhsOkVal.formula())));
  return &res;
}

static BoolValue *evaluateEquality(const Expr *LhsExpr, const Expr *RhsExpr,
                                   Environment &Env) {
  // Check the type of both sides in case an operator== is added that admits
  // different types.
  if (isStatusOrType(LhsExpr->getType()) &&
      isStatusOrType(RhsExpr->getType())) {
    auto *LhsStatusOrLoc = Env.get<RecordStorageLocation>(*LhsExpr);
    if (LhsStatusOrLoc == nullptr)
      return nullptr;
    auto *RhsStatusOrLoc = Env.get<RecordStorageLocation>(*RhsExpr);
    if (RhsStatusOrLoc == nullptr)
      return nullptr;

    return evaluateStatusOrEquality(*LhsStatusOrLoc, *RhsStatusOrLoc, Env);
  }
  if (isStatusType(LhsExpr->getType()) && isStatusType(RhsExpr->getType())) {
    auto *LhsStatusLoc = Env.get<RecordStorageLocation>(*LhsExpr);
    if (LhsStatusLoc == nullptr)
      return nullptr;

    auto *RhsStatusLoc = Env.get<RecordStorageLocation>(*RhsExpr);
    if (RhsStatusLoc == nullptr)
      return nullptr;

    return evaluateStatusEquality(*LhsStatusLoc, *RhsStatusLoc, Env);
  }
  return nullptr;
}

static void transferComparisonOperator(const CXXOperatorCallExpr *Expr,
                                       LatticeTransferState &State,
                                       bool IsNegative) {
  auto *LhsAndRhsVal =
      evaluateEquality(Expr->getArg(0), Expr->getArg(1), State.Env);
  if (LhsAndRhsVal == nullptr)
    return;

  if (IsNegative)
    State.Env.setValue(*Expr, State.Env.makeNot(*LhsAndRhsVal));
  else
    State.Env.setValue(*Expr, *LhsAndRhsVal);
}

static RecordStorageLocation *getPointeeLocation(const Expr &Expr,
                                                 Environment &Env) {
  if (auto *PointerVal = Env.get<PointerValue>(Expr))
    return dyn_cast<RecordStorageLocation>(&PointerVal->getPointeeLoc());
  return nullptr;
}

static BoolValue *evaluatePointerEquality(const Expr *LhsExpr,
                                          const Expr *RhsExpr,
                                          Environment &Env) {
  assert(LhsExpr->getType()->isPointerType());
  assert(RhsExpr->getType()->isPointerType());
  RecordStorageLocation *LhsStatusLoc = nullptr;
  RecordStorageLocation *RhsStatusLoc = nullptr;
  if (isStatusOrType(LhsExpr->getType()->getPointeeType()) &&
      isStatusOrType(RhsExpr->getType()->getPointeeType())) {
    auto *LhsStatusOrLoc = getPointeeLocation(*LhsExpr, Env);
    auto *RhsStatusOrLoc = getPointeeLocation(*RhsExpr, Env);
    if (LhsStatusOrLoc == nullptr || RhsStatusOrLoc == nullptr)
      return nullptr;
    LhsStatusLoc = &locForStatus(*LhsStatusOrLoc);
    RhsStatusLoc = &locForStatus(*RhsStatusOrLoc);
  } else if (isStatusType(LhsExpr->getType()->getPointeeType()) &&
             isStatusType(RhsExpr->getType()->getPointeeType())) {
    LhsStatusLoc = getPointeeLocation(*LhsExpr, Env);
    RhsStatusLoc = getPointeeLocation(*RhsExpr, Env);
  }
  if (LhsStatusLoc == nullptr || RhsStatusLoc == nullptr)
    return nullptr;
  auto &LhsOkVal = valForOk(*LhsStatusLoc, Env);
  auto &RhsOkVal = valForOk(*RhsStatusLoc, Env);
  auto &Res = Env.makeAtomicBoolValue();
  auto &A = Env.arena();
  Env.assume(A.makeImplies(
      Res.formula(), A.makeEquals(LhsOkVal.formula(), RhsOkVal.formula())));
  return &Res;
}

static void transferPointerComparisonOperator(const BinaryOperator *Expr,
                                              LatticeTransferState &State,
                                              bool IsNegative) {
  auto *LhsAndRhsVal =
      evaluatePointerEquality(Expr->getLHS(), Expr->getRHS(), State.Env);
  if (LhsAndRhsVal == nullptr)
    return;

  if (IsNegative)
    State.Env.setValue(*Expr, State.Env.makeNot(*LhsAndRhsVal));
  else
    State.Env.setValue(*Expr, *LhsAndRhsVal);
}

static void transferOkStatusCall(const CallExpr *Expr,
                                 const MatchFinder::MatchResult &,
                                 LatticeTransferState &State) {
  auto &OkVal =
      initializeStatus(State.Env.getResultObjectLocation(*Expr), State.Env);
  State.Env.assume(OkVal.formula());
}

static void transferNotOkStatusCall(const CallExpr *Expr,
                                    const MatchFinder::MatchResult &,
                                    LatticeTransferState &State) {
  auto &OkVal =
      initializeStatus(State.Env.getResultObjectLocation(*Expr), State.Env);
  auto &A = State.Env.arena();
  State.Env.assume(A.makeNot(OkVal.formula()));
}

static void transferEmplaceCall(const CXXMemberCallExpr *Expr,
                                const MatchFinder::MatchResult &,
                                LatticeTransferState &State) {
  RecordStorageLocation *StatusOrLoc =
      getImplicitObjectLocation(*Expr, State.Env);
  if (StatusOrLoc == nullptr)
    return;

  auto &OkVal = valForOk(locForStatus(*StatusOrLoc), State.Env);
  State.Env.assume(OkVal.formula());
}

static void transferValueAssignmentCall(const CXXOperatorCallExpr *Expr,
                                        const MatchFinder::MatchResult &,
                                        LatticeTransferState &State) {
  assert(Expr->getNumArgs() > 1);

  auto *StatusOrLoc = State.Env.get<RecordStorageLocation>(*Expr->getArg(0));
  if (StatusOrLoc == nullptr)
    return;

  auto &OkVal = initializeStatusOr(*StatusOrLoc, State.Env);
  State.Env.assume(OkVal.formula());
}

static void transferValueConstructor(const CXXConstructExpr *Expr,
                                     const MatchFinder::MatchResult &,
                                     LatticeTransferState &State) {
  auto &OkVal =
      initializeStatusOr(State.Env.getResultObjectLocation(*Expr), State.Env);
  State.Env.assume(OkVal.formula());
}

static void transferStatusOrConstructor(const CXXConstructExpr *Expr,
                                        const MatchFinder::MatchResult &,
                                        LatticeTransferState &State) {
  RecordStorageLocation &StatusOrLoc = State.Env.getResultObjectLocation(*Expr);
  RecordStorageLocation &StatusLoc = locForStatus(StatusOrLoc);

  if (State.Env.getValue(locForOk(StatusLoc)) == nullptr)
    initializeStatusOr(StatusOrLoc, State.Env);
}

static void transferStatusConstructor(const CXXConstructExpr *Expr,
                                      const MatchFinder::MatchResult &,
                                      LatticeTransferState &State) {
  RecordStorageLocation &StatusLoc = State.Env.getResultObjectLocation(*Expr);

  if (State.Env.getValue(locForOk(StatusLoc)) == nullptr)
    initializeStatus(StatusLoc, State.Env);
}

CFGMatchSwitch<LatticeTransferState>
buildTransferMatchSwitch(ASTContext &Ctx,
                         CFGMatchSwitchBuilder<LatticeTransferState> Builder) {
  using namespace ::clang::ast_matchers; // NOLINT: Too many names
  return std::move(Builder)
      .CaseOfCFGStmt<CXXMemberCallExpr>(isStatusOrMemberCallWithName("ok"),
                                        transferStatusOrOkCall)
      .CaseOfCFGStmt<CXXMemberCallExpr>(isStatusOrMemberCallWithName("status"),
                                        transferStatusCall)
      .CaseOfCFGStmt<CXXMemberCallExpr>(isStatusMemberCallWithName("ok"),
                                        transferStatusOkCall)
      .CaseOfCFGStmt<CXXMemberCallExpr>(isStatusMemberCallWithName("Update"),
                                        transferStatusUpdateCall)
      .CaseOfCFGStmt<CXXOperatorCallExpr>(
          isComparisonOperatorCall("=="),
          [](const CXXOperatorCallExpr *Expr, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            transferComparisonOperator(Expr, State,
                                       /*IsNegative=*/false);
          })
      .CaseOfCFGStmt<CXXOperatorCallExpr>(
          isComparisonOperatorCall("!="),
          [](const CXXOperatorCallExpr *Expr, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            transferComparisonOperator(Expr, State,
                                       /*IsNegative=*/true);
          })
      .CaseOfCFGStmt<BinaryOperator>(
          isPointerComparisonOperatorCall("=="),
          [](const BinaryOperator *Expr, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            transferPointerComparisonOperator(Expr, State,
                                              /*IsNegative=*/false);
          })
      .CaseOfCFGStmt<BinaryOperator>(
          isPointerComparisonOperatorCall("!="),
          [](const BinaryOperator *Expr, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            transferPointerComparisonOperator(Expr, State,
                                              /*IsNegative=*/true);
          })
      .CaseOfCFGStmt<CallExpr>(isOkStatusCall(), transferOkStatusCall)
      .CaseOfCFGStmt<CallExpr>(isNotOkStatusCall(), transferNotOkStatusCall)
      .CaseOfCFGStmt<CXXMemberCallExpr>(isStatusOrMemberCallWithName("emplace"),
                                        transferEmplaceCall)
      .CaseOfCFGStmt<CXXOperatorCallExpr>(isStatusOrValueAssignmentCall(),
                                          transferValueAssignmentCall)
      .CaseOfCFGStmt<CXXConstructExpr>(isStatusOrValueConstructor(),
                                       transferValueConstructor)
      // N.B. These need to come after all other CXXConstructExpr.
      // These are there to make sure that every Status and StatusOr object
      // have their ok boolean initialized when constructed. If we were to
      // lazily initialize them when we first access them, we can produce
      // false positives if that first access is in a control flow statement.
      // You can comment out these two constructors and see tests fail.
      .CaseOfCFGStmt<CXXConstructExpr>(isStatusOrConstructor(),
                                       transferStatusOrConstructor)
      .CaseOfCFGStmt<CXXConstructExpr>(isStatusConstructor(),
                                       transferStatusConstructor)
      .Build();
}

QualType findStatusType(const ASTContext &Ctx) {
  for (Type *Ty : Ctx.getTypes())
    if (isStatusType(QualType(Ty, 0)))
      return QualType(Ty, 0);

  return QualType();
}

UncheckedStatusOrAccessModel::UncheckedStatusOrAccessModel(ASTContext &Ctx,
                                                           Environment &Env)
    : DataflowAnalysis<UncheckedStatusOrAccessModel,
                       UncheckedStatusOrAccessModel::Lattice>(Ctx),
      TransferMatchSwitch(buildTransferMatchSwitch(Ctx, {})) {
  QualType StatusType = findStatusType(Ctx);
  Env.getDataflowAnalysisContext().setSyntheticFieldCallback(
      [StatusType](QualType Ty) -> llvm::StringMap<QualType> {
        CXXRecordDecl *RD = Ty->getAsCXXRecordDecl();
        if (RD == nullptr)
          return {};

        if (auto Fields = getSyntheticFields(Ty, StatusType, *RD);
            !Fields.empty())
          return Fields;
        return {};
      });
}

void UncheckedStatusOrAccessModel::transfer(const CFGElement &Elt, Lattice &L,
                                            Environment &Env) {
  LatticeTransferState State(L, Env);
  TransferMatchSwitch(Elt, getASTContext(), State);
}

} // namespace clang::dataflow::statusor_model
