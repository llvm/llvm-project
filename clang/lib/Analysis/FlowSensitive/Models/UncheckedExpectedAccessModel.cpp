//===-- UncheckedExpectedAccessModel.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a dataflow analysis that detects unsafe accesses to the
//  values of `std::expected` objects.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Models/UncheckedExpectedAccessModel.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Analysis/FlowSensitive/Arena.h"
#include "clang/Analysis/FlowSensitive/Formula.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"

namespace clang {
namespace dataflow {

/// Returns true if `DC` is `std` or the top-level `tl` namespace, which holds
/// `tl::expected` (https://github.com/TartanLlama/expected).
static bool isSupportedNamespace(const DeclContext &DC) {
  if (DC.isStdNamespace())
    return true;
  const auto *NS = dyn_cast<NamespaceDecl>(&DC);
  return NS != nullptr && NS->getDeclName().isIdentifier() &&
         NS->getName() == "tl" && NS->getParent()->isTranslationUnit();
}

static bool isExpectedClass(const CXXRecordDecl &RD) {
  return RD.getDeclName().isIdentifier() && RD.getName() == "expected" &&
         isSupportedNamespace(*RD.getDeclContext());
}

/// Returns the name of the expected class `RD` to show in diagnostics.
static llvm::StringRef getExpectedTypeName(const CXXRecordDecl &RD) {
  return RD.getDeclContext()->isStdNamespace() ? "std::expected"
                                               : "tl::expected";
}

// Classes derived from `std::expected` are deliberately not supported.
static bool isExpectedType(QualType Ty) {
  const CXXRecordDecl *RD = Ty->getAsCXXRecordDecl();
  return RD != nullptr && isExpectedClass(*RD);
}

/// Returns true if `Ty` is `std::expected<void, E>`.
static bool isVoidExpectedType(QualType Ty) {
  const auto *CTSD = dyn_cast_or_null<ClassTemplateSpecializationDecl>(
      Ty->getAsCXXRecordDecl());
  if (CTSD == nullptr || !isExpectedClass(*CTSD))
    return false;
  const TemplateArgument &ValueType = CTSD->getTemplateArgs()[0];
  return ValueType.getKind() == TemplateArgument::Type &&
         ValueType.getAsType()->isVoidType();
}

namespace {

using namespace ::clang::ast_matchers;

using LatticeTransferState = TransferState<NoopLattice>;

AST_MATCHER(CXXRecordDecl, expectedClass) { return isExpectedClass(Node); }

AST_MATCHER(CXXRecordDecl, unexpectedClass) {
  return Node.getDeclName().isIdentifier() && Node.getName() == "unexpected" &&
         isSupportedNamespace(*Node.getDeclContext());
}

auto expectedMemberCall(ast_matchers::internal::Matcher<NamedDecl> Name) {
  return cxxMemberCallExpr(
      callee(cxxMethodDecl(Name, ofClass(expectedClass()))));
}

auto expectedOperatorCall(ArrayRef<StringRef> Names) {
  return cxxOperatorCallExpr(hasAnyOverloadedOperatorName(Names),
                             callee(cxxMethodDecl(ofClass(expectedClass()))));
}

auto hasExpectedType() {
  return hasType(hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(cxxRecordDecl(expectedClass())))));
}

auto hasUnexpectedType() {
  return hasType(hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(cxxRecordDecl(unexpectedClass())))));
}

auto expectedComparisonCall(
    const ast_matchers::internal::Matcher<Expr> &Other) {
  return cxxOperatorCallExpr(
      hasAnyOverloadedOperatorName("==", "!="), argumentCountIs(2),
      anyOf(allOf(hasArgument(0, hasExpectedType()), hasArgument(1, Other)),
            allOf(hasArgument(0, Other), hasArgument(1, hasExpectedType()))));
}

auto expectedEqualityCall() {
  return expectedComparisonCall(hasExpectedType());
}

auto expectedValueEqualityCall() {
  return expectedComparisonCall(
      unless(anyOf(hasExpectedType(), hasUnexpectedType())));
}

auto expectedUnexpectedEqualityCall() {
  return expectedComparisonCall(hasUnexpectedType());
}

// Copy and move constructors are modeled by the framework, which copies the
// "has_value" property of the source.
auto expectedConstructor() {
  return cxxConstructExpr(hasDeclaration(cxxConstructorDecl(
      ofClass(expectedClass()),
      unless(anyOf(isCopyConstructor(), isMoveConstructor())))));
}

// `expected(unexpect_t, ...)`, `expected(const unexpected<G> &)` and
// `expected(unexpected<G> &&)`.
auto expectedErrorConstructor() {
  return cxxConstructExpr(
      expectedConstructor(),
      hasArgument(0, anyOf(hasType(namedDecl(hasAnyName("std::unexpect_t",
                                                        "tl::unexpect_t"))),
                           hasUnexpectedType())));
}

// Matches `template <class U, class G> expected(const expected<U, G> &)` and
// `template <class U, class G> expected(expected<U, G> &&)`. The argument type
// alone is not enough to recognize them: `expected(U &&)` also accepts an
// `expected` when `T` is constructible from it, e.g. for nested `expected`s.
AST_MATCHER(CXXConstructorDecl, expectedConvertingConstructorDecl) {
  const FunctionTemplateDecl *Template = Node.getPrimaryTemplate();
  if (Template == nullptr)
    return false;
  const FunctionDecl *Pattern = Template->getTemplatedDecl();
  if (Pattern->getNumParams() != 1)
    return false;
  const auto *ParamType = Pattern->getParamDecl(0)
                              ->getType()
                              .getNonReferenceType()
                              ->getAs<TemplateSpecializationType>();
  if (ParamType == nullptr)
    return false;
  const auto *ParamTemplate = dyn_cast_or_null<ClassTemplateDecl>(
      ParamType->getTemplateName().getAsTemplateDecl());
  return ParamTemplate != nullptr &&
         isExpectedClass(*ParamTemplate->getTemplatedDecl());
}

auto expectedConvertingConstructor() {
  return cxxConstructExpr(
      expectedConstructor(),
      hasDeclaration(cxxConstructorDecl(expectedConvertingConstructorDecl())));
}

// Copy and move assignments are modeled by the framework, which copies the
// "has_value" property of the source.
auto expectedAssignment() {
  return cxxOperatorCallExpr(
      hasOverloadedOperatorName("="), argumentCountIs(2),
      callee(cxxMethodDecl(ofClass(expectedClass()),
                           unless(anyOf(isCopyAssignmentOperator(),
                                        isMoveAssignmentOperator())))));
}

// `operator=(const unexpected<G> &)` and `operator=(unexpected<G> &&)`.
auto expectedErrorAssignment() {
  return cxxOperatorCallExpr(expectedAssignment(),
                             hasArgument(1, hasUnexpectedType()));
}

auto stdForwardCall() {
  return callExpr(callee(functionDecl(hasName("std::forward"))),
                  argumentCountIs(1), hasArgument(0, hasExpectedType()));
}

/// Ensures that `E` is mapped to a `BoolValue` and returns its formula.
const Formula &forceBoolValue(Environment &Env, const Expr &E) {
  auto *Value = Env.get<BoolValue>(E);
  if (Value != nullptr)
    return Value->formula();

  Value = &Env.makeAtomicBoolValue();
  Env.setValue(E, *Value);
  return Value->formula();
}

const Formula &equalityFormula(Environment &Env, const CXXOperatorCallExpr &E) {
  const Formula &Result = forceBoolValue(Env, E);
  if (E.getOperator() == OO_ExclaimEqual)
    return Env.arena().makeNot(Result);
  return Result;
}

const Expr &getExpectedOperand(const CXXOperatorCallExpr &E) {
  return isExpectedType(E.getArg(0)->getType()) ? *E.getArg(0) : *E.getArg(1);
}

StorageLocation &locForHasValue(const RecordStorageLocation &ExpectedLoc) {
  return ExpectedLoc.getSyntheticField("has_value");
}

BoolValue *getHasValue(Environment &Env, RecordStorageLocation *ExpectedLoc) {
  if (ExpectedLoc == nullptr || !isExpectedType(ExpectedLoc->getType()))
    return nullptr;
  StorageLocation &HasValueLoc = locForHasValue(*ExpectedLoc);
  auto *HasValueVal = Env.get<BoolValue>(HasValueLoc);
  if (HasValueVal == nullptr) {
    HasValueVal = &Env.makeAtomicBoolValue();
    Env.setValue(HasValueLoc, *HasValueVal);
  }
  return HasValueVal;
}

void setHasValue(RecordStorageLocation *ExpectedLoc, BoolValue &HasValueVal,
                 Environment &Env) {
  if (ExpectedLoc == nullptr || !isExpectedType(ExpectedLoc->getType()))
    return;
  Env.setValue(locForHasValue(*ExpectedLoc), HasValueVal);
}

void transferHasValueCall(const CXXMemberCallExpr *E,
                          const MatchFinder::MatchResult &,
                          LatticeTransferState &State) {
  if (BoolValue *HasValueVal =
          getHasValue(State.Env, getImplicitObjectLocation(*E, State.Env)))
    State.Env.setValue(*E, *HasValueVal);
}

void transferEmplaceCall(const CXXMemberCallExpr *E,
                         const MatchFinder::MatchResult &,
                         LatticeTransferState &State) {
  // Every `emplace` overload leaves the expected holding a value.
  setHasValue(getImplicitObjectLocation(*E, State.Env),
              State.Env.getBoolLiteralValue(true), State.Env);
}

void transferTransformCall(const CXXMemberCallExpr *E,
                           const MatchFinder::MatchResult &,
                           LatticeTransferState &State) {
  // `transform` maps the value and `transform_error` maps the error, so the
  // result holds a value if and only if the source does.
  if (BoolValue *HasValueVal =
          getHasValue(State.Env, getImplicitObjectLocation(*E, State.Env)))
    setHasValue(&State.Env.getResultObjectLocation(*E), *HasValueVal,
                State.Env);
}

void transferExpectedEqualityCall(const CXXOperatorCallExpr *E,
                                  const MatchFinder::MatchResult &,
                                  LatticeTransferState &State) {
  Environment &Env = State.Env;
  BoolValue *LHasValueVal =
      getHasValue(Env, Env.get<RecordStorageLocation>(*E->getArg(0)));
  BoolValue *RHasValueVal =
      getHasValue(Env, Env.get<RecordStorageLocation>(*E->getArg(1)));
  if (LHasValueVal == nullptr || RHasValueVal == nullptr)
    return;

  Arena &A = Env.arena();
  const Formula &EqVal = equalityFormula(Env, *E);
  const Formula &LHasValue = LHasValueVal->formula();
  const Formula &RHasValue = RHasValueVal->formula();

  // Equal expecteds either both hold a value or both hold an error. Nothing
  // more follows in general: the values (or errors) themselves may differ.
  Env.assume(A.makeImplies(EqVal, A.makeEquals(LHasValue, RHasValue)));

  // `expected<void, E>` objects that both hold a value are always equal.
  if (isVoidExpectedType(E->getArg(0)->getType()) &&
      isVoidExpectedType(E->getArg(1)->getType()))
    Env.assume(A.makeImplies(A.makeAnd(LHasValue, RHasValue), EqVal));
}

void transferExpectedValueEqualityCall(const CXXOperatorCallExpr *E,
                                       const MatchFinder::MatchResult &,
                                       LatticeTransferState &State) {
  Environment &Env = State.Env;
  BoolValue *HasValueVal =
      getHasValue(Env, Env.get<RecordStorageLocation>(getExpectedOperand(*E)));
  if (HasValueVal == nullptr)
    return;

  // `e == v` --> `e.has_value() && *e == v`
  Env.assume(Env.arena().makeImplies(equalityFormula(Env, *E),
                                     HasValueVal->formula()));
}

void transferExpectedUnexpectedEqualityCall(const CXXOperatorCallExpr *E,
                                            const MatchFinder::MatchResult &,
                                            LatticeTransferState &State) {
  Environment &Env = State.Env;
  BoolValue *HasValueVal =
      getHasValue(Env, Env.get<RecordStorageLocation>(getExpectedOperand(*E)));
  if (HasValueVal == nullptr)
    return;

  // `e == u` --> `!e.has_value() && e.error() == u.error()`
  Arena &A = Env.arena();
  Env.assume(A.makeImplies(equalityFormula(Env, *E),
                           A.makeNot(HasValueVal->formula())));
}

void constructExpected(const CXXConstructExpr &E, BoolValue &HasValueVal,
                       Environment &Env) {
  setHasValue(&Env.getResultObjectLocation(E), HasValueVal, Env);
}

void transferValueConstructor(const CXXConstructExpr *E,
                              const MatchFinder::MatchResult &,
                              LatticeTransferState &State) {
  constructExpected(*E, State.Env.getBoolLiteralValue(true), State.Env);
}

void transferErrorConstructor(const CXXConstructExpr *E,
                              const MatchFinder::MatchResult &,
                              LatticeTransferState &State) {
  constructExpected(*E, State.Env.getBoolLiteralValue(false), State.Env);
}

void transferConvertingConstructor(const CXXConstructExpr *E,
                                   const MatchFinder::MatchResult &,
                                   LatticeTransferState &State) {
  // The new expected holds a value if and only if the source does.
  if (BoolValue *HasValueVal = getHasValue(
          State.Env, State.Env.get<RecordStorageLocation>(*E->getArg(0))))
    constructExpected(*E, *HasValueVal, State.Env);
}

void assignExpected(const CXXOperatorCallExpr &E, BoolValue &HasValueVal,
                    Environment &Env) {
  auto *Loc = Env.get<RecordStorageLocation>(*E.getArg(0));
  if (Loc == nullptr)
    return;
  setHasValue(Loc, HasValueVal, Env);
  // The result of the assignment refers to the assigned object.
  Env.setStorageLocation(E, *Loc);
}

void transferValueAssignment(const CXXOperatorCallExpr *E,
                             const MatchFinder::MatchResult &,
                             LatticeTransferState &State) {
  assignExpected(*E, State.Env.getBoolLiteralValue(true), State.Env);
}

void transferErrorAssignment(const CXXOperatorCallExpr *E,
                             const MatchFinder::MatchResult &,
                             LatticeTransferState &State) {
  assignExpected(*E, State.Env.getBoolLiteralValue(false), State.Env);
}

void transferStdForwardCall(const CallExpr *E, const MatchFinder::MatchResult &,
                            LatticeTransferState &State) {
  if (StorageLocation *Loc = State.Env.getStorageLocation(*E->getArg(0)))
    State.Env.setStorageLocation(*E, *Loc);
}

// FIXME: Model `swap`.
auto buildTransferMatchSwitch() {
  return CFGMatchSwitchBuilder<LatticeTransferState>()
      // expected::has_value, expected::operator bool
      .CaseOfCFGStmt<CXXMemberCallExpr>(
          expectedMemberCall(hasAnyName("has_value", "operator bool")),
          transferHasValueCall)
      // expected::emplace
      .CaseOfCFGStmt<CXXMemberCallExpr>(expectedMemberCall(hasName("emplace")),
                                        transferEmplaceCall)
      // expected::transform, expected::transform_error, and their
      // `tl::expected` aliases expected::map, expected::map_error
      .CaseOfCFGStmt<CXXMemberCallExpr>(
          expectedMemberCall(
              hasAnyName("transform", "transform_error", "map", "map_error")),
          transferTransformCall)
      // operator==, operator!= between two expecteds
      .CaseOfCFGStmt<CXXOperatorCallExpr>(expectedEqualityCall(),
                                          transferExpectedEqualityCall)
      // operator==, operator!= between an expected and a value
      .CaseOfCFGStmt<CXXOperatorCallExpr>(expectedValueEqualityCall(),
                                          transferExpectedValueEqualityCall)
      // operator==, operator!= between an expected and an unexpected
      .CaseOfCFGStmt<CXXOperatorCallExpr>(
          expectedUnexpectedEqualityCall(),
          transferExpectedUnexpectedEqualityCall)
      // Constructors. The order matters: the last case matches every
      // constructor that is not matched by the ones before it.
      .CaseOfCFGStmt<CXXConstructExpr>(expectedErrorConstructor(),
                                       transferErrorConstructor)
      .CaseOfCFGStmt<CXXConstructExpr>(expectedConvertingConstructor(),
                                       transferConvertingConstructor)
      .CaseOfCFGStmt<CXXConstructExpr>(expectedConstructor(),
                                       transferValueConstructor)
      // Assignments. The order matters: the last case matches every
      // assignment that is not matched by the one before it.
      .CaseOfCFGStmt<CXXOperatorCallExpr>(expectedErrorAssignment(),
                                          transferErrorAssignment)
      .CaseOfCFGStmt<CXXOperatorCallExpr>(expectedAssignment(),
                                          transferValueAssignment)
      // std::forward
      .CaseOfCFGStmt<CallExpr>(stdForwardCall(), transferStdForwardCall)
      .Build();
}

/// `ExpectedClass` is the class of the called accessor.
llvm::SmallVector<UncheckedExpectedAccessDiagnostic>
diagnoseAccess(SourceLocation AccessLoc, UncheckedExpectedAccessKind Kind,
               const CXXRecordDecl &ExpectedClass,
               const RecordStorageLocation *ExpectedLoc,
               const Environment &Env) {
  if (ExpectedLoc != nullptr) {
    // Accesses through classes derived from an expected are not modeled.
    if (!isExpectedType(ExpectedLoc->getType()))
      return {};
    if (auto *HasValueVal = Env.get<BoolValue>(locForHasValue(*ExpectedLoc))) {
      // The value may only be accessed if the expected holds a value, and the
      // error only if it does not.
      const Formula &Safe = Kind == UncheckedExpectedAccessKind::Value
                                ? HasValueVal->formula()
                                : Env.arena().makeNot(HasValueVal->formula());
      if (Env.proves(Safe))
        return {};
    }
  }

  return {{AccessLoc, Kind, getExpectedTypeName(ExpectedClass)}};
}

auto buildDiagnoseMatchSwitch() {
  return CFGMatchSwitchBuilder<
             const Environment,
             llvm::SmallVector<UncheckedExpectedAccessDiagnostic>>()
      // expected::value
      .CaseOfCFGStmt<CXXMemberCallExpr>(
          expectedMemberCall(hasName("value")),
          [](const CXXMemberCallExpr *E, const MatchFinder::MatchResult &,
             const Environment &Env) {
            return diagnoseAccess(E->getExprLoc(),
                                  UncheckedExpectedAccessKind::Value,
                                  *E->getMethodDecl()->getParent(),
                                  getImplicitObjectLocation(*E, Env), Env);
          })
      // expected::operator*, expected::operator->
      .CaseOfCFGStmt<CXXOperatorCallExpr>(
          expectedOperatorCall({"*", "->"}),
          [](const CXXOperatorCallExpr *E, const MatchFinder::MatchResult &,
             const Environment &Env) {
            // `getExprLoc` of `operator->` is the start of the object
            // expression, so point at the operator token instead.
            return diagnoseAccess(
                E->getOperatorLoc(), UncheckedExpectedAccessKind::Value,
                *cast<CXXMethodDecl>(E->getCalleeDecl())->getParent(),
                Env.get<RecordStorageLocation>(*E->getArg(0)), Env);
          })
      // expected::error
      .CaseOfCFGStmt<CXXMemberCallExpr>(
          expectedMemberCall(hasName("error")),
          [](const CXXMemberCallExpr *E, const MatchFinder::MatchResult &,
             const Environment &Env) {
            return diagnoseAccess(E->getExprLoc(),
                                  UncheckedExpectedAccessKind::Error,
                                  *E->getMethodDecl()->getParent(),
                                  getImplicitObjectLocation(*E, Env), Env);
          })
      .Build();
}

} // namespace

ast_matchers::StatementMatcher
UncheckedExpectedAccessModel::callToExpectedClass() {
  return callExpr(callee(cxxMethodDecl(ofClass(expectedClass()))));
}

UncheckedExpectedAccessModel::UncheckedExpectedAccessModel(ASTContext &Ctx,
                                                           Environment &Env)
    : DataflowAnalysis<UncheckedExpectedAccessModel, NoopLattice>(Ctx),
      TransferMatchSwitch(buildTransferMatchSwitch()) {
  Env.getDataflowAnalysisContext().setSyntheticFieldCallback(
      [&Ctx](QualType Ty) -> llvm::StringMap<QualType> {
        if (!isExpectedType(Ty))
          return {};
        return {{"has_value", Ctx.BoolTy}};
      });
}

void UncheckedExpectedAccessModel::transfer(const CFGElement &Elt,
                                            NoopLattice &L, Environment &Env) {
  LatticeTransferState State(L, Env);
  TransferMatchSwitch(Elt, getASTContext(), State);
}

UncheckedExpectedAccessDiagnoser::UncheckedExpectedAccessDiagnoser()
    : DiagnoseMatchSwitch(buildDiagnoseMatchSwitch()) {}

} // namespace dataflow
} // namespace clang
