//===-- UncheckedOptionalAccessModel.cpp ------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a dataflow analysis that detects unsafe uses of optional
//  values.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Models/UncheckedOptionalAccessModel.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/MatchSwitch.h"
#include "clang/Analysis/FlowSensitive/SourceLocationsLattice.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include <cassert>
#include <memory>
#include <utility>

namespace clang {
namespace dataflow {
namespace {

using namespace ::clang::ast_matchers;
using LatticeTransferState = TransferState<SourceLocationsLattice>;

DeclarationMatcher optionalClass() {
  return classTemplateSpecializationDecl(
      anyOf(hasName("std::optional"), hasName("std::__optional_storage_base"),
            hasName("__optional_destruct_base"), hasName("absl::optional"),
            hasName("base::Optional")),
      hasTemplateArgument(0, refersToType(type().bind("T"))));
}

auto hasOptionalType() { return hasType(optionalClass()); }

auto hasOptionalOrAliasType() {
  return hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(optionalClass())));
}

auto isOptionalMemberCallWithName(
    llvm::StringRef MemberName,
    llvm::Optional<StatementMatcher> Ignorable = llvm::None) {
  auto Exception = unless(Ignorable ? expr(anyOf(*Ignorable, cxxThisExpr()))
                                    : cxxThisExpr());
  return cxxMemberCallExpr(
      on(expr(Exception)),
      callee(cxxMethodDecl(hasName(MemberName), ofClass(optionalClass()))));
}

auto isOptionalOperatorCallWithName(
    llvm::StringRef operator_name,
    llvm::Optional<StatementMatcher> Ignorable = llvm::None) {
  return cxxOperatorCallExpr(
      hasOverloadedOperatorName(operator_name),
      callee(cxxMethodDecl(ofClass(optionalClass()))),
      Ignorable ? callExpr(unless(hasArgument(0, *Ignorable))) : callExpr());
}

auto isMakeOptionalCall() {
  return callExpr(
      callee(functionDecl(hasAnyName(
          "std::make_optional", "base::make_optional", "absl::make_optional"))),
      hasOptionalType());
}

auto hasNulloptType() {
  return hasType(namedDecl(
      hasAnyName("std::nullopt_t", "absl::nullopt_t", "base::nullopt_t")));
}

auto inPlaceClass() {
  return recordDecl(
      hasAnyName("std::in_place_t", "absl::in_place_t", "base::in_place_t"));
}

auto isOptionalNulloptConstructor() {
  return cxxConstructExpr(hasOptionalType(), argumentCountIs(1),
                          hasArgument(0, hasNulloptType()));
}

auto isOptionalInPlaceConstructor() {
  return cxxConstructExpr(hasOptionalType(),
                          hasArgument(0, hasType(inPlaceClass())));
}

auto isOptionalValueOrConversionConstructor() {
  return cxxConstructExpr(
      hasOptionalType(),
      unless(hasDeclaration(
          cxxConstructorDecl(anyOf(isCopyConstructor(), isMoveConstructor())))),
      argumentCountIs(1), hasArgument(0, unless(hasNulloptType())));
}

auto isOptionalValueOrConversionAssignment() {
  return cxxOperatorCallExpr(
      hasOverloadedOperatorName("="),
      callee(cxxMethodDecl(ofClass(optionalClass()))),
      unless(hasDeclaration(cxxMethodDecl(
          anyOf(isCopyAssignmentOperator(), isMoveAssignmentOperator())))),
      argumentCountIs(2), hasArgument(1, unless(hasNulloptType())));
}

auto isOptionalNulloptAssignment() {
  return cxxOperatorCallExpr(hasOverloadedOperatorName("="),
                             callee(cxxMethodDecl(ofClass(optionalClass()))),
                             argumentCountIs(2),
                             hasArgument(1, hasNulloptType()));
}

auto isStdSwapCall() {
  return callExpr(callee(functionDecl(hasName("std::swap"))),
                  argumentCountIs(2), hasArgument(0, hasOptionalType()),
                  hasArgument(1, hasOptionalType()));
}

constexpr llvm::StringLiteral ValueOrCallID = "ValueOrCall";

auto isValueOrStringEmptyCall() {
  // `opt.value_or("").empty()`
  return cxxMemberCallExpr(
      callee(cxxMethodDecl(hasName("empty"))),
      onImplicitObjectArgument(ignoringImplicit(
          cxxMemberCallExpr(on(expr(unless(cxxThisExpr()))),
                            callee(cxxMethodDecl(hasName("value_or"),
                                                 ofClass(optionalClass()))),
                            hasArgument(0, stringLiteral(hasSize(0))))
              .bind(ValueOrCallID))));
}

auto isValueOrNotEqX() {
  auto ComparesToSame = [](ast_matchers::internal::Matcher<Stmt> Arg) {
    return hasOperands(
        ignoringImplicit(
            cxxMemberCallExpr(on(expr(unless(cxxThisExpr()))),
                              callee(cxxMethodDecl(hasName("value_or"),
                                                   ofClass(optionalClass()))),
                              hasArgument(0, Arg))
                .bind(ValueOrCallID)),
        ignoringImplicit(Arg));
  };

  // `opt.value_or(X) != X`, for X is `nullptr`, `""`, or `0`. Ideally, we'd
  // support this pattern for any expression, but the AST does not have a
  // generic expression comparison facility, so we specialize to common cases
  // seen in practice.  FIXME: define a matcher that compares values across
  // nodes, which would let us generalize this to any `X`.
  return binaryOperation(hasOperatorName("!="),
                         anyOf(ComparesToSame(cxxNullPtrLiteralExpr()),
                               ComparesToSame(stringLiteral(hasSize(0))),
                               ComparesToSame(integerLiteral(equals(0)))));
}

auto isCallReturningOptional() {
  return callExpr(callee(functionDecl(
      returns(anyOf(hasOptionalOrAliasType(),
                    referenceType(pointee(hasOptionalOrAliasType())))))));
}

/// Creates a symbolic value for an `optional` value using `HasValueVal` as the
/// symbolic value of its "has_value" property.
StructValue &createOptionalValue(Environment &Env, BoolValue &HasValueVal) {
  auto OptionalVal = std::make_unique<StructValue>();
  OptionalVal->setProperty("has_value", HasValueVal);
  return Env.takeOwnership(std::move(OptionalVal));
}

/// Returns the symbolic value that represents the "has_value" property of the
/// optional value `Val`. Returns null if `Val` is null.
BoolValue *getHasValue(Value *Val) {
  if (auto *OptionalVal = cast_or_null<StructValue>(Val)) {
    return cast<BoolValue>(OptionalVal->getProperty("has_value"));
  }
  return nullptr;
}

/// If `Type` is a reference type, returns the type of its pointee. Otherwise,
/// returns `Type` itself.
QualType stripReference(QualType Type) {
  return Type->isReferenceType() ? Type->getPointeeType() : Type;
}

/// Returns true if and only if `Type` is an optional type.
bool IsOptionalType(QualType Type) {
  if (!Type->isRecordType())
    return false;
  // FIXME: Optimize this by avoiding the `getQualifiedNameAsString` call.
  auto TypeName = Type->getAsCXXRecordDecl()->getQualifiedNameAsString();
  return TypeName == "std::optional" || TypeName == "absl::optional" ||
         TypeName == "base::Optional";
}

/// Returns the number of optional wrappers in `Type`.
///
/// For example, if `Type` is `optional<optional<int>>`, the result of this
/// function will be 2.
int countOptionalWrappers(const ASTContext &ASTCtx, QualType Type) {
  if (!IsOptionalType(Type))
    return 0;
  return 1 + countOptionalWrappers(
                 ASTCtx,
                 cast<ClassTemplateSpecializationDecl>(Type->getAsRecordDecl())
                     ->getTemplateArgs()
                     .get(0)
                     .getAsType()
                     .getDesugaredType(ASTCtx));
}

void initializeOptionalReference(const Expr *OptionalExpr,
                                 const MatchFinder::MatchResult &,
                                 LatticeTransferState &State) {
  if (auto *OptionalVal = cast_or_null<StructValue>(
          State.Env.getValue(*OptionalExpr, SkipPast::Reference))) {
    if (OptionalVal->getProperty("has_value") == nullptr) {
      OptionalVal->setProperty("has_value", State.Env.makeAtomicBoolValue());
    }
  }
}

void transferUnwrapCall(const Expr *UnwrapExpr, const Expr *ObjectExpr,
                        LatticeTransferState &State) {
  if (auto *OptionalVal = cast_or_null<StructValue>(
          State.Env.getValue(*ObjectExpr, SkipPast::ReferenceThenPointer))) {
    auto *HasValueVal = getHasValue(OptionalVal);
    assert(HasValueVal != nullptr);

    if (State.Env.flowConditionImplies(*HasValueVal))
      return;
  }

  // Record that this unwrap is *not* provably safe.
  // FIXME: include either the name of the optional (if applicable) or a source
  // range of the access for easier interpretation of the result.
  State.Lattice.getSourceLocations().insert(ObjectExpr->getBeginLoc());
}

void transferMakeOptionalCall(const CallExpr *E,
                              const MatchFinder::MatchResult &,
                              LatticeTransferState &State) {
  auto &Loc = State.Env.createStorageLocation(*E);
  State.Env.setStorageLocation(*E, Loc);
  State.Env.setValue(
      Loc, createOptionalValue(State.Env, State.Env.getBoolLiteralValue(true)));
}

void transferOptionalHasValueCall(const CXXMemberCallExpr *CallExpr,
                                  const MatchFinder::MatchResult &,
                                  LatticeTransferState &State) {
  if (auto *OptionalVal = cast_or_null<StructValue>(
          State.Env.getValue(*CallExpr->getImplicitObjectArgument(),
                             SkipPast::ReferenceThenPointer))) {
    auto *HasValueVal = getHasValue(OptionalVal);
    assert(HasValueVal != nullptr);

    auto &CallExprLoc = State.Env.createStorageLocation(*CallExpr);
    State.Env.setValue(CallExprLoc, *HasValueVal);
    State.Env.setStorageLocation(*CallExpr, CallExprLoc);
  }
}

/// `ModelPred` builds a logical formula relating the predicate in
/// `ValueOrPredExpr` to the optional's `has_value` property.
void transferValueOrImpl(const clang::Expr *ValueOrPredExpr,
                         const MatchFinder::MatchResult &Result,
                         LatticeTransferState &State,
                         BoolValue &(*ModelPred)(Environment &Env,
                                                 BoolValue &ExprVal,
                                                 BoolValue &HasValueVal)) {
  auto &Env = State.Env;

  const auto *ObjectArgumentExpr =
      Result.Nodes.getNodeAs<clang::CXXMemberCallExpr>(ValueOrCallID)
          ->getImplicitObjectArgument();

  auto *OptionalVal = cast_or_null<StructValue>(
      Env.getValue(*ObjectArgumentExpr, SkipPast::ReferenceThenPointer));
  if (OptionalVal == nullptr)
    return;
  auto *HasValueVal = getHasValue(OptionalVal);
  assert(HasValueVal != nullptr);

  auto *ExprValue = cast_or_null<BoolValue>(
      State.Env.getValue(*ValueOrPredExpr, SkipPast::None));
  if (ExprValue == nullptr) {
    auto &ExprLoc = State.Env.createStorageLocation(*ValueOrPredExpr);
    ExprValue = &State.Env.makeAtomicBoolValue();
    State.Env.setValue(ExprLoc, *ExprValue);
    State.Env.setStorageLocation(*ValueOrPredExpr, ExprLoc);
  }

  Env.addToFlowCondition(ModelPred(Env, *ExprValue, *HasValueVal));
}

void transferValueOrStringEmptyCall(const clang::Expr *ComparisonExpr,
                                    const MatchFinder::MatchResult &Result,
                                    LatticeTransferState &State) {
  return transferValueOrImpl(ComparisonExpr, Result, State,
                             [](Environment &Env, BoolValue &ExprVal,
                                BoolValue &HasValueVal) -> BoolValue & {
                               // If the result is *not* empty, then we know the
                               // optional must have been holding a value. If
                               // `ExprVal` is true, though, we don't learn
                               // anything definite about `has_value`, so we
                               // don't add any corresponding implications to
                               // the flow condition.
                               return Env.makeImplication(Env.makeNot(ExprVal),
                                                          HasValueVal);
                             });
}

void transferValueOrNotEqX(const Expr *ComparisonExpr,
                           const MatchFinder::MatchResult &Result,
                           LatticeTransferState &State) {
  transferValueOrImpl(ComparisonExpr, Result, State,
                      [](Environment &Env, BoolValue &ExprVal,
                         BoolValue &HasValueVal) -> BoolValue & {
                        // We know that if `(opt.value_or(X) != X)` then
                        // `opt.hasValue()`, even without knowing further
                        // details about the contents of `opt`.
                        return Env.makeImplication(ExprVal, HasValueVal);
                      });
}

void transferCallReturningOptional(const CallExpr *E,
                                   const MatchFinder::MatchResult &Result,
                                   LatticeTransferState &State) {
  if (State.Env.getStorageLocation(*E, SkipPast::None) != nullptr)
    return;

  auto &Loc = State.Env.createStorageLocation(*E);
  State.Env.setStorageLocation(*E, Loc);
  State.Env.setValue(
      Loc, createOptionalValue(State.Env, State.Env.makeAtomicBoolValue()));
}

void assignOptionalValue(const Expr &E, LatticeTransferState &State,
                         BoolValue &HasValueVal) {
  if (auto *OptionalLoc =
          State.Env.getStorageLocation(E, SkipPast::ReferenceThenPointer)) {
    State.Env.setValue(*OptionalLoc,
                       createOptionalValue(State.Env, HasValueVal));
  }
}

/// Returns a symbolic value for the "has_value" property of an `optional<T>`
/// value that is constructed/assigned from a value of type `U` or `optional<U>`
/// where `T` is constructible from `U`.
BoolValue &
getValueOrConversionHasValue(const FunctionDecl &F, const Expr &E,
                             const MatchFinder::MatchResult &MatchRes,
                             LatticeTransferState &State) {
  assert(F.getTemplateSpecializationArgs()->size() > 0);

  const int TemplateParamOptionalWrappersCount = countOptionalWrappers(
      *MatchRes.Context,
      stripReference(F.getTemplateSpecializationArgs()->get(0).getAsType()));
  const int ArgTypeOptionalWrappersCount =
      countOptionalWrappers(*MatchRes.Context, stripReference(E.getType()));

  // Check if this is a constructor/assignment call for `optional<T>` with
  // argument of type `U` such that `T` is constructible from `U`.
  if (TemplateParamOptionalWrappersCount == ArgTypeOptionalWrappersCount)
    return State.Env.getBoolLiteralValue(true);

  // This is a constructor/assignment call for `optional<T>` with argument of
  // type `optional<U>` such that `T` is constructible from `U`.
  if (BoolValue *Val = getHasValue(State.Env.getValue(E, SkipPast::Reference)))
    return *Val;
  return State.Env.makeAtomicBoolValue();
}

void transferValueOrConversionConstructor(
    const CXXConstructExpr *E, const MatchFinder::MatchResult &MatchRes,
    LatticeTransferState &State) {
  assert(E->getNumArgs() > 0);

  assignOptionalValue(*E, State,
                      getValueOrConversionHasValue(*E->getConstructor(),
                                                   *E->getArg(0), MatchRes,
                                                   State));
}

void transferAssignment(const CXXOperatorCallExpr *E, BoolValue &HasValueVal,
                        LatticeTransferState &State) {
  assert(E->getNumArgs() > 0);

  auto *OptionalLoc =
      State.Env.getStorageLocation(*E->getArg(0), SkipPast::Reference);
  assert(OptionalLoc != nullptr);

  State.Env.setValue(*OptionalLoc, createOptionalValue(State.Env, HasValueVal));

  // Assign a storage location for the whole expression.
  State.Env.setStorageLocation(*E, *OptionalLoc);
}

void transferValueOrConversionAssignment(
    const CXXOperatorCallExpr *E, const MatchFinder::MatchResult &MatchRes,
    LatticeTransferState &State) {
  assert(E->getNumArgs() > 1);
  transferAssignment(E,
                     getValueOrConversionHasValue(
                         *E->getDirectCallee(), *E->getArg(1), MatchRes, State),
                     State);
}

void transferNulloptAssignment(const CXXOperatorCallExpr *E,
                               const MatchFinder::MatchResult &,
                               LatticeTransferState &State) {
  transferAssignment(E, State.Env.getBoolLiteralValue(false), State);
}

void transferSwap(const StorageLocation &OptionalLoc1,
                  const StorageLocation &OptionalLoc2,
                  LatticeTransferState &State) {
  auto *OptionalVal1 = State.Env.getValue(OptionalLoc1);
  assert(OptionalVal1 != nullptr);

  auto *OptionalVal2 = State.Env.getValue(OptionalLoc2);
  assert(OptionalVal2 != nullptr);

  State.Env.setValue(OptionalLoc1, *OptionalVal2);
  State.Env.setValue(OptionalLoc2, *OptionalVal1);
}

void transferSwapCall(const CXXMemberCallExpr *E,
                      const MatchFinder::MatchResult &,
                      LatticeTransferState &State) {
  assert(E->getNumArgs() == 1);

  auto *OptionalLoc1 = State.Env.getStorageLocation(
      *E->getImplicitObjectArgument(), SkipPast::ReferenceThenPointer);
  assert(OptionalLoc1 != nullptr);

  auto *OptionalLoc2 =
      State.Env.getStorageLocation(*E->getArg(0), SkipPast::Reference);
  assert(OptionalLoc2 != nullptr);

  transferSwap(*OptionalLoc1, *OptionalLoc2, State);
}

void transferStdSwapCall(const CallExpr *E, const MatchFinder::MatchResult &,
                         LatticeTransferState &State) {
  assert(E->getNumArgs() == 2);

  auto *OptionalLoc1 =
      State.Env.getStorageLocation(*E->getArg(0), SkipPast::Reference);
  assert(OptionalLoc1 != nullptr);

  auto *OptionalLoc2 =
      State.Env.getStorageLocation(*E->getArg(1), SkipPast::Reference);
  assert(OptionalLoc2 != nullptr);

  transferSwap(*OptionalLoc1, *OptionalLoc2, State);
}

llvm::Optional<StatementMatcher>
ignorableOptional(const UncheckedOptionalAccessModelOptions &Options) {
  if (Options.IgnoreSmartPointerDereference)
    return memberExpr(hasObjectExpression(ignoringParenImpCasts(
        cxxOperatorCallExpr(anyOf(hasOverloadedOperatorName("->"),
                                  hasOverloadedOperatorName("*")),
                            unless(hasArgument(0, expr(hasOptionalType())))))));
  return llvm::None;
}

auto buildTransferMatchSwitch(
    const UncheckedOptionalAccessModelOptions &Options) {
  // FIXME: Evaluate the efficiency of matchers. If using matchers results in a
  // lot of duplicated work (e.g. string comparisons), consider providing APIs
  // that avoid it through memoization.
  auto IgnorableOptional = ignorableOptional(Options);
  return MatchSwitchBuilder<LatticeTransferState>()
      // Attach a symbolic "has_value" state to optional values that we see for
      // the first time.
      .CaseOf<Expr>(expr(anyOf(declRefExpr(), memberExpr()), hasOptionalType()),
                    initializeOptionalReference)

      // make_optional
      .CaseOf<CallExpr>(isMakeOptionalCall(), transferMakeOptionalCall)

      // optional::optional
      .CaseOf<CXXConstructExpr>(
          isOptionalInPlaceConstructor(),
          [](const CXXConstructExpr *E, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            assignOptionalValue(*E, State, State.Env.getBoolLiteralValue(true));
          })
      .CaseOf<CXXConstructExpr>(
          isOptionalNulloptConstructor(),
          [](const CXXConstructExpr *E, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            assignOptionalValue(*E, State,
                                State.Env.getBoolLiteralValue(false));
          })
      .CaseOf<CXXConstructExpr>(isOptionalValueOrConversionConstructor(),
                                transferValueOrConversionConstructor)

      // optional::operator=
      .CaseOf<CXXOperatorCallExpr>(isOptionalValueOrConversionAssignment(),
                                   transferValueOrConversionAssignment)
      .CaseOf<CXXOperatorCallExpr>(isOptionalNulloptAssignment(),
                                   transferNulloptAssignment)

      // optional::value
      .CaseOf<CXXMemberCallExpr>(
          isOptionalMemberCallWithName("value", IgnorableOptional),
          [](const CXXMemberCallExpr *E, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            transferUnwrapCall(E, E->getImplicitObjectArgument(), State);
          })

      // optional::operator*, optional::operator->
      .CaseOf<CallExpr>(
          expr(anyOf(isOptionalOperatorCallWithName("*", IgnorableOptional),
                     isOptionalOperatorCallWithName("->", IgnorableOptional))),
          [](const CallExpr *E, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            transferUnwrapCall(E, E->getArg(0), State);
          })

      // optional::has_value
      .CaseOf<CXXMemberCallExpr>(isOptionalMemberCallWithName("has_value"),
                                 transferOptionalHasValueCall)

      // optional::operator bool
      .CaseOf<CXXMemberCallExpr>(isOptionalMemberCallWithName("operator bool"),
                                 transferOptionalHasValueCall)

      // optional::emplace
      .CaseOf<CXXMemberCallExpr>(
          isOptionalMemberCallWithName("emplace"),
          [](const CXXMemberCallExpr *E, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            assignOptionalValue(*E->getImplicitObjectArgument(), State,
                                State.Env.getBoolLiteralValue(true));
          })

      // optional::reset
      .CaseOf<CXXMemberCallExpr>(
          isOptionalMemberCallWithName("reset"),
          [](const CXXMemberCallExpr *E, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            assignOptionalValue(*E->getImplicitObjectArgument(), State,
                                State.Env.getBoolLiteralValue(false));
          })

      // optional::swap
      .CaseOf<CXXMemberCallExpr>(isOptionalMemberCallWithName("swap"),
                                 transferSwapCall)

      // std::swap
      .CaseOf<CallExpr>(isStdSwapCall(), transferStdSwapCall)

      // opt.value_or("").empty()
      .CaseOf<Expr>(isValueOrStringEmptyCall(), transferValueOrStringEmptyCall)

      // opt.value_or(X) != X
      .CaseOf<Expr>(isValueOrNotEqX(), transferValueOrNotEqX)

      // returns optional
      .CaseOf<CallExpr>(isCallReturningOptional(),
                        transferCallReturningOptional)

      .Build();
}

} // namespace

ast_matchers::DeclarationMatcher
UncheckedOptionalAccessModel::optionalClassDecl() {
  return optionalClass();
}

UncheckedOptionalAccessModel::UncheckedOptionalAccessModel(
    ASTContext &Ctx, UncheckedOptionalAccessModelOptions Options)
    : DataflowAnalysis<UncheckedOptionalAccessModel, SourceLocationsLattice>(
          Ctx),
      TransferMatchSwitch(buildTransferMatchSwitch(Options)) {}

void UncheckedOptionalAccessModel::transfer(const Stmt *S,
                                            SourceLocationsLattice &L,
                                            Environment &Env) {
  LatticeTransferState State(L, Env);
  TransferMatchSwitch(*S, getASTContext(), State);
}

} // namespace dataflow
} // namespace clang
