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
#include "clang/ASTMatchers/ASTMatchersMacros.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/CFGMatchSwitch.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/Formula.h"
#include "clang/Analysis/FlowSensitive/NoopLattice.h"
#include "clang/Analysis/FlowSensitive/StorageLocation.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Basic/SourceLocation.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/ErrorHandling.h"
#include <cassert>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

namespace clang {
namespace dataflow {

static bool isTopLevelNamespaceWithName(const NamespaceDecl &NS,
                                        llvm::StringRef Name) {
  return NS.getDeclName().isIdentifier() && NS.getName() == Name &&
         NS.getParent() != nullptr && NS.getParent()->isTranslationUnit();
}

static bool hasOptionalClassName(const CXXRecordDecl &RD) {
  if (!RD.getDeclName().isIdentifier())
    return false;

  if (RD.getName() == "optional") {
    if (const auto *N = dyn_cast_or_null<NamespaceDecl>(RD.getDeclContext()))
      return N->isStdNamespace() || isTopLevelNamespaceWithName(*N, "absl");
    return false;
  }

  if (RD.getName() == "Optional") {
    // Check whether namespace is "::base" or "::folly".
    const auto *N = dyn_cast_or_null<NamespaceDecl>(RD.getDeclContext());
    return N != nullptr && (isTopLevelNamespaceWithName(*N, "base") ||
                            isTopLevelNamespaceWithName(*N, "folly"));
  }

  return false;
}

namespace {

using namespace ::clang::ast_matchers;
using LatticeTransferState = TransferState<NoopLattice>;

AST_MATCHER(CXXRecordDecl, hasOptionalClassNameMatcher) {
  return hasOptionalClassName(Node);
}

DeclarationMatcher optionalClass() {
  return classTemplateSpecializationDecl(
      hasOptionalClassNameMatcher(),
      hasTemplateArgument(0, refersToType(type().bind("T"))));
}

auto optionalOrAliasType() {
  return hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(optionalClass())));
}

/// Matches any of the spellings of the optional types and sugar, aliases, etc.
auto hasOptionalType() { return hasType(optionalOrAliasType()); }

auto isOptionalMemberCallWithNameMatcher(
    ast_matchers::internal::Matcher<NamedDecl> matcher,
    const std::optional<StatementMatcher> &Ignorable = std::nullopt) {
  auto Exception = unless(Ignorable ? expr(anyOf(*Ignorable, cxxThisExpr()))
                                    : cxxThisExpr());
  return cxxMemberCallExpr(
      on(expr(Exception,
              anyOf(hasOptionalType(),
                    hasType(pointerType(pointee(optionalOrAliasType())))))),
      callee(cxxMethodDecl(matcher)));
}

auto isOptionalOperatorCallWithName(
    llvm::StringRef operator_name,
    const std::optional<StatementMatcher> &Ignorable = std::nullopt) {
  return cxxOperatorCallExpr(
      hasOverloadedOperatorName(operator_name),
      callee(cxxMethodDecl(ofClass(optionalClass()))),
      Ignorable ? callExpr(unless(hasArgument(0, *Ignorable))) : callExpr());
}

auto isMakeOptionalCall() {
  return callExpr(callee(functionDecl(hasAnyName(
                      "std::make_optional", "base::make_optional",
                      "absl::make_optional", "folly::make_optional"))),
                  hasOptionalType());
}

auto nulloptTypeDecl() {
  return namedDecl(hasAnyName("std::nullopt_t", "absl::nullopt_t",
                              "base::nullopt_t", "folly::None"));
}

auto hasNulloptType() { return hasType(nulloptTypeDecl()); }

// `optional` or `nullopt_t`
auto hasAnyOptionalType() {
  return hasType(hasUnqualifiedDesugaredType(
      recordType(hasDeclaration(anyOf(nulloptTypeDecl(), optionalClass())))));
}

auto inPlaceClass() {
  return recordDecl(hasAnyName("std::in_place_t", "absl::in_place_t",
                               "base::in_place_t", "folly::in_place_t"));
}

auto isOptionalNulloptConstructor() {
  return cxxConstructExpr(
      hasOptionalType(),
      hasDeclaration(cxxConstructorDecl(parameterCountIs(1),
                                        hasParameter(0, hasNulloptType()))));
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

auto isNulloptConstructor() {
  return cxxConstructExpr(hasNulloptType(), argumentCountIs(1),
                          hasArgument(0, hasNulloptType()));
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

auto isStdForwardCall() {
  return callExpr(callee(functionDecl(hasName("std::forward"))),
                  argumentCountIs(1), hasArgument(0, hasOptionalType()));
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
  return callExpr(hasType(qualType(anyOf(
      optionalOrAliasType(), referenceType(pointee(optionalOrAliasType()))))));
}

template <typename L, typename R>
auto isComparisonOperatorCall(L lhs_arg_matcher, R rhs_arg_matcher) {
  return cxxOperatorCallExpr(
      anyOf(hasOverloadedOperatorName("=="), hasOverloadedOperatorName("!=")),
      argumentCountIs(2), hasArgument(0, lhs_arg_matcher),
      hasArgument(1, rhs_arg_matcher));
}

/// Ensures that `Expr` is mapped to a `BoolValue` and returns its formula.
const Formula &forceBoolValue(Environment &Env, const Expr &Expr) {
  auto *Value = cast_or_null<BoolValue>(Env.getValue(Expr, SkipPast::None));
  if (Value != nullptr)
    return Value->formula();

  auto &Loc = Env.createStorageLocation(Expr);
  Value = &Env.makeAtomicBoolValue();
  Env.setValue(Loc, *Value);
  Env.setStorageLocation(Expr, Loc);
  return Value->formula();
}

/// Sets `HasValueVal` as the symbolic value that represents the "has_value"
/// property of the optional value `OptionalVal`.
void setHasValue(Value &OptionalVal, BoolValue &HasValueVal) {
  OptionalVal.setProperty("has_value", HasValueVal);
}

/// Creates a symbolic value for an `optional` value at an existing storage
/// location. Uses `HasValueVal` as the symbolic value of the "has_value"
/// property.
StructValue &createOptionalValue(AggregateStorageLocation &Loc,
                                 BoolValue &HasValueVal, Environment &Env) {
  auto &OptionalVal = Env.create<StructValue>(Loc);
  Env.setValue(Loc, OptionalVal);
  setHasValue(OptionalVal, HasValueVal);
  return OptionalVal;
}

/// Returns the symbolic value that represents the "has_value" property of the
/// optional value `OptionalVal`. Returns null if `OptionalVal` is null.
BoolValue *getHasValue(Environment &Env, Value *OptionalVal) {
  if (OptionalVal != nullptr) {
    auto *HasValueVal =
        cast_or_null<BoolValue>(OptionalVal->getProperty("has_value"));
    if (HasValueVal == nullptr) {
      HasValueVal = &Env.makeAtomicBoolValue();
      OptionalVal->setProperty("has_value", *HasValueVal);
    }
    return HasValueVal;
  }
  return nullptr;
}

/// Returns true if and only if `Type` is an optional type.
bool isOptionalType(QualType Type) {
  if (!Type->isRecordType())
    return false;
  const CXXRecordDecl *D = Type->getAsCXXRecordDecl();
  return D != nullptr && hasOptionalClassName(*D);
}

/// Returns the number of optional wrappers in `Type`.
///
/// For example, if `Type` is `optional<optional<int>>`, the result of this
/// function will be 2.
int countOptionalWrappers(const ASTContext &ASTCtx, QualType Type) {
  if (!isOptionalType(Type))
    return 0;
  return 1 + countOptionalWrappers(
                 ASTCtx,
                 cast<ClassTemplateSpecializationDecl>(Type->getAsRecordDecl())
                     ->getTemplateArgs()
                     .get(0)
                     .getAsType()
                     .getDesugaredType(ASTCtx));
}

/// Tries to initialize the `optional`'s value (that is, contents), and return
/// its location. Returns nullptr if the value can't be represented.
StorageLocation *maybeInitializeOptionalValueMember(QualType Q,
                                                    Value &OptionalVal,
                                                    Environment &Env) {
  // The "value" property represents a synthetic field. As such, it needs
  // `StorageLocation`, like normal fields (and other variables). So, we model
  // it with a `PointerValue`, since that includes a storage location.  Once
  // the property is set, it will be shared by all environments that access the
  // `Value` representing the optional (here, `OptionalVal`).
  if (auto *ValueProp = OptionalVal.getProperty("value")) {
    auto *ValuePtr = clang::cast<PointerValue>(ValueProp);
    auto &ValueLoc = ValuePtr->getPointeeLoc();
    if (Env.getValue(ValueLoc) != nullptr)
      return &ValueLoc;

    // The property was previously set, but the value has been lost. This can
    // happen in various situations, for example:
    // - Because of an environment merge (where the two environments mapped the
    //   property to different values, which resulted in them both being
    //   discarded).
    // - When two blocks in the CFG, with neither a dominator of the other,
    //   visit the same optional value. (FIXME: This is something we can and
    //   should fix -- see also the lengthy FIXME below.)
    // - Or even when a block is revisited during testing to collect
    //   per-statement state.
    // FIXME: This situation means that the optional contents are not shared
    // between branches and the like. Practically, this lack of sharing
    // reduces the precision of the model when the contents are relevant to
    // the check, like another optional or a boolean that influences control
    // flow.
    if (ValueLoc.getType()->isRecordType()) {
      refreshStructValue(cast<AggregateStorageLocation>(ValueLoc), Env);
      return &ValueLoc;
    } else {
      auto *ValueVal = Env.createValue(ValueLoc.getType());
      if (ValueVal == nullptr)
        return nullptr;
      Env.setValue(ValueLoc, *ValueVal);
      return &ValueLoc;
    }
  }

  auto Ty = Q.getNonReferenceType();
  auto &ValueLoc = Env.createObject(Ty);
  auto &ValuePtr = Env.create<PointerValue>(ValueLoc);
  // FIXME:
  // The change we make to the `value` property below may become visible to
  // other blocks that aren't successors of the current block and therefore
  // don't see the change we made above mapping `ValueLoc` to `ValueVal`. For
  // example:
  //
  //   void target(optional<int> oo, bool b) {
  //     // `oo` is associated with a `StructValue` here, which we will call
  //     // `OptionalVal`.
  //
  //     // The `has_value` property is set on `OptionalVal` (but not the
  //     // `value` property yet).
  //     if (!oo.has_value()) return;
  //
  //     if (b) {
  //       // Let's assume we transfer the `if` branch first.
  //       //
  //       // This causes us to call `maybeInitializeOptionalValueMember()`,
  //       // which causes us to set the `value` property on `OptionalVal`
  //       // (which had not been set until this point). This `value` property
  //       // refers to a `PointerValue`, which in turn refers to a
  //       // StorageLocation` that is associated to an `IntegerValue`.
  //       oo.value();
  //     } else {
  //       // Let's assume we transfer the `else` branch after the `if` branch.
  //       //
  //       // We see the `value` property that the `if` branch set on
  //       // `OptionalVal`, but in the environment for this block, the
  //       // `StorageLocation` in the `PointerValue` is not associated with any
  //       // `Value`.
  //       oo.value();
  //     }
  //   }
  //
  // This situation is currently "saved" by the code above that checks whether
  // the `value` property is already set, and if, the `ValueLoc` is not
  // associated with a `ValueVal`, creates a new `ValueVal`.
  //
  // However, what we should really do is to make sure that the change to the
  // `value` property does not "leak" to other blocks that are not successors
  // of this block. To do this, instead of simply setting the `value` property
  // on the existing `OptionalVal`, we should create a new `Value` for the
  // optional, set the property on that, and associate the storage location that
  // is currently associated with the existing `OptionalVal` with the newly
  // created `Value` instead.
  OptionalVal.setProperty("value", ValuePtr);
  return &ValueLoc;
}

void initializeOptionalReference(const Expr *OptionalExpr,
                                 const MatchFinder::MatchResult &,
                                 LatticeTransferState &State) {
  if (auto *OptionalVal =
          State.Env.getValue(*OptionalExpr, SkipPast::Reference)) {
    if (OptionalVal->getProperty("has_value") == nullptr) {
      setHasValue(*OptionalVal, State.Env.makeAtomicBoolValue());
    }
  }
}

/// Returns true if and only if `OptionalVal` is initialized and known to be
/// empty in `Env`.
bool isEmptyOptional(const Value &OptionalVal, const Environment &Env) {
  auto *HasValueVal =
      cast_or_null<BoolValue>(OptionalVal.getProperty("has_value"));
  return HasValueVal != nullptr &&
         Env.flowConditionImplies(Env.arena().makeNot(HasValueVal->formula()));
}

/// Returns true if and only if `OptionalVal` is initialized and known to be
/// non-empty in `Env`.
bool isNonEmptyOptional(const Value &OptionalVal, const Environment &Env) {
  auto *HasValueVal =
      cast_or_null<BoolValue>(OptionalVal.getProperty("has_value"));
  return HasValueVal != nullptr &&
         Env.flowConditionImplies(HasValueVal->formula());
}

Value *getValueBehindPossiblePointer(const Expr &E, const Environment &Env) {
  Value *Val = Env.getValue(E, SkipPast::Reference);
  if (auto *PointerVal = dyn_cast_or_null<PointerValue>(Val))
    return Env.getValue(PointerVal->getPointeeLoc());
  return Val;
}

void transferUnwrapCall(const Expr *UnwrapExpr, const Expr *ObjectExpr,
                        LatticeTransferState &State) {
  if (auto *OptionalVal =
          getValueBehindPossiblePointer(*ObjectExpr, State.Env)) {
    if (State.Env.getStorageLocation(*UnwrapExpr, SkipPast::None) == nullptr)
      if (auto *Loc = maybeInitializeOptionalValueMember(
              UnwrapExpr->getType(), *OptionalVal, State.Env))
        State.Env.setStorageLocation(*UnwrapExpr, *Loc);
  }
}

void transferArrowOpCall(const Expr *UnwrapExpr, const Expr *ObjectExpr,
                         LatticeTransferState &State) {
  if (auto *OptionalVal =
          getValueBehindPossiblePointer(*ObjectExpr, State.Env)) {
    if (auto *Loc = maybeInitializeOptionalValueMember(
            UnwrapExpr->getType()->getPointeeType(), *OptionalVal, State.Env)) {
      State.Env.setValueStrict(*UnwrapExpr,
                               State.Env.create<PointerValue>(*Loc));
    }
  }
}

void transferMakeOptionalCall(const CallExpr *E,
                              const MatchFinder::MatchResult &,
                              LatticeTransferState &State) {
  createOptionalValue(State.Env.getResultObjectLocation(*E),
                      State.Env.getBoolLiteralValue(true), State.Env);
}

void transferOptionalHasValueCall(const CXXMemberCallExpr *CallExpr,
                                  const MatchFinder::MatchResult &,
                                  LatticeTransferState &State) {
  if (auto *HasValueVal = getHasValue(
          State.Env, getValueBehindPossiblePointer(
                         *CallExpr->getImplicitObjectArgument(), State.Env))) {
    auto &CallExprLoc = State.Env.createStorageLocation(*CallExpr);
    State.Env.setValue(CallExprLoc, *HasValueVal);
    State.Env.setStorageLocation(*CallExpr, CallExprLoc);
  }
}

/// `ModelPred` builds a logical formula relating the predicate in
/// `ValueOrPredExpr` to the optional's `has_value` property.
void transferValueOrImpl(
    const clang::Expr *ValueOrPredExpr, const MatchFinder::MatchResult &Result,
    LatticeTransferState &State,
    const Formula &(*ModelPred)(Environment &Env, const Formula &ExprVal,
                                const Formula &HasValueVal)) {
  auto &Env = State.Env;

  const auto *ObjectArgumentExpr =
      Result.Nodes.getNodeAs<clang::CXXMemberCallExpr>(ValueOrCallID)
          ->getImplicitObjectArgument();

  auto *HasValueVal = getHasValue(
      State.Env, getValueBehindPossiblePointer(*ObjectArgumentExpr, State.Env));
  if (HasValueVal == nullptr)
    return;

  Env.addToFlowCondition(ModelPred(Env, forceBoolValue(Env, *ValueOrPredExpr),
                                   HasValueVal->formula()));
}

void transferValueOrStringEmptyCall(const clang::Expr *ComparisonExpr,
                                    const MatchFinder::MatchResult &Result,
                                    LatticeTransferState &State) {
  return transferValueOrImpl(ComparisonExpr, Result, State,
                             [](Environment &Env, const Formula &ExprVal,
                                const Formula &HasValueVal) -> const Formula & {
                               auto &A = Env.arena();
                               // If the result is *not* empty, then we know the
                               // optional must have been holding a value. If
                               // `ExprVal` is true, though, we don't learn
                               // anything definite about `has_value`, so we
                               // don't add any corresponding implications to
                               // the flow condition.
                               return A.makeImplies(A.makeNot(ExprVal),
                                                    HasValueVal);
                             });
}

void transferValueOrNotEqX(const Expr *ComparisonExpr,
                           const MatchFinder::MatchResult &Result,
                           LatticeTransferState &State) {
  transferValueOrImpl(ComparisonExpr, Result, State,
                      [](Environment &Env, const Formula &ExprVal,
                         const Formula &HasValueVal) -> const Formula & {
                        auto &A = Env.arena();
                        // We know that if `(opt.value_or(X) != X)` then
                        // `opt.hasValue()`, even without knowing further
                        // details about the contents of `opt`.
                        return A.makeImplies(ExprVal, HasValueVal);
                      });
}

void transferCallReturningOptional(const CallExpr *E,
                                   const MatchFinder::MatchResult &Result,
                                   LatticeTransferState &State) {
  if (State.Env.getStorageLocation(*E, SkipPast::None) != nullptr)
    return;

  AggregateStorageLocation *Loc = nullptr;
  if (E->isPRValue()) {
    Loc = &State.Env.getResultObjectLocation(*E);
  } else {
    Loc = &cast<AggregateStorageLocation>(State.Env.createStorageLocation(*E));
    State.Env.setStorageLocationStrict(*E, *Loc);
  }

  createOptionalValue(*Loc, State.Env.makeAtomicBoolValue(), State.Env);
}

void constructOptionalValue(const Expr &E, Environment &Env,
                            BoolValue &HasValueVal) {
  AggregateStorageLocation &Loc = Env.getResultObjectLocation(E);
  Env.setValueStrict(E, createOptionalValue(Loc, HasValueVal, Env));
}

/// Returns a symbolic value for the "has_value" property of an `optional<T>`
/// value that is constructed/assigned from a value of type `U` or `optional<U>`
/// where `T` is constructible from `U`.
BoolValue &valueOrConversionHasValue(const FunctionDecl &F, const Expr &E,
                                     const MatchFinder::MatchResult &MatchRes,
                                     LatticeTransferState &State) {
  assert(F.getTemplateSpecializationArgs() != nullptr);
  assert(F.getTemplateSpecializationArgs()->size() > 0);

  const int TemplateParamOptionalWrappersCount =
      countOptionalWrappers(*MatchRes.Context, F.getTemplateSpecializationArgs()
                                                   ->get(0)
                                                   .getAsType()
                                                   .getNonReferenceType());
  const int ArgTypeOptionalWrappersCount = countOptionalWrappers(
      *MatchRes.Context, E.getType().getNonReferenceType());

  // Check if this is a constructor/assignment call for `optional<T>` with
  // argument of type `U` such that `T` is constructible from `U`.
  if (TemplateParamOptionalWrappersCount == ArgTypeOptionalWrappersCount)
    return State.Env.getBoolLiteralValue(true);

  // This is a constructor/assignment call for `optional<T>` with argument of
  // type `optional<U>` such that `T` is constructible from `U`.
  if (auto *HasValueVal =
          getHasValue(State.Env, State.Env.getValue(E, SkipPast::Reference)))
    return *HasValueVal;
  return State.Env.makeAtomicBoolValue();
}

void transferValueOrConversionConstructor(
    const CXXConstructExpr *E, const MatchFinder::MatchResult &MatchRes,
    LatticeTransferState &State) {
  assert(E->getNumArgs() > 0);

  constructOptionalValue(*E, State.Env,
                         valueOrConversionHasValue(*E->getConstructor(),
                                                   *E->getArg(0), MatchRes,
                                                   State));
}

void transferAssignment(const CXXOperatorCallExpr *E, BoolValue &HasValueVal,
                        LatticeTransferState &State) {
  assert(E->getNumArgs() > 0);

  if (auto *Loc = cast<AggregateStorageLocation>(
          State.Env.getStorageLocationStrict(*E->getArg(0)))) {
    createOptionalValue(*Loc, HasValueVal, State.Env);

    // Assign a storage location for the whole expression.
    State.Env.setStorageLocationStrict(*E, *Loc);
  }
}

void transferValueOrConversionAssignment(
    const CXXOperatorCallExpr *E, const MatchFinder::MatchResult &MatchRes,
    LatticeTransferState &State) {
  assert(E->getNumArgs() > 1);
  transferAssignment(E,
                     valueOrConversionHasValue(*E->getDirectCallee(),
                                               *E->getArg(1), MatchRes, State),
                     State);
}

void transferNulloptAssignment(const CXXOperatorCallExpr *E,
                               const MatchFinder::MatchResult &,
                               LatticeTransferState &State) {
  transferAssignment(E, State.Env.getBoolLiteralValue(false), State);
}

void transferSwap(AggregateStorageLocation *Loc1,
                  AggregateStorageLocation *Loc2, Environment &Env) {
  // We account for cases where one or both of the optionals are not modeled,
  // either lacking associated storage locations, or lacking values associated
  // to such storage locations.

  if (Loc1 == nullptr) {
    if (Loc2 != nullptr)
      createOptionalValue(*Loc2, Env.makeAtomicBoolValue(), Env);
    return;
  }
  if (Loc2 == nullptr) {
    createOptionalValue(*Loc1, Env.makeAtomicBoolValue(), Env);
    return;
  }

  // Both expressions have locations, though they may not have corresponding
  // values. In that case, we create a fresh value at this point. Note that if
  // two branches both do this, they will not share the value, but it at least
  // allows for local reasoning about the value. To avoid the above, we would
  // need *lazy* value allocation.
  // FIXME: allocate values lazily, instead of just creating a fresh value.
  BoolValue *BoolVal1 = getHasValue(Env, Env.getValue(*Loc1));
  if (BoolVal1 == nullptr)
    BoolVal1 = &Env.makeAtomicBoolValue();

  BoolValue *BoolVal2 = getHasValue(Env, Env.getValue(*Loc2));
  if (BoolVal2 == nullptr)
    BoolVal2 = &Env.makeAtomicBoolValue();

  createOptionalValue(*Loc1, *BoolVal2, Env);
  createOptionalValue(*Loc2, *BoolVal1, Env);
}

void transferSwapCall(const CXXMemberCallExpr *E,
                      const MatchFinder::MatchResult &,
                      LatticeTransferState &State) {
  assert(E->getNumArgs() == 1);
  auto *OtherLoc = cast_or_null<AggregateStorageLocation>(
      State.Env.getStorageLocationStrict(*E->getArg(0)));
  transferSwap(getImplicitObjectLocation(*E, State.Env), OtherLoc, State.Env);
}

void transferStdSwapCall(const CallExpr *E, const MatchFinder::MatchResult &,
                         LatticeTransferState &State) {
  assert(E->getNumArgs() == 2);
  auto *Arg0Loc = cast_or_null<AggregateStorageLocation>(
      State.Env.getStorageLocationStrict(*E->getArg(0)));
  auto *Arg1Loc = cast_or_null<AggregateStorageLocation>(
      State.Env.getStorageLocationStrict(*E->getArg(1)));
  transferSwap(Arg0Loc, Arg1Loc, State.Env);
}

void transferStdForwardCall(const CallExpr *E, const MatchFinder::MatchResult &,
                            LatticeTransferState &State) {
  assert(E->getNumArgs() == 1);

  if (auto *Loc = State.Env.getStorageLocationStrict(*E->getArg(0)))
    State.Env.setStorageLocationStrict(*E, *Loc);
}

const Formula &evaluateEquality(Arena &A, const Formula &EqVal,
                                const Formula &LHS, const Formula &RHS) {
  // Logically, an optional<T> object is composed of two values - a `has_value`
  // bit and a value of type T. Equality of optional objects compares both
  // values. Therefore, merely comparing the `has_value` bits isn't sufficient:
  // when two optional objects are engaged, the equality of their respective
  // values of type T matters. Since we only track the `has_value` bits, we
  // can't make any conclusions about equality when we know that two optional
  // objects are engaged.
  //
  // We express this as two facts about the equality:
  // a) EqVal => (LHS & RHS) v (!RHS & !LHS)
  //    If they are equal, then either both are set or both are unset.
  // b) (!LHS & !RHS) => EqVal
  //    If neither is set, then they are equal.
  // We rewrite b) as !EqVal => (LHS v RHS), for a more compact formula.
  return A.makeAnd(
      A.makeImplies(EqVal, A.makeOr(A.makeAnd(LHS, RHS),
                                    A.makeAnd(A.makeNot(LHS), A.makeNot(RHS)))),
      A.makeImplies(A.makeNot(EqVal), A.makeOr(LHS, RHS)));
}

void transferOptionalAndOptionalCmp(const clang::CXXOperatorCallExpr *CmpExpr,
                                    const MatchFinder::MatchResult &,
                                    LatticeTransferState &State) {
  Environment &Env = State.Env;
  auto &A = Env.arena();
  auto *CmpValue = &forceBoolValue(Env, *CmpExpr);
  if (auto *LHasVal = getHasValue(
          Env, Env.getValue(*CmpExpr->getArg(0), SkipPast::Reference)))
    if (auto *RHasVal = getHasValue(
            Env, Env.getValue(*CmpExpr->getArg(1), SkipPast::Reference))) {
      if (CmpExpr->getOperator() == clang::OO_ExclaimEqual)
        CmpValue = &A.makeNot(*CmpValue);
      Env.addToFlowCondition(evaluateEquality(A, *CmpValue, LHasVal->formula(),
                                              RHasVal->formula()));
    }
}

void transferOptionalAndValueCmp(const clang::CXXOperatorCallExpr *CmpExpr,
                                 const clang::Expr *E, Environment &Env) {
  auto &A = Env.arena();
  auto *CmpValue = &forceBoolValue(Env, *CmpExpr);
  if (auto *HasVal = getHasValue(Env, Env.getValue(*E, SkipPast::Reference))) {
    if (CmpExpr->getOperator() == clang::OO_ExclaimEqual)
      CmpValue = &A.makeNot(*CmpValue);
    Env.addToFlowCondition(
        evaluateEquality(A, *CmpValue, HasVal->formula(), A.makeLiteral(true)));
  }
}

std::optional<StatementMatcher>
ignorableOptional(const UncheckedOptionalAccessModelOptions &Options) {
  if (Options.IgnoreSmartPointerDereference) {
    auto SmartPtrUse = expr(ignoringParenImpCasts(cxxOperatorCallExpr(
        anyOf(hasOverloadedOperatorName("->"), hasOverloadedOperatorName("*")),
        unless(hasArgument(0, expr(hasOptionalType()))))));
    return expr(
        anyOf(SmartPtrUse, memberExpr(hasObjectExpression(SmartPtrUse))));
  }
  return std::nullopt;
}

StatementMatcher
valueCall(const std::optional<StatementMatcher> &IgnorableOptional) {
  return isOptionalMemberCallWithNameMatcher(hasName("value"),
                                             IgnorableOptional);
}

StatementMatcher
valueOperatorCall(const std::optional<StatementMatcher> &IgnorableOptional) {
  return expr(anyOf(isOptionalOperatorCallWithName("*", IgnorableOptional),
                    isOptionalOperatorCallWithName("->", IgnorableOptional)));
}

auto buildTransferMatchSwitch() {
  // FIXME: Evaluate the efficiency of matchers. If using matchers results in a
  // lot of duplicated work (e.g. string comparisons), consider providing APIs
  // that avoid it through memoization.
  return CFGMatchSwitchBuilder<LatticeTransferState>()
      // Attach a symbolic "has_value" state to optional values that we see for
      // the first time.
      .CaseOfCFGStmt<Expr>(
          expr(anyOf(declRefExpr(), memberExpr()), hasOptionalType()),
          initializeOptionalReference)

      // make_optional
      .CaseOfCFGStmt<CallExpr>(isMakeOptionalCall(), transferMakeOptionalCall)

      // optional::optional (in place)
      .CaseOfCFGStmt<CXXConstructExpr>(
          isOptionalInPlaceConstructor(),
          [](const CXXConstructExpr *E, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            constructOptionalValue(*E, State.Env,
                                   State.Env.getBoolLiteralValue(true));
          })
      // nullopt_t::nullopt_t
      .CaseOfCFGStmt<CXXConstructExpr>(
          isNulloptConstructor(),
          [](const CXXConstructExpr *E, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            constructOptionalValue(*E, State.Env,
                                   State.Env.getBoolLiteralValue(false));
          })
      // optional::optional(nullopt_t)
      .CaseOfCFGStmt<CXXConstructExpr>(
          isOptionalNulloptConstructor(),
          [](const CXXConstructExpr *E, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            constructOptionalValue(*E, State.Env,
                                   State.Env.getBoolLiteralValue(false));
          })
      // optional::optional (value/conversion)
      .CaseOfCFGStmt<CXXConstructExpr>(isOptionalValueOrConversionConstructor(),
                                       transferValueOrConversionConstructor)

      // optional::operator=
      .CaseOfCFGStmt<CXXOperatorCallExpr>(
          isOptionalValueOrConversionAssignment(),
          transferValueOrConversionAssignment)
      .CaseOfCFGStmt<CXXOperatorCallExpr>(isOptionalNulloptAssignment(),
                                          transferNulloptAssignment)

      // optional::value
      .CaseOfCFGStmt<CXXMemberCallExpr>(
          valueCall(std::nullopt),
          [](const CXXMemberCallExpr *E, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            transferUnwrapCall(E, E->getImplicitObjectArgument(), State);
          })

      // optional::operator*
      .CaseOfCFGStmt<CallExpr>(isOptionalOperatorCallWithName("*"),
                               [](const CallExpr *E,
                                  const MatchFinder::MatchResult &,
                                  LatticeTransferState &State) {
                                 transferUnwrapCall(E, E->getArg(0), State);
                               })

      // optional::operator->
      .CaseOfCFGStmt<CallExpr>(isOptionalOperatorCallWithName("->"),
                               [](const CallExpr *E,
                                  const MatchFinder::MatchResult &,
                                  LatticeTransferState &State) {
                                 transferArrowOpCall(E, E->getArg(0), State);
                               })

      // optional::has_value, optional::hasValue
      // Of the supported optionals only folly::Optional uses hasValue, but this
      // will also pass for other types
      .CaseOfCFGStmt<CXXMemberCallExpr>(
          isOptionalMemberCallWithNameMatcher(
              hasAnyName("has_value", "hasValue")),
          transferOptionalHasValueCall)

      // optional::operator bool
      .CaseOfCFGStmt<CXXMemberCallExpr>(
          isOptionalMemberCallWithNameMatcher(hasName("operator bool")),
          transferOptionalHasValueCall)

      // optional::emplace
      .CaseOfCFGStmt<CXXMemberCallExpr>(
          isOptionalMemberCallWithNameMatcher(hasName("emplace")),
          [](const CXXMemberCallExpr *E, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            if (AggregateStorageLocation *Loc =
                    getImplicitObjectLocation(*E, State.Env)) {
              createOptionalValue(*Loc, State.Env.getBoolLiteralValue(true),
                                  State.Env);
            }
          })

      // optional::reset
      .CaseOfCFGStmt<CXXMemberCallExpr>(
          isOptionalMemberCallWithNameMatcher(hasName("reset")),
          [](const CXXMemberCallExpr *E, const MatchFinder::MatchResult &,
             LatticeTransferState &State) {
            if (AggregateStorageLocation *Loc =
                    getImplicitObjectLocation(*E, State.Env)) {
              createOptionalValue(*Loc, State.Env.getBoolLiteralValue(false),
                                  State.Env);
            }
          })

      // optional::swap
      .CaseOfCFGStmt<CXXMemberCallExpr>(
          isOptionalMemberCallWithNameMatcher(hasName("swap")),
          transferSwapCall)

      // std::swap
      .CaseOfCFGStmt<CallExpr>(isStdSwapCall(), transferStdSwapCall)

      // std::forward
      .CaseOfCFGStmt<CallExpr>(isStdForwardCall(), transferStdForwardCall)

      // opt.value_or("").empty()
      .CaseOfCFGStmt<Expr>(isValueOrStringEmptyCall(),
                           transferValueOrStringEmptyCall)

      // opt.value_or(X) != X
      .CaseOfCFGStmt<Expr>(isValueOrNotEqX(), transferValueOrNotEqX)

      // Comparisons (==, !=):
      .CaseOfCFGStmt<CXXOperatorCallExpr>(
          isComparisonOperatorCall(hasAnyOptionalType(), hasAnyOptionalType()),
          transferOptionalAndOptionalCmp)
      .CaseOfCFGStmt<CXXOperatorCallExpr>(
          isComparisonOperatorCall(hasOptionalType(),
                                   unless(hasAnyOptionalType())),
          [](const clang::CXXOperatorCallExpr *Cmp,
             const MatchFinder::MatchResult &, LatticeTransferState &State) {
            transferOptionalAndValueCmp(Cmp, Cmp->getArg(0), State.Env);
          })
      .CaseOfCFGStmt<CXXOperatorCallExpr>(
          isComparisonOperatorCall(unless(hasAnyOptionalType()),
                                   hasOptionalType()),
          [](const clang::CXXOperatorCallExpr *Cmp,
             const MatchFinder::MatchResult &, LatticeTransferState &State) {
            transferOptionalAndValueCmp(Cmp, Cmp->getArg(1), State.Env);
          })

      // returns optional
      .CaseOfCFGStmt<CallExpr>(isCallReturningOptional(),
                               transferCallReturningOptional)

      .Build();
}

std::vector<SourceLocation> diagnoseUnwrapCall(const Expr *ObjectExpr,
                                               const Environment &Env) {
  if (auto *OptionalVal = getValueBehindPossiblePointer(*ObjectExpr, Env)) {
    auto *Prop = OptionalVal->getProperty("has_value");
    if (auto *HasValueVal = cast_or_null<BoolValue>(Prop)) {
      if (Env.flowConditionImplies(HasValueVal->formula()))
        return {};
    }
  }

  // Record that this unwrap is *not* provably safe.
  // FIXME: include either the name of the optional (if applicable) or a source
  // range of the access for easier interpretation of the result.
  return {ObjectExpr->getBeginLoc()};
}

auto buildDiagnoseMatchSwitch(
    const UncheckedOptionalAccessModelOptions &Options) {
  // FIXME: Evaluate the efficiency of matchers. If using matchers results in a
  // lot of duplicated work (e.g. string comparisons), consider providing APIs
  // that avoid it through memoization.
  auto IgnorableOptional = ignorableOptional(Options);
  return CFGMatchSwitchBuilder<const Environment, std::vector<SourceLocation>>()
      // optional::value
      .CaseOfCFGStmt<CXXMemberCallExpr>(
          valueCall(IgnorableOptional),
          [](const CXXMemberCallExpr *E, const MatchFinder::MatchResult &,
             const Environment &Env) {
            return diagnoseUnwrapCall(E->getImplicitObjectArgument(), Env);
          })

      // optional::operator*, optional::operator->
      .CaseOfCFGStmt<CallExpr>(valueOperatorCall(IgnorableOptional),
                               [](const CallExpr *E,
                                  const MatchFinder::MatchResult &,
                                  const Environment &Env) {
                                 return diagnoseUnwrapCall(E->getArg(0), Env);
                               })
      .Build();
}

} // namespace

ast_matchers::DeclarationMatcher
UncheckedOptionalAccessModel::optionalClassDecl() {
  return optionalClass();
}

UncheckedOptionalAccessModel::UncheckedOptionalAccessModel(ASTContext &Ctx)
    : DataflowAnalysis<UncheckedOptionalAccessModel, NoopLattice>(Ctx),
      TransferMatchSwitch(buildTransferMatchSwitch()) {}

void UncheckedOptionalAccessModel::transfer(const CFGElement &Elt,
                                            NoopLattice &L, Environment &Env) {
  LatticeTransferState State(L, Env);
  TransferMatchSwitch(Elt, getASTContext(), State);
}

ComparisonResult UncheckedOptionalAccessModel::compare(
    QualType Type, const Value &Val1, const Environment &Env1,
    const Value &Val2, const Environment &Env2) {
  if (!isOptionalType(Type))
    return ComparisonResult::Unknown;
  bool MustNonEmpty1 = isNonEmptyOptional(Val1, Env1);
  bool MustNonEmpty2 = isNonEmptyOptional(Val2, Env2);
  if (MustNonEmpty1 && MustNonEmpty2)
    return ComparisonResult::Same;
  // If exactly one is true, then they're different, no reason to check whether
  // they're definitely empty.
  if (MustNonEmpty1 || MustNonEmpty2)
    return ComparisonResult::Different;
  // Check if they're both definitely empty.
  return (isEmptyOptional(Val1, Env1) && isEmptyOptional(Val2, Env2))
             ? ComparisonResult::Same
             : ComparisonResult::Different;
}

bool UncheckedOptionalAccessModel::merge(QualType Type, const Value &Val1,
                                         const Environment &Env1,
                                         const Value &Val2,
                                         const Environment &Env2,
                                         Value &MergedVal,
                                         Environment &MergedEnv) {
  if (!isOptionalType(Type))
    return true;
  // FIXME: uses same approach as join for `BoolValues`. Requires non-const
  // values, though, so will require updating the interface.
  auto &HasValueVal = MergedEnv.makeAtomicBoolValue();
  bool MustNonEmpty1 = isNonEmptyOptional(Val1, Env1);
  bool MustNonEmpty2 = isNonEmptyOptional(Val2, Env2);
  if (MustNonEmpty1 && MustNonEmpty2)
    MergedEnv.addToFlowCondition(HasValueVal.formula());
  else if (
      // Only make the costly calls to `isEmptyOptional` if we got "unknown"
      // (false) for both calls to `isNonEmptyOptional`.
      !MustNonEmpty1 && !MustNonEmpty2 && isEmptyOptional(Val1, Env1) &&
      isEmptyOptional(Val2, Env2))
    MergedEnv.addToFlowCondition(
        MergedEnv.arena().makeNot(HasValueVal.formula()));
  setHasValue(MergedVal, HasValueVal);
  return true;
}

Value *UncheckedOptionalAccessModel::widen(QualType Type, Value &Prev,
                                           const Environment &PrevEnv,
                                           Value &Current,
                                           Environment &CurrentEnv) {
  switch (compare(Type, Prev, PrevEnv, Current, CurrentEnv)) {
  case ComparisonResult::Same:
    return &Prev;
  case ComparisonResult::Different:
    if (auto *PrevHasVal =
            cast_or_null<BoolValue>(Prev.getProperty("has_value"))) {
      if (isa<TopBoolValue>(PrevHasVal))
        return &Prev;
    }
    if (auto *CurrentHasVal =
            cast_or_null<BoolValue>(Current.getProperty("has_value"))) {
      if (isa<TopBoolValue>(CurrentHasVal))
        return &Current;
    }
    return &createOptionalValue(cast<StructValue>(Current).getAggregateLoc(),
                                CurrentEnv.makeTopBoolValue(), CurrentEnv);
  case ComparisonResult::Unknown:
    return nullptr;
  }
  llvm_unreachable("all cases covered in switch");
}

UncheckedOptionalAccessDiagnoser::UncheckedOptionalAccessDiagnoser(
    UncheckedOptionalAccessModelOptions Options)
    : DiagnoseMatchSwitch(buildDiagnoseMatchSwitch(Options)) {}

std::vector<SourceLocation> UncheckedOptionalAccessDiagnoser::diagnose(
    ASTContext &Ctx, const CFGElement *Elt, const Environment &Env) {
  return DiagnoseMatchSwitch(*Elt, Ctx, Env);
}

} // namespace dataflow
} // namespace clang
