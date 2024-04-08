//===-- NullPointerAnalysisModel.cpp ----------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines a generic null-pointer analysis model, used for finding
// pointer null-checks after the pointer has already been dereferenced.
//
// Only a limited set of operations are currently recognized. Notably, pointer
// arithmetic, null-pointer assignments and _nullable/_nonnull attributes are
// missing as of yet.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/FlowSensitive/Models/NullPointerAnalysisModel.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/Decl.h"
#include "clang/AST/Expr.h"
#include "clang/AST/Stmt.h"
#include "clang/ASTMatchers/ASTMatchFinder.h"
#include "clang/ASTMatchers/ASTMatchers.h"
#include "clang/Analysis/CFG.h"
#include "clang/Analysis/FlowSensitive/CFGMatchSwitch.h"
#include "clang/Analysis/FlowSensitive/DataflowAnalysis.h"
#include "clang/Analysis/FlowSensitive/DataflowEnvironment.h"
#include "clang/Analysis/FlowSensitive/DataflowLattice.h"
#include "clang/Analysis/FlowSensitive/MapLattice.h"
#include "clang/Analysis/FlowSensitive/NoopLattice.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"

namespace clang::dataflow {

namespace {
using namespace ast_matchers;

constexpr char kCond[] = "condition";
constexpr char kVar[] = "var";
constexpr char kValue[] = "value";
constexpr char kIsNonnull[] = "is-nonnull";
constexpr char kIsNull[] = "is-null";

enum class SatisfiabilityResult {
  // Returned when the value was not initialized yet.
  Nullptr,
  // Special value that signals that the boolean value can be anything.
  // It signals that the underlying formulas are too complex to be calculated
  // efficiently.
  Top,
  // Equivalent to the literal True in the current environment.
  True,
  // Equivalent to the literal False in the current environment.
  False,
  // Both True and False values could be produced with an appropriate set of
  // conditions.
  Unknown
};

using SR = SatisfiabilityResult;

// FIXME: These AST matchers should also be exported via the
// NullPointerAnalysisModel class, for tests
auto ptrWithBinding(llvm::StringRef VarName = kVar) {
  return traverse(TK_IgnoreUnlessSpelledInSource,
      expr(hasType(isAnyPointer())).bind(VarName));
}

auto derefMatcher() {
  return unaryOperator(hasOperatorName("*"), hasUnaryOperand(ptrWithBinding()));
}

auto arrowMatcher() {
  return memberExpr(allOf(isArrow(), hasObjectExpression(ptrWithBinding())));
}

auto castExprMatcher() {
  return castExpr(hasCastKind(CK_PointerToBoolean),
                  hasSourceExpression(ptrWithBinding()))
      .bind(kCond);
}

auto nullptrMatcher(llvm::StringRef VarName = kVar) {
  return castExpr(hasCastKind(CK_NullToPointer)).bind(VarName);
}

auto addressofMatcher() {
  return unaryOperator(hasOperatorName("&")).bind(kVar);
}

auto functionCallMatcher() {
  return callExpr(hasDeclaration(functionDecl(returns(isAnyPointer()))))
      .bind(kVar);
}

auto assignMatcher() {
  return binaryOperation(isAssignmentOperator(), hasLHS(ptrWithBinding()),
                         hasRHS(expr().bind(kValue)));
}

auto nullCheckExprMatcher() {
  return binaryOperator(hasAnyOperatorName("==", "!="),
                        hasOperands(ptrWithBinding(), nullptrMatcher(kValue)));
}

// FIXME: When TK_IgnoreUnlessSpelledInSource is removed from ptrWithBinding(),
// this matcher should be merged with nullCheckExprMatcher().
auto equalExprMatcher() {
  return binaryOperator(hasAnyOperatorName("==", "!="),
                        hasOperands(ptrWithBinding(kVar),
                                    ptrWithBinding(kValue)));
}

auto anyPointerMatcher() { return expr(hasType(isAnyPointer())).bind(kVar); }

// Only computes simple comparisons against the literals True and False in the
// environment. Faster, but produces many Unknown values.
SatisfiabilityResult shallowComputeSatisfiability(BoolValue *Val,
                                                  const Environment &Env) {
  if (!Val)
    return SR::Nullptr;

  if (isa<TopBoolValue>(Val))
    return SR::Top;

  if (Val == &Env.getBoolLiteralValue(true))
    return SR::True;

  if (Val == &Env.getBoolLiteralValue(false))
    return SR::False;

  return SR::Unknown;
}

// Computes satisfiability by using the flow condition. Slower, but more
// precise.
SatisfiabilityResult computeSatisfiability(BoolValue *Val,
                                           const Environment &Env) {
  // Early return with the fast compute values.
  if (SatisfiabilityResult ShallowResult =
          shallowComputeSatisfiability(Val, Env);
      ShallowResult != SR::Unknown) {
    return ShallowResult;
  }

  if (Env.proves(Val->formula()))
    return SR::True;

  if (Env.proves(Env.arena().makeNot(Val->formula())))
    return SR::False;

  return SR::Unknown;
}

inline BoolValue &getVal(llvm::StringRef Name, Value &RootValue) {
  return *cast<BoolValue>(RootValue.getProperty(Name));
}

// Assigns initial pointer null- and nonnull-values to a given Value.
void initializeRootValue(Value &RootValue, Environment &Env) {
  Arena &A = Env.arena();

  auto *IsNull = cast_or_null<BoolValue>(RootValue.getProperty(kIsNull));
  auto *IsNonnull = cast_or_null<BoolValue>(RootValue.getProperty(kIsNonnull));

  if (!IsNull) {
    IsNull = &A.makeAtomValue();
    RootValue.setProperty(kIsNull, *IsNull);
  }

  if (!IsNonnull) {
    IsNonnull = &A.makeAtomValue();
    RootValue.setProperty(kIsNonnull, *IsNonnull);
  }

  // If the pointer cannot have either a null or nonnull value, the state is
  // unreachable.
  // FIXME: This condition is added in all cases when getValue() is called.
  // The reason is that on a post-visit step, the initialized Values are used,
  // but the flow condition does not have this constraint yet.
  // The framework provides deduplication for constraints, so should not have a
  // performance impact.
  Env.assume(A.makeOr(IsNull->formula(), IsNonnull->formula()));
}

void setGLValue(const Expr &E, Value &Val, Environment &Env) {
  StorageLocation *SL = Env.getStorageLocation(E);
  if (!SL) {
    SL = &Env.createStorageLocation(E);
    Env.setStorageLocation(E, *SL);
  }

  Env.setValue(*SL, Val);
}

void setUnknownValue(const Expr &E, Value &Val, Environment &Env) {
  if (E.isGLValue())
    setGLValue(E, Val, Env);
  else
    Env.setValue(E, Val);
}

Value *getValue(const Expr &Var, Environment &Env) {
  if (Value *EnvVal = Env.getValue(Var)) {
    // FIXME: The framework usually creates the values for us, but without the
    // null-properties.
    initializeRootValue(*EnvVal, Env);

    return EnvVal;
  }

  Value *RootValue = Env.createValue(Var.getType());

  initializeRootValue(*RootValue, Env);

  setGLValue(Var, *RootValue, Env);

  return RootValue;
}

void matchDereferenceExpr(const Stmt *stmt,
                          const MatchFinder::MatchResult &Result,
                          Environment &Env) {
  const auto *Var = Result.Nodes.getNodeAs<Expr>(kVar);
  assert(Var != nullptr);

  Value *RootValue = getValue(*Var, Env);

  Env.assume(Env.arena().makeNot(getVal(kIsNull, *RootValue).formula()));
}

void matchNullCheckExpr(const Expr *NullCheck,
                    const MatchFinder::MatchResult &Result,
                    Environment &Env) {
  const auto *Var = Result.Nodes.getNodeAs<Expr>(kVar);
  assert(Var != nullptr);

  // (bool)p or (p != nullptr)
  bool IsNonnullOp = true;
  if (auto *BinOp = dyn_cast<BinaryOperator>(NullCheck);
      BinOp->getOpcode() == BO_EQ) {
    IsNonnullOp = false;
  }

  Value *RootValue = getValue(*Var, Env);

  Arena &A = Env.arena();
  BoolValue &IsNonnull = getVal(kIsNonnull, *RootValue);
  BoolValue &IsNull = getVal(kIsNull, *RootValue);
  BoolValue *CondValue = cast_or_null<BoolValue>(Env.getValue(*NullCheck));
  if (!CondValue) {
    CondValue = &A.makeAtomValue();
    Env.setValue(*NullCheck, *CondValue);
  }
  const Formula &CondFormula = IsNonnullOp ? CondValue->formula()
                                             : A.makeNot(CondValue->formula());

  Env.assume(A.makeImplies(CondFormula, A.makeNot(IsNull.formula())));
  Env.assume(A.makeImplies(A.makeNot(CondFormula),
                           A.makeNot(IsNonnull.formula())));
}

void matchEqualExpr(const BinaryOperator *EqualExpr,
                    const MatchFinder::MatchResult &Result,
                    Environment &Env) {
  bool IsNotEqualsOp = EqualExpr->getOpcode() == BO_NE;

  const auto *LHSVar = Result.Nodes.getNodeAs<Expr>(kVar);
  assert(LHSVar != nullptr);

  const auto *RHSVar = Result.Nodes.getNodeAs<Expr>(kValue);
  assert(RHSVar != nullptr);

  Arena &A = Env.arena();
  Value *LHSValue = getValue(*LHSVar, Env);
  Value *RHSValue = getValue(*RHSVar, Env);

  BoolValue *CondValue = cast_or_null<BoolValue>(Env.getValue(*EqualExpr));
  if (!CondValue) {
    CondValue = &A.makeAtomValue();
    Env.setValue(*EqualExpr, *CondValue);
  }

  const Formula &CondFormula = IsNotEqualsOp ? A.makeNot(CondValue->formula())
                                       : CondValue->formula();

  // If the pointers are equal, the nullability properties are the same.
  Env.assume(A.makeImplies(CondFormula, 
      A.makeAnd(A.makeEquals(getVal(kIsNull, *LHSValue).formula(),
                             getVal(kIsNull, *RHSValue).formula()),
                A.makeEquals(getVal(kIsNonnull, *LHSValue).formula(),
                             getVal(kIsNonnull, *RHSValue).formula()))));

  // If the pointers are not equal, at most one of the pointers is null.
  Env.assume(A.makeImplies(A.makeNot(CondFormula),
      A.makeNot(A.makeAnd(getVal(kIsNull, *LHSValue).formula(),
                          getVal(kIsNull, *RHSValue).formula()))));
}

void matchNullptrExpr(const Expr *expr, const MatchFinder::MatchResult &Result,
                      Environment &Env) {
  const auto *PrVar = Result.Nodes.getNodeAs<Expr>(kVar);
  assert(PrVar != nullptr);

  Value *RootValue = Env.getValue(*PrVar);
  if (!RootValue) {
    RootValue = Env.createValue(PrVar->getType());
    Env.setValue(*PrVar, *RootValue);
  }

  RootValue->setProperty(kIsNull, Env.getBoolLiteralValue(true));
  RootValue->setProperty(kIsNonnull, Env.getBoolLiteralValue(false));
}

void matchAddressofExpr(const Expr *expr,
                        const MatchFinder::MatchResult &Result,
                        Environment &Env) {
  const auto *Var = Result.Nodes.getNodeAs<Expr>(kVar);
  assert(Var != nullptr);

  Value *RootValue = Env.getValue(*Var);
  if (!RootValue) {
    RootValue = Env.createValue(Var->getType());

    if (!RootValue)
      return;

    setUnknownValue(*Var, *RootValue, Env);
  }

  RootValue->setProperty(kIsNull, Env.getBoolLiteralValue(false));
  RootValue->setProperty(kIsNonnull, Env.getBoolLiteralValue(true));
}

void matchAnyPointerExpr(const Expr *fncall,
                         const MatchFinder::MatchResult &Result,
                         Environment &Env) {
  // This is not necessarily a prvalue, since operators such as prefix ++ are
  // lvalues.
  const auto *Var = Result.Nodes.getNodeAs<Expr>(kVar);
  assert(Var != nullptr);

  // Initialize to (Unknown, Unknown)
  if (Env.getValue(*Var))
    return;

  Value *RootValue = Env.createValue(Var->getType());

  initializeRootValue(*RootValue, Env);
  setUnknownValue(*Var, *RootValue, Env);
}

NullCheckAfterDereferenceDiagnoser::ResultType
diagnoseDerefLocation(const Expr *Deref, const MatchFinder::MatchResult &Result,
                      NullCheckAfterDereferenceDiagnoser::DiagnoseArgs &Data) {
  auto [ValToDerefLoc, WarningLocToVal, Env] = Data;

  const auto *Var = Result.Nodes.getNodeAs<Expr>(kVar);
  assert(Var != nullptr);

  Value *RootValue = Env.getValue(*Var);
  if (!RootValue)
    return {};

  // Dereferences are always the highest priority when giving a single location
  // FIXME: Do not replace other dereferences, only other Expr's
  auto It = ValToDerefLoc.try_emplace(RootValue, nullptr).first;

  It->second = Deref;

  return {};
}

NullCheckAfterDereferenceDiagnoser::ResultType
diagnoseAssignLocation(const Expr *Assign,
                       const MatchFinder::MatchResult &Result,
                       NullCheckAfterDereferenceDiagnoser::DiagnoseArgs &Data) {
  auto [ValToDerefLoc, WarningLocToVal, Env] = Data;

  const auto *RHSVar = Result.Nodes.getNodeAs<Expr>(kValue);
  assert(RHSVar != nullptr);

  Value *RHSValue = Env.getValue(*RHSVar);
  if (!RHSValue)
    return {};

  auto [It, Inserted] = ValToDerefLoc.try_emplace(RHSValue, nullptr);

  if (Inserted)
    It->second = Assign;

  return {};
}

NullCheckAfterDereferenceDiagnoser::ResultType
diagnoseNullCheckExpr(const Expr *NullCheck,
      const MatchFinder::MatchResult &Result,
      const NullCheckAfterDereferenceDiagnoser::DiagnoseArgs &Data) {
  auto [ValToDerefLoc, WarningLocToVal, Env] = Data;

  const auto *Var = Result.Nodes.getNodeAs<Expr>(kVar);
  assert(Var != nullptr);

  if (Value *RootValue = Env.getValue(*Var)) {
    // FIXME: The framework usually creates the values for us, but without the
    // nullability properties.
    if (RootValue->getProperty(kIsNull) && RootValue->getProperty(kIsNonnull)) {
      bool IsNull = Env.allows(getVal(kIsNull, *RootValue).formula());
      bool IsNonnull = Env.allows(getVal(kIsNonnull, *RootValue).formula());

      if (!IsNull && IsNonnull) {
        // FIXME: Separate function
        bool Inserted =
            WarningLocToVal.try_emplace(Var->getBeginLoc(), RootValue).second;
        assert(Inserted && "multiple warnings at the same source location");
        (void)Inserted;

        return {{}, {Var->getBeginLoc()}};
      }

      if (IsNull && !IsNonnull) {
        bool Inserted =
            WarningLocToVal.try_emplace(Var->getBeginLoc(), RootValue).second;
        assert(Inserted && "multiple warnings at the same source location");
        (void)Inserted;

        return {{Var->getBeginLoc()}, {}};
      }
    }

    // If no matches are found, the null-check itself signals a special location
    auto [It, Inserted] = ValToDerefLoc.try_emplace(RootValue, nullptr);

    if (Inserted)
      It->second = NullCheck;
  }

  return {};
}

NullCheckAfterDereferenceDiagnoser::ResultType
diagnoseEqualExpr(const Expr *PtrCheck, const MatchFinder::MatchResult &Result,
                  NullCheckAfterDereferenceDiagnoser::DiagnoseArgs &Data) {
  auto [ValToDerefLoc, WarningLocToVal, Env] = Data;

  const auto *LHSVar = Result.Nodes.getNodeAs<Expr>(kVar);
  assert(LHSVar != nullptr);
  const auto *RHSVar = Result.Nodes.getNodeAs<Expr>(kValue);
  assert(RHSVar != nullptr);
  
  Arena &A = Env.arena();
  std::vector<SourceLocation> NullVarLocations;

  if (Value *LHSValue = Env.getValue(*LHSVar);
      Env.proves(A.makeNot(getVal(kIsNonnull, *LHSValue).formula()))) {
    WarningLocToVal.try_emplace(LHSVar->getBeginLoc(), LHSValue);
    NullVarLocations.push_back(LHSVar->getBeginLoc());
  }

  if (Value *RHSValue = Env.getValue(*RHSVar);
      Env.proves(A.makeNot(getVal(kIsNonnull, *RHSValue).formula()))) {
    WarningLocToVal.try_emplace(RHSVar->getBeginLoc(), RHSValue);
    NullVarLocations.push_back(RHSVar->getBeginLoc());
  }

  return {NullVarLocations, {}};
}

auto buildTransferMatchSwitch() {
  return CFGMatchSwitchBuilder<Environment>()
      .CaseOfCFGStmt<Stmt>(derefMatcher(), matchDereferenceExpr)
      .CaseOfCFGStmt<Stmt>(arrowMatcher(), matchDereferenceExpr)
      .CaseOfCFGStmt<Expr>(nullptrMatcher(), matchNullptrExpr)
      .CaseOfCFGStmt<Expr>(addressofMatcher(), matchAddressofExpr)
      .CaseOfCFGStmt<Expr>(functionCallMatcher(), matchAnyPointerExpr)
      .CaseOfCFGStmt<Expr>(anyPointerMatcher(), matchAnyPointerExpr)
      .CaseOfCFGStmt<Expr>(castExprMatcher(), matchNullCheckExpr)
      .CaseOfCFGStmt<Expr>(nullCheckExprMatcher(), matchNullCheckExpr)
      .CaseOfCFGStmt<BinaryOperator>(equalExprMatcher(), matchEqualExpr)
      .Build();
}

auto buildBranchTransferMatchSwitch() {
  return ASTMatchSwitchBuilder<Stmt, NullPointerAnalysisModel::TransferArgs>()
      // .CaseOf<CastExpr>(castExprMatcher(), matchNullCheckExpr)
      // .CaseOf<BinaryOperator>(equalExprMatcher(), matchEqualExpr)
      .Build();
}

auto buildDiagnoseMatchSwitch() {
  return CFGMatchSwitchBuilder<NullCheckAfterDereferenceDiagnoser::DiagnoseArgs,
                               NullCheckAfterDereferenceDiagnoser::ResultType>()
      .CaseOfCFGStmt<Expr>(derefMatcher(), diagnoseDerefLocation)
      .CaseOfCFGStmt<Expr>(arrowMatcher(), diagnoseDerefLocation)
      .CaseOfCFGStmt<Expr>(assignMatcher(), diagnoseAssignLocation)
      .CaseOfCFGStmt<Expr>(castExprMatcher(), diagnoseNullCheckExpr)
      .CaseOfCFGStmt<Expr>(nullCheckExprMatcher(), diagnoseNullCheckExpr)
      .CaseOfCFGStmt<Expr>(equalExprMatcher(), diagnoseEqualExpr)
      .Build();
}

} // namespace

NullPointerAnalysisModel::NullPointerAnalysisModel(ASTContext &Context)
    : DataflowAnalysis<NullPointerAnalysisModel, NoopLattice>(Context),
      TransferMatchSwitch(buildTransferMatchSwitch()),
      BranchTransferMatchSwitch(buildBranchTransferMatchSwitch()) {}

ast_matchers::StatementMatcher NullPointerAnalysisModel::ptrValueMatcher() {
  return ptrWithBinding();
}

void NullPointerAnalysisModel::transfer(const CFGElement &E, NoopLattice &State,
                                        Environment &Env) {
  TransferMatchSwitch(E, getASTContext(), Env);
}

void NullPointerAnalysisModel::transferBranch(bool Branch, const Stmt *E,
                                              NoopLattice &State,
                                              Environment &Env) {
  if (!E)
    return;

  TransferArgs Args = {Branch, Env};
  BranchTransferMatchSwitch(*E, getASTContext(), Args);
}

void NullPointerAnalysisModel::join(QualType Type, const Value &Val1,
                                    const Environment &Env1, const Value &Val2,
                                    const Environment &Env2, Value &MergedVal,
                                    Environment &MergedEnv) {
  if (!Type->isAnyPointerType())
    return;

  const auto MergeValues = [&](llvm::StringRef Name) -> BoolValue & {
    auto *LHSVar = cast_or_null<BoolValue>(Val1.getProperty(Name));
    auto *RHSVar = cast_or_null<BoolValue>(Val2.getProperty(Name));

    if (LHSVar == RHSVar)
      return *LHSVar;

    SatisfiabilityResult LHSResult = computeSatisfiability(LHSVar, Env1);
    SatisfiabilityResult RHSResult = computeSatisfiability(RHSVar, Env2);

    // Handle special cases.
    if (LHSResult == SR::Top || RHSResult == SR::Top) {
      return MergedEnv.makeTopBoolValue();
    } else if (LHSResult == RHSResult) {
      switch (LHSResult) {
      case SR::Nullptr:
        return MergedEnv.makeAtomicBoolValue();
      case SR::Top:
        return *LHSVar;
      case SR::True:
        return MergedEnv.getBoolLiteralValue(true);
      case SR::False:
        return MergedEnv.getBoolLiteralValue(false);
      case SR::Unknown:
        if (MergedEnv.proves(MergedEnv.arena().makeEquals(LHSVar->formula(),
                                                          RHSVar->formula())))
          return *LHSVar;

        return MergedEnv.makeTopBoolValue();
      }
    }

    return MergedEnv.makeTopBoolValue();
  };

  BoolValue &NonnullValue = MergeValues(kIsNonnull);
  BoolValue &NullValue = MergeValues(kIsNull);

  MergedVal.setProperty(kIsNonnull, NonnullValue);
  MergedVal.setProperty(kIsNull, NullValue);

  MergedEnv.assume(MergedEnv.makeOr(NonnullValue, NullValue).formula());
}

ComparisonResult NullPointerAnalysisModel::compare(QualType Type,
                                                   const Value &Val1,
                                                   const Environment &Env1,
                                                   const Value &Val2,
                                                   const Environment &Env2) {

  if (!Type->isAnyPointerType())
    return ComparisonResult::Unknown;

  // Evaluate values, but different values compare to Unknown.
  auto CompareValues = [&](llvm::StringRef Name) -> ComparisonResult {
    auto *LHSVar = cast_or_null<BoolValue>(Val1.getProperty(Name));
    auto *RHSVar = cast_or_null<BoolValue>(Val2.getProperty(Name));

    if (LHSVar == RHSVar)
      return ComparisonResult::Same;

    SatisfiabilityResult LHSResult = computeSatisfiability(LHSVar, Env1);
    SatisfiabilityResult RHSResult = computeSatisfiability(RHSVar, Env2);

    if (LHSResult == SR::Top || RHSResult == SR::Top)
      return ComparisonResult::Same;

    if (LHSResult == SR::Unknown || RHSResult == SR::Unknown)
      return ComparisonResult::Unknown;

    if (LHSResult == RHSResult)
      return ComparisonResult::Same;

    return ComparisonResult::Different;
  };

  ComparisonResult NullComparison = CompareValues(kIsNull);
  ComparisonResult NonnullComparison = CompareValues(kIsNonnull);

  if (NullComparison == ComparisonResult::Different ||
      NonnullComparison == ComparisonResult::Different)
    return ComparisonResult::Different;

  if (NullComparison == ComparisonResult::Unknown ||
      NonnullComparison == ComparisonResult::Unknown)
    return ComparisonResult::Unknown;

  return ComparisonResult::Same;
}

// Different in that it replaces differing boolean values with Top.
ComparisonResult compareAndReplace(QualType Type, Value &Val1,
                                   const Environment &Env1, Value &Val2,
                                   Environment &Env2) {

  if (!Type->isAnyPointerType())
    return ComparisonResult::Unknown;

  auto FastCompareValues = [&](llvm::StringRef Name) -> ComparisonResult {
    auto *LHSVar = cast_or_null<BoolValue>(Val1.getProperty(Name));
    auto *RHSVar = cast_or_null<BoolValue>(Val2.getProperty(Name));

    SatisfiabilityResult LHSResult = shallowComputeSatisfiability(LHSVar, Env1);
    SatisfiabilityResult RHSResult = shallowComputeSatisfiability(RHSVar, Env2);

    if (LHSResult == SR::Top || RHSResult == SR::Top) {
      Val2.setProperty(Name, Env2.makeTopBoolValue());
      return ComparisonResult::Same;
    }

    if (LHSResult == SR::Unknown || RHSResult == SR::Unknown)
      return ComparisonResult::Unknown;

    if (LHSResult == RHSResult)
      return ComparisonResult::Same;

    Val2.setProperty(Name, Env2.makeTopBoolValue());
    return ComparisonResult::Different;
  };

  ComparisonResult NullComparison = FastCompareValues(kIsNull);
  ComparisonResult NonnullComparison = FastCompareValues(kIsNonnull);

  if (NullComparison == ComparisonResult::Different ||
      NonnullComparison == ComparisonResult::Different)
    return ComparisonResult::Different;

  if (NullComparison == ComparisonResult::Unknown ||
      NonnullComparison == ComparisonResult::Unknown)
    return ComparisonResult::Unknown;

  return ComparisonResult::Same;
}

Value *NullPointerAnalysisModel::widen(QualType Type, Value &Prev,
                                       const Environment &PrevEnv,
                                       Value &Current,
                                       Environment &CurrentEnv) {
  if (!Type->isAnyPointerType())
    return nullptr;

  switch (compareAndReplace(Type, Prev, PrevEnv, Current, CurrentEnv)) {
  case ComparisonResult::Same:
    return &Prev;
  case ComparisonResult::Unknown:
    return nullptr;
  case ComparisonResult::Different:
    return &Current;
  }
}

NullCheckAfterDereferenceDiagnoser::NullCheckAfterDereferenceDiagnoser()
    : DiagnoseMatchSwitch(buildDiagnoseMatchSwitch()) {}

NullCheckAfterDereferenceDiagnoser::ResultType
NullCheckAfterDereferenceDiagnoser::diagnose(ASTContext &Ctx,
                                             const CFGElement *Elt,
                                             const Environment &Env) {
  DiagnoseArgs Args = {ValToDerefLoc, WarningLocToVal, Env};
  return DiagnoseMatchSwitch(*Elt, Ctx, Args);
}

} // namespace clang::dataflow
