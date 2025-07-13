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
using Diagnoser = NullCheckAfterDereferenceDiagnoser;

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

enum class CompareResult {
  Same,
  Different,
  Top,
  Unknown
};

using SR = SatisfiabilityResult;
using CR = CompareResult;

// FIXME: These AST matchers should also be exported via the
// NullPointerAnalysisModel class, for tests
auto ptrWithBinding(llvm::StringRef VarName = kVar) {
  return expr(hasType(isAnyPointer())).bind(VarName);
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
  return callExpr(
      callee(functionDecl(hasAnyParameter(anyOf(hasType(pointerType()),
                                                hasType(referenceType()))))));
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

inline BoolValue &getVal(llvm::StringRef Name, Value &PtrValue) {
  return *cast<BoolValue>(PtrValue.getProperty(Name));
}

// Assigns initial pointer null- and nonnull-values to a given Value.
void initializeNullnessProperties(Value &PtrValue, Environment &Env) {
  Arena &A = Env.arena();

  auto *IsNull = cast_or_null<BoolValue>(PtrValue.getProperty(kIsNull));
  auto *IsNonnull = cast_or_null<BoolValue>(PtrValue.getProperty(kIsNonnull));

  if (!IsNull) {
    IsNull = &A.makeAtomValue();
    PtrValue.setProperty(kIsNull, *IsNull);
  }

  if (!IsNonnull) {
    IsNonnull = &A.makeAtomValue();
    PtrValue.setProperty(kIsNonnull, *IsNonnull);
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

// Gets a pointer's value, and initializes it to (Unknown, Unknown) if it hasn't
// been initialized already.
Value *getValue(const Expr &Var, Environment &Env) {
  if (Value *EnvVal = Env.getValue(Var)) {
    // FIXME: The framework usually creates the values for us, but without the
    // null-properties.
    initializeNullnessProperties(*EnvVal, Env);

    return EnvVal;
  }

  return nullptr;
}

bool hasTopOrNullValue(const Value *Val, const Environment &Env) {
  return !Val || isa_and_present<TopBoolValue>(Val->getProperty(kIsNull)) ||
         isa_and_present<TopBoolValue>(Val->getProperty(kIsNonnull));
}

void matchDereferenceExpr(const Stmt *stmt,
                          const MatchFinder::MatchResult &Result,
                          Environment &Env) {
  const auto *Var = Result.Nodes.getNodeAs<Expr>(kVar);
  assert(Var != nullptr);

  Value *PtrValue = getValue(*Var, Env);
  if (hasTopOrNullValue(PtrValue, Env))
    return;

  BoolValue &IsNull = getVal(kIsNull, *PtrValue);

  Env.assume(Env.arena().makeNot(IsNull.formula()));
}

void matchNullCheckExpr(const Expr *NullCheck,
                    const MatchFinder::MatchResult &Result,
                    Environment &Env) {
  const auto *Var = Result.Nodes.getNodeAs<Expr>(kVar);
  assert(Var != nullptr);

  // (bool)p or (p != nullptr)
  bool IsNonnullOp = true;
  if (const auto *BinOp = dyn_cast<BinaryOperator>(NullCheck);
      BinOp->getOpcode() == BO_EQ) {
    IsNonnullOp = false;
  }

  Value *PtrValue = getValue(*Var, Env);
  if (hasTopOrNullValue(PtrValue, Env))
    return;

  Arena &A = Env.arena();
  BoolValue &IsNonnull = getVal(kIsNonnull, *PtrValue);
  BoolValue &IsNull = getVal(kIsNull, *PtrValue);

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

  if (hasTopOrNullValue(LHSValue, Env) || hasTopOrNullValue(RHSValue, Env))
    return;

  BoolValue &LHSNonnull = getVal(kIsNonnull, *LHSValue);
  BoolValue &LHSNull = getVal(kIsNull, *LHSValue);
  BoolValue &RHSNonnull = getVal(kIsNonnull, *RHSValue);
  BoolValue &RHSNull = getVal(kIsNull, *RHSValue);

  BoolValue *CondValue = cast_or_null<BoolValue>(Env.getValue(*EqualExpr));
  if (!CondValue) {
    CondValue = &A.makeAtomValue();
    Env.setValue(*EqualExpr, *CondValue);
  }

  const Formula &CondFormula = IsNotEqualsOp ? A.makeNot(CondValue->formula())
                                       : CondValue->formula();

  // FIXME: Simplify formulas
  // If the pointers are equal, the nullability properties are the same.
  Env.assume(A.makeImplies(CondFormula, 
      A.makeAnd(A.makeEquals(LHSNull.formula(), RHSNull.formula()),
                A.makeEquals(LHSNonnull.formula(), RHSNonnull.formula()))));

  // If the pointers are not equal, at most one of the pointers is null.
  Env.assume(A.makeImplies(A.makeNot(CondFormula),
      A.makeNot(A.makeAnd(LHSNull.formula(), RHSNull.formula()))));
}

void matchNullptrExpr(const Expr *expr, const MatchFinder::MatchResult &Result,
                      Environment &Env) {
  const auto *PrVar = Result.Nodes.getNodeAs<Expr>(kVar);
  assert(PrVar != nullptr);

  Value *PtrValue = Env.getValue(*PrVar);
  if (!PtrValue) {
    PtrValue = Env.createValue(PrVar->getType());
    assert(PtrValue && "Failed to create nullptr value");
    Env.setValue(*PrVar, *PtrValue);
  }

  PtrValue->setProperty(kIsNull, Env.getBoolLiteralValue(true));
  PtrValue->setProperty(kIsNonnull, Env.getBoolLiteralValue(false));
}

void matchAddressofExpr(const Expr *expr,
                        const MatchFinder::MatchResult &Result,
                        Environment &Env) {
  const auto *Var = Result.Nodes.getNodeAs<Expr>(kVar);
  assert(Var != nullptr);

  // FIXME: Use atoms or export to separate function
  Value *PtrValue = Env.getValue(*Var);
  if (!PtrValue) {
    PtrValue = Env.createValue(Var->getType());

    if (!PtrValue)
      return;

    setUnknownValue(*Var, *PtrValue, Env);
  }

  PtrValue->setProperty(kIsNull, Env.getBoolLiteralValue(false));
  PtrValue->setProperty(kIsNonnull, Env.getBoolLiteralValue(true));
}

void matchPtrArgFunctionExpr(const CallExpr *fncall,
                             const MatchFinder::MatchResult &Result,
                             Environment &Env) {
  for (const auto *Arg : fncall->arguments()) {
    // FIXME: Add handling for reference types as arguments
    if (Arg->getType()->isPointerType()) {
      PointerValue *OuterValue = cast_or_null<PointerValue>(
          Env.getValue(*Arg));

      if (!OuterValue)
        continue;

      QualType InnerType = Arg->getType()->getPointeeType();
      if (!InnerType->isPointerType())
        continue;

      StorageLocation &InnerLoc = OuterValue->getPointeeLoc();
      
      PointerValue *InnerValue =
          cast_or_null<PointerValue>(Env.getValue(InnerLoc));

      if (!InnerValue)
        continue;
      
      Value *NewValue = Env.createValue(InnerType);
      assert(NewValue && "Failed to re-initialize a pointer's value");

      Env.setValue(InnerLoc, *NewValue);

    // FIXME: Recursively invalidate all member pointers of eg. a struct
    // Should be part of the framework, most likely.
    }
  }

  if (fncall->getCallReturnType(*Result.Context)->isPointerType() &&
      !Env.getValue(*fncall)) {
    Value *PtrValue = Env.createValue( 
        fncall->getCallReturnType(*Result.Context));
    if (!PtrValue)
      return;

    setUnknownValue(*fncall, *PtrValue, Env);
  }
}

void matchAnyPointerExpr(const Expr *fncall,
                         const MatchFinder::MatchResult &Result,
                         Environment &Env) {
  // This is not necessarily a prvalue, since operators such as prefix ++ are
  // lvalues.
  const auto *Var = Result.Nodes.getNodeAs<Expr>(kVar);
  assert(Var != nullptr);

  // In some cases, prvalues are not automatically initialized
  // Initialize these values, but don't set null-ness values for performance
  if (Env.getValue(*Var))
    return;

  Value *PtrValue = Env.createValue(Var->getType());
  if (!PtrValue)
    return;

  setUnknownValue(*Var, *PtrValue, Env);
}

Diagnoser::ResultType
diagnoseDerefLocation(const Expr *Deref, const MatchFinder::MatchResult &Result,
                      Diagnoser::DiagnoseArgs &Data) {
  auto [ValToDerefLoc, WarningLocToVal, Env] = Data;

  const auto *Var = Result.Nodes.getNodeAs<Expr>(kVar);
  assert(Var != nullptr);

  Value *PtrValue = Env.getValue(*Var);
  if (!PtrValue)
    return {};

  // Dereferences are always the highest priority when giving a single location
  // FIXME: Do not replace other dereferences, only other Expr's
  auto It = ValToDerefLoc.try_emplace(PtrValue, nullptr).first;

  It->second = Deref;

  return {};
}

Diagnoser::ResultType diagnoseAssignLocation(const Expr *Assign,
                                             const MatchFinder::MatchResult &Result,
                                             Diagnoser::DiagnoseArgs &Data) {
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
      const Diagnoser::DiagnoseArgs &Data) {
  auto [ValToDerefLoc, WarningLocToVal, Env] = Data;

  const auto *Var = Result.Nodes.getNodeAs<Expr>(kVar);
  assert(Var != nullptr);

  if (Value *PtrValue = Env.getValue(*Var)) {
    // FIXME: The framework usually creates the values for us, but without the
    // nullability properties.
    if (PtrValue->getProperty(kIsNull) && PtrValue->getProperty(kIsNonnull)) {
      bool IsNull = Env.allows(getVal(kIsNull, *PtrValue).formula());
      bool IsNonnull = Env.allows(getVal(kIsNonnull, *PtrValue).formula());

      if (!IsNull && IsNonnull) {
        // FIXME: Separate function
        bool Inserted =
            WarningLocToVal.try_emplace(Var->getBeginLoc(), PtrValue).second;
        assert(Inserted && "multiple warnings at the same source location");
        (void)Inserted;

        return {{Var->getBeginLoc(), Diagnoser::DiagnosticType::CheckAfterDeref}};
      }

      if (IsNull && !IsNonnull) {
        bool Inserted =
            WarningLocToVal.try_emplace(Var->getBeginLoc(), PtrValue).second;
        assert(Inserted && "multiple warnings at the same source location");
        (void)Inserted;

        return {{Var->getBeginLoc(), Diagnoser::DiagnosticType::CheckWhenNull}};
      }
    }

    // If no matches are found, the null-check itself signals a special location
    auto [It, Inserted] = ValToDerefLoc.try_emplace(PtrValue, nullptr);

    if (Inserted)
      It->second = NullCheck;
  }

  return {};
}

NullCheckAfterDereferenceDiagnoser::ResultType
diagnoseEqualExpr(const Expr *PtrCheck, const MatchFinder::MatchResult &Result,
                  Diagnoser::DiagnoseArgs &Data) {
  auto [ValToDerefLoc, WarningLocToVal, Env] = Data;

  const auto *LHSVar = Result.Nodes.getNodeAs<Expr>(kVar);
  assert(LHSVar != nullptr);
  const auto *RHSVar = Result.Nodes.getNodeAs<Expr>(kValue);
  assert(RHSVar != nullptr);
  
  Arena &A = Env.arena();
  llvm::SmallVector<Diagnoser::DiagnosticEntry> NullVarLocations;

  if (Value *LHSValue = Env.getValue(*LHSVar);
      LHSValue->getProperty(kIsNonnull) && 
      Env.proves(A.makeNot(getVal(kIsNonnull, *LHSValue).formula()))) {
    WarningLocToVal.try_emplace(LHSVar->getBeginLoc(), LHSValue);
    NullVarLocations.push_back({LHSVar->getBeginLoc(), Diagnoser::DiagnosticType::CheckWhenNull});
  }

  if (Value *RHSValue = Env.getValue(*RHSVar);
      RHSValue->getProperty(kIsNonnull) && 
      Env.proves(A.makeNot(getVal(kIsNonnull, *RHSValue).formula()))) {
    WarningLocToVal.try_emplace(RHSVar->getBeginLoc(), RHSValue);
    NullVarLocations.push_back({RHSVar->getBeginLoc(), Diagnoser::DiagnosticType::CheckWhenNull});
  }

  return NullVarLocations;
}

auto buildTransferMatchSwitch() {
  return CFGMatchSwitchBuilder<Environment>()
      .CaseOfCFGStmt<Stmt>(derefMatcher(), matchDereferenceExpr)
      .CaseOfCFGStmt<Stmt>(arrowMatcher(), matchDereferenceExpr)
      .CaseOfCFGStmt<Expr>(nullptrMatcher(), matchNullptrExpr)
      .CaseOfCFGStmt<Expr>(addressofMatcher(), matchAddressofExpr)
      .CaseOfCFGStmt<CallExpr>(functionCallMatcher(), matchPtrArgFunctionExpr)
      .CaseOfCFGStmt<Expr>(anyPointerMatcher(), matchAnyPointerExpr)
      .CaseOfCFGStmt<Expr>(castExprMatcher(), matchNullCheckExpr)
      .CaseOfCFGStmt<Expr>(nullCheckExprMatcher(), matchNullCheckExpr)
      .CaseOfCFGStmt<BinaryOperator>(equalExprMatcher(), matchEqualExpr)
      .Build();
}

auto buildDiagnoseMatchSwitch() {
  return CFGMatchSwitchBuilder<Diagnoser::DiagnoseArgs, Diagnoser::ResultType>()
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
      TransferMatchSwitch(buildTransferMatchSwitch()) {}

ast_matchers::StatementMatcher NullPointerAnalysisModel::ptrValueMatcher() {
  return ptrWithBinding();
}

void NullPointerAnalysisModel::transfer(const CFGElement &E, NoopLattice &State,
                                        Environment &Env) {
  TransferMatchSwitch(E, getASTContext(), Env);
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

    const auto SimplifyVar = [&](BoolValue *VarToSimplify,
                                 const Environment &Env) -> BoolValue * {
      SatisfiabilityResult SatResult =
          computeSatisfiability(VarToSimplify, Env);
      switch (SatResult) {
      case SR::Nullptr:
        return nullptr;
      case SR::Top:
        return &MergedEnv.makeTopBoolValue();
      case SR::True:
        return &MergedEnv.getBoolLiteralValue(true);
      case SR::False:
        return &MergedEnv.getBoolLiteralValue(false);
      case SR::Unknown:
        return VarToSimplify;
      }
    };

    LHSVar = SimplifyVar(LHSVar, Env1);
    RHSVar = SimplifyVar(RHSVar, Env2);

    // Handle special cases.
    if (LHSVar == RHSVar)
      return LHSVar ? *LHSVar : MergedEnv.makeAtomicBoolValue();
    else if (isa_and_nonnull<TopBoolValue>(LHSVar) || isa_and_nonnull<TopBoolValue>(RHSVar))
      return MergedEnv.makeTopBoolValue();

    if (!LHSVar)
      LHSVar = &MergedEnv.makeAtomicBoolValue();

    if (!RHSVar)
      RHSVar = &MergedEnv.makeAtomicBoolValue();

    assert(LHSVar != nullptr && RHSVar != nullptr);

    if (MergedEnv.proves(
            MergedEnv.arena().makeEquals(LHSVar->formula(), RHSVar->formula())))
      return *LHSVar;

    BoolValue &ReturnVar = MergedEnv.makeAtomicBoolValue();
    Arena &A = MergedEnv.arena();

    MergedEnv.assume(A.makeOr(
        A.makeAnd(A.makeAtomRef(Env1.getFlowConditionToken()),
                  A.makeEquals(ReturnVar.formula(), LHSVar->formula())),
        A.makeAnd(A.makeAtomRef(Env2.getFlowConditionToken()),
                  A.makeEquals(ReturnVar.formula(), RHSVar->formula()))));

    return ReturnVar;
  };

  BoolValue &NonnullValue = MergeValues(kIsNonnull);
  BoolValue &NullValue = MergeValues(kIsNull);

  if (isa<TopBoolValue>(NonnullValue) || isa<TopBoolValue>(NullValue)) {
    MergedVal.setProperty(kIsNonnull, MergedEnv.makeTopBoolValue());
    MergedVal.setProperty(kIsNull, MergedEnv.makeTopBoolValue());
  } else {
    MergedVal.setProperty(kIsNonnull, NonnullValue);
    MergedVal.setProperty(kIsNull, NullValue);
  
    MergedEnv.assume(MergedEnv.makeOr(NonnullValue, NullValue).formula());
  }
}

ComparisonResult NullPointerAnalysisModel::compare(QualType Type,
                                                   const Value &Val1,
                                                   const Environment &Env1,
                                                   const Value &Val2,
                                                   const Environment &Env2) {

  if (!Type->isAnyPointerType())
    return ComparisonResult::Unknown;

  // Evaluate values, but different values compare to Unknown.
  auto CompareValues = [&](llvm::StringRef Name) -> CR {
    auto *LHSVar = cast_or_null<BoolValue>(Val1.getProperty(Name));
    auto *RHSVar = cast_or_null<BoolValue>(Val2.getProperty(Name));

    const auto SimplifyVar = [&](BoolValue *VarToSimplify,
                                 const Environment &Env) -> BoolValue * {
      SatisfiabilityResult SatResult =
          computeSatisfiability(VarToSimplify, Env);
      switch (SatResult) {
      case SR::Nullptr:
        return nullptr;
      case SR::Top:
        return &MergedEnv.makeTopBoolValue();
      case SR::True:
        return &MergedEnv.getBoolLiteralValue(true);
      case SR::False:
        return &MergedEnv.getBoolLiteralValue(false);
      case SR::Unknown:
        return VarToSimplify;
      }
    };

    LHSVar = SimplifyVar(LHSVar, Env1);
    RHSVar = SimplifyVar(RHSVar, Env2);

    // Handle special cases.
    if (isa_and_nonnull<TopBoolValue>(LHSVar) || isa_and_nonnull<TopBoolValue>(RHSVar))
      return CR::Top;
    else if (LHSVar == RHSVar)
      return LHSVar ? CR::Same : CR::Unknown;

    return CR::Different;
  };

  CR NullComparison = CompareValues(kIsNull);
  CR NonnullComparison = CompareValues(kIsNonnull);

  if (NullComparison == CR::Top || NonnullComparison == CR::Top)
    return ComparisonResult::Same;

  if (NullComparison == CR::Different ||
      NonnullComparison == CR::Different)
    return ComparisonResult::Different;

  if (NullComparison == CR::Unknown ||
      NonnullComparison == CR::Unknown)
    return ComparisonResult::Unknown;

  return ComparisonResult::Same;
}

// Different in that it replaces differing boolean values with Top.
ComparisonResult compareAndReplace(QualType Type, Value &Val1,
                                   const Environment &Env1, Value &Val2,
                                   Environment &Env2) {

  if (!Type->isAnyPointerType())
    return ComparisonResult::Unknown;

  // Evaluate values, but different values compare to Unknown.
  auto FastCompareValues = [&](llvm::StringRef Name) -> CR {
    auto *LHSVar = cast_or_null<BoolValue>(Val1.getProperty(Name));
    auto *RHSVar = cast_or_null<BoolValue>(Val2.getProperty(Name));

    const auto SimplifyVar = [&](BoolValue *VarToSimplify,
                                 const Environment &Env) -> BoolValue * {
      SatisfiabilityResult SatResult =
          shallowComputeSatisfiability(VarToSimplify, Env);
      switch (SatResult) {
      case SR::Nullptr:
        return nullptr;
      case SR::Top:
        return &MergedEnv.makeTopBoolValue();
      case SR::True:
        return &MergedEnv.getBoolLiteralValue(true);
      case SR::False:
        return &MergedEnv.getBoolLiteralValue(false);
      case SR::Unknown:
        return VarToSimplify;
      }
    };

    LHSVar = SimplifyVar(LHSVar, Env1);
    RHSVar = SimplifyVar(RHSVar, Env2);

    // Handle special cases.
    if (isa_and_nonnull<TopBoolValue>(LHSVar) || isa_and_nonnull<TopBoolValue>(RHSVar)) {
      Val2.setProperty(Name, Env2.makeTopBoolValue());
      return CR::Top;
    } else if (LHSVar == RHSVar)
      return LHSVar ? CR::Same : CR::Unknown;

    return CR::Different;
  };
  CR NullComparison = FastCompareValues(kIsNull);
  CR NonnullComparison = FastCompareValues(kIsNonnull);

  if (NullComparison == CR::Top || NonnullComparison == CR::Top)
    return ComparisonResult::Same;

  if (NullComparison == CR::Different ||
      NonnullComparison == CR::Different)
    return ComparisonResult::Different;

  if (NullComparison == CR::Unknown ||
      NonnullComparison == CR::Unknown)
    return ComparisonResult::Unknown;

  return ComparisonResult::Same;
}

std::optional<WidenResult>
NullPointerAnalysisModel::widen(QualType Type, Value &Prev,
                                const Environment &PrevEnv, Value &Current,
                                Environment &CurrentEnv) {
  if (!Type->isAnyPointerType())
    return std::nullopt;

  switch (compareAndReplace(Type, Prev, PrevEnv, Current, CurrentEnv)) {
  case ComparisonResult::Same:
    return WidenResult{&Prev, LatticeEffect::Unchanged};
  case ComparisonResult::Unknown:
    return std::nullopt;
  case ComparisonResult::Different:
    return WidenResult{&Current, LatticeEffect::Changed};
  }
}

NullCheckAfterDereferenceDiagnoser::NullCheckAfterDereferenceDiagnoser()
    : DiagnoseMatchSwitch(buildDiagnoseMatchSwitch()) {}

NullCheckAfterDereferenceDiagnoser::ResultType
NullCheckAfterDereferenceDiagnoser::operator()(
    const CFGElement &Elt, ASTContext &Ctx, 
    const TransferStateForDiagnostics<NoopLattice> &State) {
  DiagnoseArgs Args = {ValToDerefLoc, WarningLocToVal, State.Env};
  return DiagnoseMatchSwitch(Elt, Ctx, Args);
}

} // namespace clang::dataflow
