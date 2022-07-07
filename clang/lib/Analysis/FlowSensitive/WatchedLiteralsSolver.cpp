//===- WatchedLiteralsSolver.cpp --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines a SAT solver implementation that can be used by dataflow
//  analyses.
//
//===----------------------------------------------------------------------===//

#include <cassert>
#include <cstdint>
#include <iterator>
#include <queue>
#include <vector>

#include "clang/Analysis/FlowSensitive/Solver.h"
#include "clang/Analysis/FlowSensitive/Value.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"

namespace clang {
namespace dataflow {

// `WatchedLiteralsSolver` is an implementation of Algorithm D from Knuth's
// The Art of Computer Programming Volume 4: Satisfiability, Fascicle 6. It is
// based on the backtracking DPLL algorithm [1], keeps references to a single
// "watched" literal per clause, and uses a set of "active" variables to perform
// unit propagation.
//
// The solver expects that its input is a boolean formula in conjunctive normal
// form that consists of clauses of at least one literal. A literal is either a
// boolean variable or its negation. Below we define types, data structures, and
// utilities that are used to represent boolean formulas in conjunctive normal
// form.
//
// [1] https://en.wikipedia.org/wiki/DPLL_algorithm

/// Boolean variables are represented as positive integers.
using Variable = uint32_t;

/// A null boolean variable is used as a placeholder in various data structures
/// and algorithms.
static constexpr Variable NullVar = 0;

/// Literals are represented as positive integers. Specifically, for a boolean
/// variable `V` that is represented as the positive integer `I`, the positive
/// literal `V` is represented as the integer `2*I` and the negative literal
/// `!V` is represented as the integer `2*I+1`.
using Literal = uint32_t;

/// A null literal is used as a placeholder in various data structures and
/// algorithms.
static constexpr Literal NullLit = 0;

/// Returns the positive literal `V`.
static constexpr Literal posLit(Variable V) { return 2 * V; }

/// Returns the negative literal `!V`.
static constexpr Literal negLit(Variable V) { return 2 * V + 1; }

/// Returns the negated literal `!L`.
static constexpr Literal notLit(Literal L) { return L ^ 1; }

/// Returns the variable of `L`.
static constexpr Variable var(Literal L) { return L >> 1; }

/// Clause identifiers are represented as positive integers.
using ClauseID = uint32_t;

/// A null clause identifier is used as a placeholder in various data structures
/// and algorithms.
static constexpr ClauseID NullClause = 0;

/// A boolean formula in conjunctive normal form.
struct BooleanFormula {
  /// `LargestVar` is equal to the largest positive integer that represents a
  /// variable in the formula.
  const Variable LargestVar;

  /// Literals of all clauses in the formula.
  ///
  /// The element at index 0 stands for the literal in the null clause. It is
  /// set to 0 and isn't used. Literals of clauses in the formula start from the
  /// element at index 1.
  ///
  /// For example, for the formula `(L1 v L2) ^ (L2 v L3 v L4)` the elements of
  /// `Clauses` will be `[0, L1, L2, L2, L3, L4]`.
  std::vector<Literal> Clauses;

  /// Start indices of clauses of the formula in `Clauses`.
  ///
  /// The element at index 0 stands for the start index of the null clause. It
  /// is set to 0 and isn't used. Start indices of clauses in the formula start
  /// from the element at index 1.
  ///
  /// For example, for the formula `(L1 v L2) ^ (L2 v L3 v L4)` the elements of
  /// `ClauseStarts` will be `[0, 1, 3]`. Note that the literals of the first
  /// clause always start at index 1. The start index for the literals of the
  /// second clause depends on the size of the first clause and so on.
  std::vector<size_t> ClauseStarts;

  /// Maps literals (indices of the vector) to clause identifiers (elements of
  /// the vector) that watch the respective literals.
  ///
  /// For a given clause, its watched literal is always its first literal in
  /// `Clauses`. This invariant is maintained when watched literals change.
  std::vector<ClauseID> WatchedHead;

  /// Maps clause identifiers (elements of the vector) to identifiers of other
  /// clauses that watch the same literals, forming a set of linked lists.
  ///
  /// The element at index 0 stands for the identifier of the clause that
  /// follows the null clause. It is set to 0 and isn't used. Identifiers of
  /// clauses in the formula start from the element at index 1.
  std::vector<ClauseID> NextWatched;

  /// Stores the variable identifier and value location for atomic booleans in
  /// the formula.
  llvm::DenseMap<Variable, AtomicBoolValue *> Atomics;

  explicit BooleanFormula(Variable LargestVar,
                          llvm::DenseMap<Variable, AtomicBoolValue *> Atomics)
      : LargestVar(LargestVar), Atomics(std::move(Atomics)) {
    Clauses.push_back(0);
    ClauseStarts.push_back(0);
    NextWatched.push_back(0);
    const size_t NumLiterals = 2 * LargestVar + 1;
    WatchedHead.resize(NumLiterals + 1, 0);
  }

  /// Adds the `L1 v L2 v L3` clause to the formula. If `L2` or `L3` are
  /// `NullLit` they are respectively omitted from the clause.
  ///
  /// Requirements:
  ///
  ///  `L1` must not be `NullLit`.
  ///
  ///  All literals in the input that are not `NullLit` must be distinct.
  void addClause(Literal L1, Literal L2 = NullLit, Literal L3 = NullLit) {
    // The literals are guaranteed to be distinct from properties of BoolValue
    // and the construction in `buildBooleanFormula`.
    assert(L1 != NullLit && L1 != L2 && L1 != L3 &&
           (L2 != L3 || L2 == NullLit));

    const ClauseID C = ClauseStarts.size();
    const size_t S = Clauses.size();
    ClauseStarts.push_back(S);

    Clauses.push_back(L1);
    if (L2 != NullLit)
      Clauses.push_back(L2);
    if (L3 != NullLit)
      Clauses.push_back(L3);

    // Designate the first literal as the "watched" literal of the clause.
    NextWatched.push_back(WatchedHead[L1]);
    WatchedHead[L1] = C;
  }

  /// Returns the number of literals in clause `C`.
  size_t clauseSize(ClauseID C) const {
    return C == ClauseStarts.size() - 1 ? Clauses.size() - ClauseStarts[C]
                                        : ClauseStarts[C + 1] - ClauseStarts[C];
  }

  /// Returns the literals of clause `C`.
  llvm::ArrayRef<Literal> clauseLiterals(ClauseID C) const {
    return llvm::ArrayRef<Literal>(&Clauses[ClauseStarts[C]], clauseSize(C));
  }
};

/// Converts the conjunction of `Vals` into a formula in conjunctive normal
/// form where each clause has at least one and at most three literals.
BooleanFormula buildBooleanFormula(const llvm::DenseSet<BoolValue *> &Vals) {
  // The general strategy of the algorithm implemented below is to map each
  // of the sub-values in `Vals` to a unique variable and use these variables in
  // the resulting CNF expression to avoid exponential blow up. The number of
  // literals in the resulting formula is guaranteed to be linear in the number
  // of sub-values in `Vals`.

  // Map each sub-value in `Vals` to a unique variable.
  llvm::DenseMap<BoolValue *, Variable> SubValsToVar;
  // Store variable identifiers and value location of atomic booleans.
  llvm::DenseMap<Variable, AtomicBoolValue *> Atomics;
  Variable NextVar = 1;
  {
    std::queue<BoolValue *> UnprocessedSubVals;
    for (BoolValue *Val : Vals)
      UnprocessedSubVals.push(Val);
    while (!UnprocessedSubVals.empty()) {
      Variable Var = NextVar;
      BoolValue *Val = UnprocessedSubVals.front();
      UnprocessedSubVals.pop();

      if (!SubValsToVar.try_emplace(Val, Var).second)
        continue;
      ++NextVar;

      // Visit the sub-values of `Val`.
      switch (Val->getKind()) {
      case Value::Kind::Conjunction: {
        auto *C = cast<ConjunctionValue>(Val);
        UnprocessedSubVals.push(&C->getLeftSubValue());
        UnprocessedSubVals.push(&C->getRightSubValue());
        break;
      }
      case Value::Kind::Disjunction: {
        auto *D = cast<DisjunctionValue>(Val);
        UnprocessedSubVals.push(&D->getLeftSubValue());
        UnprocessedSubVals.push(&D->getRightSubValue());
        break;
      }
      case Value::Kind::Negation: {
        auto *N = cast<NegationValue>(Val);
        UnprocessedSubVals.push(&N->getSubVal());
        break;
      }
      case Value::Kind::AtomicBool: {
        Atomics[Var] = cast<AtomicBoolValue>(Val);
        break;
      }
      default:
        llvm_unreachable("buildBooleanFormula: unhandled value kind");
      }
    }
  }

  auto GetVar = [&SubValsToVar](const BoolValue *Val) {
    auto ValIt = SubValsToVar.find(Val);
    assert(ValIt != SubValsToVar.end());
    return ValIt->second;
  };

  BooleanFormula Formula(NextVar - 1, std::move(Atomics));
  std::vector<bool> ProcessedSubVals(NextVar, false);

  // Add a conjunct for each variable that represents a top-level conjunction
  // value in `Vals`.
  for (BoolValue *Val : Vals)
    Formula.addClause(posLit(GetVar(Val)));

  // Add conjuncts that represent the mapping between newly-created variables
  // and their corresponding sub-values.
  std::queue<BoolValue *> UnprocessedSubVals;
  for (BoolValue *Val : Vals)
    UnprocessedSubVals.push(Val);
  while (!UnprocessedSubVals.empty()) {
    const BoolValue *Val = UnprocessedSubVals.front();
    UnprocessedSubVals.pop();
    const Variable Var = GetVar(Val);

    if (ProcessedSubVals[Var])
      continue;
    ProcessedSubVals[Var] = true;

    if (auto *C = dyn_cast<ConjunctionValue>(Val)) {
      const Variable LeftSubVar = GetVar(&C->getLeftSubValue());
      const Variable RightSubVar = GetVar(&C->getRightSubValue());

      // `X <=> (A ^ B)` is equivalent to `(!X v A) ^ (!X v B) ^ (X v !A v !B)`
      // which is already in conjunctive normal form. Below we add each of the
      // conjuncts of the latter expression to the result.
      Formula.addClause(negLit(Var), posLit(LeftSubVar));
      Formula.addClause(negLit(Var), posLit(RightSubVar));
      Formula.addClause(posLit(Var), negLit(LeftSubVar), negLit(RightSubVar));

      // Visit the sub-values of `Val`.
      UnprocessedSubVals.push(&C->getLeftSubValue());
      UnprocessedSubVals.push(&C->getRightSubValue());
    } else if (auto *D = dyn_cast<DisjunctionValue>(Val)) {
      const Variable LeftSubVar = GetVar(&D->getLeftSubValue());
      const Variable RightSubVar = GetVar(&D->getRightSubValue());

      // `X <=> (A v B)` is equivalent to `(!X v A v B) ^ (X v !A) ^ (X v !B)`
      // which is already in conjunctive normal form. Below we add each of the
      // conjuncts of the latter expression to the result.
      Formula.addClause(negLit(Var), posLit(LeftSubVar), posLit(RightSubVar));
      Formula.addClause(posLit(Var), negLit(LeftSubVar));
      Formula.addClause(posLit(Var), negLit(RightSubVar));

      // Visit the sub-values of `Val`.
      UnprocessedSubVals.push(&D->getLeftSubValue());
      UnprocessedSubVals.push(&D->getRightSubValue());
    } else if (auto *N = dyn_cast<NegationValue>(Val)) {
      const Variable SubVar = GetVar(&N->getSubVal());

      // `X <=> !Y` is equivalent to `(!X v !Y) ^ (X v Y)` which is already in
      // conjunctive normal form. Below we add each of the conjuncts of the
      // latter expression to the result.
      Formula.addClause(negLit(Var), negLit(SubVar));
      Formula.addClause(posLit(Var), posLit(SubVar));

      // Visit the sub-values of `Val`.
      UnprocessedSubVals.push(&N->getSubVal());
    }
  }

  return Formula;
}

class WatchedLiteralsSolverImpl {
  /// A boolean formula in conjunctive normal form that the solver will attempt
  /// to prove satisfiable. The formula will be modified in the process.
  BooleanFormula Formula;

  /// The search for a satisfying assignment of the variables in `Formula` will
  /// proceed in levels, starting from 1 and going up to `Formula.LargestVar`
  /// (inclusive). The current level is stored in `Level`. At each level the
  /// solver will assign a value to an unassigned variable. If this leads to a
  /// consistent partial assignment, `Level` will be incremented. Otherwise, if
  /// it results in a conflict, the solver will backtrack by decrementing
  /// `Level` until it reaches the most recent level where a decision was made.
  size_t Level = 0;

  /// Maps levels (indices of the vector) to variables (elements of the vector)
  /// that are assigned values at the respective levels.
  ///
  /// The element at index 0 isn't used. Variables start from the element at
  /// index 1.
  std::vector<Variable> LevelVars;

  /// State of the solver at a particular level.
  enum class State : uint8_t {
    /// Indicates that the solver made a decision.
    Decision = 0,

    /// Indicates that the solver made a forced move.
    Forced = 1,
  };

  /// State of the solver at a particular level. It keeps track of previous
  /// decisions that the solver can refer to when backtracking.
  ///
  /// The element at index 0 isn't used. States start from the element at index
  /// 1.
  std::vector<State> LevelStates;

  enum class Assignment : int8_t {
    Unassigned = -1,
    AssignedFalse = 0,
    AssignedTrue = 1
  };

  /// Maps variables (indices of the vector) to their assignments (elements of
  /// the vector).
  ///
  /// The element at index 0 isn't used. Variable assignments start from the
  /// element at index 1.
  std::vector<Assignment> VarAssignments;

  /// A set of unassigned variables that appear in watched literals in
  /// `Formula`. The vector is guaranteed to contain unique elements.
  std::vector<Variable> ActiveVars;

public:
  explicit WatchedLiteralsSolverImpl(const llvm::DenseSet<BoolValue *> &Vals)
      : Formula(buildBooleanFormula(Vals)), LevelVars(Formula.LargestVar + 1),
        LevelStates(Formula.LargestVar + 1) {
    assert(!Vals.empty());

    // Initialize the state at the root level to a decision so that in
    // `reverseForcedMoves` we don't have to check that `Level >= 0` on each
    // iteration.
    LevelStates[0] = State::Decision;

    // Initialize all variables as unassigned.
    VarAssignments.resize(Formula.LargestVar + 1, Assignment::Unassigned);

    // Initialize the active variables.
    for (Variable Var = Formula.LargestVar; Var != NullVar; --Var) {
      if (isWatched(posLit(Var)) || isWatched(negLit(Var)))
        ActiveVars.push_back(Var);
    }
  }

  Solver::Result solve() && {
    size_t I = 0;
    while (I < ActiveVars.size()) {
      // Assert that the following invariants hold:
      // 1. All active variables are unassigned.
      // 2. All active variables form watched literals.
      // 3. Unassigned variables that form watched literals are active.
      // FIXME: Consider replacing these with test cases that fail if the any
      // of the invariants is broken. That might not be easy due to the
      // transformations performed by `buildBooleanFormula`.
      assert(activeVarsAreUnassigned());
      assert(activeVarsFormWatchedLiterals());
      assert(unassignedVarsFormingWatchedLiteralsAreActive());

      const Variable ActiveVar = ActiveVars[I];

      // Look for unit clauses that contain the active variable.
      const bool unitPosLit = watchedByUnitClause(posLit(ActiveVar));
      const bool unitNegLit = watchedByUnitClause(negLit(ActiveVar));
      if (unitPosLit && unitNegLit) {
        // We found a conflict!

        // Backtrack and rewind the `Level` until the most recent non-forced
        // assignment.
        reverseForcedMoves();

        // If the root level is reached, then all possible assignments lead to
        // a conflict.
        if (Level == 0)
          return Solver::Result::Unsatisfiable();

        // Otherwise, take the other branch at the most recent level where a
        // decision was made.
        LevelStates[Level] = State::Forced;
        const Variable Var = LevelVars[Level];
        VarAssignments[Var] = VarAssignments[Var] == Assignment::AssignedTrue
                                  ? Assignment::AssignedFalse
                                  : Assignment::AssignedTrue;

        updateWatchedLiterals();
      } else if (unitPosLit || unitNegLit) {
        // We found a unit clause! The value of its unassigned variable is
        // forced.
        ++Level;

        LevelVars[Level] = ActiveVar;
        LevelStates[Level] = State::Forced;
        VarAssignments[ActiveVar] =
            unitPosLit ? Assignment::AssignedTrue : Assignment::AssignedFalse;

        // Remove the variable that was just assigned from the set of active
        // variables.
        if (I + 1 < ActiveVars.size()) {
          // Replace the variable that was just assigned with the last active
          // variable for efficient removal.
          ActiveVars[I] = ActiveVars.back();
        } else {
          // This was the last active variable. Repeat the process from the
          // beginning.
          I = 0;
        }
        ActiveVars.pop_back();

        updateWatchedLiterals();
      } else if (I + 1 == ActiveVars.size()) {
        // There are no remaining unit clauses in the formula! Make a decision
        // for one of the active variables at the current level.
        ++Level;

        LevelVars[Level] = ActiveVar;
        LevelStates[Level] = State::Decision;
        VarAssignments[ActiveVar] = decideAssignment(ActiveVar);

        // Remove the variable that was just assigned from the set of active
        // variables.
        ActiveVars.pop_back();

        updateWatchedLiterals();

        // This was the last active variable. Repeat the process from the
        // beginning.
        I = 0;
      } else {
        ++I;
      }
    }
    return Solver::Result::Satisfiable(buildSolution());
  }

private:
  /// Returns a satisfying truth assignment to the atomic values in the boolean
  /// formula.
  llvm::DenseMap<AtomicBoolValue *, Solver::Result::Assignment>
  buildSolution() {
    llvm::DenseMap<AtomicBoolValue *, Solver::Result::Assignment> Solution;
    for (auto [Var, Val] : Formula.Atomics) {
      // A variable may have a definite true/false assignment, or it may be
      // unassigned indicating its truth value does not affect the result of
      // the formula. Unassigned variables are assigned to true as a default.
      Solution[Val] = VarAssignments[Var] == Assignment::AssignedFalse
                          ? Solver::Result::Assignment::AssignedFalse
                          : Solver::Result::Assignment::AssignedTrue;
    }
    return Solution;
  }

  /// Reverses forced moves until the most recent level where a decision was
  /// made on the assignment of a variable.
  void reverseForcedMoves() {
    for (; LevelStates[Level] == State::Forced; --Level) {
      const Variable Var = LevelVars[Level];

      VarAssignments[Var] = Assignment::Unassigned;

      // If the variable that we pass through is watched then we add it to the
      // active variables.
      if (isWatched(posLit(Var)) || isWatched(negLit(Var)))
        ActiveVars.push_back(Var);
    }
  }

  /// Updates watched literals that are affected by a variable assignment.
  void updateWatchedLiterals() {
    const Variable Var = LevelVars[Level];

    // Update the watched literals of clauses that currently watch the literal
    // that falsifies `Var`.
    const Literal FalseLit = VarAssignments[Var] == Assignment::AssignedTrue
                                 ? negLit(Var)
                                 : posLit(Var);
    ClauseID FalseLitWatcher = Formula.WatchedHead[FalseLit];
    Formula.WatchedHead[FalseLit] = NullClause;
    while (FalseLitWatcher != NullClause) {
      const ClauseID NextFalseLitWatcher = Formula.NextWatched[FalseLitWatcher];

      // Pick the first non-false literal as the new watched literal.
      const size_t FalseLitWatcherStart = Formula.ClauseStarts[FalseLitWatcher];
      size_t NewWatchedLitIdx = FalseLitWatcherStart + 1;
      while (isCurrentlyFalse(Formula.Clauses[NewWatchedLitIdx]))
        ++NewWatchedLitIdx;
      const Literal NewWatchedLit = Formula.Clauses[NewWatchedLitIdx];
      const Variable NewWatchedLitVar = var(NewWatchedLit);

      // Swap the old watched literal for the new one in `FalseLitWatcher` to
      // maintain the invariant that the watched literal is at the beginning of
      // the clause.
      Formula.Clauses[NewWatchedLitIdx] = FalseLit;
      Formula.Clauses[FalseLitWatcherStart] = NewWatchedLit;

      // If the new watched literal isn't watched by any other clause and its
      // variable isn't assigned we need to add it to the active variables.
      if (!isWatched(NewWatchedLit) && !isWatched(notLit(NewWatchedLit)) &&
          VarAssignments[NewWatchedLitVar] == Assignment::Unassigned)
        ActiveVars.push_back(NewWatchedLitVar);

      Formula.NextWatched[FalseLitWatcher] = Formula.WatchedHead[NewWatchedLit];
      Formula.WatchedHead[NewWatchedLit] = FalseLitWatcher;

      // Go to the next clause that watches `FalseLit`.
      FalseLitWatcher = NextFalseLitWatcher;
    }
  }

  /// Returns true if and only if one of the clauses that watch `Lit` is a unit
  /// clause.
  bool watchedByUnitClause(Literal Lit) const {
    for (ClauseID LitWatcher = Formula.WatchedHead[Lit];
         LitWatcher != NullClause;
         LitWatcher = Formula.NextWatched[LitWatcher]) {
      llvm::ArrayRef<Literal> Clause = Formula.clauseLiterals(LitWatcher);

      // Assert the invariant that the watched literal is always the first one
      // in the clause.
      // FIXME: Consider replacing this with a test case that fails if the
      // invariant is broken by `updateWatchedLiterals`. That might not be easy
      // due to the transformations performed by `buildBooleanFormula`.
      assert(Clause.front() == Lit);

      if (isUnit(Clause))
        return true;
    }
    return false;
  }

  /// Returns true if and only if `Clause` is a unit clause.
  bool isUnit(llvm::ArrayRef<Literal> Clause) const {
    return llvm::all_of(Clause.drop_front(),
                        [this](Literal L) { return isCurrentlyFalse(L); });
  }

  /// Returns true if and only if `Lit` evaluates to `false` in the current
  /// partial assignment.
  bool isCurrentlyFalse(Literal Lit) const {
    return static_cast<int8_t>(VarAssignments[var(Lit)]) ==
           static_cast<int8_t>(Lit & 1);
  }

  /// Returns true if and only if `Lit` is watched by a clause in `Formula`.
  bool isWatched(Literal Lit) const {
    return Formula.WatchedHead[Lit] != NullClause;
  }

  /// Returns an assignment for an unassigned variable.
  Assignment decideAssignment(Variable Var) const {
    return !isWatched(posLit(Var)) || isWatched(negLit(Var))
               ? Assignment::AssignedFalse
               : Assignment::AssignedTrue;
  }

  /// Returns a set of all watched literals.
  llvm::DenseSet<Literal> watchedLiterals() const {
    llvm::DenseSet<Literal> WatchedLiterals;
    for (Literal Lit = 2; Lit < Formula.WatchedHead.size(); Lit++) {
      if (Formula.WatchedHead[Lit] == NullClause)
        continue;
      WatchedLiterals.insert(Lit);
    }
    return WatchedLiterals;
  }

  /// Returns true if and only if all active variables are unassigned.
  bool activeVarsAreUnassigned() const {
    return llvm::all_of(ActiveVars, [this](Variable Var) {
      return VarAssignments[Var] == Assignment::Unassigned;
    });
  }

  /// Returns true if and only if all active variables form watched literals.
  bool activeVarsFormWatchedLiterals() const {
    const llvm::DenseSet<Literal> WatchedLiterals = watchedLiterals();
    return llvm::all_of(ActiveVars, [&WatchedLiterals](Variable Var) {
      return WatchedLiterals.contains(posLit(Var)) ||
             WatchedLiterals.contains(negLit(Var));
    });
  }

  /// Returns true if and only if all unassigned variables that are forming
  /// watched literals are active.
  bool unassignedVarsFormingWatchedLiteralsAreActive() const {
    const llvm::DenseSet<Variable> ActiveVarsSet(ActiveVars.begin(),
                                                 ActiveVars.end());
    for (Literal Lit : watchedLiterals()) {
      const Variable Var = var(Lit);
      if (VarAssignments[Var] != Assignment::Unassigned)
        continue;
      if (ActiveVarsSet.contains(Var))
        continue;
      return false;
    }
    return true;
  }
};

Solver::Result WatchedLiteralsSolver::solve(llvm::DenseSet<BoolValue *> Vals) {
  return Vals.empty() ? Solver::Result::Satisfiable({{}})
                      : WatchedLiteralsSolverImpl(Vals).solve();
}

} // namespace dataflow
} // namespace clang
