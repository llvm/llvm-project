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

#include "clang/Analysis/FlowSensitive/Formula.h"
#include "clang/Analysis/FlowSensitive/Solver.h"
#include "clang/Analysis/FlowSensitive/WatchedLiteralsSolver.h"
#include "llvm/ADT/ArrayRef.h"
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
struct CNFFormula {
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

  /// Stores the variable identifier and Atom for atomic booleans in the
  /// formula.
  llvm::DenseMap<Variable, Atom> Atomics;

  explicit CNFFormula(Variable LargestVar,
                      llvm::DenseMap<Variable, Atom> Atomics)
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
    // The literals are guaranteed to be distinct from properties of Formula
    // and the construction in `buildCNF`.
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
CNFFormula buildCNF(const llvm::ArrayRef<const Formula *> &Vals) {
  // The general strategy of the algorithm implemented below is to map each
  // of the sub-values in `Vals` to a unique variable and use these variables in
  // the resulting CNF expression to avoid exponential blow up. The number of
  // literals in the resulting formula is guaranteed to be linear in the number
  // of sub-formulas in `Vals`.

  // Map each sub-formula in `Vals` to a unique variable.
  llvm::DenseMap<const Formula *, Variable> SubValsToVar;
  // Store variable identifiers and Atom of atomic booleans.
  llvm::DenseMap<Variable, Atom> Atomics;
  Variable NextVar = 1;
  {
    std::queue<const Formula *> UnprocessedSubVals;
    for (const Formula *Val : Vals)
      UnprocessedSubVals.push(Val);
    while (!UnprocessedSubVals.empty()) {
      Variable Var = NextVar;
      const Formula *Val = UnprocessedSubVals.front();
      UnprocessedSubVals.pop();

      if (!SubValsToVar.try_emplace(Val, Var).second)
        continue;
      ++NextVar;

      for (const Formula *F : Val->operands())
        UnprocessedSubVals.push(F);
      if (Val->kind() == Formula::AtomRef)
        Atomics[Var] = Val->getAtom();
    }
  }

  auto GetVar = [&SubValsToVar](const Formula *Val) {
    auto ValIt = SubValsToVar.find(Val);
    assert(ValIt != SubValsToVar.end());
    return ValIt->second;
  };

  CNFFormula CNF(NextVar - 1, std::move(Atomics));
  std::vector<bool> ProcessedSubVals(NextVar, false);

  // Add a conjunct for each variable that represents a top-level formula in
  // `Vals`.
  for (const Formula *Val : Vals)
    CNF.addClause(posLit(GetVar(Val)));

  // Add conjuncts that represent the mapping between newly-created variables
  // and their corresponding sub-formulas.
  std::queue<const Formula *> UnprocessedSubVals;
  for (const Formula *Val : Vals)
    UnprocessedSubVals.push(Val);
  while (!UnprocessedSubVals.empty()) {
    const Formula *Val = UnprocessedSubVals.front();
    UnprocessedSubVals.pop();
    const Variable Var = GetVar(Val);

    if (ProcessedSubVals[Var])
      continue;
    ProcessedSubVals[Var] = true;

    switch (Val->kind()) {
    case Formula::AtomRef:
      break;
    case Formula::And: {
      const Variable LHS = GetVar(Val->operands()[0]);
      const Variable RHS = GetVar(Val->operands()[1]);

      if (LHS == RHS) {
        // `X <=> (A ^ A)` is equivalent to `(!X v A) ^ (X v !A)` which is
        // already in conjunctive normal form. Below we add each of the
        // conjuncts of the latter expression to the result.
        CNF.addClause(negLit(Var), posLit(LHS));
        CNF.addClause(posLit(Var), negLit(LHS));
      } else {
        // `X <=> (A ^ B)` is equivalent to `(!X v A) ^ (!X v B) ^ (X v !A v !B)`
        // which is already in conjunctive normal form. Below we add each of the
        // conjuncts of the latter expression to the result.
        CNF.addClause(negLit(Var), posLit(LHS));
        CNF.addClause(negLit(Var), posLit(RHS));
        CNF.addClause(posLit(Var), negLit(LHS), negLit(RHS));
      }
      break;
    }
    case Formula::Or: {
      const Variable LHS = GetVar(Val->operands()[0]);
      const Variable RHS = GetVar(Val->operands()[1]);

      if (LHS == RHS) {
        // `X <=> (A v A)` is equivalent to `(!X v A) ^ (X v !A)` which is
        // already in conjunctive normal form. Below we add each of the
        // conjuncts of the latter expression to the result.
        CNF.addClause(negLit(Var), posLit(LHS));
        CNF.addClause(posLit(Var), negLit(LHS));
      } else {
        // `X <=> (A v B)` is equivalent to `(!X v A v B) ^ (X v !A) ^ (X v
        // !B)` which is already in conjunctive normal form. Below we add each
        // of the conjuncts of the latter expression to the result.
        CNF.addClause(negLit(Var), posLit(LHS), posLit(RHS));
        CNF.addClause(posLit(Var), negLit(LHS));
        CNF.addClause(posLit(Var), negLit(RHS));
      }
      break;
    }
    case Formula::Not: {
      const Variable Operand = GetVar(Val->operands()[0]);

      // `X <=> !Y` is equivalent to `(!X v !Y) ^ (X v Y)` which is
      // already in conjunctive normal form. Below we add each of the
      // conjuncts of the latter expression to the result.
      CNF.addClause(negLit(Var), negLit(Operand));
      CNF.addClause(posLit(Var), posLit(Operand));
      break;
    }
    case Formula::Implies: {
      const Variable LHS = GetVar(Val->operands()[0]);
      const Variable RHS = GetVar(Val->operands()[1]);

      // `X <=> (A => B)` is equivalent to
      // `(X v A) ^ (X v !B) ^ (!X v !A v B)` which is already in
      // conjunctive normal form. Below we add each of the conjuncts of
      // the latter expression to the result.
      CNF.addClause(posLit(Var), posLit(LHS));
      CNF.addClause(posLit(Var), negLit(RHS));
      CNF.addClause(negLit(Var), negLit(LHS), posLit(RHS));
      break;
    }
    case Formula::Equal: {
      const Variable LHS = GetVar(Val->operands()[0]);
      const Variable RHS = GetVar(Val->operands()[1]);

      if (LHS == RHS) {
        // `X <=> (A <=> A)` is equvalent to `X` which is already in
        // conjunctive normal form. Below we add each of the conjuncts of the
        // latter expression to the result.
        CNF.addClause(posLit(Var));

        // No need to visit the sub-values of `Val`.
        continue;
      }
      // `X <=> (A <=> B)` is equivalent to
      // `(X v A v B) ^ (X v !A v !B) ^ (!X v A v !B) ^ (!X v !A v B)` which
      // is already in conjunctive normal form. Below we add each of the
      // conjuncts of the latter expression to the result.
      CNF.addClause(posLit(Var), posLit(LHS), posLit(RHS));
      CNF.addClause(posLit(Var), negLit(LHS), negLit(RHS));
      CNF.addClause(negLit(Var), posLit(LHS), negLit(RHS));
      CNF.addClause(negLit(Var), negLit(LHS), posLit(RHS));
      break;
    }
    }
    for (const Formula *Child : Val->operands())
      UnprocessedSubVals.push(Child);
  }

  return CNF;
}

class WatchedLiteralsSolverImpl {
  /// A boolean formula in conjunctive normal form that the solver will attempt
  /// to prove satisfiable. The formula will be modified in the process.
  CNFFormula CNF;

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
  explicit WatchedLiteralsSolverImpl(
      const llvm::ArrayRef<const Formula *> &Vals)
      : CNF(buildCNF(Vals)), LevelVars(CNF.LargestVar + 1),
        LevelStates(CNF.LargestVar + 1) {
    assert(!Vals.empty());

    // Initialize the state at the root level to a decision so that in
    // `reverseForcedMoves` we don't have to check that `Level >= 0` on each
    // iteration.
    LevelStates[0] = State::Decision;

    // Initialize all variables as unassigned.
    VarAssignments.resize(CNF.LargestVar + 1, Assignment::Unassigned);

    // Initialize the active variables.
    for (Variable Var = CNF.LargestVar; Var != NullVar; --Var) {
      if (isWatched(posLit(Var)) || isWatched(negLit(Var)))
        ActiveVars.push_back(Var);
    }
  }

  // Returns the `Result` and the number of iterations "remaining" from
  // `MaxIterations` (that is, `MaxIterations` - iterations in this call).
  std::pair<Solver::Result, std::int64_t> solve(std::int64_t MaxIterations) && {
    size_t I = 0;
    while (I < ActiveVars.size()) {
      if (MaxIterations == 0)
        return std::make_pair(Solver::Result::TimedOut(), 0);
      --MaxIterations;

      // Assert that the following invariants hold:
      // 1. All active variables are unassigned.
      // 2. All active variables form watched literals.
      // 3. Unassigned variables that form watched literals are active.
      // FIXME: Consider replacing these with test cases that fail if the any
      // of the invariants is broken. That might not be easy due to the
      // transformations performed by `buildCNF`.
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
          return std::make_pair(Solver::Result::Unsatisfiable(), MaxIterations);

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
    return std::make_pair(Solver::Result::Satisfiable(buildSolution()), MaxIterations);
  }

private:
  /// Returns a satisfying truth assignment to the atoms in the boolean formula.
  llvm::DenseMap<Atom, Solver::Result::Assignment> buildSolution() {
    llvm::DenseMap<Atom, Solver::Result::Assignment> Solution;
    for (auto &Atomic : CNF.Atomics) {
      // A variable may have a definite true/false assignment, or it may be
      // unassigned indicating its truth value does not affect the result of
      // the formula. Unassigned variables are assigned to true as a default.
      Solution[Atomic.second] =
          VarAssignments[Atomic.first] == Assignment::AssignedFalse
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
    ClauseID FalseLitWatcher = CNF.WatchedHead[FalseLit];
    CNF.WatchedHead[FalseLit] = NullClause;
    while (FalseLitWatcher != NullClause) {
      const ClauseID NextFalseLitWatcher = CNF.NextWatched[FalseLitWatcher];

      // Pick the first non-false literal as the new watched literal.
      const size_t FalseLitWatcherStart = CNF.ClauseStarts[FalseLitWatcher];
      size_t NewWatchedLitIdx = FalseLitWatcherStart + 1;
      while (isCurrentlyFalse(CNF.Clauses[NewWatchedLitIdx]))
        ++NewWatchedLitIdx;
      const Literal NewWatchedLit = CNF.Clauses[NewWatchedLitIdx];
      const Variable NewWatchedLitVar = var(NewWatchedLit);

      // Swap the old watched literal for the new one in `FalseLitWatcher` to
      // maintain the invariant that the watched literal is at the beginning of
      // the clause.
      CNF.Clauses[NewWatchedLitIdx] = FalseLit;
      CNF.Clauses[FalseLitWatcherStart] = NewWatchedLit;

      // If the new watched literal isn't watched by any other clause and its
      // variable isn't assigned we need to add it to the active variables.
      if (!isWatched(NewWatchedLit) && !isWatched(notLit(NewWatchedLit)) &&
          VarAssignments[NewWatchedLitVar] == Assignment::Unassigned)
        ActiveVars.push_back(NewWatchedLitVar);

      CNF.NextWatched[FalseLitWatcher] = CNF.WatchedHead[NewWatchedLit];
      CNF.WatchedHead[NewWatchedLit] = FalseLitWatcher;

      // Go to the next clause that watches `FalseLit`.
      FalseLitWatcher = NextFalseLitWatcher;
    }
  }

  /// Returns true if and only if one of the clauses that watch `Lit` is a unit
  /// clause.
  bool watchedByUnitClause(Literal Lit) const {
    for (ClauseID LitWatcher = CNF.WatchedHead[Lit]; LitWatcher != NullClause;
         LitWatcher = CNF.NextWatched[LitWatcher]) {
      llvm::ArrayRef<Literal> Clause = CNF.clauseLiterals(LitWatcher);

      // Assert the invariant that the watched literal is always the first one
      // in the clause.
      // FIXME: Consider replacing this with a test case that fails if the
      // invariant is broken by `updateWatchedLiterals`. That might not be easy
      // due to the transformations performed by `buildCNF`.
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
    return CNF.WatchedHead[Lit] != NullClause;
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
    for (Literal Lit = 2; Lit < CNF.WatchedHead.size(); Lit++) {
      if (CNF.WatchedHead[Lit] == NullClause)
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

Solver::Result
WatchedLiteralsSolver::solve(llvm::ArrayRef<const Formula *> Vals) {
  if (Vals.empty())
    return Solver::Result::Satisfiable({{}});
  auto [Res, Iterations] =
      WatchedLiteralsSolverImpl(Vals).solve(MaxIterations);
  MaxIterations = Iterations;
  return Res;
}

} // namespace dataflow
} // namespace clang
