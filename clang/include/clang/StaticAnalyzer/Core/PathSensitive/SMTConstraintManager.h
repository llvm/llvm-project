//== SMTConstraintManager.h -------------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  This file defines a SMT generic API, which will be the base class for
//  every SMT solver specific class.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SMTCONSTRAINTMANAGER_H
#define LLVM_CLANG_STATICANALYZER_CORE_PATHSENSITIVE_SMTCONSTRAINTMANAGER_H

#include "clang/StaticAnalyzer/Core/PathSensitive/RangedConstraintManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SMTSolver.h"

namespace clang {
namespace ento {

class SMTConstraintManager : public clang::ento::SimpleConstraintManager {
  SMTSolverRef &Solver;

public:
  SMTConstraintManager(clang::ento::SubEngine *SE, clang::ento::SValBuilder &SB,
                       SMTSolverRef &S)
      : SimpleConstraintManager(SE, SB), Solver(S) {}
  virtual ~SMTConstraintManager() = default;

  //===------------------------------------------------------------------===//
  // Implementation for interface from SimpleConstraintManager.
  //===------------------------------------------------------------------===//

  ProgramStateRef assumeSym(ProgramStateRef state, SymbolRef Sym,
                            bool Assumption) override;

  ProgramStateRef assumeSymInclusiveRange(ProgramStateRef State, SymbolRef Sym,
                                          const llvm::APSInt &From,
                                          const llvm::APSInt &To,
                                          bool InRange) override;

  ProgramStateRef assumeSymUnsupported(ProgramStateRef State, SymbolRef Sym,
                                       bool Assumption) override;

  //===------------------------------------------------------------------===//
  // Implementation for interface from ConstraintManager.
  //===------------------------------------------------------------------===//

  ConditionTruthVal checkNull(ProgramStateRef State, SymbolRef Sym) override;

  const llvm::APSInt *getSymVal(ProgramStateRef State,
                                SymbolRef Sym) const override;

  /// Dumps SMT formula
  LLVM_DUMP_METHOD void dump() const { Solver->dump(); }

protected:
  // Check whether a new model is satisfiable, and update the program state.
  virtual ProgramStateRef assumeExpr(ProgramStateRef State, SymbolRef Sym,
                                     const SMTExprRef &Exp) = 0;

  /// Given a program state, construct the logical conjunction and add it to
  /// the solver
  virtual void addStateConstraints(ProgramStateRef State) const = 0;

  // Generate and check a Z3 model, using the given constraint.
  ConditionTruthVal checkModel(ProgramStateRef State,
                               const SMTExprRef &Exp) const;
}; // end class SMTConstraintManager

} // namespace ento
} // namespace clang

#endif
