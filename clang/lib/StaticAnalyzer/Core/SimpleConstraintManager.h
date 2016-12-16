//== SimpleConstraintManager.h ----------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Code shared between BasicConstraintManager and RangeConstraintManager.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CORE_SIMPLECONSTRAINTMANAGER_H
#define LLVM_CLANG_LIB_STATICANALYZER_CORE_SIMPLECONSTRAINTMANAGER_H

#include "clang/StaticAnalyzer/Core/PathSensitive/ConstraintManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"

namespace clang {

namespace ento {

class SimpleConstraintManager : public ConstraintManager {
  SubEngine *SU;
  SValBuilder &SVB;

public:
  SimpleConstraintManager(SubEngine *SE, SValBuilder &SB) : SU(SE), SVB(SB) {}
  ~SimpleConstraintManager() override;

  //===------------------------------------------------------------------===//
  // Common implementation for the interface provided by ConstraintManager.
  //===------------------------------------------------------------------===//

  ProgramStateRef assume(ProgramStateRef State, DefinedSVal Cond,
                         bool Assumption) override;

  ProgramStateRef assume(ProgramStateRef State, NonLoc Cond, bool Assumption);

  ProgramStateRef assumeInclusiveRange(ProgramStateRef State, NonLoc Value,
                                       const llvm::APSInt &From,
                                       const llvm::APSInt &To,
                                       bool InRange) override;

  ProgramStateRef assumeSymRel(ProgramStateRef State, const SymExpr *LHS,
                               BinaryOperator::Opcode Op,
                               const llvm::APSInt &Int);

  ProgramStateRef assumeSymWithinInclusiveRange(ProgramStateRef State,
                                                SymbolRef Sym,
                                                const llvm::APSInt &From,
                                                const llvm::APSInt &To,
                                                bool InRange);

protected:
  //===------------------------------------------------------------------===//
  // Interface that subclasses must implement.
  //===------------------------------------------------------------------===//

  // Each of these is of the form "$Sym+Adj <> V", where "<>" is the comparison
  // operation for the method being invoked.
  virtual ProgramStateRef assumeSymNE(ProgramStateRef State, SymbolRef Sym,
                                      const llvm::APSInt &V,
                                      const llvm::APSInt &Adjustment) = 0;

  virtual ProgramStateRef assumeSymEQ(ProgramStateRef State, SymbolRef Sym,
                                      const llvm::APSInt &V,
                                      const llvm::APSInt &Adjustment) = 0;

  virtual ProgramStateRef assumeSymLT(ProgramStateRef State, SymbolRef Sym,
                                      const llvm::APSInt &V,
                                      const llvm::APSInt &Adjustment) = 0;

  virtual ProgramStateRef assumeSymGT(ProgramStateRef State, SymbolRef Sym,
                                      const llvm::APSInt &V,
                                      const llvm::APSInt &Adjustment) = 0;

  virtual ProgramStateRef assumeSymLE(ProgramStateRef State, SymbolRef Sym,
                                      const llvm::APSInt &V,
                                      const llvm::APSInt &Adjustment) = 0;

  virtual ProgramStateRef assumeSymGE(ProgramStateRef State, SymbolRef Sym,
                                      const llvm::APSInt &V,
                                      const llvm::APSInt &Adjustment) = 0;

  virtual ProgramStateRef assumeSymbolWithinInclusiveRange(
      ProgramStateRef State, SymbolRef Sym, const llvm::APSInt &From,
      const llvm::APSInt &To, const llvm::APSInt &Adjustment) = 0;

  virtual ProgramStateRef assumeSymbolOutOfInclusiveRange(
      ProgramStateRef State, SymbolRef Sym, const llvm::APSInt &From,
      const llvm::APSInt &To, const llvm::APSInt &Adjustment) = 0;

  //===------------------------------------------------------------------===//
  // Internal implementation.
  //===------------------------------------------------------------------===//

  BasicValueFactory &getBasicVals() const { return SVB.getBasicValueFactory(); }
  SymbolManager &getSymbolManager() const { return SVB.getSymbolManager(); }

  bool canReasonAbout(SVal X) const override;

  ProgramStateRef assumeAux(ProgramStateRef State, NonLoc Cond,
                            bool Assumption);

  ProgramStateRef assumeAuxForSymbol(ProgramStateRef State, SymbolRef Sym,
                                     bool Assumption);
};

} // end GR namespace

} // end clang namespace

#endif
