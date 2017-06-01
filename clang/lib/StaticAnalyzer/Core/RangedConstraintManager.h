//== RangedConstraintManager.h ----------------------------------*- C++ -*--==//
//
//                     The LLVM Compiler Infrastructure
//
// This file is distributed under the University of Illinois Open Source
// License. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
//
//  Ranged constraint manager, built on SimpleConstraintManager.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_LIB_STATICANALYZER_CORE_RANGEDCONSTRAINTMANAGER_H
#define LLVM_CLANG_LIB_STATICANALYZER_CORE_RANGEDCONSTRAINTMANAGER_H

#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SimpleConstraintManager.h"

namespace clang {

namespace ento {

class RangedConstraintManager : public SimpleConstraintManager {
public:
  RangedConstraintManager(SubEngine *SE, SValBuilder &SB)
      : SimpleConstraintManager(SE, SB) {}

  ~RangedConstraintManager() override;

  //===------------------------------------------------------------------===//
  // Implementation for interface from SimpleConstraintManager.
  //===------------------------------------------------------------------===//

  ProgramStateRef assumeSym(ProgramStateRef State, SymbolRef Sym,
                            bool Assumption) override;

  ProgramStateRef assumeSymInclusiveRange(ProgramStateRef State, SymbolRef Sym,
                                          const llvm::APSInt &From,
                                          const llvm::APSInt &To,
                                          bool InRange) override;

  ProgramStateRef assumeSymUnsupported(ProgramStateRef State, SymbolRef Sym,
                                       bool Assumption) override;

protected:
  /// Assume a constraint between a symbolic expression and a concrete integer.
  virtual ProgramStateRef assumeSymRel(ProgramStateRef State, SymbolRef Sym,
                               BinaryOperator::Opcode op,
                               const llvm::APSInt &Int);

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

  virtual ProgramStateRef assumeSymWithinInclusiveRange(
      ProgramStateRef State, SymbolRef Sym, const llvm::APSInt &From,
      const llvm::APSInt &To, const llvm::APSInt &Adjustment) = 0;

  virtual ProgramStateRef assumeSymOutsideInclusiveRange(
      ProgramStateRef State, SymbolRef Sym, const llvm::APSInt &From,
      const llvm::APSInt &To, const llvm::APSInt &Adjustment) = 0;

  //===------------------------------------------------------------------===//
  // Internal implementation.
  //===------------------------------------------------------------------===//
private:
  static void computeAdjustment(SymbolRef &Sym, llvm::APSInt &Adjustment);
};

} // end GR namespace

} // end clang namespace

#endif
