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

namespace clang {
namespace ento {

class SMTConstraintManager : public clang::ento::SimpleConstraintManager {

public:
  SMTConstraintManager(clang::ento::SubEngine *SE, clang::ento::SValBuilder &SB)
      : SimpleConstraintManager(SE, SB) {}
  virtual ~SMTConstraintManager() = default;

  /// Converts the ranged constraints of a set of symbols to SMT
  ///
  /// \param CR The set of constraints.
  virtual void addRangeConstraints(clang::ento::ConstraintRangeTy CR) = 0;

  /// Checks if the added constraints are satisfiable
  virtual clang::ento::ConditionTruthVal isModelFeasible() = 0;

}; // end class SMTConstraintManager

} // namespace ento
} // namespace clang

#endif
