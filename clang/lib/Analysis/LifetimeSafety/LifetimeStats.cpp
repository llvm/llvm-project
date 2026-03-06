//===- LifetimeStats.cpp - Lifetime Safety Statistics -*------------ C++-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the data structures and utility function for collection of
// staticstics related to Lifetimesafety analysis.
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety/LifetimeStats.h"
#include "clang/AST/TypeBase.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::lifetimes {
void printStats(const LifetimeSafetyStats &Stats) {
  llvm::errs() << "\n*** LifetimeSafety Missing Origin per QualType: "
                  "(QualType : count) :\n\n";
  unsigned TotalMissingOrigins = 0;
  for (const auto &[ExprType, MissingOriginCount] :
       Stats.ExprTypeToMissingOriginCount) {
    QualType QT = QualType(ExprType, 0);
    llvm::errs() << QT.getAsString() << " : " << MissingOriginCount << '\n';
    TotalMissingOrigins += MissingOriginCount;
  }
  llvm::errs() << "\n\n*** LifetimeSafety Missing Origin per StmtClassName: "
                  "(StmtClassName : count) :\n\n";
  for (const auto &[ExprStmtClassName, MissingOriginCount] :
       Stats.ExprStmtClassToMissingOriginCount) {
    llvm::errs() << ExprStmtClassName << " : " << MissingOriginCount << '\n';
  }
  llvm::errs() << "\nTotal missing origins: " << TotalMissingOrigins << "\n";
  llvm::errs() << "\n****************************************\n";
}
} // namespace clang::lifetimes
