//===- LifetimeStats.cpp - Miscellaneous statistics related to C++ Lifetime
//Safety analysis -*--------- C++-*-===//
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
#include "llvm/Support/raw_ostream.h"

namespace clang::lifetimes {
void printStats(const LifetimeSafetyStats &Stats) {
  llvm::errs() << "\n*** LifetimeSafety Missing Origin Stats "
                  "(expression_type : count) :\n\n";
  unsigned TotalMissingOrigins = 0;
  for (const auto &[expr, count] : Stats.MissingOriginCount) {
    llvm::errs() << expr << " : " << count << '\n';
    TotalMissingOrigins += count;
  }
  llvm::errs() << "Total missing origins: " << TotalMissingOrigins << "\n";
  llvm::errs() << "\n****************************************\n";
}
} // namespace clang::lifetimes