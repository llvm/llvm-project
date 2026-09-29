//===- LifetimeStats.h - Lifetime Safety Statistics -------------*- C++-* -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the data structures and utility function for collection of
// statistics related to Lifetime Safety analysis.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LIFETIMESTATS_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LIFETIMESTATS_H

#include "clang/AST/TypeBase.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/StringMap.h"

namespace clang::lifetimes {
/// A structure to hold the statistics related to LifetimeAnalysis.
/// These are accumulated across all analyzed functions and printed
/// when -print-stats is enabled.
struct LifetimeSafetyStats {
  /// A map from `StmtClassName` to their missing origin counts.
  llvm::StringMap<unsigned> ExprStmtClassToMissingOriginCount;
  /// A map from `QualType` to their missing origin counts.
  llvm::DenseMap<const clang::Type *, unsigned> ExprTypeToMissingOriginCount;
};

/// Utility function to print missing origin stats.
void printStats(const LifetimeSafetyStats &Stats);
} // namespace clang::lifetimes

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LIFETIMESTATS_H
