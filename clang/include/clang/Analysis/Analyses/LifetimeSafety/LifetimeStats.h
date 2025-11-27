//===- LifetimeStats.h - Miscellaneous statistics related to lifetime safety
//analysis ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------------------------------===//
//
// This file declares the data structures and utility function for collection of
// staticstics related to Lifetimesafety analysis.
//
//===----------------------------------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LIFETIMESTATS_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LIFETIMESTATS_H

#include "llvm/ADT/StringMap.h"

namespace clang::lifetimes {
/// A structure to hold the statistics related to LifetimeAnalysis.
/// Currently it holds only the missing origin details.
struct LifetimeSafetyStats {
  /// A map from `ExpressionClassName<QualType>` to their missing origin
  /// counts.
  llvm::StringMap<unsigned> MissingOriginCount;
};

// utility function to print missing origin stats.
void printStats(const LifetimeSafetyStats &Stats);
} // namespace clang::lifetimes

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_LIFETIMESAFETY_LIFETIMESTATS_H