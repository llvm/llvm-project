//===- AnalysisName.h -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Strong typedef identifying a whole-program analysis and its result type.
// Distinct from SummaryName, which identifies per-entity EntitySummary types.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_ANALYSISNAME_H
#define LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_ANALYSISNAME_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>

namespace clang::ssaf {

/// Uniquely identifies a whole-program analysis and the AnalysisResult it
/// produces. Used as the key in WPASuite and AnalysisRegistry.
///
/// Distinct from SummaryName, which is used by EntitySummary types for routing
/// through the LUSummary.
class AnalysisName {
public:
  explicit AnalysisName(std::string Name) : Name(std::move(Name)) {}

  bool operator==(const AnalysisName &Other) const {
    return Name == Other.Name;
  }
  bool operator!=(const AnalysisName &Other) const { return !(*this == Other); }
  bool operator<(const AnalysisName &Other) const { return Name < Other.Name; }

  /// Explicit conversion to the underlying string representation.
  llvm::StringRef str() const { return Name; }

private:
  std::string Name;
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const AnalysisName &AN);

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSISFRAMEWORK_CORE_WHOLEPROGRAMANALYSIS_ANALYSISNAME_H
