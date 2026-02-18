//===- SummaryName.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_SUMMARYNAME_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_SUMMARYNAME_H

#include "llvm/ADT/StringRef.h"
#include <string>

namespace clang::ssaf {

/// Uniquely identifies an analysis summary.
///
/// This is the key to refer to an analysis or to name a builder to build an
/// analysis.
class SummaryName {
public:
  explicit SummaryName(std::string Name) : Name(std::move(Name)) {}

  bool operator==(const SummaryName &Other) const { return Name == Other.Name; }
  bool operator!=(const SummaryName &Other) const { return !(*this == Other); }
  bool operator<(const SummaryName &Other) const { return Name < Other.Name; }

  /// Explicit conversion to the underlying string representation.
  llvm::StringRef str() const { return Name; }

private:
  std::string Name;
  friend class TestFixture;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_SUMMARYNAME_H
