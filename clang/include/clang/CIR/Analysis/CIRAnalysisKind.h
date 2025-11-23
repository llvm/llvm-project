//===--- CIRAnalysisKind.h - CIR Analysis Pass Kinds -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// Defines the CIR analysis pass kinds enum and related utilities.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_CIR_SEMA_CIRANALYSISKIND_H
#define LLVM_CLANG_CIR_SEMA_CIRANALYSISKIND_H

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>
#include <vector>

namespace cir {

/// Enumeration of available CIR semantic analysis passes
enum class CIRAnalysisKind : unsigned {
  Unrecognized = 0,
  FallThrough = 1 << 0,      // Fallthrough warning analysis
  UnreachableCode = 1 << 1,  // Unreachable code detection
  NullCheck = 1 << 2,        // Null pointer checks
  UninitializedVar = 1 << 3, // Uninitialized variable detection
  // Add more analysis passes here as needed
};

/// A set of CIR analysis passes (bitmask)
class CIRAnalysisSet {
  unsigned mask = 0;

public:
  CIRAnalysisSet() = default;
  explicit CIRAnalysisSet(CIRAnalysisKind kind)
      : mask(static_cast<unsigned>(kind)) {}
  explicit CIRAnalysisSet(unsigned mask) : mask(mask) {}

  /// Check if a specific analysis is enabled
  bool has(CIRAnalysisKind kind) const {
    return (mask & static_cast<unsigned>(kind)) != 0;
  }

  /// Enable a specific analysis
  void enable(CIRAnalysisKind kind) {
    mask |= static_cast<unsigned>(kind);
  }

  /// Disable a specific analysis
  void disable(CIRAnalysisKind kind) {
    mask &= ~static_cast<unsigned>(kind);
  }

  /// Check if any analysis is enabled
  bool hasAny() const { return mask != 0; }

  /// Check if no analysis is enabled
  bool empty() const { return mask == 0; }

  /// Get the raw mask value
  unsigned getMask() const { return mask; }

  /// Union with another set
  CIRAnalysisSet &operator|=(const CIRAnalysisSet &other) {
    mask |= other.mask;
    return *this;
  }

  /// Union operator
  CIRAnalysisSet operator|(const CIRAnalysisSet &other) const {
    return CIRAnalysisSet(mask | other.mask);
  }

  /// Intersection with another set
  CIRAnalysisSet &operator&=(const CIRAnalysisSet &other) {
    mask &= other.mask;
    return *this;
  }

  /// Intersection operator
  CIRAnalysisSet operator&(const CIRAnalysisSet &other) const {
    return CIRAnalysisSet(mask & other.mask);
  }

  bool operator==(const CIRAnalysisSet &other) const {
    return mask == other.mask;
  }

  bool operator!=(const CIRAnalysisSet &other) const {
    return mask != other.mask;
  }

  /// Print the analysis set to an output stream
  void print(llvm::raw_ostream &OS) const;
};

/// Parse a single analysis name into a CIRAnalysisKind
/// Returns std::nullopt if the name is not recognized
CIRAnalysisKind parseCIRAnalysisKind(llvm::StringRef name);

/// Parse a list of analysis names (from command line) into a CIRAnalysisSet
/// Handles comma and semicolon separators
/// Invalid names are ignored and optionally reported via InvalidNames
CIRAnalysisSet parseCIRAnalysisList(
    const std::vector<std::string> &analysisList,
    llvm::SmallVectorImpl<std::string> *invalidNames = nullptr);

} // namespace cir

#endif // LLVM_CLANG_CIR_SEMA_CIRANALYSISKIND_H
