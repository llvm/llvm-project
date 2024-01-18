//===- ExtractAPI/AvailabilityInfo.h - Availability Info --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the AvailabilityInfo struct that collects availability
/// attributes of a symbol.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_EXTRACTAPI_AVAILABILITY_INFO_H
#define LLVM_CLANG_EXTRACTAPI_AVAILABILITY_INFO_H

#include "clang/AST/Decl.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/raw_ostream.h"

namespace clang {
namespace extractapi {

/// Stores availability attributes of a symbol.
struct AvailabilityInfo {
  /// The domain for which this availability info item applies
  std::string Domain;
  VersionTuple Introduced;
  VersionTuple Deprecated;
  VersionTuple Obsoleted;
  bool UnconditionallyDeprecated = false;
  bool UnconditionallyUnavailable = false;

  AvailabilityInfo() = default;

  /// Determine if this AvailabilityInfo represents the default availability.
  bool isDefault() const { return *this == AvailabilityInfo(); }
  /// Check if the symbol is unconditionally deprecated.
  ///
  /// i.e. \code __attribute__((deprecated)) \endcode
  bool isUnconditionallyDeprecated() const { return UnconditionallyDeprecated; }
  /// Check if the symbol is unconditionally unavailable.
  ///
  /// i.e. \code __attribute__((unavailable)) \endcode
  bool isUnconditionallyUnavailable() const {
    return UnconditionallyUnavailable;
  }

  AvailabilityInfo(StringRef Domain, VersionTuple I, VersionTuple D,
                   VersionTuple O, bool UD, bool UU)
      : Domain(Domain), Introduced(I), Deprecated(D), Obsoleted(O),
        UnconditionallyDeprecated(UD), UnconditionallyUnavailable(UU) {}

  friend bool operator==(const AvailabilityInfo &Lhs,
                         const AvailabilityInfo &Rhs);

public:
  static AvailabilityInfo createFromDecl(const Decl *Decl);
};

inline bool operator==(const AvailabilityInfo &Lhs,
                       const AvailabilityInfo &Rhs) {
  return std::tie(Lhs.Introduced, Lhs.Deprecated, Lhs.Obsoleted,
                  Lhs.UnconditionallyDeprecated,
                  Lhs.UnconditionallyUnavailable) ==
         std::tie(Rhs.Introduced, Rhs.Deprecated, Rhs.Obsoleted,
                  Rhs.UnconditionallyDeprecated,
                  Rhs.UnconditionallyUnavailable);
}

} // namespace extractapi
} // namespace clang

#endif // LLVM_CLANG_EXTRACTAPI_AVAILABILITY_INFO_H
