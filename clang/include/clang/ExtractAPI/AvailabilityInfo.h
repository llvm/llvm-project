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

#include "llvm/Support/Error.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/raw_ostream.h"

using llvm::VersionTuple;

namespace clang {
namespace extractapi {

/// Stores availability attributes of a symbol.
struct AvailabilityInfo {
  VersionTuple Introduced;
  VersionTuple Deprecated;
  VersionTuple Obsoleted;
  bool Unavailable{false};
  bool UnconditionallyDeprecated{false};
  bool UnconditionallyUnavailable{false};

  /// Determine if this AvailabilityInfo represents the default availability.
  bool isDefault() const { return *this == AvailabilityInfo(); }

  /// Check if the symbol is unavailable.
  bool isUnavailable() const { return Unavailable; }

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

  AvailabilityInfo() = default;

  AvailabilityInfo(VersionTuple I, VersionTuple D, VersionTuple O, bool U,
                   bool UD, bool UU)
      : Introduced(I), Deprecated(D), Obsoleted(O), Unavailable(U),
        UnconditionallyDeprecated(UD), UnconditionallyUnavailable(UU) {}

  friend bool operator==(const AvailabilityInfo &Lhs,
                         const AvailabilityInfo &Rhs);
};

inline bool operator==(const AvailabilityInfo &Lhs,
                       const AvailabilityInfo &Rhs) {
  return std::tie(Lhs.Introduced, Lhs.Deprecated, Lhs.Obsoleted,
                  Lhs.Unavailable, Lhs.UnconditionallyDeprecated,
                  Lhs.UnconditionallyUnavailable) ==
         std::tie(Rhs.Introduced, Rhs.Deprecated, Rhs.Obsoleted,
                  Rhs.Unavailable, Rhs.UnconditionallyDeprecated,
                  Rhs.UnconditionallyUnavailable);
}

} // namespace extractapi
} // namespace clang

#endif // LLVM_CLANG_EXTRACTAPI_AVAILABILITY_INFO_H
