//===- SymbolGraph/AvailabilityInfo.h - Availability Info -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// \brief Defines the Availability Info for a declaration.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SYMBOLGRAPH_AVAILABILITY_INFO_H
#define LLVM_CLANG_SYMBOLGRAPH_AVAILABILITY_INFO_H

#include "llvm/Support/Error.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/raw_ostream.h"

using llvm::VersionTuple;

namespace clang {
namespace symbolgraph {

struct AvailabilityInfo {
  VersionTuple Introduced;
  VersionTuple Deprecated;
  VersionTuple Obsoleted;
  bool Unavailable{false};
  bool UnconditionallyDeprecated{false};
  bool UnconditionallyUnavailable{false};

  explicit AvailabilityInfo(bool Unavailable = false)
      : Unavailable(Unavailable) {}

  AvailabilityInfo(VersionTuple I, VersionTuple D, VersionTuple O, bool U,
                   bool UD, bool UU)
      : Introduced(I), Deprecated(D), Obsoleted(O), Unavailable(U),
        UnconditionallyDeprecated(UD), UnconditionallyUnavailable(UU) {}

  bool isDefault() const { return *this == AvailabilityInfo(); }
  bool isUnavailable() const { return Unavailable; }
  bool isUnconditionallyDeprecated() const { return UnconditionallyDeprecated; }
  bool isUnconditionallyUnavailable() const {
    return UnconditionallyUnavailable;
  }

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

} // namespace symbolgraph
} // namespace clang

#endif // LLVM_CLANG_SYMBOLGRAPH_AVAILABILITY_INFO_H
