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
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/VersionTuple.h"
#include "llvm/Support/raw_ostream.h"

using llvm::VersionTuple;

namespace clang {
namespace extractapi {

/// Stores availability attributes of a symbol in a given domain.
struct AvailabilityInfo {
  /// The domain for which this availability info item applies
  std::string Domain;
  VersionTuple Introduced;
  VersionTuple Deprecated;
  VersionTuple Obsoleted;

  AvailabilityInfo() = default;

  AvailabilityInfo(StringRef Domain, VersionTuple I, VersionTuple D,
                   VersionTuple O)
      : Domain(Domain), Introduced(I), Deprecated(D), Obsoleted(O) {}
};

class AvailabilitySet {
private:
  using AvailabilityList = llvm::SmallVector<AvailabilityInfo, 4>;
  AvailabilityList Availabilities;

  bool UnconditionallyDeprecated = false;
  bool UnconditionallyUnavailable = false;

public:
  AvailabilitySet(const Decl *Decl);
  AvailabilitySet() = default;

  AvailabilityList::const_iterator begin() const {
    return Availabilities.begin();
  }

  AvailabilityList::const_iterator end() const { return Availabilities.end(); }

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

  /// Determine if this AvailabilitySet represents default availability.
  bool isDefault() const { return Availabilities.empty(); }
};

} // namespace extractapi
} // namespace clang

#endif // LLVM_CLANG_EXTRACTAPI_AVAILABILITY_INFO_H
