//===- BuildNamespace.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the BuildNamespace class that represents a build namespace
// in the Scalable Static Analysis Framework.
//
// Build namespaces provide an abstraction for grouping program entities (such
// as those in a shared library or compilation unit) to enable analysis of
// software projects constructed from individual components.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_MODEL_BUILDNAMESPACE_H
#define LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_MODEL_BUILDNAMESPACE_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

namespace clang::ssaf {

/// Represents a hierarchical build namespace.
///
/// A BuildNamespace is an ordered sequence of name strings that qualifies
/// program entities through multiple steps of the build process. For example,
/// an entity may be qualified first by a compilation-unit name and then by a
/// link-unit name.
///
/// A default-constructed BuildNamespace has zero levels and denotes "no
/// qualification". A single-level BuildNamespace can be constructed from a
/// StringRef.
class BuildNamespace {
  std::vector<std::string> Names;

public:
  BuildNamespace() = default;

  explicit BuildNamespace(llvm::StringRef Name) : Names{Name.str()} {}

  explicit BuildNamespace(std::vector<std::string> Names)
      : Names(std::move(Names)) {}

  /// Returns a new BuildNamespace with \p Namespace appended.
  BuildNamespace makeQualified(BuildNamespace Namespace) const {
    auto Copy = *this;
    Copy.Names.reserve(Copy.Names.size() + Namespace.Names.size());
    llvm::append_range(Copy.Names, Namespace.Names);
    return Copy;
  }

  bool empty() const;

  bool operator==(const BuildNamespace &Other) const;
  bool operator!=(const BuildNamespace &Other) const;
  bool operator<(const BuildNamespace &Other) const;

  friend class EntityLinker;
  friend class SerializationFormat;
  friend class TestFixture;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                       const BuildNamespace &BN);
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const BuildNamespace &BN);

} // namespace clang::ssaf

#endif // LLVM_CLANG_SCALABLESTATICANALYSIS_CORE_MODEL_BUILDNAMESPACE_H
