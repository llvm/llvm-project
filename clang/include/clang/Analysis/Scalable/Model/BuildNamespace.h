//===- BuildNamespace.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines BuildNamespace and NestedBuildNamespace classes that
// represent build namespaces in the Scalable Static Analysis Framework.
//
// Build namespaces provide an abstraction for grouping program entities (such
// as those in a shared library or compilation unit) to enable analysis of
// software projects constructed from individual components.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_BUILDNAMESPACE_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_BUILDNAMESPACE_H

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"
#include <optional>
#include <string>
#include <vector>

namespace clang::ssaf {

enum class BuildNamespaceKind : unsigned short { CompilationUnit, LinkUnit };

llvm::StringRef toString(BuildNamespaceKind BNK);

std::optional<BuildNamespaceKind> parseBuildNamespaceKind(llvm::StringRef Str);

/// Represents a single namespace in the build process.
///
/// A BuildNamespace groups program entities, such as those belonging to a
/// compilation unit or link unit (e.g., a shared library). Each namespace has a
/// kind (CompilationUnit or LinkUnit) and a unique identifier name within that
/// kind.
///
/// BuildNamespaces can be composed into NestedBuildNamespace to represent
/// hierarchical namespace structures that model how software is constructed
/// from its components.
class BuildNamespace {
  BuildNamespaceKind Kind;
  std::string Name;

  auto asTuple() const { return std::tie(Kind, Name); }

public:
  BuildNamespace(BuildNamespaceKind Kind, llvm::StringRef Name)
      : Kind(Kind), Name(Name.str()) {}

  /// Creates a BuildNamespace representing a compilation unit.
  ///
  /// \param CompilationId The unique identifier for the compilation unit.
  /// \returns A BuildNamespace with CompilationUnit kind.
  static BuildNamespace makeCompilationUnit(llvm::StringRef CompilationId);

  bool operator==(const BuildNamespace &Other) const;
  bool operator!=(const BuildNamespace &Other) const;
  bool operator<(const BuildNamespace &Other) const;

  friend class EntityLinker;
  friend class SerializationFormat;
  friend class TestFixture;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                       const BuildNamespace &BN);
};

/// Represents a hierarchical sequence of build namespaces.
///
/// A NestedBuildNamespace captures namespace qualification for program entities
/// by maintaining an ordered sequence of BuildNamespace steps. This models how
/// entities are organized through multiple steps of the build process, such as
/// first being part of a compilation unit, then incorporated into a link unit.
///
/// For example, an entity might be qualified by a compilation unit namespace
/// followed by a shared library namespace.
class NestedBuildNamespace {
  std::vector<BuildNamespace> Namespaces;

public:
  NestedBuildNamespace() = default;

  explicit NestedBuildNamespace(const std::vector<BuildNamespace> &Namespaces)
      : Namespaces(Namespaces) {}

  explicit NestedBuildNamespace(const BuildNamespace &N) {
    Namespaces.push_back(N);
  }

  /// Creates a NestedBuildNamespace representing a compilation unit.
  ///
  /// \param CompilationId The unique identifier for the compilation unit.
  /// \returns A NestedBuildNamespace containing a single CompilationUnit
  ///          BuildNamespace.
  static NestedBuildNamespace
  makeCompilationUnit(llvm::StringRef CompilationId);

  /// Creates a new NestedBuildNamespace by appending additional namespace.
  ///
  /// \param Namespace The namespace to append.
  NestedBuildNamespace makeQualified(NestedBuildNamespace Namespace) const {
    auto Copy = *this;
    Copy.Namespaces.reserve(Copy.Namespaces.size() +
                            Namespace.Namespaces.size());
    llvm::append_range(Copy.Namespaces, Namespace.Namespaces);
    return Copy;
  }

  bool empty() const;

  bool operator==(const NestedBuildNamespace &Other) const;
  bool operator!=(const NestedBuildNamespace &Other) const;
  bool operator<(const NestedBuildNamespace &Other) const;

  friend class SerializationFormat;
  friend class TestFixture;
  friend llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                                       const NestedBuildNamespace &NBN);
};

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const BuildNamespace &BN);
llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const NestedBuildNamespace &NBN);

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_MODEL_BUILDNAMESPACE_H
