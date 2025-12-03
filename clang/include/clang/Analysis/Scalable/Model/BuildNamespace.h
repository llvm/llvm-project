//===- BuildNamespace.h -----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_SCALABLE_BUILD_NAMESPACE_H
#define LLVM_CLANG_ANALYSIS_SCALABLE_BUILD_NAMESPACE_H

#include "llvm/ADT/StringRef.h"
#include <optional>
#include <string>
#include <vector>

namespace clang::ssaf {

enum class BuildNamespaceKind : unsigned short {
  CompilationUnit,
  LinkUnit
};

llvm::StringRef toString(BuildNamespaceKind BNK);

std::optional<BuildNamespaceKind> parseBuildNamespaceKind(llvm::StringRef Str);

/// Represents a single step in the build process.
class BuildNamespace {
  BuildNamespaceKind Kind;
  std::string Name;
public:
  BuildNamespace(BuildNamespaceKind Kind, llvm::StringRef Name)
    : Kind(Kind), Name(Name.str()) {}

  static BuildNamespace makeTU(llvm::StringRef CompilationId);

  bool operator==(const BuildNamespace& Other) const;
  bool operator!=(const BuildNamespace& Other) const;
  bool operator<(const BuildNamespace& Other) const;

  friend class SerializationFormat;
};

/// Represents a sequence of steps in the build process.
class NestedBuildNamespace {
  friend class SerializationFormat;

  std::vector<BuildNamespace> Namespaces;

public:
  NestedBuildNamespace() = default;

  explicit NestedBuildNamespace(const std::vector<BuildNamespace>& Namespaces)
    : Namespaces(Namespaces) {}

  explicit NestedBuildNamespace(const BuildNamespace& N) {
    Namespaces.push_back(N);
  }

  static NestedBuildNamespace makeTU(llvm::StringRef CompilationId);

  NestedBuildNamespace makeQualified(NestedBuildNamespace Namespace) {
    auto Copy = *this;
    for (const auto& N : Namespace.Namespaces)
      Copy.Namespaces.push_back(N);
    return Copy;
  }

  bool empty() const;

  bool operator==(const NestedBuildNamespace& Other) const;
  bool operator!=(const NestedBuildNamespace& Other) const;
  bool operator<(const NestedBuildNamespace& Other) const;

  friend class JSONWriter;
  friend class LinkUnitResolution;
};

} // namespace clang::ssaf

#endif // LLVM_CLANG_ANALYSIS_SCALABLE_BUILD_NAMESPACE_H
