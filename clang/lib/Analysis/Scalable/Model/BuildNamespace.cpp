//===- BuildNamespace.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "llvm/Support/ErrorHandling.h"
#include <tuple>

namespace clang::ssaf {

llvm::StringRef toString(BuildNamespaceKind BNK) {
  switch (BNK) {
  case BuildNamespaceKind::CompilationUnit:
    return "compilation_unit";
  case BuildNamespaceKind::LinkUnit:
    return "link_unit";
  }
  llvm_unreachable("Unknown BuildNamespaceKind");
}

std::optional<BuildNamespaceKind> parseBuildNamespaceKind(llvm::StringRef Str) {
  if (Str == "compilation_unit")
    return BuildNamespaceKind::CompilationUnit;
  if (Str == "link_unit")
    return BuildNamespaceKind::LinkUnit;
  return std::nullopt;
}

BuildNamespace
BuildNamespace::makeCompilationUnit(llvm::StringRef CompilationId) {
  return BuildNamespace{BuildNamespaceKind::CompilationUnit,
                        CompilationId.str()};
}

bool BuildNamespace::operator==(const BuildNamespace &Other) const {
  return asTuple() == Other.asTuple();
}

bool BuildNamespace::operator!=(const BuildNamespace &Other) const {
  return !(*this == Other);
}

bool BuildNamespace::operator<(const BuildNamespace &Other) const {
  return asTuple() < Other.asTuple();
}

NestedBuildNamespace
NestedBuildNamespace::makeCompilationUnit(llvm::StringRef CompilationId) {
  NestedBuildNamespace Result;
  Result.Namespaces.push_back(
      BuildNamespace::makeCompilationUnit(CompilationId));
  return Result;
}

bool NestedBuildNamespace::empty() const { return Namespaces.empty(); }

bool NestedBuildNamespace::operator==(const NestedBuildNamespace &Other) const {
  return Namespaces == Other.Namespaces;
}

bool NestedBuildNamespace::operator!=(const NestedBuildNamespace &Other) const {
  return !(*this == Other);
}

bool NestedBuildNamespace::operator<(const NestedBuildNamespace &Other) const {
  return Namespaces < Other.Namespaces;
}

} // namespace clang::ssaf
