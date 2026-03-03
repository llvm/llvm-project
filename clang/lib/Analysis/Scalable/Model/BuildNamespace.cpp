//===- BuildNamespace.cpp ---------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Model/BuildNamespace.h"
#include "../ModelStringConversions.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/ErrorHandling.h"
#include <tuple>

namespace clang::ssaf {

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

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, BuildNamespaceKind BNK) {
  return OS << buildNamespaceKindToString(BNK);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const BuildNamespace &BN) {
  return OS << "BuildNamespace(" << BN.Kind << ", " << BN.Name << ")";
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const NestedBuildNamespace &NBN) {
  OS << "NestedBuildNamespace([";
  llvm::interleaveComma(NBN.Namespaces, OS);
  OS << "])";
  return OS;
}

} // namespace clang::ssaf
