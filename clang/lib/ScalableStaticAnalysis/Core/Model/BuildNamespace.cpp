//===- BuildNamespace.cpp -------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysis/Core/Model/BuildNamespace.h"
#include "llvm/ADT/STLExtras.h"

namespace clang::ssaf {

bool BuildNamespace::operator==(const BuildNamespace &Other) const {
  return Name == Other.Name;
}

bool BuildNamespace::operator!=(const BuildNamespace &Other) const {
  return !(*this == Other);
}

bool BuildNamespace::operator<(const BuildNamespace &Other) const {
  return Name < Other.Name;
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

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const BuildNamespace &BN) {
  return OS << "BuildNamespace(" << BN.Name << ")";
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const NestedBuildNamespace &NBN) {
  OS << "NestedBuildNamespace([";
  llvm::interleaveComma(NBN.Namespaces, OS);
  OS << "])";
  return OS;
}

} // namespace clang::ssaf
