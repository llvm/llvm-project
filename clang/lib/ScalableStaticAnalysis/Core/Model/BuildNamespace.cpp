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

bool BuildNamespace::empty() const { return Names.empty(); }

bool BuildNamespace::operator==(const BuildNamespace &Other) const {
  return Names == Other.Names;
}

bool BuildNamespace::operator!=(const BuildNamespace &Other) const {
  return !(*this == Other);
}

bool BuildNamespace::operator<(const BuildNamespace &Other) const {
  return Names < Other.Names;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const BuildNamespace &BN) {
  OS << "BuildNamespace([";
  llvm::interleaveComma(BN.Names, OS);
  OS << "])";
  return OS;
}

} // namespace clang::ssaf
