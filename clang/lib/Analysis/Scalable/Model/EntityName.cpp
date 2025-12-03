//===- EntityName.cpp -------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Model/EntityName.h"

namespace clang::ssaf {

EntityName::EntityName(llvm::StringRef USR, llvm::StringRef Suffix,
                       NestedBuildNamespace Namespace)
  : USR(USR.str()), Suffix(Suffix), Namespace(std::move(Namespace)) {}

bool EntityName::operator==(const EntityName& Other) const {
  return USR == Other.USR &&
         Suffix == Other.Suffix &&
         Namespace == Other.Namespace;
}

bool EntityName::operator!=(const EntityName& Other) const {
  return !(*this == Other);
}

bool EntityName::operator<(const EntityName& Other) const {
  if (USR != Other.USR)
    return USR < Other.USR;
  if (Suffix != Other.Suffix)
    return Suffix.str() < Other.Suffix.str();
  return Namespace < Other.Namespace;
}

EntityName EntityName::makeQualified(NestedBuildNamespace Namespace) const {
  auto Copy = *this;
  Copy.Namespace = Copy.Namespace.makeQualified(Namespace);

  return Copy;
}

} // namespace clang::ssaf
