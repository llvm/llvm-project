//===- EntityLinkage.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Model/EntityLinkage.h"
#include "llvm/Support/ErrorHandling.h"

namespace clang::ssaf {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              EntityLinkage::LinkageType Linkage) {
  switch (Linkage) {
  case EntityLinkage::LinkageType::None:
    OS << "None";
    break;
  case EntityLinkage::LinkageType::Internal:
    OS << "Internal";
    break;
  case EntityLinkage::LinkageType::External:
    OS << "External";
    break;
  }
  return OS;
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const EntityLinkage &Linkage) {
  OS << "EntityLinkage(" << Linkage.getLinkage() << ")";
  return OS;
}

bool EntityLinkage::operator==(const EntityLinkage &Other) const {
  return Linkage == Other.Linkage;
}

bool EntityLinkage::operator!=(const EntityLinkage &Other) const {
  return !(*this == Other);
}

} // namespace clang::ssaf
