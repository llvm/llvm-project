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
    return OS << "None";
  case EntityLinkage::LinkageType::Internal:
    return OS << "Internal";
  case EntityLinkage::LinkageType::External:
    return OS << "External";
  }
  llvm_unreachable("Unhandled EntityLinkage::LinkageType variant");
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const EntityLinkage &Linkage) {
  return OS << "EntityLinkage(" << Linkage.getLinkage() << ")";
}

bool EntityLinkage::operator==(const EntityLinkage &Other) const {
  return Linkage == Other.Linkage;
}

bool EntityLinkage::operator!=(const EntityLinkage &Other) const {
  return !(*this == Other);
}

} // namespace clang::ssaf
