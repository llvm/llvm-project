//===- EntityLinkage.cpp ----------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Model/EntityLinkage.h"

#include "../ModelStringConversions.h"

namespace clang::ssaf {

bool EntityLinkage::operator==(const EntityLinkage &Other) const {
  return Linkage == Other.Linkage;
}

bool EntityLinkage::operator!=(const EntityLinkage &Other) const {
  return !(*this == Other);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              EntityLinkageType Linkage) {
  return OS << entityLinkageTypeToString(Linkage);
}

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS,
                              const EntityLinkage &Linkage) {
  return OS << "EntityLinkage(" << Linkage.getLinkage() << ")";
}

} // namespace clang::ssaf
