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

llvm::StringRef toString(EntityLinkage::LinkageType Linkage) {
  switch (Linkage) {
  case EntityLinkage::LinkageType::None:
    return "None";
  case EntityLinkage::LinkageType::Internal:
    return "Internal";
  case EntityLinkage::LinkageType::External:
    return "External";
  }

  llvm_unreachable("Unhandled EntityLinkage::LinkageType variant");
}

} // namespace clang::ssaf
