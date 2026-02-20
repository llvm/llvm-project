//===- SummaryName.cpp ------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Scalable/Model/SummaryName.h"

namespace clang::ssaf {

llvm::raw_ostream &operator<<(llvm::raw_ostream &OS, const SummaryName &SN) {
  OS << "SummaryName(" << SN.str() << ")";
  return OS;
}

} // namespace clang::ssaf
