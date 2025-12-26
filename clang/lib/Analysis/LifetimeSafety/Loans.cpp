//===- Loans.cpp - Loan Implementation --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Analysis/Analyses/LifetimeSafety/Loans.h"

namespace clang::lifetimes::internal {

void PathLoan::dump(llvm::raw_ostream &OS) const {
  OS << getID() << " (Path: ";
  OS << Path.D->getNameAsString() << ")";
}

void PlaceholderLoan::dump(llvm::raw_ostream &OS) const {
  OS << getID() << " (Placeholder loan)";
}

} // namespace clang::lifetimes::internal
