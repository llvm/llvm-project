//===-------- QueryProfile.cpp - clang-query --------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "QueryProfile.h"
#include "llvm/Support/raw_ostream.h"

namespace clang::query {

void QueryProfile::printUserFriendlyTable(llvm::raw_ostream &OS) {
  TG->print(OS);
  OS.flush();
}

QueryProfile::~QueryProfile() {
  TG.emplace("clang-query", "clang-query matcher profiling", Records);
  printUserFriendlyTable(llvm::errs());
}

} // namespace clang::query
