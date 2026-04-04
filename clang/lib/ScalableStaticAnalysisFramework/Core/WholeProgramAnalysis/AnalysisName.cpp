//===- AnalysisName.cpp ---------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/ScalableStaticAnalysisFramework/Core/WholeProgramAnalysis/AnalysisName.h"

using namespace clang::ssaf;

llvm::raw_ostream &clang::ssaf::operator<<(llvm::raw_ostream &OS,
                                           const AnalysisName &AN) {
  return OS << "AnalysisName(" << AN.str() << ")";
}
