//===- CIRAliasAnalysis.cpp - CIR Alias Analysis Suite --------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/CIR/Dialect/Analysis/CIRAliasAnalysis.h"
#include "clang/CIR/Dialect/Analysis/CIRBasicAliasAnalysis.h"

void cir::registerCIRAliasAnalyses(mlir::AliasAnalysis &aa) {
  aa.addAnalysisImplementation(CIRBasicAliasAnalysis());
}
