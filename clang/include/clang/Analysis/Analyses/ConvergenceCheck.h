//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Analyse implicit convergence in the CFG.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_ANALYSIS_ANALYSES_CONVERGENCECHECK_H
#define LLVM_CLANG_ANALYSIS_ANALYSES_CONVERGENCECHECK_H

namespace clang {
class AnalysisDeclContext;
class Sema;
class Stmt;

void analyzeForConvergence(Sema &S, AnalysisDeclContext &AC);

} // end namespace clang

#endif // LLVM_CLANG_ANALYSIS_ANALYSES_CONVERGENCECHECK_H
