//===- CIRAliasAnalysis.h - CIR Alias Analysis Suite ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the registration function for the full suite of CIR alias
// analysis implementations. Callers that want all CIR analyses should use
// registerCIRAliasAnalyses() rather than adding individual implementations.
//
//===----------------------------------------------------------------------===//

#ifndef CLANG_CIR_DIALECT_ANALYSIS_CIRALIASANALYSIS_H
#define CLANG_CIR_DIALECT_ANALYSIS_CIRALIASANALYSIS_H

#include "mlir/Analysis/AliasAnalysis.h"

namespace cir {

/// Register all CIR alias analysis implementations with `aa`.
///
/// Passes that want full CIR alias information should call this rather than
/// adding individual implementations:
///
///   mlir::AliasAnalysis aa(funcOp);
///   cir::registerCIRAliasAnalyses(aa);
///
void registerCIRAliasAnalyses(mlir::AliasAnalysis &aa);

} // namespace cir

#endif // CLANG_CIR_DIALECT_ANALYSIS_CIRALIASANALYSIS_H
