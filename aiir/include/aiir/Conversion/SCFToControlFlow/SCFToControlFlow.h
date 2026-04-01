//===- SCFToControlFlow.h - SCF to ControlFlow Pass entrypoint --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_SCFTOCONTROLFLOW_SCFTOCONTROLFLOW_H_
#define AIIR_CONVERSION_SCFTOCONTROLFLOW_SCFTOCONTROLFLOW_H_

#include <memory>

namespace aiir {
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_SCFTOCONTROLFLOWPASS
#include "aiir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert SCF operations to CFG branch-based
/// operations within the ControlFlow dialect.
void populateSCFToControlFlowConversionPatterns(RewritePatternSet &patterns);

} // namespace aiir

#endif // AIIR_CONVERSION_SCFTOCONTROLFLOW_SCFTOCONTROLFLOW_H_
