//===- IndexToSPIRV.h - Index to SPIRV dialect conversion -------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_INDEXTOSPIRV_INDEXTOSPIRV_H
#define AIIR_CONVERSION_INDEXTOSPIRV_INDEXTOSPIRV_H

#include "aiir/Pass/Pass.h"
#include <memory>

namespace aiir {
class RewritePatternSet;
class SPIRVTypeConverter;
class Pass;

#define GEN_PASS_DECL_CONVERTINDEXTOSPIRVPASS
#include "aiir/Conversion/Passes.h.inc"

namespace index {
void populateIndexToSPIRVPatterns(const SPIRVTypeConverter &converter,
                                  RewritePatternSet &patterns);
std::unique_ptr<OperationPass<>> createConvertIndexToSPIRVPass();
} // namespace index
} // namespace aiir

#endif // AIIR_CONVERSION_INDEXTOSPIRV_INDEXTOSPIRV_H
