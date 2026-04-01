//===- ArithToSPIRV.h - Convert Arith to SPIRV dialect -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_ARITHTOSPIRV_ARITHTOSPIRV_H
#define AIIR_CONVERSION_ARITHTOSPIRV_ARITHTOSPIRV_H

#include "aiir/Pass/Pass.h"
#include <memory>

namespace aiir {

class SPIRVTypeConverter;
class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_CONVERTARITHTOSPIRVPASS
#include "aiir/Conversion/Passes.h.inc"

namespace arith {
void populateArithToSPIRVPatterns(const SPIRVTypeConverter &typeConverter,
                                  RewritePatternSet &patterns);

std::unique_ptr<OperationPass<>> createConvertArithToSPIRVPass();
} // namespace arith
} // namespace aiir

#endif // AIIR_CONVERSION_ARITHTOSPIRV_ARITHTOSPIRV_H
