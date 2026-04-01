//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_MPITOLLVM_H
#define AIIR_CONVERSION_MPITOLLVM_H

#include "aiir/IR/DialectRegistry.h"

namespace aiir {

class LLVMTypeConverter;
class RewritePatternSet;

namespace mpi {

void populateMPIToLLVMConversionPatterns(LLVMTypeConverter &converter,
                                         RewritePatternSet &patterns);

void registerConvertMPIToLLVMInterface(DialectRegistry &registry);

} // namespace mpi
} // namespace aiir

#endif // AIIR_CONVERSION_MPITOLLVM_H
