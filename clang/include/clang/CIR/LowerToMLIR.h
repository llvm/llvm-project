//====- LowerToMLIR.h- Lowering from CIR to MLIR --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares functions for lowering CIR modules to MLIR.
//
//===----------------------------------------------------------------------===//
#ifndef CLANG_CIR_LOWERTOMLIR_H
#define CLANG_CIR_LOWERTOMLIR_H

namespace cir {

void populateCIRLoopToSCFConversionPatterns(mlir::RewritePatternSet &patterns,
                                            mlir::TypeConverter &converter);
} // namespace cir

#endif // CLANG_CIR_LOWERTOMLIR_H_
