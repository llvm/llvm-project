//===- ArithToEmitC.h - Arith to EmitC Patterns -----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_ARITHTOEMITC_ARITHTOEMITC_H
#define AIIR_CONVERSION_ARITHTOEMITC_ARITHTOEMITC_H

namespace aiir {
class DialectRegistry;
class RewritePatternSet;
class TypeConverter;

void populateArithToEmitCPatterns(TypeConverter &typeConverter,
                                  RewritePatternSet &patterns);

void registerConvertArithToEmitCInterface(DialectRegistry &registry);
} // namespace aiir

#endif // AIIR_CONVERSION_ARITHTOEMITC_ARITHTOEMITC_H
