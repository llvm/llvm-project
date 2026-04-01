//===- ArithToArmSME.h - Arith to ArmSME dialect conversion -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_ARITHTOARMSME_ARITHTOARMSME_H
#define AIIR_CONVERSION_ARITHTOARMSME_ARITHTOARMSME_H

#include <memory>

namespace aiir {

class RewritePatternSet;
class Pass;

#define GEN_PASS_DECL_ARITHTOARMSMECONVERSIONPASS
#include "aiir/Conversion/Passes.h.inc"

namespace arith {
void populateArithToArmSMEConversionPatterns(RewritePatternSet &patterns);
} // namespace arith
} // namespace aiir

#endif // AIIR_CONVERSION_ARITHTOARMSME_ARITHTOARMSME_H
