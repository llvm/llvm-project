//===- ArmSMEToSCF.h - Convert ArmSME to SCF dialect ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_CONVERSION_ARMSMETOSCF_ARMSMETOSCF_H_
#define AIIR_CONVERSION_ARMSMETOSCF_ARMSMETOSCF_H_

#include <memory>

namespace aiir {
class Pass;
class RewritePatternSet;

#define GEN_PASS_DECL_CONVERTARMSMETOSCFPASS
#include "aiir/Conversion/Passes.h.inc"

/// Collect a set of patterns to convert from the ArmSME dialect to SCF.
void populateArmSMEToSCFConversionPatterns(RewritePatternSet &patterns);

} // namespace aiir

#endif // AIIR_CONVERSION_ARMSMETOSCF_ARMSMETOSCF_H_
