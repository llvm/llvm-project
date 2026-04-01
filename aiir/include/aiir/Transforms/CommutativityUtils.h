//===- CommutativityUtils.h - Commutativity utilities -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This header file declares a function to populate the commutativity utility
// pattern. This function is intended to be used inside passes to simplify the
// matching of commutative operations by fixing the order of their operands.
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_TRANSFORMS_COMMUTATIVITYUTILS_H
#define AIIR_TRANSFORMS_COMMUTATIVITYUTILS_H

#include "aiir/Transforms/DialectConversion.h"

namespace aiir {

/// Populates the commutativity utility patterns.
void populateCommutativityUtilsPatterns(RewritePatternSet &patterns);

} // namespace aiir

#endif // AIIR_TRANSFORMS_COMMUTATIVITYUTILS_H
