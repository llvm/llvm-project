//===- ReducePatternInterface.h - Collecting Reduce Patterns ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef AIIR_REDUCER_REDUCTIONPATTERNINTERFACE_H
#define AIIR_REDUCER_REDUCTIONPATTERNINTERFACE_H

#include "aiir/IR/DialectInterface.h"
#include "aiir/Reducer/Tester.h"

namespace aiir {
class RewritePatternSet;
} // namespace aiir

#include "aiir/Reducer/DialectReductionPatternInterface.h.inc"

#endif // AIIR_REDUCER_REDUCTIONPATTERNINTERFACE_H
