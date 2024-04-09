//===-- Differential test for fmodf128 ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BinaryOpSingleOutputDiff.h"

#include "src/math/fmodf128.h"

#include <math.h>

BINARY_OP_SINGLE_OUTPUT_PERF(float, LIBC_NAMESPACE::fmodf128, ::fmodf128,
                             "fmodf128_perf.log")
