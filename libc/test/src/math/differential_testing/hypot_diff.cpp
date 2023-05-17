//===-- Differential test for hypot ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BinaryOpSingleOutputDiff.h"

#include "src/math/hypot.h"

#include <math.h>

BINARY_OP_SINGLE_OUTPUT_DIFF(double, __llvm_libc::hypot, ::hypot,
                             "hypot_diff.log")
