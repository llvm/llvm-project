//===-- Differential test for expm1f --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SingleInputSingleOutputDiff.h"

#include "src/math/expm1f.h"

#include <math.h>

SINGLE_INPUT_SINGLE_OUTPUT_DIFF(float, __llvm_libc::expm1f, ::expm1f,
                                "expm1f_diff.log")
