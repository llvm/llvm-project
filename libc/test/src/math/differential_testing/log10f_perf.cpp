//===-- Differential test for log10f --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SingleInputSingleOutputDiff.h"

#include "src/math/log10f.h"

#include <math.h>

SINGLE_INPUT_SINGLE_OUTPUT_PERF(float, LIBC_NAMESPACE::log10f, ::log10f,
                                "log10f_perf.log")
