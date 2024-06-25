//===-- Differential test for roundf---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SingleInputSingleOutputPerf.h"

#include "src/math/roundf.h"

#include <math.h>

SINGLE_INPUT_SINGLE_OUTPUT_PERF(float, LIBC_NAMESPACE::roundf, ::roundf,
                                "roundf_perf.log")
