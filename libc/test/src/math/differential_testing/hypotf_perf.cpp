//===-- Differential test for hypotf --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "BinaryOpSingleOutputDiff.h"

#include "src/math/hypotf.h"

#include <math.h>

BINARY_OP_SINGLE_OUTPUT_PERF(float, LIBC_NAMESPACE::hypotf, ::hypotf,
                             "hypotf_perf.log")
