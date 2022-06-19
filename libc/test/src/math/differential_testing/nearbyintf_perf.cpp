//===-- Differential test for nearbyintf-----------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "SingleInputSingleOutputDiff.h"

#include "src/math/nearbyintf.h"

#include <math.h>

SINGLE_INPUT_SINGLE_OUTPUT_PERF(float, __llvm_libc::nearbyintf, ::nearbyintf,
                                "nearbyintf_perf.log")
