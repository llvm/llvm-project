//===-- Implementation of math utils --------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "math_utils.h"

namespace __llvm_libc {

const float XFlowValues<float>::overflow_value =
    as_float(0x70000000); // 0x1p97f
const float XFlowValues<float>::underflow_value =
    as_float(0x10000000); // 0x1p97f
const float XFlowValues<float>::may_underflow_value =
    as_float(0x1a200000); // 0x1.4p-75f

const double XFlowValues<double>::overflow_value =
    as_double(0x7000000000000000); // 0x1p769
const double XFlowValues<double>::underflow_value =
    as_double(0x1000000000000000); // 0x1p-767
const double XFlowValues<double>::may_underflow_value =
    as_double(0x1e58000000000000); // 0x1.8p-538

} // namespace __llvm_libc
