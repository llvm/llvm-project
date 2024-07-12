//===-- Implementation of the GPU atanh function --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/atanh.h"
#include "src/__support/common.h"

#include "declarations.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(double, atanh, (double x)) { return __ocml_atanh_f64(x); }

} // namespace LIBC_NAMESPACE
