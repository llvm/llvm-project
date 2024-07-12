//===-- Single-precision 2^x function -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/exp2f.h"
#include "src/__support/common.h" // for LLVM_LIBC_FUNCTION
#include "src/math/generic/exp2f_impl.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float, exp2f, (float x)) { return generic::exp2f(x); }

} // namespace LIBC_NAMESPACE
