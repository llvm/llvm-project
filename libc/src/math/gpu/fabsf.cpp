//===-- Implementation of the fabsf function for GPU ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fabsf.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float, fabsf, (float x)) { return __builtin_fabsf(x); }

} // namespace LIBC_NAMESPACE
