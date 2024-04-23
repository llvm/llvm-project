//===-- Implementation of the fmaf function for GPU -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/fmaf.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(float, fmaf, (float x, float y, float z)) {
  return __builtin_fmaf(x, y, z);
}

} // namespace LIBC_NAMESPACE
