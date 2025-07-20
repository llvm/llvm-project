//===-- Implementation of bf16fmaf function -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/bf16fmaf.h"

#include "src/__support/FPUtil/FMA.h"
#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, bf16fmaf, (float x, float y, float z)) {
  return fputil::fma<bfloat16>(x, y, z);
}

} // namespace LIBC_NAMESPACE_DECL
