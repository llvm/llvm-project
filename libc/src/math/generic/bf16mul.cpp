//===-- Implementation of bf16mul function --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/bf16mul.h"

#include "src/__support/FPUtil/bfloat16.h"
#include "src/__support/FPUtil/generic/mul.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(bfloat16, bf16mul, (double x, double y)) {
  return fputil::generic::mul<bfloat16>(x, y);
}

} // namespace LIBC_NAMESPACE_DECL
