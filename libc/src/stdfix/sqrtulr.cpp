//===-- Implementation of sqrtulr function  -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sqrtulr.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/sqrt.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(unsigned long fract, sqrtulr, (unsigned long fract x)) {
  return fixed_point::sqrt(x);
}

} // namespace LIBC_NAMESPACE
