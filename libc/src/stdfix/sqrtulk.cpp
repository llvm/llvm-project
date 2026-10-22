//===-- Implementation of sqrtulk function  -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "sqrtulk.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/sqrt.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(unsigned long accum, sqrtulk, (unsigned long accum x)) {
  return fixed_point::sqrt(x);
}

} // namespace LIBC_NAMESPACE_DECL
