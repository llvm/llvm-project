//===-- Implementation of divilk function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "divilk.h"
#include "include/llvm-libc-macros/stdfix-macros.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(long int, divilk, (long int n, long accum d)) {
  return fixed_point::divifx<long int, long accum>(n, d);
}

} // namespace LIBC_NAMESPACE_DECL
