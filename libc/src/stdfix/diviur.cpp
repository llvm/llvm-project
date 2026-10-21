//===-- Implementation of diviur function ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "diviur.h"
#include "include/llvm-libc-macros/stdfix-macros.h"
#include "src/__support/common.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(unsigned int, diviur, (unsigned int n, unsigned fract d)) {
  return fixed_point::divifx<unsigned int, unsigned fract>(n, d);
}

} // namespace LIBC_NAMESPACE_DECL
