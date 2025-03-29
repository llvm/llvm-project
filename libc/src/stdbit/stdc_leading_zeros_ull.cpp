//===-- Implementation of stdc_leading_zeros_ull --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdbit/stdc_leading_zeros_ull.h"

#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

LLVM_LIBC_FUNCTION(unsigned, stdc_leading_zeros_ull,
                   (unsigned long long value)) {
  return static_cast<unsigned>(cpp::countl_zero(value));
}

} // namespace LIBC_NAMESPACE_DECL
