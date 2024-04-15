//===-- Implementation of stdc_bit_ceil_ull -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdbit/stdc_bit_ceil_ull.h"

#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(unsigned long long, stdc_bit_ceil_ull,
                   (unsigned long long value)) {
  return cpp::bit_ceil(value);
}

} // namespace LIBC_NAMESPACE
