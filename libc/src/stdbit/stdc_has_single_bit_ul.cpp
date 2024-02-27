//===-- Implementation of stdc_has_single_bit_ul --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdbit/stdc_has_single_bit_ul.h"

#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(bool, stdc_has_single_bit_ul, (unsigned long value)) {
  return cpp::has_single_bit(value);
}

} // namespace LIBC_NAMESPACE
