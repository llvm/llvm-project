//===-- Implementation of stdc_leading_ones_ull ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdbit/stdc_leading_ones_ull.h"

#include "src/__support/CPP/bit.h"
#include "src/__support/common.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(unsigned long long, stdc_leading_ones_ull,
                   (unsigned long long value)) {
  return static_cast<unsigned long long>(cpp::countl_one(value));
}

} // namespace LIBC_NAMESPACE
