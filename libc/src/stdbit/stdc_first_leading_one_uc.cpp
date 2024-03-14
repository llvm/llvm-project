//===-- Implementation of stdc_first_leading_one_uc -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdbit/stdc_first_leading_one_uc.h"

#include "src/__support/common.h"
#include "src/__support/math_extras.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(unsigned, stdc_first_leading_one_uc, (unsigned char value)) {
  return static_cast<unsigned>(first_leading_one(value));
}

} // namespace LIBC_NAMESPACE
