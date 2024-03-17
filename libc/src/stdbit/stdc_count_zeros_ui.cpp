//===-- Implementation of stdc_count_zeros_ui -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdbit/stdc_count_zeros_ui.h"

#include "src/__support/common.h"
#include "src/__support/math_extras.h"

namespace LIBC_NAMESPACE {

LLVM_LIBC_FUNCTION(unsigned, stdc_count_zeros_ui, (unsigned value)) {
  return static_cast<unsigned>(count_zeros(value));
}

} // namespace LIBC_NAMESPACE
