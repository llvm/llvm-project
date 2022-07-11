//===-- Implementation of imaxabs -----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/inttypes/imaxabs.h"
#include "src/__support/common.h"
#include "src/__support/integer_operations.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(intmax_t, imaxabs, (intmax_t j)) { return integer_abs(j); }

} // namespace __llvm_libc
