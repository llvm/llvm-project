//===-- Implementation of srand -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/srand.h"
#include "src/__support/common.h"
#include "src/stdlib/rand_util.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(void, srand, (unsigned int seed)) { rand_next = seed; }

} // namespace __llvm_libc
