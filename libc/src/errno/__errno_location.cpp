//===-- Implementation of __errno_location --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/errno/__errno_location.h"

#include "src/__support/common.h"

namespace __llvm_libc {

static thread_local int __errno = 0;

// __errno_location is not really an entry point but we still want it to behave
// like an entry point because the errno macro resolves to the C symbol
// "__errno_location".
LLVM_LIBC_FUNCTION(int *, __errno_location, ()) { return &__errno; }

} // namespace __llvm_libc
