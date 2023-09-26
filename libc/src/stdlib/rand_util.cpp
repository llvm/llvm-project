//===-- Shared utility for rand -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/rand_util.h"
#include "src/__support/macros/attributes.h"

namespace LIBC_NAMESPACE {

// C standard 7.10p2: If 'rand' is called before 'srand' it is to proceed as if
// the 'srand' function was called with a value of '1'.
LIBC_THREAD_LOCAL unsigned long rand_next = 1;

} // namespace LIBC_NAMESPACE
