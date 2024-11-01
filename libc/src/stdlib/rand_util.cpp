//===-- Shared utility for rand -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/rand_util.h"
#include "src/__support/macros/attributes.h"

namespace __llvm_libc {

LIBC_THREAD_LOCAL unsigned long rand_next;

} // namespace __llvm_libc
