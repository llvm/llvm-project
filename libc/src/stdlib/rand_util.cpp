//===-- Shared utility for rand -------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdlib/rand_util.h"
#include "src/__support/CPP/atomic.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

// C standard 7.10p2: If 'rand' is called before 'srand' it is to
// proceed as if the 'srand' function was called with a value of '1'.
cpp::Atomic<unsigned long> rand_next = 1;

} // namespace LIBC_NAMESPACE_DECL
