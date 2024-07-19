//===-- Test fake definition for __libc_heaplimit -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/cstddef.h"
#include "src/__support/macros/config.h"

namespace LIBC_NAMESPACE_DECL {

extern "C" {
// This isn't used in the unit tests, but it must be defined for the non-fake
// version of the heap to work.
cpp::byte __libc_heap_limit;
}

} // namespace LIBC_NAMESPACE_DECL
