//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// void* operator new(std::size_t);
// void* operator new(std::size_t, std::align_val_t);
// void* operator new[](std::size_t);
// void* operator new[](std::size_t, std::align_val_t);

// This test ensures that we abort the program instead of returning nullptr
// when we fail to satisfy the allocation request. The throwing versions of
// `operator new` must never return nullptr on failure to allocate (per the
// Standard) and the compiler actually relies on that for optimizations.
// Returning nullptr from the throwing `operator new` can basically result
// in miscompiles.

// REQUIRES: has-unix-headers
// REQUIRES: no-exceptions
// UNSUPPORTED: c++03, c++11, c++14

#include <cstddef>
#include <limits>
#include <new>

#include "check_assertion.h"

int main(int, char**) {
  EXPECT_ANY_DEATH((void)operator new(std::numeric_limits<std::size_t>::max()));
  EXPECT_ANY_DEATH((void)operator new(std::numeric_limits<std::size_t>::max(), static_cast<std::align_val_t>(32)));
  EXPECT_ANY_DEATH((void)operator new[](std::numeric_limits<std::size_t>::max()));
  EXPECT_ANY_DEATH((void)operator new[](std::numeric_limits<std::size_t>::max(), static_cast<std::align_val_t>(32)));
  return 0;
}
