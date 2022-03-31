//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// Verify TEST_WORKAROUND_MSVC_BROKEN_IS_TRIVIALLY_COPYABLE.

#include <type_traits>

#include "test_macros.h"
#include "test_workarounds.h"

struct S {
  S(S const&) = default;
  S(S&&) = default;
  S& operator=(S const&) = delete;
  S& operator=(S&&) = delete;
};

int main(int, char**) {
#ifdef TEST_WORKAROUND_MSVC_BROKEN_IS_TRIVIALLY_COPYABLE
  static_assert(!std::is_trivially_copyable<S>::value, "");
#else
  static_assert(std::is_trivially_copyable<S>::value, "");
#endif

  return 0;
}
