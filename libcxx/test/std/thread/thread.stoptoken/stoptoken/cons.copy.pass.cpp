//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17

// stop_token(const stop_token&) noexcept;

#include <cassert>
#include <stop_token>
#include <type_traits>

#include "test_macros.h"

static_assert(std::is_nothrow_copy_constructible_v<std::stop_token>);

int main(int, char**) {
  {
    std::stop_source source;
    auto st = source.get_token();
    std::stop_token copy{st};

    assert(st == copy);

    assert(st.stop_possible());
    assert(!st.stop_requested());

    assert(copy.stop_possible());
    assert(!copy.stop_requested());

    source.request_stop();
    assert(st.stop_possible());
    assert(st.stop_requested());

    assert(copy.stop_possible());
    assert(copy.stop_requested());

  }

  return 0;
}
