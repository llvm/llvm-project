//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: libcpp-has-no-experimental-stop_token
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: availability-synchronization_library-missing

// stop_source(const stop_source&) noexcept;

#include <cassert>
#include <optional>
#include <stop_token>
#include <type_traits>

#include "test_macros.h"

static_assert(std::is_nothrow_copy_constructible_v<std::stop_source>);

int main(int, char**) {
  {
    std::stop_source source;
    std::stop_source copy{source};

    assert(source == copy);

    assert(source.stop_possible());
    assert(!source.stop_requested());

    assert(copy.stop_possible());
    assert(!copy.stop_requested());

    source.request_stop();
    assert(source.stop_possible());
    assert(source.stop_requested());

    assert(copy.stop_possible());
    assert(copy.stop_requested());
  }

  // source counter incremented
  {
    std::optional<std::stop_source> source(std::in_place);
    auto st = source->get_token();
    assert(st.stop_possible());

    std::optional<std::stop_source> copy{source};
    source.reset();

    assert(st.stop_possible());

    copy.reset();
    assert(!st.stop_possible());
  }

  // copy from empty
  {
    std::stop_source ss1{std::nostopstate};
    std::stop_source copy{ss1};
    assert(!copy.stop_possible());
  }

  return 0;
}
