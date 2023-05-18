//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// <future>
//
// class future_error
//
// const error_code& code() const noexcept;

#include <cassert>
#include <future>
#include <utility>

#include "test_macros.h"

int main(int, char**) {
  ASSERT_NOEXCEPT(std::declval<std::future_error const&>().code());
  ASSERT_SAME_TYPE(decltype(std::declval<std::future_error const&>().code()), std::error_code const&);

  // Before C++17, we can't construct std::future_error directly in a standards-conforming way
#if TEST_STD_VER >= 17
  {
    std::future_error const f(std::future_errc::broken_promise);
    std::error_code const& code = f.code();
    assert(code == std::make_error_code(std::future_errc::broken_promise));
  }
  {
    std::future_error const f(std::future_errc::future_already_retrieved);
    std::error_code const& code = f.code();
    assert(code == std::make_error_code(std::future_errc::future_already_retrieved));
  }
  {
    std::future_error const f(std::future_errc::promise_already_satisfied);
    std::error_code const& code = f.code();
    assert(code == std::make_error_code(std::future_errc::promise_already_satisfied));
  }
  {
    std::future_error const f(std::future_errc::no_state);
    std::error_code const& code = f.code();
    assert(code == std::make_error_code(std::future_errc::no_state));
  }
#endif

  return 0;
}
