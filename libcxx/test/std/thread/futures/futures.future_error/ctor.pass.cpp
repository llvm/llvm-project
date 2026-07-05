//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14

// <future>

// class future_error
//
// explicit future_error(future_errc e); // since C++17

#include <cassert>
#include <future>
#include <type_traits>

int main(int, char**) {
  {
    std::future_error f(std::future_errc::broken_promise);
    assert(f.code() == std::make_error_code(std::future_errc::broken_promise));
  }
  {
    std::future_error f(std::future_errc::future_already_retrieved);
    assert(f.code() == std::make_error_code(std::future_errc::future_already_retrieved));
  }
  {
    std::future_error f(std::future_errc::promise_already_satisfied);
    assert(f.code() == std::make_error_code(std::future_errc::promise_already_satisfied));
  }
  {
    std::future_error f(std::future_errc::no_state);
    assert(f.code() == std::make_error_code(std::future_errc::no_state));
  }

  // Make sure the constructor is explicit
  static_assert(!std::is_convertible_v<std::future_errc, std::future_error>);

  return 0;
}
