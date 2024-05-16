//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads

// LWG 2056 changed the values of future_errc, so if we're using new headers
// with an old library we'll get incorrect messages.
//
// XFAIL: stdlib=apple-libc++ && target={{.+}}-apple-macosx10.{{9|10|11}}

// VC Runtime's std::exception::what() method is not marked as noexcept, so
// this fails.
// UNSUPPORTED: target=x86_64-pc-windows-msvc

// <future>
//
// class future_error
//
// const char* what() const noexcept;

#include <cassert>
#include <future>
#include <string_view>
#include <utility>

#include "test_macros.h"

int main(int, char**) {
  ASSERT_NOEXCEPT(std::declval<std::future_error const&>().what());
  ASSERT_SAME_TYPE(decltype(std::declval<std::future_error const&>().what()), char const*);

  // Before C++17, we can't construct std::future_error directly in a standards-conforming way
#if TEST_STD_VER >= 17
  {
    std::future_error const f(std::future_errc::broken_promise);
    [[maybe_unused]] char const* what = f.what();
    LIBCPP_ASSERT(what == std::string_view{"The associated promise has been destructed prior "
                                           "to the associated state becoming ready."});
  }
  {
    std::future_error f(std::future_errc::future_already_retrieved);
    [[maybe_unused]] char const* what = f.what();
    LIBCPP_ASSERT(what == std::string_view{"The future has already been retrieved from "
                                           "the promise or packaged_task."});
  }
  {
    std::future_error f(std::future_errc::promise_already_satisfied);
    [[maybe_unused]] char const* what = f.what();
    LIBCPP_ASSERT(what == std::string_view{"The state of the promise has already been set."});
  }
  {
    std::future_error f(std::future_errc::no_state);
    [[maybe_unused]] char const* what = f.what();
    LIBCPP_ASSERT(what == std::string_view{"Operation not permitted on an object without "
                                           "an associated state."});
  }
#endif

  return 0;
}
