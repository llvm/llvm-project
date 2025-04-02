//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <system_error>

// class error_category

// const error_category& system_category();

#include <system_error>
#include <cassert>
#include <string>
#include <cerrno>

#include "test_macros.h"

// See https://llvm.org/D65667
struct StaticInit {
  const std::error_category* ec;
  ~StaticInit() {
    std::string str = ec->name();
    assert(str == "system");
  }
};
static StaticInit foo;

int main(int, char**) {
  {
    const std::error_category& e_cat1 = std::system_category();
    std::error_condition e_cond       = e_cat1.default_error_condition(5);
#ifdef _WIN32
    // Windows' system error 5 is ERROR_ACCESS_DENIED, which maps to generic code permission_denied.
    LIBCPP_ASSERT(e_cond.value() == static_cast<int>(std::errc::permission_denied));
#else
    LIBCPP_ASSERT(e_cond.value() == 5);
#endif
    LIBCPP_ASSERT(e_cond.category() == std::generic_category());
    assert(e_cat1.equivalent(5, e_cond));

    e_cond = e_cat1.default_error_condition(5000);
    LIBCPP_ASSERT(e_cond.value() == 5000);
    LIBCPP_ASSERT(e_cond.category() == std::system_category());
    assert(e_cat1.equivalent(5000, e_cond));
  }

  // Test the result of message(int cond) when given a bad error condition
  {
    errno                             = E2BIG; // something that message will never generate
    const std::error_category& e_cat1 = std::system_category();
    const std::string msg             = e_cat1.message(-1);
    // Exact message format varies by platform.  We can't detect
    // some of these (Musl in particular) using the preprocessor,
    // so accept a few sensible messages.  Newlib unfortunately
    // responds with an empty message, which we probably want to
    // treat as a failure code otherwise, but we can detect that
    // with the preprocessor.
#if defined(_NEWLIB_VERSION)
    const bool is_newlib = true;
#else
    const bool is_newlib = false;
#endif
    (void)is_newlib;
    LIBCPP_ASSERT(msg.rfind("Error -1 occurred", 0) == 0       // AIX
                  || msg.rfind("No error information", 0) == 0 // Musl
                  || msg.rfind("Unknown error", 0) == 0        // Glibc
                  || (is_newlib && msg.empty()));
    assert(errno == E2BIG);
  }

  {
    foo.ec        = &std::system_category();
    std::string m = foo.ec->name();
    assert(m == "system");
  }

  return 0;
}
