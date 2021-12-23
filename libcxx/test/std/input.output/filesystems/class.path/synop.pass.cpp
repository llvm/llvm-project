//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03

// <filesystem>

// class path

// typedef ... value_type;
// typedef basic_string<value_type> string_type;
// static constexpr value_type preferred_separator = ...;

#include "filesystem_include.h"
#include <cassert>
#include <string>
#include <type_traits>

#include "test_macros.h"


int main(int, char**) {
  using namespace fs;
#ifdef _WIN32
  ASSERT_SAME_TYPE(path::value_type, wchar_t);
#else
  ASSERT_SAME_TYPE(path::value_type, char);
#endif
  ASSERT_SAME_TYPE(path::string_type, std::basic_string<path::value_type>);
  {
    ASSERT_SAME_TYPE(const path::value_type, decltype(path::preferred_separator));
#ifdef _WIN32
    static_assert(path::preferred_separator == '\\', "");
#else
    static_assert(path::preferred_separator == '/', "");
#endif
    // Make preferred_separator ODR used by taking its address.
    const path::value_type* dummy = &path::preferred_separator;
    ((void)dummy);
  }

  return 0;
}
