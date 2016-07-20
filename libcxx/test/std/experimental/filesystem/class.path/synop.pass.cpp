//===----------------------------------------------------------------------===//
//
//                     The LLVM Compiler Infrastructure
//
// This file is dual licensed under the MIT and the University of Illinois Open
// Source Licenses. See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03

// <experimental/filesystem>

// class path

// typedef ... value_type;
// typedef basic_string<value_type> string_type;
// static constexpr value_type preferred_separator = ...;

#include <experimental/filesystem>
#include <type_traits>
#include <cassert>

#include "test_macros.h"

namespace fs = std::experimental::filesystem;

int main() {
  using namespace fs;
  ASSERT_SAME_TYPE(path::value_type, char);
  ASSERT_SAME_TYPE(path::string_type, std::basic_string<path::value_type>);
  {
    ASSERT_SAME_TYPE(const path::value_type, decltype(path::preferred_separator));
    static_assert(path::preferred_separator == '/', "");
    // Make preferred_separator ODR used by taking it's address.
    const char* dummy = &path::preferred_separator;
    ((void)dummy);
  }
}
