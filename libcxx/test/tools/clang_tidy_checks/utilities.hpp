//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LIBCXX_TEST_TOOLS_CLANG_TIDY_CHECKS_UTILITIES_HPP
#define LIBCXX_TEST_TOOLS_CLANG_TIDY_CHECKS_UTILITIES_HPP

#include <string_view>

inline bool is_ugly_name(std::string_view str) {
  if (str.size() < 2)
    return false;
  if (str[0] == '_' && str[1] >= 'A' && str[1] <= 'Z')
    return true;
  return str.find("__") != std::string_view::npos;
}

#endif // LIBCXX_TEST_TOOLS_CLANG_TIDY_CHECKS_UTILITIES_HPP
