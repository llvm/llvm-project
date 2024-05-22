//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <locale>

// class messages_base
// {
// public:
//     typedef unspecified catalog;
// };

#include <cstdint>
#include <locale>
#include <type_traits>

#include "assert_macros.h"

#ifdef _LIBCPP_VERSION
ASSERT_SAME_TYPE(std::messages_base::catalog, std::intptr_t);
#endif

// Check that we implement LWG2028
static_assert(std::is_signed<std::messages_base::catalog>::value, "");
static_assert(std::is_integral<std::messages_base::catalog>::value, "");

int main(int, char**) {
  std::messages_base mb;

  return 0;
}
