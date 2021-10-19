//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_SUPPORT_DEBUG_MACROS_H
#define TEST_SUPPORT_DEBUG_MACROS_H

#include <__debug>
#include <cassert>
#include <string>

static const char* expected_libcpp_assert_message = 0;

static void test_debug_function(std::__libcpp_debug_info const& info) {
  if (0 == std::strcmp(info.__msg_, expected_libcpp_assert_message))
    std::exit(0);
  std::fprintf(stderr, "%s\n", info.what().c_str());
  std::abort();
}

#define TEST_LIBCPP_ASSERT_FAILURE(expr, m)                                                                            \
  do {                                                                                                                 \
    ::expected_libcpp_assert_message = m;                                                                              \
    std::__libcpp_set_debug_function(&::test_debug_function);                                                          \
    (void)(expr);                                                                                                      \
    assert(false);                                                                                                     \
  } while (false)

#endif // TEST_SUPPORT_DEBUG_MACROS_H
