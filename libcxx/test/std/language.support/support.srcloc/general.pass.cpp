//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17

#include <source_location>

#include <cassert>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "test_macros.h"

static_assert(std::is_nothrow_move_constructible_v<std::source_location>, "support.srcloc.cons (1.1)");
static_assert(std::is_nothrow_move_assignable_v<std::source_location>, "support.srcloc.cons (1.2)");
static_assert(std::is_nothrow_swappable_v<std::source_location>, "support.srcloc.cons (1.3)");

ASSERT_NOEXCEPT(std::source_location());
ASSERT_NOEXCEPT(std::source_location::current());

// Note: the standard doesn't strictly require the particular values asserted
// here, but does "suggest" them.  Additional tests for details of how the
// implementation of current() chooses which location to report for more complex
// scenarios are in the Clang test-suite, and not replicated here.

// A default-constructed value.
constexpr std::source_location empty;
static_assert(empty.line() == 0);
static_assert(empty.column() == 0);
static_assert(empty.file_name()[0] == '\0');
static_assert(empty.function_name()[0] == '\0');

ASSERT_NOEXCEPT(empty.line());
ASSERT_NOEXCEPT(empty.column());
ASSERT_NOEXCEPT(empty.file_name());
ASSERT_NOEXCEPT(empty.function_name());
std::same_as<std::uint_least32_t> auto line   = empty.line();
std::same_as<std::uint_least32_t> auto column = empty.column();
std::same_as<const char*> auto file      = empty.file_name();
std::same_as<const char*> auto function  = empty.function_name();

// A simple use of current() outside a function.
constexpr std::source_location cur =
#line 1000 "ss"
    std::source_location::current();
static_assert(cur.line() == 1000);
static_assert(cur.column() > 0);
static_assert(cur.file_name()[0] == 's' && cur.file_name()[1] == 's' && cur.file_name()[2] == '\0');
static_assert(cur.function_name()[0] == '\0');

// and inside a function.
int main(int, char**) {
  auto local =
#line 2000
      std::source_location::current();
  assert(strcmp(local.file_name(), "ss") == 0);
  assert(strstr(local.function_name(), "main") != nullptr);
  assert(local.line() == 2000);
  assert(local.column() > 0);

  // Finally, the type should be copy-constructible
  auto local2 = cur;
  assert(strcmp(local2.file_name(), cur.file_name()) == 0);
  assert(strcmp(local2.function_name(), cur.function_name()) == 0);
  assert(local2.line() == cur.line());
  assert(local2.column() == cur.column());

  // and copy-assignable.
  local = cur;
  assert(strcmp(local.file_name(), cur.file_name()) == 0);
  assert(strcmp(local.function_name(), cur.function_name()) == 0);
  assert(local.line() == cur.line());
  assert(local.column() == cur.column());

  return 0;
}
