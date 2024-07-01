//===-- Unittests for at_quick_exit ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/utility.h"
#include "src/stdlib/at_quick_exit.h"
#include "src/stdlib/quick_exit.h"
#include "test/UnitTest/Test.h"

static int a;
TEST(LlvmLibcAtQuickExit, Basic) {
  // In case tests ever run multiple times.
  a = 0;

  auto test = [] {
    int status = LIBC_NAMESPACE::at_quick_exit(+[] {
      if (a != 1)
        __builtin_trap();
    });
    status |= LIBC_NAMESPACE::at_quick_exit(+[] { a++; });
    if (status)
      __builtin_trap();

    LIBC_NAMESPACE::quick_exit(0);
  };
  EXPECT_EXITS(test, 0);
}

TEST(LlvmLibcAtQuickExit, AtQuickExitCallsSysExit) {
  auto test = [] {
    LIBC_NAMESPACE::at_quick_exit(+[] { _Exit(1); });
    LIBC_NAMESPACE::quick_exit(0);
  };
  EXPECT_EXITS(test, 1);
}

static int size;
static LIBC_NAMESPACE::cpp::array<int, 256> arr;

template <int... Ts>
void register_at_quick_exit_handlers(
    LIBC_NAMESPACE::cpp::integer_sequence<int, Ts...>) {
  (LIBC_NAMESPACE::at_quick_exit(+[] { arr[size++] = Ts; }), ...);
}

template <int count> constexpr auto get_test() {
  return [] {
    LIBC_NAMESPACE::at_quick_exit(+[] {
      if (size != count)
        __builtin_trap();
      for (int i = 0; i < count; i++)
        if (arr[i] != count - 1 - i)
          __builtin_trap();
    });
    register_at_quick_exit_handlers(
        LIBC_NAMESPACE::cpp::make_integer_sequence<int, count>{});
    LIBC_NAMESPACE::quick_exit(0);
  };
}

TEST(LlvmLibcAtQuickExit, ReverseOrder) {
  // In case tests ever run multiple times.
  size = 0;

  auto test = get_test<32>();
  EXPECT_EXITS(test, 0);
}

TEST(LlvmLibcAtQuickExit, Many) {
  // In case tests ever run multiple times.
  size = 0;

  auto test = get_test<256>();
  EXPECT_EXITS(test, 0);
}

TEST(LlvmLibcAtQuickExit, HandlerCallsAtQuickExit) {
  auto test = [] {
    LIBC_NAMESPACE::at_quick_exit(+[] {
      LIBC_NAMESPACE::at_quick_exit(+[] { LIBC_NAMESPACE::quick_exit(1); });
    });
    LIBC_NAMESPACE::quick_exit(0);
  };
  EXPECT_EXITS(test, 1);
}
