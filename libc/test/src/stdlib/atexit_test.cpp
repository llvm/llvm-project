//===-- Unittests for atexit ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/utility.h"
#include "src/stdlib/atexit.h"
#include "src/stdlib/exit.h"
#include "test/UnitTest/Test.h"

static int a;
TEST(LlvmLibcAtExit, Basic) {
  // In case tests ever run multiple times.
  a = 0;

  auto test = [] {
    int status = LIBC_NAMESPACE::atexit(+[] {
      if (a != 1)
        __builtin_trap();
    });
    status |= LIBC_NAMESPACE::atexit(+[] { a++; });
    if (status)
      __builtin_trap();

    LIBC_NAMESPACE::exit(0);
  };
  EXPECT_EXITS(test, 0);
}

TEST(LlvmLibcAtExit, AtExitCallsSysExit) {
  auto test = [] {
    LIBC_NAMESPACE::atexit(+[] { _Exit(1); });
    LIBC_NAMESPACE::exit(0);
  };
  EXPECT_EXITS(test, 1);
}

static int size;
static LIBC_NAMESPACE::cpp::array<int, 256> arr;

template <int... Ts>
void register_atexit_handlers(
    LIBC_NAMESPACE::cpp::integer_sequence<int, Ts...>) {
  (LIBC_NAMESPACE::atexit(+[] { arr[size++] = Ts; }), ...);
}

template <int count> constexpr auto getTest() {
  return [] {
    LIBC_NAMESPACE::atexit(+[] {
      if (size != count)
        __builtin_trap();
      for (int i = 0; i < count; i++)
        if (arr[i] != count - 1 - i)
          __builtin_trap();
    });
    register_atexit_handlers(
        LIBC_NAMESPACE::cpp::make_integer_sequence<int, count>{});
    LIBC_NAMESPACE::exit(0);
  };
}

TEST(LlvmLibcAtExit, ReverseOrder) {
  // In case tests ever run multiple times.
  size = 0;

  auto test = getTest<32>();
  EXPECT_EXITS(test, 0);
}

TEST(LlvmLibcAtExit, Many) {
  // In case tests ever run multiple times.
  size = 0;

  auto test = getTest<256>();
  EXPECT_EXITS(test, 0);
}

TEST(LlvmLibcAtExit, HandlerCallsAtExit) {
  auto test = [] {
    LIBC_NAMESPACE::atexit(
        +[] { LIBC_NAMESPACE::atexit(+[] { LIBC_NAMESPACE::exit(1); }); });
    LIBC_NAMESPACE::exit(0);
  };
  EXPECT_EXITS(test, 1);
}
