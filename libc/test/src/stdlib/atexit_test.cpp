//===-- Unittests for atexit ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/Array.h"
#include "src/__support/CPP/Utility.h"
#include "src/stdlib/atexit.h"
#include "src/stdlib/exit.h"
#include "utils/UnitTest/Test.h"

static int a;
TEST(LlvmLibcAtExit, Basic) {
  // In case tests ever run multiple times.
  a = 0;

  auto test = [] {
    int status = __llvm_libc::atexit(+[] {
      if (a != 1)
        __builtin_trap();
    });
    status |= __llvm_libc::atexit(+[] { a++; });
    if (status)
      __builtin_trap();

    __llvm_libc::exit(0);
  };
  EXPECT_EXITS(test, 0);
}

TEST(LlvmLibcAtExit, AtExitCallsSysExit) {
  auto test = [] {
    __llvm_libc::atexit(+[] { _Exit(1); });
    __llvm_libc::exit(0);
  };
  EXPECT_EXITS(test, 1);
}

static int size;
static __llvm_libc::cpp::Array<int, 256> arr;

template <int... Ts>
void register_atexit_handlers(__llvm_libc::cpp::IntegerSequence<int, Ts...>) {
  (__llvm_libc::atexit(+[] { arr[size++] = Ts; }), ...);
}

template <int count> constexpr auto getTest() {
  return [] {
    __llvm_libc::atexit(+[] {
      if (size != count)
        __builtin_trap();
      for (int i = 0; i < count; i++)
        if (arr[i] != count - 1 - i)
          __builtin_trap();
    });
    register_atexit_handlers(
        __llvm_libc::cpp::MakeIntegerSequence<int, count>{});
    __llvm_libc::exit(0);
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
    __llvm_libc::atexit(+[] {
      __llvm_libc::atexit(+[] { __llvm_libc::exit(1); });
    });
    __llvm_libc::exit(0);
  };
  EXPECT_EXITS(test, 1);
}
