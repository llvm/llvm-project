//===-- Unittests for atexit ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/CPP/array.h"
#include "src/__support/CPP/utility.h"
#include "src/stdlib/_Exit.h"
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
    LIBC_NAMESPACE::atexit(+[] { LIBC_NAMESPACE::_Exit(1); });
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

extern "C" {
int __cxa_atexit(void (*callback)(void *), void *payload, void *dso);
void __cxa_finalize(void *dso);
}

TEST(LlvmLibcAtExit, CxaFinalizeDso) {
  size = 0;

  auto test = [] {
    static int dso1 = 1;
    static int dso2 = 2;

    auto record =
        +[](void *arg) { arr[size++] = *reinterpret_cast<int *>(arg); };

    static int v1 = 1, v2 = 2, v3 = 3;
    if (__cxa_atexit(record, &v1, &dso1) != 0)
      __builtin_trap();
    if (__cxa_atexit(record, &v2, &dso2) != 0)
      __builtin_trap();
    if (__cxa_atexit(record, &v3, &dso1) != 0)
      __builtin_trap();
    if (LIBC_NAMESPACE::atexit(+[] { arr[size++] = 4; }) != 0)
      __builtin_trap();

    // Finalize only dso1: should run v3 then v1 (LIFO), leaving dso2 and
    // atexit.
    __cxa_finalize(&dso1);
    if (size != 2 || arr[0] != 3 || arr[1] != 1)
      __builtin_trap();

    // Calling __cxa_finalize(&dso1) a second time should be a no-op.
    __cxa_finalize(&dso1);
    if (size != 2)
      __builtin_trap();

    // Finalize dso2 and verify it runs.
    __cxa_finalize(&dso2);
    if (size != 3 || arr[2] != 2)
      __builtin_trap();

    LIBC_NAMESPACE::exit(0);
  };
  EXPECT_EXITS(test, 0);
}

TEST(LlvmLibcAtExit, CxaFinalizeReentrant) {
  size = 0;

  auto test = [] {
    static int dso = 1;
    static int v1 = 1, v2 = 2;

    auto record =
        +[](void *arg) { arr[size++] = *reinterpret_cast<int *>(arg); };

    auto register_more = +[](void *) {
      arr[size++] = 1;
      if (__cxa_atexit(
              +[](void *arg) { arr[size++] = *reinterpret_cast<int *>(arg); },
              &v2, &dso) != 0)
        __builtin_trap();
    };

    if (__cxa_atexit(record, &v1, &dso) != 0)
      __builtin_trap();
    if (__cxa_atexit(register_more, nullptr, &dso) != 0)
      __builtin_trap();

    // register_more runs first (records 1) and registers v2 for the same dso.
    // Restarting the scan should run v2 (records 2) before older v1 (records
    // 1).
    __cxa_finalize(&dso);
    if (size != 3 || arr[0] != 1 || arr[1] != 2 || arr[2] != 1)
      __builtin_trap();

    LIBC_NAMESPACE::exit(0);
  };
  EXPECT_EXITS(test, 0);
}
