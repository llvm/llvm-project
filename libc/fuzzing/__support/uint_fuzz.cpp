//===-- uint_fuzz.cpp -----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc unsigned integer utilities.
///
//===----------------------------------------------------------------------===//
#include "src/__support/CPP/bit.h"
#include "src/__support/big_int.h"
#include "src/string/memory_utils/inline_memcpy.h"

using namespace LIBC_NAMESPACE;

// Helper function when using gdb / lldb to set a breakpoint and inspect values.
template <typename T> void debug_and_trap(const char *msg, T a, T b) {
  __builtin_trap();
}

#define DEBUG_AND_TRAP()

#define TEST_BINOP(OP)                                                         \
  if ((a OP b) != (static_cast<T>(BigInt(a) OP BigInt(b))))                    \
    debug_and_trap(#OP, a, b);

#define TEST_SHIFTOP(OP)                                                       \
  if ((a OP b) != (static_cast<T>(BigInt(a) OP b)))                            \
    debug_and_trap(#OP, a, b);

#define TEST_FUNCTION(FUN)                                                     \
  if (FUN(a) != FUN(BigInt(a)))                                                \
    debug_and_trap(#FUN, a, b);

// Test that basic arithmetic operations of BigInt behave like their scalar
// counterparts.
template <typename T, typename BigInt> void run_tests(T a, T b) {
  TEST_BINOP(+)
  TEST_BINOP(-)
  TEST_BINOP(*)
  if (b != 0)
    TEST_BINOP(/)
  if (b >= 0 && b < cpp::numeric_limits<T>::digits) {
    TEST_SHIFTOP(<<)
    TEST_SHIFTOP(>>)
  }
  if constexpr (!BigInt::SIGNED) {
    TEST_FUNCTION(cpp::has_single_bit)
    TEST_FUNCTION(cpp::countr_zero)
    TEST_FUNCTION(cpp::countl_zero)
    TEST_FUNCTION(cpp::countl_one)
    TEST_FUNCTION(cpp::countr_one)
  }
}

// Reads a T from libfuzzer data.
template <typename T> T read(const uint8_t *data, size_t &remainder) {
  T out = 0;
  constexpr size_t T_SIZE = sizeof(T);
  const size_t copy_size = remainder < T_SIZE ? remainder : T_SIZE;
  inline_memcpy(&out, data, copy_size);
  remainder -= copy_size;
  return out;
}

template <typename T, typename BigInt>
void run_tests(const uint8_t *data, size_t size) {
  const auto a = read<T>(data, size);
  const auto b = read<T>(data, size);
  run_tests<T, BigInt>(a, b);
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  // unsigned
  run_tests<uint64_t, BigInt<64, false, uint16_t>>(data, size);
  // signed
  run_tests<int64_t, BigInt<64, true, uint16_t>>(data, size);
  return 0;
}
