//===-- printf_fixed_conv_fuzz.cpp ----------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// Fuzzing test for llvm-libc printf %f/e/g/a implementations.
///
//===----------------------------------------------------------------------===//
#include "src/stdio/snprintf.h"

#include "include/llvm-libc-macros/stdfix-macros.h"
#include "src/__support/fixed_point/fx_bits.h"
#include "src/__support/fixed_point/fx_rep.h"

#include <stddef.h>
#include <stdint.h>

#include "utils/MPFRWrapper/mpfr_inc.h"

constexpr int MAX_SIZE = 10000;

inline bool simple_streq(char *first, char *second, int length) {
  for (int i = 0; i < length; ++i)
    if (first[i] != second[i])
      return false;

  return true;
}

inline int clamp(int num, int max) {
  if (num > max)
    return max;
  if (num < -max)
    return -max;
  return num;
}

enum class TestResult {
  Success,
  BufferSizeFailed,
  LengthsDiffer,
  StringsNotEqual,
};

template <typename F>
inline TestResult test_vals(const char *fmt, uint64_t num, int prec,
                            int width) {
  typename LIBC_NAMESPACE::fixed_point::FXRep<F>::StorageType raw_num = num;

  auto raw_num_bits = LIBC_NAMESPACE::fixed_point::FXBits<F>(raw_num);

  // This needs to be a float with enough bits of precision to hold the fixed
  // point number.
  static_assert(sizeof(long double) > sizeof(long accum));

  // build a long double that is equivalent to the fixed point number.
  long double ld_num =
      static_cast<long double>(raw_num_bits.get_integral()) +
      (static_cast<long double>(raw_num_bits.get_fraction()) /
       static_cast<long double>(1ll << raw_num_bits.get_exponent()));

  if (raw_num_bits.get_sign())
    ld_num = -ld_num;

  // Call snprintf on a nullptr to get the buffer size.
  int buffer_size = LIBC_NAMESPACE::snprintf(nullptr, 0, fmt, width, prec, num);

  if (buffer_size < 0)
    return TestResult::BufferSizeFailed;

  char *test_buff = new char[buffer_size + 1];
  char *reference_buff = new char[buffer_size + 1];

  int test_result = 0;
  int reference_result = 0;

  test_result = LIBC_NAMESPACE::snprintf(test_buff, buffer_size + 1, fmt, width,
                                         prec, num);

  // The fixed point format is defined to be %f equivalent.
  reference_result = mpfr_snprintf(reference_buff, buffer_size + 1, "%*.*Lf",
                                   width, prec, ld_num);

  // All of these calls should return that they wrote the same amount.
  if (test_result != reference_result || test_result != buffer_size)
    return TestResult::LengthsDiffer;

  if (!simple_streq(test_buff, reference_buff, buffer_size))
    return TestResult::StringsNotEqual;

  delete[] test_buff;
  delete[] reference_buff;
  return TestResult::Success;
}

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  // const uint8_t raw_data[] = {0x8d,0x43,0x40,0x0,0x0,0x0,};
  // data = raw_data;
  // size = sizeof(raw_data);
  int prec = 0;
  int width = 0;

  LIBC_NAMESPACE::fixed_point::FXRep<long accum>::StorageType raw_num = 0;

  // Copy as many bytes of data as will fit into num, prec, and with. Any extras
  // are ignored.
  for (size_t cur = 0; cur < size; ++cur) {
    if (cur < sizeof(raw_num)) {
      raw_num = (raw_num << 8) + data[cur];
    } else if (cur < sizeof(raw_num) + sizeof(prec)) {
      prec = (prec << 8) + data[cur];
    } else if (cur < sizeof(raw_num) + sizeof(prec) + sizeof(width)) {
      width = (width << 8) + data[cur];
    }
  }

  width = clamp(width, MAX_SIZE);
  prec = clamp(prec, MAX_SIZE);

  TestResult result;
  result = test_vals<long accum>("%*.*lk", raw_num, prec, width);
  if (result != TestResult::Success)
    __builtin_trap();

  result = test_vals<unsigned long accum>("%*.*lK", raw_num, prec, width);
  if (result != TestResult::Success)
    __builtin_trap();

  return 0;
}
