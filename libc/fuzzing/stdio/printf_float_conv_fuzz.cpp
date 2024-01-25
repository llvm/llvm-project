//===-- printf_float_conv_fuzz.cpp ----------------------------------------===//
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

#include "src/__support/FPUtil/FPBits.h"

#include <stddef.h>
#include <stdint.h>

#include "utils/MPFRWrapper/mpfr_inc.h"

constexpr int MAX_SIZE = 10000;

inline bool simple_streq(char *first, char *second, int length) {
  for (int i = 0; i < length; ++i) {
    if (first[i] != second[i]) {
      return false;
    }
  }
  return true;
}

inline int simple_strlen(const char *str) {
  int i = 0;
  for (; *str; ++str, ++i) {
    ;
  }
  return i;
}

enum class TestResult {
  Success,
  BufferSizeFailed,
  LengthsDiffer,
  StringsNotEqual,
};

template <typename F>
inline TestResult test_vals(const char *fmt, F num, int prec, int width) {
  // Call snprintf on a nullptr to get the buffer size.
  int buffer_size = LIBC_NAMESPACE::snprintf(nullptr, 0, fmt, width, prec, num);

  if (buffer_size < 0) {
    return TestResult::BufferSizeFailed;
  }

  char *test_buff = new char[buffer_size + 1];
  char *reference_buff = new char[buffer_size + 1];

  int test_result = 0;
  int reference_result = 0;

  test_result = LIBC_NAMESPACE::snprintf(test_buff, buffer_size + 1, fmt, width,
                                         prec, num);
  reference_result =
      mpfr_snprintf(reference_buff, buffer_size + 1, fmt, width, prec, num);

  // All of these calls should return that they wrote the same amount.
  if (test_result != reference_result || test_result != buffer_size) {
    return TestResult::LengthsDiffer;
  }

  if (!simple_streq(test_buff, reference_buff, buffer_size)) {
    return TestResult::StringsNotEqual;
  }

  delete[] test_buff;
  delete[] reference_buff;
  return TestResult::Success;
}

constexpr char const *fmt_arr[] = {
    "%*.*f", "%*.*e", "%*.*g", "%*.*a", "%*.*Lf", "%*.*Le", "%*.*Lg", "%*.*La",
};

extern "C" int LLVMFuzzerTestOneInput(const uint8_t *data, size_t size) {
  // const uint8_t raw_data[] = {0x30,0x27,0x1,0x0,0x0,0x0,0x0,0x0,0x24};
  // data = raw_data;
  // size = sizeof(raw_data);
  double num = 0.0;
  int prec = 0;
  int width = 0;

  LIBC_NAMESPACE::fputil::FPBits<double>::StorageType raw_num = 0;

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

  num = LIBC_NAMESPACE::fputil::FPBits<double>(raw_num).get_val();

  // While we could create a "ld_raw_num" from additional bytes, it's much
  // easier to stick with simply casting num to long double. This avoids the
  // issues around 80 bit long doubles, especially unnormal and pseudo-denormal
  // numbers, which MPFR doesn't handle well.
  long double ld_num = static_cast<long double>(num);

  if (width > MAX_SIZE) {
    width = MAX_SIZE;
  } else if (width < -MAX_SIZE) {
    width = -MAX_SIZE;
  }

  if (prec > MAX_SIZE) {
    prec = MAX_SIZE;
  } else if (prec < -MAX_SIZE) {
    prec = -MAX_SIZE;
  }

  for (size_t cur_fmt = 0; cur_fmt < sizeof(fmt_arr) / sizeof(char *);
       ++cur_fmt) {
    int fmt_len = simple_strlen(fmt_arr[cur_fmt]);
    TestResult result;
    if (fmt_arr[cur_fmt][fmt_len - 2] == 'L') {
      result = test_vals<long double>(fmt_arr[cur_fmt], ld_num, prec, width);
    } else {
      result = test_vals<double>(fmt_arr[cur_fmt], num, prec, width);
    }
    if (result != TestResult::Success) {
      __builtin_trap();
    }
  }
  return 0;
}
