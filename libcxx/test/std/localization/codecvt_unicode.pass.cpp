//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS -D_LIBCPP_ENABLE_CXX26_REMOVED_CODECVT
// XFAIL: stdlib=system && target={{.+}}-apple-macosx{{10.9|10.10|10.11|10.12|10.13|10.14|10.15|11.0|12.0|13.0}}

#include <algorithm>
#include <cassert>
#include <codecvt>
#include <locale>

#include "test_macros.h"

struct test_offsets_ok {
  size_t in_size;
  size_t out_size;
};
struct test_offsets_partial {
  size_t in_size;
  size_t out_size;
  size_t expected_in_next;
  size_t expected_out_next;
};

template <class CharT>
struct test_offsets_error {
  size_t in_size;
  size_t out_size;
  size_t expected_in_next;
  size_t expected_out_next;
  CharT replace_char;
  size_t replace_pos;
};

#define array_size(x) (sizeof(x) / sizeof(x)[0])

using std::begin;
using std::char_traits;
using std::codecvt_base;
using std::copy;
using std::end;

template <class InternT, class ExternT>
void utf8_to_utf32_in_ok(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  // UTF-8 string of 1-byte CP, 2-byte CP, 3-byte CP and 4-byte CP
  const unsigned char input[] = "b\u0448\uAAAA\U0010AAAA";
  const char32_t expected[]   = {'b', 0x0448, 0xAAAA, 0x10AAAA, 0};
  static_assert(array_size(input) == 11, "");
  static_assert(array_size(expected) == 5, "");

  ExternT in[array_size(input)];
  InternT exp[array_size(expected)];
  copy(begin(input), end(input), begin(in));
  copy(begin(expected), end(expected), begin(exp));
  assert(char_traits<ExternT>::length(in) == 10);
  assert(char_traits<InternT>::length(exp) == 4);
  test_offsets_ok offsets[] = {{0, 0}, {1, 1}, {3, 2}, {6, 3}, {10, 4}};
  for (test_offsets_ok* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_ok t                = *it;
    InternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    mbstate_t state          = {};
    const ExternT* in_next   = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.ok);
    assert(in_next == in + t.in_size);
    assert(out_next == out + t.out_size);
    assert(char_traits<InternT>::compare(out, exp, t.out_size) == 0);
    if (t.out_size < array_size(out))
      assert(out[t.out_size] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, t.out_size);
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.in_size);
  }

  for (test_offsets_ok* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_ok t            = *it;
    InternT out[array_size(exp)] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    mbstate_t state          = {};
    const ExternT* in_next   = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, end(out), out_next);
    assert(res == cvt.ok);
    assert(in_next == in + t.in_size);
    assert(out_next == out + t.out_size);
    assert(char_traits<InternT>::compare(out, exp, t.out_size) == 0);
    if (t.out_size < array_size(out))
      assert(out[t.out_size] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, array_size(out));
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.in_size);
  }
}

template <class InternT, class ExternT>
void utf8_to_utf32_in_partial(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  // UTF-8 string of 1-byte CP, 2-byte CP, 3-byte CP and 4-byte CP
  const unsigned char input[] = "b\u0448\uAAAA\U0010AAAA";
  const char32_t expected[]   = {'b', 0x0448, 0xAAAA, 0x10AAAA, 0};
  static_assert(array_size(input) == 11, "");
  static_assert(array_size(expected) == 5, "");

  ExternT in[array_size(input)];
  InternT exp[array_size(expected)];
  copy(begin(input), end(input), begin(in));
  copy(begin(expected), end(expected), begin(exp));
  assert(char_traits<ExternT>::length(in) == 10);
  assert(char_traits<InternT>::length(exp) == 4);

  test_offsets_partial offsets[] = {
      {1, 0, 0, 0}, // no space for first CP

      {3, 1, 1, 1}, // no space for second CP
      {2, 2, 1, 1}, // incomplete second CP
      {2, 1, 1, 1}, // incomplete second CP, and no space for it

      {6, 2, 3, 2}, // no space for third CP
      {4, 3, 3, 2}, // incomplete third CP
      {5, 3, 3, 2}, // incomplete third CP
      {4, 2, 3, 2}, // incomplete third CP, and no space for it
      {5, 2, 3, 2}, // incomplete third CP, and no space for it

      {10, 3, 6, 3}, // no space for fourth CP
      {7, 4, 6, 3},  // incomplete fourth CP
      {8, 4, 6, 3},  // incomplete fourth CP
      {9, 4, 6, 3},  // incomplete fourth CP
      {7, 3, 6, 3},  // incomplete fourth CP, and no space for it
      {8, 3, 6, 3},  // incomplete fourth CP, and no space for it
      {9, 3, 6, 3},  // incomplete fourth CP, and no space for it
  };

  for (test_offsets_partial* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_partial t           = *it;
    InternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    mbstate_t state          = {};
    const ExternT* in_next   = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.partial);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<InternT>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, t.out_size);
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.expected_in_next);
  }
}

template <class InternT, class ExternT>
void utf8_to_utf32_in_error(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  // UTF-8 string of 1-byte CP, 2-byte CP, 3-byte CP, 4-byte CP
  const unsigned char input[] = "b\u0448\uD700\U0010AAAA";
  const char32_t expected[]   = {'b', 0x0448, 0xD700, 0x10AAAA, 0};
  static_assert(array_size(input) == 11, "");
  static_assert(array_size(expected) == 5, "");

  ExternT in[array_size(input)];
  InternT exp[array_size(expected)];
  copy(begin(input), end(input), begin(in));
  copy(begin(expected), end(expected), begin(exp));
  assert(char_traits<ExternT>::length(in) == 10);
  assert(char_traits<InternT>::length(exp) == 4);

  // There are 5 classes of errors in UTF-8 decoding
  // 1. Missing leading byte
  // 2. Missing trailing byte
  // 3. Surrogate CP
  // 4. Overlong sequence
  // 5. CP out of Unicode range
  test_offsets_error<unsigned char> offsets[] = {

      // 1. Missing leading byte. We will replace the leading byte with
      // non-leading byte, such as a byte that is always invalid or a trailing
      // byte.

      // replace leading byte with invalid byte
      {1, 4, 0, 0, 0xFF, 0},
      {3, 4, 1, 1, 0xFF, 1},
      {6, 4, 3, 2, 0xFF, 3},
      {10, 4, 6, 3, 0xFF, 6},

      // replace leading byte with trailing byte
      {1, 4, 0, 0, 0b10101010, 0},
      {3, 4, 1, 1, 0b10101010, 1},
      {6, 4, 3, 2, 0b10101010, 3},
      {10, 4, 6, 3, 0b10101010, 6},

      // 2. Missing trailing byte. We will replace the trailing byte with
      // non-trailing byte, such as a byte that is always invalid or a leading
      // byte (simple ASCII byte in our case).

      // replace first trailing byte with ASCII byte
      {3, 4, 1, 1, 'z', 2},
      {6, 4, 3, 2, 'z', 4},
      {10, 4, 6, 3, 'z', 7},

      // replace first trailing byte with invalid byte
      {3, 4, 1, 1, 0xFF, 2},
      {6, 4, 3, 2, 0xFF, 4},
      {10, 4, 6, 3, 0xFF, 7},

      // replace second trailing byte with ASCII byte
      {6, 4, 3, 2, 'z', 5},
      {10, 4, 6, 3, 'z', 8},

      // replace second trailing byte with invalid byte
      {6, 4, 3, 2, 0xFF, 5},
      {10, 4, 6, 3, 0xFF, 8},

      // replace third trailing byte
      {10, 4, 6, 3, 'z', 9},
      {10, 4, 6, 3, 0xFF, 9},

      // 2.1 The following test-cases raise doubt whether error or partial should
      // be returned. For example, we have 4-byte sequence with valid leading
      // byte. If we hide the last byte we need to return partial. But, if the
      // second or third byte, which are visible to the call to codecvt, are
      // malformed then error should be returned.

      // replace first trailing byte with ASCII byte, also incomplete at end
      {5, 4, 3, 2, 'z', 4},
      {8, 4, 6, 3, 'z', 7},
      {9, 4, 6, 3, 'z', 7},

      // replace first trailing byte with invalid byte, also incomplete at end
      {5, 4, 3, 2, 0xFF, 4},
      {8, 4, 6, 3, 0xFF, 7},
      {9, 4, 6, 3, 0xFF, 7},

      // replace second trailing byte with ASCII byte, also incomplete at end
      {9, 4, 6, 3, 'z', 8},

      // replace second trailing byte with invalid byte, also incomplete at end
      {9, 4, 6, 3, 0xFF, 8},

      // 3. Surrogate CP. We modify the second byte (first trailing) of the 3-byte
      // CP U+D700
      {6, 4, 3, 2, 0b10100000, 4}, // turn U+D700 into U+D800
      {6, 4, 3, 2, 0b10101100, 4}, // turn U+D700 into U+DB00
      {6, 4, 3, 2, 0b10110000, 4}, // turn U+D700 into U+DC00
      {6, 4, 3, 2, 0b10111100, 4}, // turn U+D700 into U+DF00

      // 4. Overlong sequence. The CPs in the input are chosen such as modifying
      // just the leading byte is enough to make them overlong, i.e. for the
      // 3-byte and 4-byte CP the second byte (first trailing) has enough leading
      // zeroes.
      {3, 4, 1, 1, 0b11000000, 1},  // make the 2-byte CP overlong
      {3, 4, 1, 1, 0b11000001, 1},  // make the 2-byte CP overlong
      {6, 4, 3, 2, 0b11100000, 3},  // make the 3-byte CP overlong
      {10, 4, 6, 3, 0b11110000, 6}, // make the 4-byte CP overlong

      // 5. CP above range
      // turn U+10AAAA into U+14AAAA by changing its leading byte
      {10, 4, 6, 3, 0b11110101, 6},
      // turn U+10AAAA into U+11AAAA by changing its 2nd byte
      {10, 4, 6, 3, 0b10011010, 7},
  };
  for (test_offsets_error<unsigned char>* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_error<unsigned char> t = *it;
    InternT out[array_size(exp) - 1]    = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    ExternT old_char  = in[t.replace_pos];
    in[t.replace_pos] = t.replace_char;

    mbstate_t state          = {};
    const ExternT* in_next   = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.error);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<InternT>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, t.out_size);
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.expected_in_next);

    in[t.replace_pos] = old_char;
  }
}

template <class InternT, class ExternT>
void utf8_to_utf32_in(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  utf8_to_utf32_in_ok(cvt);
  utf8_to_utf32_in_partial(cvt);
  utf8_to_utf32_in_error(cvt);
}

template <class InternT, class ExternT>
void utf32_to_utf8_out_ok(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  // UTF-8 string of 1-byte CP, 2-byte CP, 3-byte CP and 4-byte CP
  const char32_t input[]         = {'b', 0x0448, 0xAAAA, 0x10AAAA, 0};
  const unsigned char expected[] = "b\u0448\uAAAA\U0010AAAA";
  static_assert(array_size(input) == 5, "");
  static_assert(array_size(expected) == 11, "");

  InternT in[array_size(input)];
  ExternT exp[array_size(expected)];
  copy(begin(input), end(input), begin(in));
  copy(begin(expected), end(expected), begin(exp));
  assert(char_traits<InternT>::length(in) == 4);
  assert(char_traits<ExternT>::length(exp) == 10);

  test_offsets_ok offsets[] = {{0, 0}, {1, 1}, {2, 3}, {3, 6}, {4, 10}};
  for (test_offsets_ok* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_ok t                = *it;
    ExternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    mbstate_t state          = {};
    const InternT* in_next   = nullptr;
    ExternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.out(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.ok);
    assert(in_next == in + t.in_size);
    assert(out_next == out + t.out_size);
    assert(char_traits<ExternT>::compare(out, exp, t.out_size) == 0);
    if (t.out_size < array_size(out))
      assert(out[t.out_size] == 0);
  }
}

template <class InternT, class ExternT>
void utf32_to_utf8_out_partial(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  // UTF-8 string of 1-byte CP, 2-byte CP, 3-byte CP and 4-byte CP
  const char32_t input[]         = {'b', 0x0448, 0xAAAA, 0x10AAAA, 0};
  const unsigned char expected[] = "b\u0448\uAAAA\U0010AAAA";
  static_assert(array_size(input) == 5, "");
  static_assert(array_size(expected) == 11, "");

  InternT in[array_size(input)];
  ExternT exp[array_size(expected)];
  copy(begin(input), end(input), begin(in));
  copy(begin(expected), end(expected), begin(exp));
  assert(char_traits<InternT>::length(in) == 4);
  assert(char_traits<ExternT>::length(exp) == 10);

  test_offsets_partial offsets[] = {
      {1, 0, 0, 0}, // no space for first CP

      {2, 1, 1, 1}, // no space for second CP
      {2, 2, 1, 1}, // no space for second CP

      {3, 3, 2, 3}, // no space for third CP
      {3, 4, 2, 3}, // no space for third CP
      {3, 5, 2, 3}, // no space for third CP

      {4, 6, 3, 6}, // no space for fourth CP
      {4, 7, 3, 6}, // no space for fourth CP
      {4, 8, 3, 6}, // no space for fourth CP
      {4, 9, 3, 6}, // no space for fourth CP
  };
  for (test_offsets_partial* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_partial t           = *it;
    ExternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    mbstate_t state          = {};
    const InternT* in_next   = nullptr;
    ExternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.out(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.partial);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<ExternT>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);
  }
}

template <class InternT, class ExternT>
void utf32_to_utf8_out_error(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  // UTF-8 string of 1-byte CP, 2-byte CP, 3-byte CP and 4-byte CP
  const char32_t input[]         = {'b', 0x0448, 0xAAAA, 0x10AAAA, 0};
  const unsigned char expected[] = "b\u0448\uAAAA\U0010AAAA";
  static_assert(array_size(input) == 5, "");
  static_assert(array_size(expected) == 11, "");

  InternT in[array_size(input)];
  ExternT exp[array_size(expected)];
  copy(begin(input), end(input), begin(in));
  copy(begin(expected), end(expected), begin(exp));
  assert(char_traits<InternT>::length(in) == 4);
  assert(char_traits<ExternT>::length(exp) == 10);

  test_offsets_error<InternT> offsets[] = {

      // Surrogate CP
      {4, 10, 0, 0, 0xD800, 0},
      {4, 10, 1, 1, 0xDBFF, 1},
      {4, 10, 2, 3, 0xDC00, 2},
      {4, 10, 3, 6, 0xDFFF, 3},

      // CP out of range
      {4, 10, 0, 0, 0x00110000, 0},
      {4, 10, 1, 1, 0x00110000, 1},
      {4, 10, 2, 3, 0x00110000, 2},
      {4, 10, 3, 6, 0x00110000, 3}};

  for (test_offsets_error<InternT>* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_error<InternT> t    = *it;
    ExternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    InternT old_char  = in[t.replace_pos];
    in[t.replace_pos] = t.replace_char;

    mbstate_t state          = {};
    const InternT* in_next   = nullptr;
    ExternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.out(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.error);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<ExternT>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);

    in[t.replace_pos] = old_char;
  }
}

template <class InternT, class ExternT>
void utf32_to_utf8_out(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  utf32_to_utf8_out_ok(cvt);
  utf32_to_utf8_out_partial(cvt);
  utf32_to_utf8_out_error(cvt);
}

template <class InternT, class ExternT>
void test_utf8_utf32_cvt(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  utf8_to_utf32_in(cvt);
  utf32_to_utf8_out(cvt);
}

template <class InternT, class ExternT>
void utf8_to_utf16_in_ok(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  // UTF-8 string of 1-byte CP, 2-byte CP, 3-byte CP and 4-byte CP
  const unsigned char input[] = "b\u0448\uAAAA\U0010AAAA";
  const char16_t expected[]   = {'b', 0x0448, 0xAAAA, 0xDBEA, 0xDEAA, 0};
  static_assert(array_size(input) == 11, "");
  static_assert(array_size(expected) == 6, "");

  ExternT in[array_size(input)];
  InternT exp[array_size(expected)];
  copy(begin(input), end(input), begin(in));
  copy(begin(expected), end(expected), begin(exp));
  assert(char_traits<ExternT>::length(in) == 10);
  assert(char_traits<InternT>::length(exp) == 5);

  test_offsets_ok offsets[] = {{0, 0}, {1, 1}, {3, 2}, {6, 3}, {10, 5}};
  for (test_offsets_ok* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_ok t                = *it;
    InternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    mbstate_t state          = {};
    const ExternT* in_next   = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.ok);
    assert(in_next == in + t.in_size);
    assert(out_next == out + t.out_size);
    assert(char_traits<InternT>::compare(out, exp, t.out_size) == 0);
    if (t.out_size < array_size(out))
      assert(out[t.out_size] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, t.out_size);
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.in_size);
  }

  for (test_offsets_ok* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_ok t            = *it;
    InternT out[array_size(exp)] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    mbstate_t state          = {};
    const ExternT* in_next   = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, end(out), out_next);
    assert(res == cvt.ok);
    assert(in_next == in + t.in_size);
    assert(out_next == out + t.out_size);
    assert(char_traits<InternT>::compare(out, exp, t.out_size) == 0);
    if (t.out_size < array_size(out))
      assert(out[t.out_size] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, array_size(out));
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.in_size);
  }
}

template <class InternT, class ExternT>
void utf8_to_utf16_in_partial(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  // UTF-8 string of 1-byte CP, 2-byte CP, 3-byte CP and 4-byte CP
  const unsigned char input[] = "b\u0448\uAAAA\U0010AAAA";
  const char16_t expected[]   = {'b', 0x0448, 0xAAAA, 0xDBEA, 0xDEAA, 0};
  static_assert(array_size(input) == 11, "");
  static_assert(array_size(expected) == 6, "");

  ExternT in[array_size(input)];
  InternT exp[array_size(expected)];
  copy(begin(input), end(input), begin(in));
  copy(begin(expected), end(expected), begin(exp));
  assert(char_traits<ExternT>::length(in) == 10);
  assert(char_traits<InternT>::length(exp) == 5);

  test_offsets_partial offsets[] = {
      {1, 0, 0, 0}, // no space for first CP

      {3, 1, 1, 1}, // no space for second CP
      {2, 2, 1, 1}, // incomplete second CP
      {2, 1, 1, 1}, // incomplete second CP, and no space for it

      {6, 2, 3, 2}, // no space for third CP
      {4, 3, 3, 2}, // incomplete third CP
      {5, 3, 3, 2}, // incomplete third CP
      {4, 2, 3, 2}, // incomplete third CP, and no space for it
      {5, 2, 3, 2}, // incomplete third CP, and no space for it

      {10, 3, 6, 3}, // no space for fourth CP
      {10, 4, 6, 3}, // no space for fourth CP
      {7, 5, 6, 3},  // incomplete fourth CP
      {8, 5, 6, 3},  // incomplete fourth CP
      {9, 5, 6, 3},  // incomplete fourth CP
      {7, 3, 6, 3},  // incomplete fourth CP, and no space for it
      {8, 3, 6, 3},  // incomplete fourth CP, and no space for it
      {9, 3, 6, 3},  // incomplete fourth CP, and no space for it
      {7, 4, 6, 3},  // incomplete fourth CP, and no space for it
      {8, 4, 6, 3},  // incomplete fourth CP, and no space for it
      {9, 4, 6, 3},  // incomplete fourth CP, and no space for it

  };

  for (test_offsets_partial* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_partial t           = *it;
    InternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    mbstate_t state          = {};
    const ExternT* in_next   = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.partial);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<InternT>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, t.out_size);
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.expected_in_next);
  }
}

template <class InternT, class ExternT>
void utf8_to_utf16_in_error(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  // UTF-8 string of 1-byte CP, 2-byte CP, 3-byte CP, 4-byte CP
  const unsigned char input[] = "b\u0448\uD700\U0010AAAA";
  const char16_t expected[]   = {'b', 0x0448, 0xD700, 0xDBEA, 0xDEAA, 0};
  static_assert(array_size(input) == 11, "");
  static_assert(array_size(expected) == 6, "");

  ExternT in[array_size(input)];
  InternT exp[array_size(expected)];
  copy(begin(input), end(input), begin(in));
  copy(begin(expected), end(expected), begin(exp));
  assert(char_traits<ExternT>::length(in) == 10);
  assert(char_traits<InternT>::length(exp) == 5);

  // There are 5 classes of errors in UTF-8 decoding
  // 1. Missing leading byte
  // 2. Missing trailing byte
  // 3. Surrogate CP
  // 4. Overlong sequence
  // 5. CP out of Unicode range
  test_offsets_error<unsigned char> offsets[] = {

      // 1. Missing leading byte. We will replace the leading byte with
      // non-leading byte, such as a byte that is always invalid or a trailing
      // byte.

      // replace leading byte with invalid byte
      {1, 5, 0, 0, 0xFF, 0},
      {3, 5, 1, 1, 0xFF, 1},
      {6, 5, 3, 2, 0xFF, 3},
      {10, 5, 6, 3, 0xFF, 6},

      // replace leading byte with trailing byte
      {1, 5, 0, 0, 0b10101010, 0},
      {3, 5, 1, 1, 0b10101010, 1},
      {6, 5, 3, 2, 0b10101010, 3},
      {10, 5, 6, 3, 0b10101010, 6},

      // 2. Missing trailing byte. We will replace the trailing byte with
      // non-trailing byte, such as a byte that is always invalid or a leading
      // byte (simple ASCII byte in our case).

      // replace first trailing byte with ASCII byte
      {3, 5, 1, 1, 'z', 2},
      {6, 5, 3, 2, 'z', 4},
      {10, 5, 6, 3, 'z', 7},

      // replace first trailing byte with invalid byte
      {3, 5, 1, 1, 0xFF, 2},
      {6, 5, 3, 2, 0xFF, 4},
      {10, 5, 6, 3, 0xFF, 7},

      // replace second trailing byte with ASCII byte
      {6, 5, 3, 2, 'z', 5},
      {10, 5, 6, 3, 'z', 8},

      // replace second trailing byte with invalid byte
      {6, 5, 3, 2, 0xFF, 5},
      {10, 5, 6, 3, 0xFF, 8},

      // replace third trailing byte
      {10, 5, 6, 3, 'z', 9},
      {10, 5, 6, 3, 0xFF, 9},

      // 2.1 The following test-cases raise doubt whether error or partial should
      // be returned. For example, we have 4-byte sequence with valid leading
      // byte. If we hide the last byte we need to return partial. But, if the
      // second or third byte, which are visible to the call to codecvt, are
      // malformed then error should be returned.

      // replace first trailing byte with ASCII byte, also incomplete at end
      {5, 5, 3, 2, 'z', 4},
      {8, 5, 6, 3, 'z', 7},
      {9, 5, 6, 3, 'z', 7},

      // replace first trailing byte with invalid byte, also incomplete at end
      {5, 5, 3, 2, 0xFF, 4},
      {8, 5, 6, 3, 0xFF, 7},
      {9, 5, 6, 3, 0xFF, 7},

      // replace second trailing byte with ASCII byte, also incomplete at end
      {9, 5, 6, 3, 'z', 8},

      // replace second trailing byte with invalid byte, also incomplete at end
      {9, 5, 6, 3, 0xFF, 8},

      // 3. Surrogate CP. We modify the second byte (first trailing) of the 3-byte
      // CP U+D700
      {6, 5, 3, 2, 0b10100000, 4}, // turn U+D700 into U+D800
      {6, 5, 3, 2, 0b10101100, 4}, // turn U+D700 into U+DB00
      {6, 5, 3, 2, 0b10110000, 4}, // turn U+D700 into U+DC00
      {6, 5, 3, 2, 0b10111100, 4}, // turn U+D700 into U+DF00

      // 4. Overlong sequence. The CPs in the input are chosen such as modifying
      // just the leading byte is enough to make them overlong, i.e. for the
      // 3-byte and 4-byte CP the second byte (first trailing) has enough leading
      // zeroes.
      {3, 5, 1, 1, 0b11000000, 1},  // make the 2-byte CP overlong
      {3, 5, 1, 1, 0b11000001, 1},  // make the 2-byte CP overlong
      {6, 5, 3, 2, 0b11100000, 3},  // make the 3-byte CP overlong
      {10, 5, 6, 3, 0b11110000, 6}, // make the 4-byte CP overlong

      // 5. CP above range
      // turn U+10AAAA into U+14AAAA by changing its leading byte
      {10, 5, 6, 3, 0b11110101, 6},
      // turn U+10AAAA into U+11AAAA by changing its 2nd byte
      {10, 5, 6, 3, 0b10011010, 7},
  };
  for (test_offsets_error<unsigned char>* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_error<unsigned char> t = *it;
    InternT out[array_size(exp) - 1]    = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    ExternT old_char  = in[t.replace_pos];
    in[t.replace_pos] = t.replace_char;

    mbstate_t state          = {};
    const ExternT* in_next   = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.error);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<InternT>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, t.out_size);
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.expected_in_next);

    in[t.replace_pos] = old_char;
  }
}

template <class InternT, class ExternT>
void utf8_to_utf16_in(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  utf8_to_utf16_in_ok(cvt);
  utf8_to_utf16_in_partial(cvt);
  utf8_to_utf16_in_error(cvt);
}

template <class InternT, class ExternT>
void utf16_to_utf8_out_ok(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  // UTF-8 string of 1-byte CP, 2-byte CP, 3-byte CP and 4-byte CP
  const char16_t input[]         = {'b', 0x0448, 0xAAAA, 0xDBEA, 0xDEAA, 0};
  const unsigned char expected[] = "b\u0448\uAAAA\U0010AAAA";
  static_assert(array_size(input) == 6, "");
  static_assert(array_size(expected) == 11, "");

  InternT in[array_size(input)];
  ExternT exp[array_size(expected)];
  copy(begin(input), end(input), begin(in));
  copy(begin(expected), end(expected), begin(exp));
  assert(char_traits<InternT>::length(in) == 5);
  assert(char_traits<ExternT>::length(exp) == 10);

  test_offsets_ok offsets[] = {{0, 0}, {1, 1}, {2, 3}, {3, 6}, {5, 10}};
  for (test_offsets_ok* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_ok t                = *it;
    ExternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    mbstate_t state          = {};
    const InternT* in_next   = nullptr;
    ExternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.out(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.ok);
    assert(in_next == in + t.in_size);
    assert(out_next == out + t.out_size);
    assert(char_traits<ExternT>::compare(out, exp, t.out_size) == 0);
    if (t.out_size < array_size(out))
      assert(out[t.out_size] == 0);
  }
}

template <class InternT, class ExternT>
void utf16_to_utf8_out_partial(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  // UTF-8 string of 1-byte CP, 2-byte CP, 3-byte CP and 4-byte CP
  const char16_t input[]         = {'b', 0x0448, 0xAAAA, 0xDBEA, 0xDEAA, 0};
  const unsigned char expected[] = "b\u0448\uAAAA\U0010AAAA";
  static_assert(array_size(input) == 6, "");
  static_assert(array_size(expected) == 11, "");

  InternT in[array_size(input)];
  ExternT exp[array_size(expected)];
  copy(begin(input), end(input), begin(in));
  copy(begin(expected), end(expected), begin(exp));
  assert(char_traits<InternT>::length(in) == 5);
  assert(char_traits<ExternT>::length(exp) == 10);

  test_offsets_partial offsets[] = {
      {1, 0, 0, 0}, // no space for first CP

      {2, 1, 1, 1}, // no space for second CP
      {2, 2, 1, 1}, // no space for second CP

      {3, 3, 2, 3}, // no space for third CP
      {3, 4, 2, 3}, // no space for third CP
      {3, 5, 2, 3}, // no space for third CP

      {5, 6, 3, 6}, // no space for fourth CP
      {5, 7, 3, 6}, // no space for fourth CP
      {5, 8, 3, 6}, // no space for fourth CP
      {5, 9, 3, 6}, // no space for fourth CP

      {4, 10, 3, 6}, // incomplete fourth CP

      {4, 6, 3, 6}, // incomplete fourth CP, and no space for it
      {4, 7, 3, 6}, // incomplete fourth CP, and no space for it
      {4, 8, 3, 6}, // incomplete fourth CP, and no space for it
      {4, 9, 3, 6}, // incomplete fourth CP, and no space for it
  };
  for (test_offsets_partial* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_partial t           = *it;
    ExternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    mbstate_t state          = {};
    const InternT* in_next   = nullptr;
    ExternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.out(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.partial);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<ExternT>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);
  }
}

template <class InternT, class ExternT>
void utf16_to_utf8_out_error(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  // UTF-8 string of 1-byte CP, 2-byte CP, 3-byte CP and 4-byte CP
  const char16_t input[]         = {'b', 0x0448, 0xAAAA, 0xDBEA, 0xDEAA, 0};
  const unsigned char expected[] = "b\u0448\uAAAA\U0010AAAA";
  static_assert(array_size(input) == 6, "");
  static_assert(array_size(expected) == 11, "");

  InternT in[array_size(input)];
  ExternT exp[array_size(expected)];
  copy(begin(input), end(input), begin(in));
  copy(begin(expected), end(expected), begin(exp));
  assert(char_traits<InternT>::length(in) == 5);
  assert(char_traits<ExternT>::length(exp) == 10);

  // The only possible error in UTF-16 is unpaired surrogate code units.
  // So we replace valid code points (scalar values) with lone surrogate CU.
  test_offsets_error<InternT> offsets[] = {
      {5, 10, 0, 0, 0xD800, 0},
      {5, 10, 0, 0, 0xDBFF, 0},
      {5, 10, 0, 0, 0xDC00, 0},
      {5, 10, 0, 0, 0xDFFF, 0},

      {5, 10, 1, 1, 0xD800, 1},
      {5, 10, 1, 1, 0xDBFF, 1},
      {5, 10, 1, 1, 0xDC00, 1},
      {5, 10, 1, 1, 0xDFFF, 1},

      {5, 10, 2, 3, 0xD800, 2},
      {5, 10, 2, 3, 0xDBFF, 2},
      {5, 10, 2, 3, 0xDC00, 2},
      {5, 10, 2, 3, 0xDFFF, 2},

      // make the leading surrogate a trailing one
      {5, 10, 3, 6, 0xDC00, 3},
      {5, 10, 3, 6, 0xDFFF, 3},

      // make the trailing surrogate a leading one
      {5, 10, 3, 6, 0xD800, 4},
      {5, 10, 3, 6, 0xDBFF, 4},

      // make the trailing surrogate a BMP char
      {5, 10, 3, 6, 'z', 4},
  };

  for (test_offsets_error<InternT>* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_error<InternT> t    = *it;
    ExternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    InternT old_char  = in[t.replace_pos];
    in[t.replace_pos] = t.replace_char;

    mbstate_t state          = {};
    const InternT* in_next   = nullptr;
    ExternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.out(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.error);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<ExternT>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);

    in[t.replace_pos] = old_char;
  }
}

template <class InternT, class ExternT>
void utf16_to_utf8_out(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  utf16_to_utf8_out_ok(cvt);
  utf16_to_utf8_out_partial(cvt);
  utf16_to_utf8_out_error(cvt);
}

template <class InternT, class ExternT>
void test_utf8_utf16_cvt(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  utf8_to_utf16_in(cvt);
  utf16_to_utf8_out(cvt);
}

template <class InternT, class ExternT>
void utf8_to_ucs2_in_ok(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  // UTF-8 string of 1-byte CP, 2-byte CP and 3-byte CP
  const unsigned char input[] = "b\u0448\uAAAA";
  const char16_t expected[]   = {'b', 0x0448, 0xAAAA, 0};
  static_assert(array_size(input) == 7, "");
  static_assert(array_size(expected) == 4, "");

  ExternT in[array_size(input)];
  InternT exp[array_size(expected)];
  copy(begin(input), end(input), begin(in));
  copy(begin(expected), end(expected), begin(exp));
  assert(char_traits<ExternT>::length(in) == 6);
  assert(char_traits<InternT>::length(exp) == 3);

  test_offsets_ok offsets[] = {{0, 0}, {1, 1}, {3, 2}, {6, 3}};
  for (test_offsets_ok* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_ok t                = *it;
    InternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    mbstate_t state          = {};
    const ExternT* in_next   = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.ok);
    assert(in_next == in + t.in_size);
    assert(out_next == out + t.out_size);
    assert(char_traits<InternT>::compare(out, exp, t.out_size) == 0);
    if (t.out_size < array_size(out))
      assert(out[t.out_size] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, t.out_size);
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.in_size);
  }

  for (test_offsets_ok* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_ok t            = *it;
    InternT out[array_size(exp)] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    mbstate_t state          = {};
    const ExternT* in_next   = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, end(out), out_next);
    assert(res == cvt.ok);
    assert(in_next == in + t.in_size);
    assert(out_next == out + t.out_size);
    assert(char_traits<InternT>::compare(out, exp, t.out_size) == 0);
    if (t.out_size < array_size(out))
      assert(out[t.out_size] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, array_size(out));
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.in_size);
  }
}

template <class InternT, class ExternT>
void utf8_to_ucs2_in_partial(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  // UTF-8 string of 1-byte CP, 2-byte CP and 3-byte CP
  const unsigned char input[] = "b\u0448\uAAAA";
  const char16_t expected[]   = {'b', 0x0448, 0xAAAA, 0};
  static_assert(array_size(input) == 7, "");
  static_assert(array_size(expected) == 4, "");

  ExternT in[array_size(input)];
  InternT exp[array_size(expected)];
  copy(begin(input), end(input), begin(in));
  copy(begin(expected), end(expected), begin(exp));
  assert(char_traits<ExternT>::length(in) == 6);
  assert(char_traits<InternT>::length(exp) == 3);

  test_offsets_partial offsets[] = {
      {1, 0, 0, 0}, // no space for first CP

      {3, 1, 1, 1}, // no space for second CP
      {2, 2, 1, 1}, // incomplete second CP
      {2, 1, 1, 1}, // incomplete second CP, and no space for it

      {6, 2, 3, 2}, // no space for third CP
      {4, 3, 3, 2}, // incomplete third CP
      {5, 3, 3, 2}, // incomplete third CP
      {4, 2, 3, 2}, // incomplete third CP, and no space for it
      {5, 2, 3, 2}, // incomplete third CP, and no space for it
  };

  for (test_offsets_partial* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_partial t           = *it;
    InternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    mbstate_t state          = {};
    const ExternT* in_next   = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.partial);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<InternT>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, t.out_size);
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.expected_in_next);
  }
}

template <class InternT, class ExternT>
void utf8_to_ucs2_in_error(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  const unsigned char input[] = "b\u0448\uD700\U0010AAAA";
  const char16_t expected[]   = {'b', 0x0448, 0xD700, 0xDBEA, 0xDEAA, 0};
  static_assert(array_size(input) == 11, "");
  static_assert(array_size(expected) == 6, "");

  ExternT in[array_size(input)];
  InternT exp[array_size(expected)];
  copy(begin(input), end(input), begin(in));
  copy(begin(expected), end(expected), begin(exp));
  assert(char_traits<ExternT>::length(in) == 10);
  assert(char_traits<InternT>::length(exp) == 5);

  // There are 5 classes of errors in UTF-8 decoding
  // 1. Missing leading byte
  // 2. Missing trailing byte
  // 3. Surrogate CP
  // 4. Overlong sequence
  // 5. CP out of Unicode range
  test_offsets_error<unsigned char> offsets[] = {

      // 1. Missing leading byte. We will replace the leading byte with
      // non-leading byte, such as a byte that is always invalid or a trailing
      // byte.

      // replace leading byte with invalid byte
      {1, 5, 0, 0, 0xFF, 0},
      {3, 5, 1, 1, 0xFF, 1},
      {6, 5, 3, 2, 0xFF, 3},
      {10, 5, 6, 3, 0xFF, 6},

      // replace leading byte with trailing byte
      {1, 5, 0, 0, 0b10101010, 0},
      {3, 5, 1, 1, 0b10101010, 1},
      {6, 5, 3, 2, 0b10101010, 3},
      {10, 5, 6, 3, 0b10101010, 6},

      // 2. Missing trailing byte. We will replace the trailing byte with
      // non-trailing byte, such as a byte that is always invalid or a leading
      // byte (simple ASCII byte in our case).

      // replace first trailing byte with ASCII byte
      {3, 5, 1, 1, 'z', 2},
      {6, 5, 3, 2, 'z', 4},
      {10, 5, 6, 3, 'z', 7},

      // replace first trailing byte with invalid byte
      {3, 5, 1, 1, 0xFF, 2},
      {6, 5, 3, 2, 0xFF, 4},
      {10, 5, 6, 3, 0xFF, 7},

      // replace second trailing byte with ASCII byte
      {6, 5, 3, 2, 'z', 5},
      {10, 5, 6, 3, 'z', 8},

      // replace second trailing byte with invalid byte
      {6, 5, 3, 2, 0xFF, 5},
      {10, 5, 6, 3, 0xFF, 8},

      // replace third trailing byte
      {10, 5, 6, 3, 'z', 9},
      {10, 5, 6, 3, 0xFF, 9},

      // 2.1 The following test-cases raise doubt whether error or partial should
      // be returned. For example, we have 4-byte sequence with valid leading
      // byte. If we hide the last byte we need to return partial. But, if the
      // second or third byte, which are visible to the call to codecvt, are
      // malformed then error should be returned.

      // replace first trailing byte with ASCII byte, also incomplete at end
      {5, 5, 3, 2, 'z', 4},
      {8, 5, 6, 3, 'z', 7},
      {9, 5, 6, 3, 'z', 7},

      // replace first trailing byte with invalid byte, also incomplete at end
      {5, 5, 3, 2, 0xFF, 4},
      {8, 5, 6, 3, 0xFF, 7},
      {9, 5, 6, 3, 0xFF, 7},

      // replace second trailing byte with ASCII byte, also incomplete at end
      {9, 5, 6, 3, 'z', 8},

      // replace second trailing byte with invalid byte, also incomplete at end
      {9, 5, 6, 3, 0xFF, 8},

      // 3. Surrogate CP. We modify the second byte (first trailing) of the 3-byte
      // CP U+D700
      {6, 5, 3, 2, 0b10100000, 4}, // turn U+D700 into U+D800
      {6, 5, 3, 2, 0b10101100, 4}, // turn U+D700 into U+DB00
      {6, 5, 3, 2, 0b10110000, 4}, // turn U+D700 into U+DC00
      {6, 5, 3, 2, 0b10111100, 4}, // turn U+D700 into U+DF00

      // 4. Overlong sequence. The CPs in the input are chosen such as modifying
      // just the leading byte is enough to make them overlong, i.e. for the
      // 3-byte and 4-byte CP the second byte (first trailing) has enough leading
      // zeroes.
      {3, 5, 1, 1, 0b11000000, 1},  // make the 2-byte CP overlong
      {3, 5, 1, 1, 0b11000001, 1},  // make the 2-byte CP overlong
      {6, 5, 3, 2, 0b11100000, 3},  // make the 3-byte CP overlong
      {10, 5, 6, 3, 0b11110000, 6}, // make the 4-byte CP overlong

      // 5. CP above range
      // turn U+10AAAA into U+14AAAA by changing its leading byte
      {10, 5, 6, 3, 0b11110101, 6},
      // turn U+10AAAA into U+11AAAA by changing its 2nd byte
      {10, 5, 6, 3, 0b10011010, 7},
      // Don't replace anything, show full 4-byte CP U+10AAAA
      {10, 4, 6, 3, 'b', 0},
      {10, 5, 6, 3, 'b', 0},
      // Don't replace anything, show incomplete 4-byte CP at the end. It's still
      // out of UCS2 range just by seeing the first byte.
      {7, 4, 6, 3, 'b', 0}, // incomplete fourth CP
      {8, 4, 6, 3, 'b', 0}, // incomplete fourth CP
      {9, 4, 6, 3, 'b', 0}, // incomplete fourth CP
      {7, 5, 6, 3, 'b', 0}, // incomplete fourth CP
      {8, 5, 6, 3, 'b', 0}, // incomplete fourth CP
      {9, 5, 6, 3, 'b', 0}, // incomplete fourth CP
  };
  for (test_offsets_error<unsigned char>* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_error<unsigned char> t = *it;
    InternT out[array_size(exp) - 1]    = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    ExternT old_char  = in[t.replace_pos];
    in[t.replace_pos] = t.replace_char;

    mbstate_t state          = {};
    const ExternT* in_next   = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.error);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<InternT>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, t.out_size);
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.expected_in_next);

    in[t.replace_pos] = old_char;
  }
}

template <class InternT, class ExternT>
void utf8_to_ucs2_in(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  utf8_to_ucs2_in_ok(cvt);
  utf8_to_ucs2_in_partial(cvt);
  utf8_to_ucs2_in_error(cvt);
}

template <class InternT, class ExternT>
void ucs2_to_utf8_out_ok(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  // UTF-8 string of 1-byte CP, 2-byte CP and 3-byte CP
  const char16_t input[]         = {'b', 0x0448, 0xAAAA, 0};
  const unsigned char expected[] = "b\u0448\uAAAA";
  static_assert(array_size(input) == 4, "");
  static_assert(array_size(expected) == 7, "");

  InternT in[array_size(input)];
  ExternT exp[array_size(expected)];
  copy(begin(input), end(input), begin(in));
  copy(begin(expected), end(expected), begin(exp));
  assert(char_traits<InternT>::length(in) == 3);
  assert(char_traits<ExternT>::length(exp) == 6);

  test_offsets_ok offsets[] = {{0, 0}, {1, 1}, {2, 3}, {3, 6}};
  for (test_offsets_ok* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_ok t                = *it;
    ExternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    mbstate_t state          = {};
    const InternT* in_next   = nullptr;
    ExternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.out(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.ok);
    assert(in_next == in + t.in_size);
    assert(out_next == out + t.out_size);
    assert(char_traits<ExternT>::compare(out, exp, t.out_size) == 0);
    if (t.out_size < array_size(out))
      assert(out[t.out_size] == 0);
  }
}

template <class InternT, class ExternT>
void ucs2_to_utf8_out_partial(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  // UTF-8 string of 1-byte CP, 2-byte CP and 3-byte CP
  const char16_t input[]         = {'b', 0x0448, 0xAAAA, 0};
  const unsigned char expected[] = "b\u0448\uAAAA";
  static_assert(array_size(input) == 4, "");
  static_assert(array_size(expected) == 7, "");

  InternT in[array_size(input)];
  ExternT exp[array_size(expected)];
  copy(begin(input), end(input), begin(in));
  copy(begin(expected), end(expected), begin(exp));
  assert(char_traits<InternT>::length(in) == 3);
  assert(char_traits<ExternT>::length(exp) == 6);

  test_offsets_partial offsets[] = {
      {1, 0, 0, 0}, // no space for first CP

      {2, 1, 1, 1}, // no space for second CP
      {2, 2, 1, 1}, // no space for second CP

      {3, 3, 2, 3}, // no space for third CP
      {3, 4, 2, 3}, // no space for third CP
      {3, 5, 2, 3}, // no space for third CP
  };
  for (test_offsets_partial* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_partial t           = *it;
    ExternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    mbstate_t state          = {};
    const InternT* in_next   = nullptr;
    ExternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.out(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.partial);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<ExternT>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);
  }
}

template <class InternT, class ExternT>
void ucs2_to_utf8_out_error(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  const char16_t input[]         = {'b', 0x0448, 0xAAAA, 0xDBEA, 0xDEAA, 0};
  const unsigned char expected[] = "b\u0448\uAAAA\U0010AAAA";
  static_assert(array_size(input) == 6, "");
  static_assert(array_size(expected) == 11, "");

  InternT in[array_size(input)];
  ExternT exp[array_size(expected)];
  copy(begin(input), end(input), begin(in));
  copy(begin(expected), end(expected), begin(exp));
  assert(char_traits<InternT>::length(in) == 5);
  assert(char_traits<ExternT>::length(exp) == 10);

  test_offsets_error<InternT> offsets[] = {
      {3, 6, 0, 0, 0xD800, 0},
      {3, 6, 0, 0, 0xDBFF, 0},
      {3, 6, 0, 0, 0xDC00, 0},
      {3, 6, 0, 0, 0xDFFF, 0},

      {3, 6, 1, 1, 0xD800, 1},
      {3, 6, 1, 1, 0xDBFF, 1},
      {3, 6, 1, 1, 0xDC00, 1},
      {3, 6, 1, 1, 0xDFFF, 1},

      {3, 6, 2, 3, 0xD800, 2},
      {3, 6, 2, 3, 0xDBFF, 2},
      {3, 6, 2, 3, 0xDC00, 2},
      {3, 6, 2, 3, 0xDFFF, 2},

      // make the leading surrogate a trailing one
      {5, 10, 3, 6, 0xDC00, 3},
      {5, 10, 3, 6, 0xDFFF, 3},

      // make the trailing surrogate a leading one
      {5, 10, 3, 6, 0xD800, 4},
      {5, 10, 3, 6, 0xDBFF, 4},

      // make the trailing surrogate a BMP char
      {5, 10, 3, 6, 'z', 4},

      // don't replace anything in the test cases bellow, just show the surrogate
      // pair (fourth CP) fully or partially
      {5, 10, 3, 6, 'b', 0},
      {5, 7, 3, 6, 'b', 0}, // no space for fourth CP
      {5, 8, 3, 6, 'b', 0}, // no space for fourth CP
      {5, 9, 3, 6, 'b', 0}, // no space for fourth CP

      {4, 10, 3, 6, 'b', 0}, // incomplete fourth CP
      {4, 7, 3, 6, 'b', 0},  // incomplete fourth CP, and no space for it
      {4, 8, 3, 6, 'b', 0},  // incomplete fourth CP, and no space for it
      {4, 9, 3, 6, 'b', 0},  // incomplete fourth CP, and no space for it
  };

  for (test_offsets_error<InternT>* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_error<InternT> t    = *it;
    ExternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    InternT old_char  = in[t.replace_pos];
    in[t.replace_pos] = t.replace_char;

    mbstate_t state          = {};
    const InternT* in_next   = nullptr;
    ExternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.out(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.error);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<ExternT>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);

    in[t.replace_pos] = old_char;
  }
}

template <class InternT, class ExternT>
void ucs2_to_utf8_out(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  ucs2_to_utf8_out_ok(cvt);
  ucs2_to_utf8_out_partial(cvt);
  ucs2_to_utf8_out_error(cvt);
}

template <class InternT, class ExternT>
void test_utf8_ucs2_cvt(const std::codecvt<InternT, ExternT, mbstate_t>& cvt) {
  utf8_to_ucs2_in(cvt);
  ucs2_to_utf8_out(cvt);
}

enum utf16_endianess { utf16_big_endian, utf16_little_endian };

template <class Iter1, class Iter2>
Iter2 utf16_to_bytes(Iter1 f, Iter1 l, Iter2 o, utf16_endianess e) {
  if (e == utf16_big_endian)
    for (; f != l; ++f) {
      *o++ = (*f >> 8) & 0xFF;
      *o++ = *f & 0xFF;
    }
  else
    for (; f != l; ++f) {
      *o++ = *f & 0xFF;
      *o++ = (*f >> 8) & 0xFF;
    }
  return o;
}

template <class InternT>
void utf16_to_utf32_in_ok(const std::codecvt<InternT, char, mbstate_t>& cvt, utf16_endianess endianess) {
  const char16_t input[]    = {'b', 0x0448, 0xAAAA, 0xDBEA, 0xDEAA, 0};
  const char32_t expected[] = {'b', 0x0448, 0xAAAA, 0x10AAAA, 0};
  static_assert(array_size(input) == 6, "");
  static_assert(array_size(expected) == 5, "");

  char in[array_size(input) * 2];
  InternT exp[array_size(expected)];
  utf16_to_bytes(begin(input), end(input), begin(in), endianess);
  copy(begin(expected), end(expected), begin(exp));

  test_offsets_ok offsets[] = {{0, 0}, {2, 1}, {4, 2}, {6, 3}, {10, 4}};
  for (test_offsets_ok* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_ok t                = *it;
    InternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    mbstate_t state          = {};
    const char* in_next      = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.ok);
    assert(in_next == in + t.in_size);
    assert(out_next == out + t.out_size);
    assert(char_traits<InternT>::compare(out, exp, t.out_size) == 0);
    if (t.out_size < array_size(out))
      assert(out[t.out_size] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, t.out_size);
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.in_size);
  }

  for (test_offsets_ok* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_ok t            = *it;
    InternT out[array_size(exp)] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    mbstate_t state          = {};
    const char* in_next      = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, end(out), out_next);
    assert(res == cvt.ok);
    assert(in_next == in + t.in_size);
    assert(out_next == out + t.out_size);
    assert(char_traits<InternT>::compare(out, exp, t.out_size) == 0);
    if (t.out_size < array_size(out))
      assert(out[t.out_size] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, array_size(out));
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.in_size);
  }
}

template <class InternT>
void utf16_to_utf32_in_partial(const std::codecvt<InternT, char, mbstate_t>& cvt, utf16_endianess endianess) {
  const char16_t input[]    = {'b', 0x0448, 0xAAAA, 0xDBEA, 0xDEAA, 0};
  const char32_t expected[] = {'b', 0x0448, 0xAAAA, 0x10AAAA, 0};
  static_assert(array_size(input) == 6, "");
  static_assert(array_size(expected) == 5, "");

  char in[array_size(input) * 2];
  InternT exp[array_size(expected)];
  utf16_to_bytes(begin(input), end(input), begin(in), endianess);
  copy(begin(expected), end(expected), begin(exp));

  test_offsets_partial offsets[] = {
      {2, 0, 0, 0}, // no space for first CP
      {1, 1, 0, 0}, // incomplete first CP
      {1, 0, 0, 0}, // incomplete first CP, and no space for it

      {4, 1, 2, 1}, // no space for second CP
      {3, 2, 2, 1}, // incomplete second CP
      {3, 1, 2, 1}, // incomplete second CP, and no space for it

      {6, 2, 4, 2}, // no space for third CP
      {5, 3, 4, 2}, // incomplete third CP
      {5, 2, 4, 2}, // incomplete third CP, and no space for it

      {10, 3, 6, 3}, // no space for fourth CP
      {7, 4, 6, 3},  // incomplete fourth CP
      {8, 4, 6, 3},  // incomplete fourth CP
      {9, 4, 6, 3},  // incomplete fourth CP
      {7, 3, 6, 3},  // incomplete fourth CP, and no space for it
      {8, 3, 6, 3},  // incomplete fourth CP, and no space for it
      {9, 3, 6, 3},  // incomplete fourth CP, and no space for it
  };

  for (test_offsets_partial* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_partial t           = *it;
    InternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    mbstate_t state          = {};
    const char* in_next      = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.partial);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<InternT>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, t.out_size);
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.expected_in_next);
  }
}

template <class InternT>
void utf16_to_utf32_in_error(const std::codecvt<InternT, char, mbstate_t>& cvt, utf16_endianess endianess) {
  char16_t input[]          = {'b', 0x0448, 0xAAAA, 0xDBEA, 0xDEAA, 0};
  const char32_t expected[] = {'b', 0x0448, 0xAAAA, 0x10AAAA, 0};
  static_assert(array_size(input) == 6, "");
  static_assert(array_size(expected) == 5, "");

  InternT exp[array_size(expected)];
  copy(begin(expected), end(expected), begin(exp));

  // The only possible error in UTF-16 is unpaired surrogate code units.
  // So we replace valid code points (scalar values) with lone surrogate CU.
  test_offsets_error<char16_t> offsets[] = {
      {10, 4, 0, 0, 0xD800, 0},
      {10, 4, 0, 0, 0xDBFF, 0},
      {10, 4, 0, 0, 0xDC00, 0},
      {10, 4, 0, 0, 0xDFFF, 0},

      {10, 4, 2, 1, 0xD800, 1},
      {10, 4, 2, 1, 0xDBFF, 1},
      {10, 4, 2, 1, 0xDC00, 1},
      {10, 4, 2, 1, 0xDFFF, 1},

      {10, 4, 4, 2, 0xD800, 2},
      {10, 4, 4, 2, 0xDBFF, 2},
      {10, 4, 4, 2, 0xDC00, 2},
      {10, 4, 4, 2, 0xDFFF, 2},

      // make the leading surrogate a trailing one
      {10, 4, 6, 3, 0xDC00, 3},
      {10, 4, 6, 3, 0xDFFF, 3},

      // make the trailing surrogate a leading one
      {10, 4, 6, 3, 0xD800, 4},
      {10, 4, 6, 3, 0xDBFF, 4},

      // make the trailing surrogate a BMP char
      {10, 4, 6, 3, 'z', 4},
  };

  for (test_offsets_error<char16_t>* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_error<char16_t> t = *it;
    char in[array_size(input) * 2];
    InternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    char16_t old_char    = input[t.replace_pos];
    input[t.replace_pos] = t.replace_char; // replace in input, not in in
    utf16_to_bytes(begin(input), end(input), begin(in), endianess);

    mbstate_t state          = {};
    const char* in_next      = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.error);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<InternT>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, t.out_size);
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.expected_in_next);

    input[t.replace_pos] = old_char;
  }
}

template <class InternT>
void utf32_to_utf16_out_ok(const std::codecvt<InternT, char, mbstate_t>& cvt, utf16_endianess endianess) {
  const char32_t input[]    = {'b', 0x0448, 0xAAAA, 0x10AAAA, 0};
  const char16_t expected[] = {'b', 0x0448, 0xAAAA, 0xDBEA, 0xDEAA, 0};
  static_assert(array_size(input) == 5, "");
  static_assert(array_size(expected) == 6, "");

  InternT in[array_size(input)];
  char exp[array_size(expected) * 2];
  copy(begin(input), end(input), begin(in));
  utf16_to_bytes(begin(expected), end(expected), begin(exp), endianess);

  test_offsets_ok offsets[] = {{0, 0}, {1, 2}, {2, 4}, {3, 6}, {4, 10}};
  for (test_offsets_ok* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_ok t             = *it;
    char out[array_size(exp) - 2] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    mbstate_t state          = {};
    const InternT* in_next   = nullptr;
    char* out_next           = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.out(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.ok);
    assert(in_next == in + t.in_size);
    assert(out_next == out + t.out_size);
    assert(char_traits<char>::compare(out, exp, t.out_size) == 0);
    if (t.out_size < array_size(out))
      assert(out[t.out_size] == 0);
  }
}

template <class InternT>
void utf32_to_utf16_out_partial(const std::codecvt<InternT, char, mbstate_t>& cvt, utf16_endianess endianess) {
  const char32_t input[]    = {'b', 0x0448, 0xAAAA, 0x10AAAA, 0};
  const char16_t expected[] = {'b', 0x0448, 0xAAAA, 0xDBEA, 0xDEAA, 0};
  static_assert(array_size(input) == 5, "");
  static_assert(array_size(expected) == 6, "");

  InternT in[array_size(input)];
  char exp[array_size(expected) * 2];
  copy(begin(input), end(input), begin(in));
  utf16_to_bytes(begin(expected), end(expected), begin(exp), endianess);

  test_offsets_partial offsets[] = {
      {1, 0, 0, 0}, // no space for first CP
      {1, 1, 0, 0}, // no space for first CP

      {2, 2, 1, 2}, // no space for second CP
      {2, 3, 1, 2}, // no space for second CP

      {3, 4, 2, 4}, // no space for third CP
      {3, 5, 2, 4}, // no space for third CP

      {4, 6, 3, 6}, // no space for fourth CP
      {4, 7, 3, 6}, // no space for fourth CP
      {4, 8, 3, 6}, // no space for fourth CP
      {4, 9, 3, 6}, // no space for fourth CP
  };
  for (test_offsets_partial* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_partial t        = *it;
    char out[array_size(exp) - 2] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    mbstate_t state          = {};
    const InternT* in_next   = nullptr;
    char* out_next           = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.out(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.partial);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<char>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);
  }
}

template <class InternT>
void utf32_to_utf16_out_error(const std::codecvt<InternT, char, mbstate_t>& cvt, utf16_endianess endianess) {
  const char32_t input[]    = {'b', 0x0448, 0xAAAA, 0x10AAAA, 0};
  const char16_t expected[] = {'b', 0x0448, 0xAAAA, 0xDBEA, 0xDEAA, 0};
  static_assert(array_size(input) == 5, "");
  static_assert(array_size(expected) == 6, "");

  InternT in[array_size(input)];
  char exp[array_size(expected) * 2];
  copy(begin(input), end(input), begin(in));
  utf16_to_bytes(begin(expected), end(expected), begin(exp), endianess);

  test_offsets_error<InternT> offsets[] = {

      // Surrogate CP
      {4, 10, 0, 0, 0xD800, 0},
      {4, 10, 1, 2, 0xDBFF, 1},
      {4, 10, 2, 4, 0xDC00, 2},
      {4, 10, 3, 6, 0xDFFF, 3},

      // CP out of range
      {4, 10, 0, 0, 0x00110000, 0},
      {4, 10, 1, 2, 0x00110000, 1},
      {4, 10, 2, 4, 0x00110000, 2},
      {4, 10, 3, 6, 0x00110000, 3}};

  for (test_offsets_error<InternT>* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_error<InternT> t = *it;
    char out[array_size(exp) - 2] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    InternT old_char  = in[t.replace_pos];
    in[t.replace_pos] = t.replace_char;

    mbstate_t state          = {};
    const InternT* in_next   = nullptr;
    char* out_next           = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.out(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.error);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<char>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);

    in[t.replace_pos] = old_char;
  }
}

template <class InternT>
void test_utf16_utf32_cvt(const std::codecvt<InternT, char, mbstate_t>& cvt, utf16_endianess endianess) {
  utf16_to_utf32_in_ok(cvt, endianess);
  utf16_to_utf32_in_partial(cvt, endianess);
  utf16_to_utf32_in_error(cvt, endianess);
  utf32_to_utf16_out_ok(cvt, endianess);
  utf32_to_utf16_out_partial(cvt, endianess);
  utf32_to_utf16_out_error(cvt, endianess);
}

template <class InternT>
void utf16_to_ucs2_in_ok(const std::codecvt<InternT, char, mbstate_t>& cvt, utf16_endianess endianess) {
  const char16_t input[]    = {'b', 0x0448, 0xAAAA, 0};
  const char16_t expected[] = {'b', 0x0448, 0xAAAA, 0};
  static_assert(array_size(input) == 4, "");
  static_assert(array_size(expected) == 4, "");

  char in[array_size(input) * 2];
  InternT exp[array_size(expected)];
  utf16_to_bytes(begin(input), end(input), begin(in), endianess);
  copy(begin(expected), end(expected), begin(exp));

  test_offsets_ok offsets[] = {{0, 0}, {2, 1}, {4, 2}, {6, 3}};
  for (test_offsets_ok* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_ok t                = *it;
    InternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    mbstate_t state          = {};
    const char* in_next      = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.ok);
    assert(in_next == in + t.in_size);
    assert(out_next == out + t.out_size);
    assert(char_traits<InternT>::compare(out, exp, t.out_size) == 0);
    if (t.out_size < array_size(out))
      assert(out[t.out_size] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, t.out_size);
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.in_size);
  }

  for (test_offsets_ok* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_ok t            = *it;
    InternT out[array_size(exp)] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    mbstate_t state          = {};
    const char* in_next      = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, end(out), out_next);
    assert(res == cvt.ok);
    assert(in_next == in + t.in_size);
    assert(out_next == out + t.out_size);
    assert(char_traits<InternT>::compare(out, exp, t.out_size) == 0);
    if (t.out_size < array_size(out))
      assert(out[t.out_size] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, array_size(out));
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.in_size);
  }
}

template <class InternT>
void utf16_to_ucs2_in_partial(const std::codecvt<InternT, char, mbstate_t>& cvt, utf16_endianess endianess) {
  const char16_t input[]    = {'b', 0x0448, 0xAAAA, 0};
  const char16_t expected[] = {'b', 0x0448, 0xAAAA, 0};
  static_assert(array_size(input) == 4, "");
  static_assert(array_size(expected) == 4, "");

  char in[array_size(input) * 2];
  InternT exp[array_size(expected)];
  utf16_to_bytes(begin(input), end(input), begin(in), endianess);
  copy(begin(expected), end(expected), begin(exp));

  test_offsets_partial offsets[] = {
      {2, 0, 0, 0}, // no space for first CP
      {1, 1, 0, 0}, // incomplete first CP
      {1, 0, 0, 0}, // incomplete first CP, and no space for it

      {4, 1, 2, 1}, // no space for second CP
      {3, 2, 2, 1}, // incomplete second CP
      {3, 1, 2, 1}, // incomplete second CP, and no space for it

      {6, 2, 4, 2}, // no space for third CP
      {5, 3, 4, 2}, // incomplete third CP
      {5, 2, 4, 2}, // incomplete third CP, and no space for it
  };

  for (test_offsets_partial* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_partial t           = *it;
    InternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    mbstate_t state          = {};
    const char* in_next      = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.partial);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<InternT>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, t.out_size);
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.expected_in_next);
  }
}

template <class InternT>
void utf16_to_ucs2_in_error(const std::codecvt<InternT, char, mbstate_t>& cvt, utf16_endianess endianess) {
  char16_t input[]          = {'b', 0x0448, 0xAAAA, 0xDBEA, 0xDEAA, 0};
  const char16_t expected[] = {'b', 0x0448, 0xAAAA, 0xDBEA, 0xDEAA, 0};
  static_assert(array_size(input) == 6, "");
  static_assert(array_size(expected) == 6, "");

  InternT exp[array_size(expected)];
  copy(begin(expected), end(expected), begin(exp));

  // The only possible error in UTF-16 is unpaired surrogate code units.
  // Additionally, because the target encoding is UCS-2, a proper pair of
  // surrogates is also error. Simply, any surrogate CU is error.
  test_offsets_error<char16_t> offsets[] = {
      {6, 3, 0, 0, 0xD800, 0},
      {6, 3, 0, 0, 0xDBFF, 0},
      {6, 3, 0, 0, 0xDC00, 0},
      {6, 3, 0, 0, 0xDFFF, 0},

      {6, 3, 2, 1, 0xD800, 1},
      {6, 3, 2, 1, 0xDBFF, 1},
      {6, 3, 2, 1, 0xDC00, 1},
      {6, 3, 2, 1, 0xDFFF, 1},

      {6, 3, 4, 2, 0xD800, 2},
      {6, 3, 4, 2, 0xDBFF, 2},
      {6, 3, 4, 2, 0xDC00, 2},
      {6, 3, 4, 2, 0xDFFF, 2},

      // make the leading surrogate a trailing one
      {10, 5, 6, 3, 0xDC00, 3},
      {10, 5, 6, 3, 0xDFFF, 3},

      // make the trailing surrogate a leading one
      {10, 5, 6, 3, 0xD800, 4},
      {10, 5, 6, 3, 0xDBFF, 4},

      // make the trailing surrogate a BMP char
      {10, 5, 6, 3, 'z', 4},

      // don't replace anything in the test cases bellow, just show the surrogate
      // pair (fourth CP) fully or partially (just the first surrogate)
      {10, 5, 6, 3, 'b', 0},
      {8, 5, 6, 3, 'b', 0},
      {9, 5, 6, 3, 'b', 0},

      {10, 4, 6, 3, 'b', 0},
      {8, 4, 6, 3, 'b', 0},
      {9, 4, 6, 3, 'b', 0},
  };

  for (test_offsets_error<char16_t>* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_error<char16_t> t = *it;
    char in[array_size(input) * 2];
    InternT out[array_size(exp) - 1] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    char16_t old_char    = input[t.replace_pos];
    input[t.replace_pos] = t.replace_char; // replace in input, not in in
    utf16_to_bytes(begin(input), end(input), begin(in), endianess);

    mbstate_t state          = {};
    const char* in_next      = nullptr;
    InternT* out_next        = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.in(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.error);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<InternT>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);

    state   = mbstate_t();
    int len = cvt.length(state, in, in + t.in_size, t.out_size);
    assert(len >= 0);
    assert(static_cast<size_t>(len) == t.expected_in_next);

    input[t.replace_pos] = old_char;
  }
}

template <class InternT>
void ucs2_to_utf16_out_ok(const std::codecvt<InternT, char, mbstate_t>& cvt, utf16_endianess endianess) {
  const char16_t input[]    = {'b', 0x0448, 0xAAAA, 0};
  const char16_t expected[] = {'b', 0x0448, 0xAAAA, 0};
  static_assert(array_size(input) == 4, "");
  static_assert(array_size(expected) == 4, "");

  InternT in[array_size(input)];
  char exp[array_size(expected) * 2];
  copy(begin(input), end(input), begin(in));
  utf16_to_bytes(begin(expected), end(expected), begin(exp), endianess);

  test_offsets_ok offsets[] = {{0, 0}, {1, 2}, {2, 4}, {3, 6}};
  for (test_offsets_ok* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_ok t             = *it;
    char out[array_size(exp) - 2] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    mbstate_t state          = {};
    const InternT* in_next   = nullptr;
    char* out_next           = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.out(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.ok);
    assert(in_next == in + t.in_size);
    assert(out_next == out + t.out_size);
    assert(char_traits<char>::compare(out, exp, t.out_size) == 0);
    if (t.out_size < array_size(out))
      assert(out[t.out_size] == 0);
  }
}

template <class InternT>
void ucs2_to_utf16_out_partial(const std::codecvt<InternT, char, mbstate_t>& cvt, utf16_endianess endianess) {
  const char16_t input[]    = {'b', 0x0448, 0xAAAA, 0};
  const char16_t expected[] = {'b', 0x0448, 0xAAAA, 0};
  static_assert(array_size(input) == 4, "");
  static_assert(array_size(expected) == 4, "");

  InternT in[array_size(input)];
  char exp[array_size(expected) * 2];
  copy(begin(input), end(input), begin(in));
  utf16_to_bytes(begin(expected), end(expected), begin(exp), endianess);

  test_offsets_partial offsets[] = {
      {1, 0, 0, 0}, // no space for first CP
      {1, 1, 0, 0}, // no space for first CP

      {2, 2, 1, 2}, // no space for second CP
      {2, 3, 1, 2}, // no space for second CP

      {3, 4, 2, 4}, // no space for third CP
      {3, 5, 2, 4}, // no space for third CP
  };
  for (test_offsets_partial* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_partial t        = *it;
    char out[array_size(exp) - 2] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    mbstate_t state          = {};
    const InternT* in_next   = nullptr;
    char* out_next           = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.out(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.partial);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<char>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);
  }
}

template <class InternT>
void ucs2_to_utf16_out_error(const std::codecvt<InternT, char, mbstate_t>& cvt, utf16_endianess endianess) {
  const char16_t input[]    = {'b', 0x0448, 0xAAAA, 0xDBEA, 0xDEAA, 0};
  const char16_t expected[] = {'b', 0x0448, 0xAAAA, 0xDBEA, 0xDEAA, 0};
  static_assert(array_size(input) == 6, "");
  static_assert(array_size(expected) == 6, "");

  InternT in[array_size(input)];
  char exp[array_size(expected) * 2];
  copy(begin(input), end(input), begin(in));
  utf16_to_bytes(begin(expected), end(expected), begin(exp), endianess);

  test_offsets_error<InternT> offsets[] = {
      {3, 6, 0, 0, 0xD800, 0},
      {3, 6, 0, 0, 0xDBFF, 0},
      {3, 6, 0, 0, 0xDC00, 0},
      {3, 6, 0, 0, 0xDFFF, 0},

      {3, 6, 1, 2, 0xD800, 1},
      {3, 6, 1, 2, 0xDBFF, 1},
      {3, 6, 1, 2, 0xDC00, 1},
      {3, 6, 1, 2, 0xDFFF, 1},

      {3, 6, 2, 4, 0xD800, 2},
      {3, 6, 2, 4, 0xDBFF, 2},
      {3, 6, 2, 4, 0xDC00, 2},
      {3, 6, 2, 4, 0xDFFF, 2},

      // make the leading surrogate a trailing one
      {5, 10, 3, 6, 0xDC00, 3},
      {5, 10, 3, 6, 0xDFFF, 3},

      // make the trailing surrogate a leading one
      {5, 10, 3, 6, 0xD800, 4},
      {5, 10, 3, 6, 0xDBFF, 4},

      // make the trailing surrogate a BMP char
      {5, 10, 3, 6, 'z', 4},

      // don't replace anything in the test cases bellow, just show the surrogate
      // pair (fourth CP) fully or partially (just the first surrogate)
      {5, 10, 3, 6, 'b', 0},
      {5, 8, 3, 6, 'b', 0},
      {5, 9, 3, 6, 'b', 0},

      {4, 10, 3, 6, 'b', 0},
      {4, 8, 3, 6, 'b', 0},
      {4, 9, 3, 6, 'b', 0},
  };

  for (test_offsets_error<InternT>* it = begin(offsets); it != end(offsets); ++it) {
    test_offsets_error<InternT> t = *it;
    char out[array_size(exp) - 2] = {};
    assert(t.in_size <= array_size(in));
    assert(t.out_size <= array_size(out));
    assert(t.expected_in_next <= t.in_size);
    assert(t.expected_out_next <= t.out_size);
    InternT old_char  = in[t.replace_pos];
    in[t.replace_pos] = t.replace_char;

    mbstate_t state          = {};
    const InternT* in_next   = nullptr;
    char* out_next           = nullptr;
    codecvt_base::result res = codecvt_base::ok;

    res = cvt.out(state, in, in + t.in_size, in_next, out, out + t.out_size, out_next);
    assert(res == cvt.error);
    assert(in_next == in + t.expected_in_next);
    assert(out_next == out + t.expected_out_next);
    assert(char_traits<char>::compare(out, exp, t.expected_out_next) == 0);
    if (t.expected_out_next < array_size(out))
      assert(out[t.expected_out_next] == 0);

    in[t.replace_pos] = old_char;
  }
}

template <class InternT>
void test_utf16_ucs2_cvt(const std::codecvt<InternT, char, mbstate_t>& cvt, utf16_endianess endianess) {
  utf16_to_ucs2_in_ok(cvt, endianess);
  utf16_to_ucs2_in_partial(cvt, endianess);
  utf16_to_ucs2_in_error(cvt, endianess);
  ucs2_to_utf16_out_ok(cvt, endianess);
  ucs2_to_utf16_out_partial(cvt, endianess);
  ucs2_to_utf16_out_error(cvt, endianess);
}

using std::codecvt;
using std::codecvt_utf16;
using std::codecvt_utf8;
using std::codecvt_utf8_utf16;
using std::has_facet;
using std::locale;
using std::use_facet;

void test_utf8_utf32_codecvts() {
  typedef codecvt<char32_t, char, mbstate_t> codecvt_c32;
  const locale& loc_c = locale::classic();
  assert(has_facet<codecvt_c32>(loc_c));

  const codecvt_c32& cvt = use_facet<codecvt_c32>(loc_c);
  test_utf8_utf32_cvt(cvt);

  codecvt_utf8<char32_t> cvt2;
  test_utf8_utf32_cvt(cvt2);

#if !defined(TEST_HAS_NO_WIDE_CHARACTERS) && !defined(TEST_SHORT_WCHAR)
  codecvt_utf8<wchar_t> cvt3;
  test_utf8_utf32_cvt(cvt3);
#endif

#ifndef TEST_HAS_NO_CHAR8_T
  typedef codecvt<char32_t, char8_t, mbstate_t> codecvt_c32_c8;
  assert(has_facet<codecvt_c32_c8>(loc_c));
  const codecvt_c32_c8& cvt4 = use_facet<codecvt_c32_c8>(loc_c);
  test_utf8_utf32_cvt(cvt4);
#endif
}

void test_utf8_utf16_codecvts() {
  typedef codecvt<char16_t, char, mbstate_t> codecvt_c16;
  const locale& loc_c = locale::classic();
  assert(has_facet<codecvt_c16>(loc_c));

  const codecvt_c16& cvt = use_facet<codecvt_c16>(loc_c);
  test_utf8_utf16_cvt(cvt);

  codecvt_utf8_utf16<char16_t> cvt2;
  test_utf8_utf16_cvt(cvt2);

  codecvt_utf8_utf16<char32_t> cvt3;
  test_utf8_utf16_cvt(cvt3);

#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  codecvt_utf8_utf16<wchar_t> cvt4;
  test_utf8_utf16_cvt(cvt4);
#endif

#ifndef TEST_HAS_NO_CHAR8_T
  typedef codecvt<char16_t, char8_t, mbstate_t> codecvt_c16_c8;
  assert(has_facet<codecvt_c16_c8>(loc_c));
  const codecvt_c16_c8& cvt5 = use_facet<codecvt_c16_c8>(loc_c);
  test_utf8_utf16_cvt(cvt5);
#endif
}

void test_utf8_ucs2_codecvts() {
  codecvt_utf8<char16_t> cvt;
  test_utf8_ucs2_cvt(cvt);

#if !defined(TEST_HAS_NO_WIDE_CHARACTERS) && defined(TEST_SHORT_WCHAR)
  codecvt_utf8<wchar_t> cvt2;
  test_utf8_ucs2_cvt(cvt2);
#endif
}

void test_utf16_utf32_codecvts() {
  codecvt_utf16<char32_t> cvt;
  test_utf16_utf32_cvt(cvt, utf16_big_endian);

  codecvt_utf16<char32_t, 0x10FFFF, std::little_endian> cvt2;
  test_utf16_utf32_cvt(cvt2, utf16_little_endian);

#if !defined(TEST_HAS_NO_WIDE_CHARACTERS) && !defined(TEST_SHORT_WCHAR)
  codecvt_utf16<wchar_t> cvt3;
  test_utf16_utf32_cvt(cvt3, utf16_big_endian);

  codecvt_utf16<wchar_t, 0x10FFFF, std::little_endian> cvt4;
  test_utf16_utf32_cvt(cvt4, utf16_little_endian);
#endif
}

void test_utf16_ucs2_codecvts() {
  codecvt_utf16<char16_t> cvt;
  test_utf16_ucs2_cvt(cvt, utf16_big_endian);

  codecvt_utf16<char16_t, 0x10FFFF, std::little_endian> cvt2;
  test_utf16_ucs2_cvt(cvt2, utf16_little_endian);

#if !defined(TEST_HAS_NO_WIDE_CHARACTERS) && defined(TEST_SHORT_WCHAR)
  codecvt_utf16<wchar_t> cvt3;
  test_utf16_ucs2_cvt(cvt3, utf16_big_endian);

  codecvt_utf16<wchar_t, 0x10FFFF, std::little_endian> cvt4;
  test_utf16_ucs2_cvt(cvt4, utf16_little_endian);
#endif
}

int main() {
  test_utf8_utf32_codecvts();
  test_utf8_utf16_codecvts();
  test_utf8_ucs2_codecvts();
  test_utf16_utf32_codecvts();
  test_utf16_ucs2_codecvts();
}
