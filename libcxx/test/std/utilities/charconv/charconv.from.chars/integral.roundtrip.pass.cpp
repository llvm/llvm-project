//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// ADDITIONAL_COMPILE_FLAGS(has-fconstexpr-steps): -fconstexpr-steps=12712420

// <charconv>

// constexpr from_chars_result from_chars(const char* first, const char* last,
//                                        Integral& value, int base = 10)

#include <cassert>
#include <charconv>
#include <system_error>

#include "charconv_test_helpers.h"
#include "test_macros.h"
#include "type_algorithms.h"

template <class X, class T, class... Ts>
TEST_CONSTEXPR_CXX23 void test_roundtrip(T v, Ts... args) {
  std::from_chars_result r2;
  std::to_chars_result r;
  X x = 0xc;
  char buf[150];

  if (fits_in<X>(v)) {
    r = std::to_chars(buf, buf + sizeof(buf), v, args...);
    assert(r.ec == std::errc{});

    r2 = std::from_chars(buf, r.ptr, x, args...);
    assert(r2.ptr == r.ptr);
    assert(x == X(v));
  } else {
    r = std::to_chars(buf, buf + sizeof(buf), v, args...);
    assert(r.ec == std::errc{});

    r2 = std::from_chars(buf, r.ptr, x, args...);

    TEST_DIAGNOSTIC_PUSH
    TEST_MSVC_DIAGNOSTIC_IGNORED(4127) // conditional expression is constant

    if (std::is_signed<T>::value && v < 0 && std::is_unsigned<X>::value) {
      assert(x == 0xc);
      assert(r2.ptr == buf);
      assert(r2.ec == std::errc::invalid_argument);
    } else {
      assert(x == 0xc);
      assert(r2.ptr == r.ptr);
      assert(r2.ec == std::errc::result_out_of_range);
    }

    TEST_DIAGNOSTIC_POP
  }
}

struct test_basics {
  template <typename T>
  TEST_CONSTEXPR_CXX23 void operator()() {
    test_roundtrip<T>(0);
    test_roundtrip<T>(42);
    test_roundtrip<T>(32768);
    test_roundtrip<T>(0, 10);
    test_roundtrip<T>(42, 10);
    test_roundtrip<T>(32768, 10);
    test_roundtrip<T>(0xf, 16);
    test_roundtrip<T>(0xdeadbeaf, 16);
    test_roundtrip<T>(0755, 8);

    for (int b = 2; b < 37; ++b) {
      using xl = std::numeric_limits<T>;

      test_roundtrip<T>(1, b);
      test_roundtrip<T>(-1, b);
      test_roundtrip<T>(xl::lowest(), b);
      test_roundtrip<T>((xl::max)(), b);
      test_roundtrip<T>((xl::max)() / 2, b);
    }
  }
};

struct test_signed {
  template <typename T>
  TEST_CONSTEXPR_CXX23 void operator()() {
    test_roundtrip<T>(-1);
    test_roundtrip<T>(-12);
    test_roundtrip<T>(-1, 10);
    test_roundtrip<T>(-12, 10);
    test_roundtrip<T>(-21734634, 10);
    test_roundtrip<T>(-2647, 2);
    test_roundtrip<T>(-0xcc1, 16);

    for (int b = 2; b < 37; ++b) {
      using xl = std::numeric_limits<T>;

      test_roundtrip<T>(0, b);
      test_roundtrip<T>(xl::lowest(), b);
      test_roundtrip<T>((xl::max)(), b);
    }
  }
};

TEST_CONSTEXPR_CXX23 bool test() {
  types::for_each(types::integer_types{}, test_basics{});
  types::for_each(types::signed_integer_types{}, test_signed{});

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 20
  static_assert(test());
#endif

  return 0;
}
