//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14

// XFAIL: availability-fp_from_chars-missing

// from_chars_result from_chars(const char* first, const char* last,
//                              float& value, chars_format fmt = chars_format::general)
//
// from_chars_result from_chars(const char* first, const char* last,
//                              double& value, chars_format fmt = chars_format::general)

#include <array>
#include <charconv>
#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <system_error>

#include "charconv_test_helpers.h"
#include "test_macros.h"

template <class F>
void test_infinity(std::chars_format fmt) {
  const char* s = "-InFiNiTyXXX";
  { // I
    F value                       = 0.25;
    std::from_chars_result result = std::from_chars(s + 1, s + 2, value, fmt);

    assert(result.ec == std::errc::invalid_argument);
    assert(result.ptr == s + 1);
    assert(value == F(0.25));
  }
  { // In
    F value                       = 0.25;
    std::from_chars_result result = std::from_chars(s + 1, s + 3, value, fmt);

    assert(result.ec == std::errc::invalid_argument);
    assert(result.ptr == s + 1);
    assert(value == F(0.25));
  }
  { // InF
    F value                       = 0.25;
    std::from_chars_result result = std::from_chars(s + 1, s + 4, value, fmt);

    assert(result.ec == std::errc{});
    assert(result.ptr == s + 4);
    assert(value == std::numeric_limits<F>::infinity());
  }
  { // -InF
    F value                       = 0.25;
    std::from_chars_result result = std::from_chars(s, s + 4, value, fmt);

    assert(result.ec == std::errc{});
    assert(result.ptr == s + 4);
    assert(value == -std::numeric_limits<F>::infinity());
  }
  { // InFi
    F value                       = 0.25;
    std::from_chars_result result = std::from_chars(s + 1, s + 5, value, fmt);

    assert(result.ec == std::errc{});
    assert(result.ptr == s + 4);
    assert(value == std::numeric_limits<F>::infinity());
  }
  { // -InFiN
    F value                       = 0.25;
    std::from_chars_result result = std::from_chars(s, s + 6, value, fmt);

    assert(result.ec == std::errc{});
    assert(result.ptr == s + 4);
    assert(value == -std::numeric_limits<F>::infinity());
  }
  { // InFiNi
    F value                       = 0.25;
    std::from_chars_result result = std::from_chars(s + 1, s + 7, value, fmt);

    assert(result.ec == std::errc{});
    assert(result.ptr == s + 4);
    assert(value == std::numeric_limits<F>::infinity());
  }
  { // -InFiNiT
    F value                       = 0.25;
    std::from_chars_result result = std::from_chars(s, s + 8, value, fmt);

    assert(result.ec == std::errc{});
    assert(result.ptr == s + 4);
    assert(value == -std::numeric_limits<F>::infinity());
  }
  { // InFiNiTy
    F value                       = 0.25;
    std::from_chars_result result = std::from_chars(s + 1, s + 9, value, fmt);

    assert(result.ec == std::errc{});
    assert(result.ptr == s + 9);
    assert(value == std::numeric_limits<F>::infinity());
  }
  { // -InFiNiTy
    F value                       = 0.25;
    std::from_chars_result result = std::from_chars(s, s + 9, value, fmt);

    assert(result.ec == std::errc{});
    assert(result.ptr == s + 9);
    assert(value == -std::numeric_limits<F>::infinity());
  }
  { // InFiNiTyXXX
    F value                       = 0.25;
    std::from_chars_result result = std::from_chars(s + 1, s + 12, value, fmt);

    assert(result.ec == std::errc{});
    assert(result.ptr == s + 9);
    assert(value == std::numeric_limits<F>::infinity());
  }
  { // -InFiNiTyXXX
    F value                       = 0.25;
    std::from_chars_result result = std::from_chars(s, s + 12, value, fmt);

    assert(result.ec == std::errc{});
    assert(result.ptr == s + 9);
    assert(value == -std::numeric_limits<F>::infinity());
  }
}

template <class F>
void test_nan(std::chars_format fmt) {
  {
    const char* s = "-NaN(1_A)XXX";
    { // N
      F value                       = 0.25;
      std::from_chars_result result = std::from_chars(s + 1, s + 2, value, fmt);

      assert(result.ec == std::errc::invalid_argument);
      assert(result.ptr == s + 1);
      assert(value == F(0.25));
    }
    { // Na
      F value                       = 0.25;
      std::from_chars_result result = std::from_chars(s + 1, s + 3, value, fmt);

      assert(result.ec == std::errc::invalid_argument);
      assert(result.ptr == s + 1);
      assert(value == F(0.25));
    }
    { // NaN
      F value                       = 0.25;
      std::from_chars_result result = std::from_chars(s + 1, s + 4, value, fmt);

      assert(result.ec == std::errc{});
      assert(result.ptr == s + 4);
      assert(std::isnan(value));
      assert(!std::signbit(value));
    }
    { // -NaN
      F value                       = 0.25;
      std::from_chars_result result = std::from_chars(s + 0, s + 4, value, fmt);

      assert(result.ec == std::errc{});
      assert(result.ptr == s + 4);
      assert(std::isnan(value));
      assert(std::signbit(value));
    }
    { // NaN(
      F value                       = 0.25;
      std::from_chars_result result = std::from_chars(s + 1, s + 5, value, fmt);

      assert(result.ec == std::errc{});
      assert(result.ptr == s + 4);
      assert(std::isnan(value));
      assert(!std::signbit(value));
    }
    { // -NaN(1
      F value                       = 0.25;
      std::from_chars_result result = std::from_chars(s, s + 6, value, fmt);

      assert(result.ec == std::errc{});
      assert(result.ptr == s + 4);
      assert(std::isnan(value));
      assert(std::signbit(value));
    }
    { // NaN(1_
      F value                       = 0.25;
      std::from_chars_result result = std::from_chars(s + 1, s + 7, value, fmt);

      assert(result.ec == std::errc{});
      assert(result.ptr == s + 4);
      assert(std::isnan(value));
      assert(!std::signbit(value));
    }
    { // -NaN(1_A
      F value                       = 0.25;
      std::from_chars_result result = std::from_chars(s, s + 8, value, fmt);

      assert(result.ec == std::errc{});
      assert(result.ptr == s + 4);
      assert(std::isnan(value));
      assert(std::signbit(value));
    }
    { // NaN(1_A)
      F value                       = 0.25;
      std::from_chars_result result = std::from_chars(s + 1, s + 9, value, fmt);

      assert(result.ec == std::errc{});
      assert(result.ptr == s + 9);
      assert(std::isnan(value));
      assert(!std::signbit(value));
    }
    { // -NaN(1_A)
      F value                       = 0.25;
      std::from_chars_result result = std::from_chars(s, s + 9, value, fmt);

      assert(result.ec == std::errc{});
      assert(result.ptr == s + 9);
      assert(std::isnan(value));
      assert(std::signbit(value));
    }
    { // NaN(1_A)XXX
      F value                       = 0.25;
      std::from_chars_result result = std::from_chars(s + 1, s + 12, value, fmt);

      assert(result.ec == std::errc{});
      assert(result.ptr == s + 9);
      assert(std::isnan(value));
      assert(!std::signbit(value));
    }
    { // -NaN(1_A)XXX
      F value                       = 0.25;
      std::from_chars_result result = std::from_chars(s, s + 12, value, fmt);

      assert(result.ec == std::errc{});
      assert(result.ptr == s + 9);
      assert(std::isnan(value));
      assert(std::signbit(value));
    }
  }
  {
    const char* s                 = "NaN()";
    F value                       = 0.25;
    std::from_chars_result result = std::from_chars(s, s + std::strlen(s), value, fmt);

    assert(result.ec == std::errc{});
    assert(result.ptr == s + 5);
    assert(std::isnan(value));
    assert(!std::signbit(value));
  }
  { // validates a n-char-sequences with an invalid value
    std::array s = {'N', 'a', 'N', '(', ' ', ')'};
    s[4]         = 'a';
    {
      F value                       = 0.25;
      std::from_chars_result result = std::from_chars(s.data(), s.data() + s.size(), value, fmt);

      assert(result.ec == std::errc{});
      assert(result.ptr == s.data() + s.size());
      assert(std::isnan(value));
      assert(!std::signbit(value));
    }
    for (auto c : "!@#$%^&*(-=+[]{}|\\;:'\",./<>?~` \t\v\r\n") {
      F value                       = 0.25;
      s[4]                          = c;
      std::from_chars_result result = std::from_chars(s.data(), s.data() + s.size(), value, fmt);

      assert(result.ec == std::errc{});
      assert(result.ptr == s.data() + 3);
      assert(std::isnan(value));
      assert(!std::signbit(value));
    }
  }
}

template <class F>
void test_fmt_independent(std::chars_format fmt) {
  test_infinity<F>(fmt);
  test_nan<F>(fmt);

  { // first == last
    F value                       = 0.25;
    std::from_chars_result result = std::from_chars(nullptr, nullptr, value, fmt);

    assert(result.ec == std::errc::invalid_argument);
    assert(result.ptr == nullptr);
    assert(value == F(0.25));
  }
  { // only a sign
    F value                       = 0.25;
    const char* s                 = "-";
    std::from_chars_result result = std::from_chars(s, s + std::strlen(s), value, fmt);

    assert(result.ec == std::errc::invalid_argument);
    assert(result.ptr == s);
    assert(value == F(0.25));
  }
  { // only decimal separator
    F value                       = 0.25;
    const char* s                 = ".";
    std::from_chars_result result = std::from_chars(s, s + std::strlen(s), value, fmt);

    assert(result.ec == std::errc::invalid_argument);
    assert(result.ptr == s);
    assert(value == F(0.25));
  }
  { // sign and decimal separator
    F value                       = 0.25;
    const char* s                 = "-.";
    std::from_chars_result result = std::from_chars(s, s + std::strlen(s), value, fmt);

    assert(result.ec == std::errc::invalid_argument);
    assert(result.ptr == s);
    assert(value == F(0.25));
  }
  { // + sign is not allowed
    F value                       = 0.25;
    const char* s                 = "+0.25";
    std::from_chars_result result = std::from_chars(s, s + std::strlen(s), value, fmt);

    assert(result.ec == std::errc::invalid_argument);
    assert(result.ptr == s);
    assert(value == F(0.25));
  }
}

template <class F>
struct test_basics {
  void operator()() {
    for (auto fmt : {std::chars_format::scientific,
                     std::chars_format::fixed,
                     /*std::chars_format::hex,*/ std::chars_format::general})
      test_fmt_independent<F>(fmt);
  }
};

template <class F>
struct test_fixed {
  void operator()() {
    std::from_chars_result r;
    F x = 0.25;

    // *** Failures

    { // Starts with invalid character
      std::array s = {' ', '1'};
      for (auto c : "abcdefghijklmnopqrstuvwxyz"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "`~!@#$%^&*()_=[]{}\\|;:'\",/<>? \t\v\r\n") {
        s[0] = c;
        r    = std::from_chars(s.data(), s.data() + s.size(), x, std::chars_format::fixed);

        assert(r.ec == std::errc::invalid_argument);
        assert(r.ptr == s.data());
        assert(x == F(0.25));
      }
    }

    // *** Success

    { // number followed by non-numeric values
      const char* s = "001x";

      // the expected form of the subject sequence is a nonempty sequence of
      // decimal digits optionally containing a decimal-point character, then
      // an optional exponent part as defined in 6.4.4.3, excluding any digit
      // separators (6.4.4.2); (C23 7.24.1.5)
      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 3);
      assert(x == F(1.0));
    }
    { // no leading digit
      const char* s = ".5";

      // the expected form of the subject sequence is a nonempty sequence of
      // decimal digits optionally containing a decimal-point character, then
      // an optional exponent part as defined in 6.4.4.3, excluding any digit
      // separators (6.4.4.2); (C23 7.24.1.5)
      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 2);
      assert(x == F(0.5));
    }
    { // negative sign and no leading digit
      const char* s = "-.5";

      // the expected form of the subject sequence is a nonempty sequence of
      // decimal digits optionally containing a decimal-point character, then
      // an optional exponent part as defined in 6.4.4.3, excluding any digit
      // separators (6.4.4.2); (C23 7.24.1.5)
      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 3);
      assert(x == F(-0.5));
    }

    { // double decimal point
      const char* s = "1.25.78";

      // This number is halfway between two float values.
      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 4);
      assert(x == F(1.25));
    }
    { // exponent no sign
      const char* s = "1.5e10";
      r             = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);

      assert(r.ec == std::errc{});
      assert(r.ptr == s + 3);
      assert(x == F(1.5));
    }
    { // exponent capitalized no sign
      const char* s = "1.5E10";
      r             = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);

      assert(r.ec == std::errc{});
      assert(r.ptr == s + 3);
      assert(x == F(1.5));
    }
    { // exponent + sign
      const char* s = "1.5e+10";
      r             = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);

      assert(r.ec == std::errc{});
      assert(r.ptr == s + 3);
      assert(x == F(1.5));
    }
    { // exponent - sign
      const char* s = "1.5e-10";
      r             = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);

      assert(r.ec == std::errc{});
      assert(r.ptr == s + 3);
      assert(x == F(1.5));
    }
    { // Exponent no number
      const char* s = "1.5e";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 3);
      assert(x == F(1.5));
    }
    { // Exponent sign no number
      {
        const char* s = "1.5e+";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 3);
        assert(x == F(1.5));
      }
      {
        const char* s = "1.5e-";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 3);
        assert(x == F(1.5));
      }
    }
    { // Exponent with whitespace
      {
        const char* s = "1.5e +1";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 3);
        assert(x == F(1.5));
      }
      {
        const char* s = "1.5e+ 1";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 3);
        assert(x == F(1.5));
      }
      {
        const char* s = "1.5e -1";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 3);
        assert(x == F(1.5));
      }
      {
        const char* s = "1.5e- 1";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 3);
        assert(x == F(1.5));
      }
    }
    { // double exponent
      const char* s = "1.25e0e12";
      r             = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);

      assert(r.ec == std::errc{});
      assert(r.ptr == s + 4);
      assert(x == F(1.25));
    }
    { // Exponent double sign
      {
        const char* s = "1.25e++12";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 4);
        assert(x == F(1.25));
      }
      {
        const char* s = "1.25e+-12";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 4);
        assert(x == F(1.25));
      }
      {
        const char* s = "1.25e-+12";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 4);
        assert(x == F(1.25));
      }
      {
        const char* s = "1.25e--12";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 4);
        assert(x == F(1.25));
      }
    }
    { // exponent hex prefix
      const char* s = "1.25e0x12";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 4);
      assert(x == F(1.25));
    }
    { // This number is halfway between two float values.
      const char* s = "20040229";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 8);
      assert(x == F(20040229));
    }
    { // Shifting mantissa exponent and no exponent
      const char* s = "123.456";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 7);
      assert(x == F(1.23456e2));
    }
    { // Shifting mantissa exponent and an exponent
      const char* s = "123.456e3";
      r             = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);

      assert(r.ec == std::errc{});
      assert(r.ptr == s + 7);
      assert(x == F(123.456));
    }
    { // Mantissa overflow
      {
        const char* s = "0.111111111111111111111111111111111111111111";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + std::strlen(s));
        assert(x == F(0.111111111111111111111111111111111111111111));
      }
      {
        const char* s = "111111111111.111111111111111111111111111111111111111111";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + std::strlen(s));
        assert(x == F(111111111111.111111111111111111111111111111111111111111));
      }
    }
    { // Negative value
      const char* s = "-0.25";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::fixed);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + std::strlen(s));
      assert(x == F(-0.25));
    }
  }
};

template <class F>
struct test_scientific {
  void operator()() {
    std::from_chars_result r;
    F x = 0.25;

    // *** Failures

    { // Starts with invalid character
      std::array s = {' ', '1', 'e', '0'};
      for (auto c : "abcdefghijklmnopqrstuvwxyz"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "`~!@#$%^&*()_=[]{}\\|;:'\",/<>? \t\v\r\n") {
        s[0] = c;
        r    = std::from_chars(s.data(), s.data() + s.size(), x, std::chars_format::scientific);

        assert(r.ec == std::errc::invalid_argument);
        assert(r.ptr == s.data());
        assert(x == F(0.25));
      }
    }
    { // No exponent
      const char* s = "1.23";
      r             = std::from_chars(s, s + strlen(s), x, std::chars_format::scientific);

      assert(r.ec == std::errc::invalid_argument);
      assert(r.ptr == s);
      assert(x == F(0.25));
    }
    { // Exponent no number
      const char* s = "1.23e";
      r             = std::from_chars(s, s + strlen(s), x, std::chars_format::scientific);

      assert(r.ec == std::errc::invalid_argument);
      assert(r.ptr == s);
      assert(x == F(0.25));
    }
    { // Exponent sign no number
      {
        const char* s = "1.5e+";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
        assert(r.ec == std::errc::invalid_argument);
        assert(r.ptr == s);
        assert(x == F(0.25));
      }
      {
        const char* s = "1.5e-";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
        assert(r.ec == std::errc::invalid_argument);
        assert(r.ptr == s);
        assert(x == F(0.25));
      }
    }
    { // Exponent with whitespace
      {
        const char* s = "1.5e +1";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
        assert(r.ec == std::errc::invalid_argument);
        assert(r.ptr == s);
        assert(x == F(0.25));
      }
      {
        const char* s = "1.5e+ 1";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
        assert(r.ec == std::errc::invalid_argument);
        assert(r.ptr == s);
        assert(x == F(0.25));
      }
      {
        const char* s = "1.5e -1";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
        assert(r.ec == std::errc::invalid_argument);
        assert(r.ptr == s);
        assert(x == F(0.25));
      }
      {
        const char* s = "1.5e- 1";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
        assert(r.ec == std::errc::invalid_argument);
        assert(r.ptr == s);
        assert(x == F(0.25));
      }
    }
    { // exponent double sign
      {
        const char* s = "1.25e++12";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
        assert(r.ec == std::errc::invalid_argument);
        assert(r.ptr == s);
        assert(x == F(0.25));
      }
      {
        const char* s = "1.25e+-12";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
        assert(r.ec == std::errc::invalid_argument);
        assert(r.ptr == s);
        assert(x == F(0.25));
      }
      {
        const char* s = "1.25e-+12";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
        assert(r.ec == std::errc::invalid_argument);
        assert(r.ptr == s);
        assert(x == F(0.25));
      }
      {
        const char* s = "1.25e--12";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
        assert(r.ec == std::errc::invalid_argument);
        assert(r.ptr == s);
        assert(x == F(0.25));
      }
    }

    // *** Success

    { // number followed by non-numeric values
      const char* s = "001e0x";

      // the expected form of the subject sequence is a nonempty sequence of
      // decimal digits optionally containing a decimal-point character, then
      // an optional exponent part as defined in 6.4.4.3, excluding any digit
      // separators (6.4.4.2); (C23 7.24.1.5)
      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 5);
      assert(x == F(1.0));
    }

    { // double decimal point
      const char* s = "1.25e0.78";

      // This number is halfway between two float values.
      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 6);
      assert(x == F(1.25));
    }

    { // exponent no sign
      const char* s = "1.5e10";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 6);
      assert(x == F(1.5e10));
    }
    { // exponent capitalized no sign
      const char* s = "1.5E10";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 6);
      assert(x == F(1.5e10));
    }
    { // exponent + sign
      const char* s = "1.5e+10";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 7);
      assert(x == F(1.5e10));
    }
    { // exponent - sign
      const char* s = "1.5e-10";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 7);
      assert(x == F(1.5e-10));
    }
    { // exponent hex prefix -> e0
      const char* s = "1.25e0x12";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 6);
      assert(x == F(1.25));
    }
    { // double exponent
      const char* s = "1.25e0e12";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 6);
      assert(x == F(1.25));
    }
    { // This number is halfway between two float values.
      const char* s = "20040229e0";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 10);
      assert(x == F(20040229));
    }
    { // Shifting mantissa exponent and an exponent
      const char* s = "123.456e3";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 9);
      assert(x == F(1.23456e5));
    }
    { // Mantissa overflow
      {
        const char* s = "0.111111111111111111111111111111111111111111e0";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + std::strlen(s));
        assert(x == F(0.111111111111111111111111111111111111111111));
      }
      {
        const char* s = "111111111111.111111111111111111111111111111111111111111e0";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + std::strlen(s));
        assert(x == F(111111111111.111111111111111111111111111111111111111111));
      }
    }
    { // Negative value
      const char* s = "-0.25e0";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + std::strlen(s));
      assert(x == F(-0.25));
    }
    { // value is too big -> +inf
      const char* s = "1e9999999999999999999999999999999999999999";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
      assert(r.ec == std::errc::result_out_of_range);
      assert(r.ptr == s + strlen(s));
      assert(x == std::numeric_limits<F>::infinity());
    }
    { // negative value is too big -> -inf
      const char* s = "-1e9999999999999999999999999999999999999999";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
      assert(r.ec == std::errc::result_out_of_range);
      assert(r.ptr == s + strlen(s));
      assert(x == -std::numeric_limits<F>::infinity());
    }
    { // value is too small -> 0
      const char* s = "1e-9999999999999999999999999999999999999999";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
      assert(r.ec == std::errc::result_out_of_range);
      assert(r.ptr == s + strlen(s));
      assert(x == F(0.0));
    }
    { // negative value is too small -> -0
      const char* s = "-1e-9999999999999999999999999999999999999999";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::scientific);
      assert(r.ec == std::errc::result_out_of_range);
      assert(r.ptr == s + strlen(s));
      assert(x == F(-0.0));
    }
  }
};

template <class F>
struct test_general {
  void operator()() {
    std::from_chars_result r;
    F x = 0.25;

    // *** Failures

    { // Starts with invalid character
      std::array s = {' ', '1'};
      for (auto c : "abcdefghijklmnopqrstuvwxyz"
                    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                    "`~!@#$%^&*()_=[]{}\\|;:'\",/<>? \t\v\r\n") {
        s[0] = c;
        r    = std::from_chars(s.data(), s.data() + s.size(), x);

        assert(r.ec == std::errc::invalid_argument);
        assert(r.ptr == s.data());
        assert(x == F(0.25));
      }
    }

    // *** Success

    { // number followed by non-numeric values
      const char* s = "001x";

      // the expected form of the subject sequence is a nonempty sequence of
      // decimal digits optionally containing a decimal-point character, then
      // an optional exponent part as defined in 6.4.4.3, excluding any digit
      // separators (6.4.4.2); (C23 7.24.1.5)
      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 3);
      assert(x == F(1.0));
    }
    { // no leading digit
      const char* s = ".5e0";

      // the expected form of the subject sequence is a nonempty sequence of
      // decimal digits optionally containing a decimal-point character, then
      // an optional exponent part as defined in 6.4.4.3, excluding any digit
      // separators (6.4.4.2); (C23 7.24.1.5)
      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 4);
      assert(x == F(0.5));
    }
    { // negative sign and no leading digit
      const char* s = "-.5e0";

      // the expected form of the subject sequence is a nonempty sequence of
      // decimal digits optionally containing a decimal-point character, then
      // an optional exponent part as defined in 6.4.4.3, excluding any digit
      // separators (6.4.4.2); (C23 7.24.1.5)
      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 5);
      assert(x == F(-0.5));
    }
    { // no leading digit
      const char* s = ".5";

      // the expected form of the subject sequence is a nonempty sequence of
      // decimal digits optionally containing a decimal-point character, then
      // an optional exponent part as defined in 6.4.4.3, excluding any digit
      // separators (6.4.4.2); (C23 7.24.1.5)
      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 2);
      assert(x == F(0.5));
    }
    { // negative sign and no leading digit
      const char* s = "-.5";

      // the expected form of the subject sequence is a nonempty sequence of
      // decimal digits optionally containing a decimal-point character, then
      // an optional exponent part as defined in 6.4.4.3, excluding any digit
      // separators (6.4.4.2); (C23 7.24.1.5)
      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 3);
      assert(x == F(-0.5));
    }
    { // double decimal point
      const char* s = "1.25.78";

      // This number is halfway between two float values.
      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 4);
      assert(x == F(1.25));
    }
    { // exponent no sign
      const char* s = "1.5e10";

      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 6);
      assert(x == F(1.5e10));
    }
    { // exponent capitalized no sign
      const char* s = "1.5E10";

      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 6);
      assert(x == F(1.5e10));
    }
    { // exponent + sign
      const char* s = "1.5e+10";

      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 7);
      assert(x == F(1.5e10));
    }
    { // exponent - sign
      const char* s = "1.5e-10";

      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 7);
      assert(x == F(1.5e-10));
    }
    { // Exponent no number
      const char* s = "1.5e";

      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 3);
      assert(x == F(1.5));
    }
    { // Exponent sign no number
      {
        const char* s = "1.5e+";

        r = std::from_chars(s, s + std::strlen(s), x);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 3);
        assert(x == F(1.5));
      }
      {
        const char* s = "1.5e-";

        r = std::from_chars(s, s + std::strlen(s), x);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 3);
        assert(x == F(1.5));
      }
    }
    { // Exponent with whitespace
      {
        const char* s = "1.5e +1";

        r = std::from_chars(s, s + std::strlen(s), x);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 3);
        assert(x == F(1.5));
      }
      {
        const char* s = "1.5e+ 1";

        r = std::from_chars(s, s + std::strlen(s), x);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 3);
        assert(x == F(1.5));
      }
      {
        const char* s = "1.5e -1";

        r = std::from_chars(s, s + std::strlen(s), x);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 3);
        assert(x == F(1.5));
      }
      {
        const char* s = "1.5e- 1";

        r = std::from_chars(s, s + std::strlen(s), x);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 3);
        assert(x == F(1.5));
      }
    }
    { // exponent double sign
      {
        const char* s = "1.25e++12";

        r = std::from_chars(s, s + std::strlen(s), x);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 4);
        assert(x == F(1.25));
      }
      {
        const char* s = "1.25e+-12";

        r = std::from_chars(s, s + std::strlen(s), x);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 4);
        assert(x == F(1.25));
      }
      {
        const char* s = "1.25e-+12";

        r = std::from_chars(s, s + std::strlen(s), x);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 4);
        assert(x == F(1.25));
      }
      {
        const char* s = "1.25e--12";

        r = std::from_chars(s, s + std::strlen(s), x);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 4);
        assert(x == F(1.25));
      }
    }
    { // exponent hex prefix -> e0
      const char* s = "1.25e0x12";

      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 6);
      assert(x == F(1.25));
    }
    { // double exponent
      const char* s = "1.25e0e12";

      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 6);
      assert(x == F(1.25));
    }
    { // This number is halfway between two float values.
      const char* s = "20040229";

      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 8);
      assert(x == F(20040229));
    }
    { // Shifting mantissa exponent and no exponent
      const char* s = "123.456";

      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 7);
      assert(x == F(1.23456e2));
    }
    { // Shifting mantissa exponent and an exponent
      const char* s = "123.456e3";

      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 9);
      assert(x == F(1.23456e5));
    }
    { // Mantissa overflow
      {
        const char* s = "0.111111111111111111111111111111111111111111";

        r = std::from_chars(s, s + std::strlen(s), x);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + std::strlen(s));
        assert(x == F(0.111111111111111111111111111111111111111111));
      }
      {
        const char* s = "111111111111.111111111111111111111111111111111111111111";

        r = std::from_chars(s, s + std::strlen(s), x);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + std::strlen(s));
        assert(x == F(111111111111.111111111111111111111111111111111111111111));
      }
    }
    { // Negative value
      const char* s = "-0.25";

      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + std::strlen(s));
      assert(x == F(-0.25));
    }
    { // value is too big -> +inf
      const char* s = "1e9999999999999999999999999999999999999999";

      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc::result_out_of_range);
      assert(r.ptr == s + strlen(s));
      assert(x == std::numeric_limits<F>::infinity());
    }
    { // negative value is too big -> -inf
      const char* s = "-1e9999999999999999999999999999999999999999";

      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc::result_out_of_range);
      assert(r.ptr == s + strlen(s));
      assert(x == -std::numeric_limits<F>::infinity());
    }
    { // value is too small -> 0
      const char* s = "1e-9999999999999999999999999999999999999999";

      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc::result_out_of_range);
      assert(r.ptr == s + strlen(s));
      assert(x == F(0.0));
    }
    { // negative value is too small -> -0
      const char* s = "-1e-9999999999999999999999999999999999999999";

      r = std::from_chars(s, s + std::strlen(s), x);
      assert(r.ec == std::errc::result_out_of_range);
      assert(r.ptr == s + strlen(s));
      assert(x == F(-0.0));
    }
  }
};

template <class F>
struct test_hex {
  void operator()() {
    std::from_chars_result r;
    F x = 0.25;

    // *** Failures

    { // Starts with invalid character
      std::array s = {' ', '1', 'e', '0'};
      for (auto c : "ghijklmnopqrstuvwxyz"
                    "GHIJKLMNOPQRSTUVWXYZ"
                    "`~!@#$%^&*()_=[]{}\\|;:'\",/<>? \t\v\r\n") {
        s[0] = c;
        r    = std::from_chars(s.data(), s.data() + s.size(), x, std::chars_format::hex);

        assert(r.ec == std::errc::invalid_argument);
        assert(r.ptr == s.data());
        assert(x == F(0.25));
      }
    }

    // *** Success

    { // number followed by non-numeric values
      const char* s = "001x";

      // the expected form of the subject sequence is a nonempty sequence of
      // decimal digits optionally containing a decimal-point character, then
      // an optional exponent part as defined in 6.4.4.3, excluding any digit
      // separators (6.4.4.2); (C23 7.24.1.5)
      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 3);
      assert(x == F(1.0));
    }
    { // no leading digit
      const char* s = ".5p0";

      // the expected form of the subject sequence is a nonempty sequence of
      // decimal digits optionally containing a decimal-point character, then
      // an optional exponent part as defined in 6.4.4.3, excluding any digit
      // separators (6.4.4.2); (C23 7.24.1.5)
      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 4);
      assert(x == F(0x0.5p0));
    }
    { // negative sign and no leading digit
      const char* s = "-.5p0";

      // the expected form of the subject sequence is a nonempty sequence of
      // decimal digits optionally containing a decimal-point character, then
      // an optional exponent part as defined in 6.4.4.3, excluding any digit
      // separators (6.4.4.2); (C23 7.24.1.5)
      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 5);
      assert(x == F(-0x0.5p0));
    }
    { // no leading digit
      const char* s = ".5";

      // the expected form of the subject sequence is a nonempty sequence of
      // decimal digits optionally containing a decimal-point character, then
      // an optional exponent part as defined in 6.4.4.3, excluding any digit
      // separators (6.4.4.2); (C23 7.24.1.5)
      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 2);
      assert(x == F(0x0.5p0));
    }
    { // negative sign and no leading digit
      const char* s = "-.5";

      // the expected form of the subject sequence is a nonempty sequence of
      // decimal digits optionally containing a decimal-point character, then
      // an optional exponent part as defined in 6.4.4.3, excluding any digit
      // separators (6.4.4.2); (C23 7.24.1.5)
      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 3);
      assert(x == F(-0x0.5p0));
    }
    { // double decimal point
      const char* s = "1.25.78";

      // This number is halfway between two float values.
      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 4);
      assert(x == F(0x1.25p0));
    }
    { // exponent no sign
      const char* s = "1.5p10";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 6);
      assert(x == F(0x1.5p10));
    }
    { // exponent capitalized no sign
      const char* s = "1.5P10";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 6);
      assert(x == F(0x1.5p10));
    }
    { // exponent + sign
      const char* s = "1.5p+10";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 7);
      assert(x == F(0x1.5p10));
    }
    { // exponent - sign
      const char* s = "1.5p-10";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 7);
      assert(x == F(0x1.5p-10));
    }
    { // Exponent no number
      const char* s = "1.5p";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 3);
      assert(x == F(0x1.5p0));
    }
    { // Exponent sign no number
      {
        const char* s = "1.5p+";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 3);
        assert(x == F(0x1.5p0));
      }
      {
        const char* s = "1.5p-";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 3);
        assert(x == F(0x1.5p0));
      }
    }
    { // Exponent with whitespace
      {
        const char* s = "1.5p +1";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 3);
        assert(x == F(0x1.5p0));
      }
      {
        const char* s = "1.5p+ 1";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 3);
        assert(x == F(0x1.5p0));
      }
      {
        const char* s = "1.5p -1";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 3);
        assert(x == F(0x1.5p0));
      }
      {
        const char* s = "1.5p- 1";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 3);
        assert(x == F(0x1.5p0));
      }
    }
    { // Exponent double sign
      {
        const char* s = "1.25p++12";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 4);
        assert(x == F(0x1.25p0));
      }
      {
        const char* s = "1.25p+-12";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 4);
        assert(x == F(0x1.25p0));
      }
      {
        const char* s = "1.25p-+12";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 4);
        assert(x == F(0x1.25p0));
      }
      {
        const char* s = "1.25p--12";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + 4);
        assert(x == F(0x1.25p0));
      }
    }
    { // exponent hex prefix -> p0
      const char* s = "1.25p0x12";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 6);
      assert(x == F(0x1.25p0));
    }
    { // double exponent
      const char* s = "1.25p0p12";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 6);
      assert(x == F(0x1.25p0));
    }
    { // This number is halfway between two float values.
      const char* s = "131CA25";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 7);
      assert(x == F(0x131CA25p0));
    }
    { // Shifting mantissa exponent and no exponent
      const char* s = "123.456";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 7);
      assert(x == F(0x123.456p0));
    }
    { // Shifting mantissa exponent and an exponent
      const char* s = "123.456p3";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + 9);
      assert(x == F(0x123.456p3));
    }
    { // Mantissa overflow
      {
        const char* s = "0.111111111111111111111111111111111111111111";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + std::strlen(s));
        assert(x == F(0x0.111111111111111111111111111111111111111111p0));
      }
      {
        const char* s = "111111111111.111111111111111111111111111111111111111111";

        r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
        assert(r.ec == std::errc{});
        assert(r.ptr == s + std::strlen(s));
        assert(x == F(0x111111111111.111111111111111111111111111111111111111111p0));
      }
    }
    { // Negative value
      const char* s = "-0.25";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc{});
      assert(r.ptr == s + std::strlen(s));
      assert(x == F(-0x0.25p0));
    }
    { // value is too big -> +inf
      const char* s = "1p9999999999999999999999999999999999999999";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc::result_out_of_range);
      assert(r.ptr == s + strlen(s));
      assert(x == std::numeric_limits<F>::infinity());
    }
    { // negative value is too big -> -inf
      const char* s = "-1p9999999999999999999999999999999999999999";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc::result_out_of_range);
      assert(r.ptr == s + strlen(s));
      assert(x == -std::numeric_limits<F>::infinity());
    }
    { // value is too small -> 0
      const char* s = "1p-9999999999999999999999999999999999999999";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc::result_out_of_range);
      assert(r.ptr == s + strlen(s));
      assert(x == F(0.0));
    }
    { // negative value is too small -> -0
      const char* s = "-1p-9999999999999999999999999999999999999999";

      r = std::from_chars(s, s + std::strlen(s), x, std::chars_format::hex);
      assert(r.ec == std::errc::result_out_of_range);
      assert(r.ptr == s + strlen(s));
      assert(x == F(-0.0));
    }
  }
};

// The test
//   test/std/utilities/charconv/charconv.msvc/test.cpp
// uses random values. This tests contains errors found by this test.
void test_random_errors() {
  {
    const char* s    = "4.219902180869891e-2788";
    const char* last = s + std::strlen(s) - 1;

    // last + 1 contains a digit. When that value is parsed the exponent is
    // e-2788 which returns std::errc::result_out_of_range and the value 0.
    // the proper exponent is e-278, which can be represented by a double.

    double value                  = 0.25;
    std::from_chars_result result = std::from_chars(s, last, value);

    assert(result.ec == std::errc{});
    assert(result.ptr == last);
    assert(value == 4.219902180869891e-278);
  }
  {
    const char* s    = "7.411412e-39U";
    const char* last = s + std::strlen(s) - 1;

    float value                   = 0.25;
    std::from_chars_result result = std::from_chars(s, last, value);

    assert(result.ec == std::errc{});
    assert(result.ptr == last);
    assert(value == 7.411412e-39F);
  }
}

int main(int, char**) {
  run<test_basics>(all_floats);
  run<test_scientific>(all_floats);
  run<test_fixed>(all_floats);
  run<test_general>(all_floats);

  run<test_hex>(all_floats);

  test_random_errors();

  return 0;
}
