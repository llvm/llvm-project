//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// XFAIL: availability-char8_t_support-missing

// This test runs in C++20, but we have deprecated codecvt<char(16|32), char, mbstate_t> in C++20.
// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS -D_LIBCPP_ENABLE_CXX26_REMOVED_CODECVT

// <locale>

// template <> class codecvt<char32_t, char, mbstate_t>
// template <> class codecvt<char16_t, char, mbstate_t>
// template <> class codecvt<char32_t, char8_t, mbstate_t> // C++20
// template <> class codecvt<char16_t, char8_t, mbstate_t> // C++20
// template <> class codecvt<char32_t, char16_t, mbstate_t>  // extension

// sanity check

#include <locale>
#include <codecvt>
#include <cassert>

#include <stdio.h>

#include "test_macros.h"

int main(int, char**) {
  typedef std::codecvt<char32_t, char, std::mbstate_t> F32_8;
  typedef std::codecvt<char16_t, char, std::mbstate_t> F16_8;
  typedef std::codecvt_utf16<char32_t> F32_16;
  std::locale l = std::locale(std::locale::classic(), new F32_16);
  const F32_8& f32_8 = std::use_facet<F32_8>(std::locale::classic());
  const F32_16& f32_16 = std::use_facet<F32_16>(l);
  const F16_8& f16_8 = std::use_facet<F16_8>(std::locale::classic());
  std::mbstate_t mbs32_8 = {};
  std::mbstate_t mbs16_8 = {};
  std::mbstate_t mbs32_16 = {};
  F32_8::intern_type* c32p;
  F16_8::intern_type* c16p;
  F32_8::extern_type* c8p;
  const F32_8::intern_type* c_c32p;
  const F16_8::intern_type* c_c16p;
  const F32_8::extern_type* c_c8p;
  F32_8::intern_type c32;
  F16_8::intern_type c16[2];
  char c16c[4];
  char* c16cp;
  F32_8::extern_type c8[4];

#if TEST_STD_VER > 17
  typedef std::codecvt<char32_t, char8_t, std::mbstate_t> F32_8T;
  typedef std::codecvt<char16_t, char8_t, std::mbstate_t> F16_8T;
  const F32_8T& f32_8t = std::use_facet<F32_8T>(std::locale::classic());
  const F16_8T& f16_8t = std::use_facet<F16_8T>(std::locale::classic());
  std::mbstate_t mbs32_8t = {};
  std::mbstate_t mbs16_8t = {};
  F32_8T::extern_type* c8tp;
  const F32_8T::extern_type* c_c8tp;
  F32_8T::extern_type c8t[4];
#endif

  for (F32_8::intern_type c32x = 0; c32x < 0x110003; ++c32x) {
    if ((0xD800 <= c32x && c32x < 0xE000) || c32x >= 0x110000) {
#ifndef _MSVC_STL_VERSION // Don't ask; eldritch horrors.
      assert(f32_16.out(mbs32_16, &c32x, &c32x + 1, c_c32p, c16c + 0, c16c + 4,
                        c16cp) == F32_8::error);
      assert(f32_8.out(mbs32_8, &c32x, &c32x + 1, c_c32p, c8, c8 + 4, c8p) ==
             F32_8::error);
#endif

#if TEST_STD_VER > 17
      assert(f32_8t.out(mbs32_8t, &c32x, &c32x + 1, c_c32p, c8t, c8t + 4,
                        c8tp) == F32_8T::error);
#endif
    } else {
      assert(f32_16.out(mbs32_16, &c32x, &c32x + 1, c_c32p, c16c, c16c + 4,
                        c16cp) == F32_8::ok);
      assert(c_c32p - &c32x == 1);
      if (c32x < 0x10000)
        assert(c16cp - c16c == 2);
      else
        assert(c16cp - c16c == 4);
      for (int i = 0; i < (c16cp - c16c) / 2; ++i)
        c16[i] = (char16_t)((unsigned char)c16c[2 * i] << 8 |
                            (unsigned char)c16c[2 * i + 1]);
      c_c16p = c16 + (c16cp - c16c) / 2;
      assert(f16_8.out(mbs16_8, c16, c_c16p, c_c16p, c8, c8 + 4, c8p) ==
             F32_8::ok);
      if (c32x < 0x10000)
        assert(c_c16p - c16 == 1);
      else
        assert(c_c16p - c16 == 2);
      if (c32x < 0x80)
        assert(c8p - c8 == 1);
      else if (c32x < 0x800)
        assert(c8p - c8 == 2);
      else if (c32x < 0x10000)
        assert(c8p - c8 == 3);
      else
        assert(c8p - c8 == 4);
      c_c8p = c8p;
      assert(f32_8.in(mbs32_8, c8, c_c8p, c_c8p, &c32, &c32 + 1, c32p) ==
             F32_8::ok);
      if (c32x < 0x80)
        assert(c_c8p - c8 == 1);
      else if (c32x < 0x800)
        assert(c_c8p - c8 == 2);
      else if (c32x < 0x10000)
        assert(c_c8p - c8 == 3);
      else
        assert(c_c8p - c8 == 4);
      assert(c32p - &c32 == 1);
      assert(c32 == c32x);
      assert(f32_8.out(mbs32_8, &c32x, &c32x + 1, c_c32p, c8, c8 + 4, c8p) ==
             F32_8::ok);
      assert(c_c32p - &c32x == 1);
      if (c32x < 0x80)
        assert(c8p - c8 == 1);
      else if (c32x < 0x800)
        assert(c8p - c8 == 2);
      else if (c32x < 0x10000)
        assert(c8p - c8 == 3);
      else
        assert(c8p - c8 == 4);
      c_c8p = c8p;
      assert(f16_8.in(mbs16_8, c8, c_c8p, c_c8p, c16, c16 + 2, c16p) ==
             F32_8::ok);
      if (c32x < 0x80)
        assert(c_c8p - c8 == 1);
      else if (c32x < 0x800)
        assert(c_c8p - c8 == 2);
      else if (c32x < 0x10000)
        assert(c_c8p - c8 == 3);
      else
        assert(c_c8p - c8 == 4);
      if (c32x < 0x10000)
        assert(c16p - c16 == 1);
      else
        assert(c16p - c16 == 2);
      for (int i = 0; i < c16p - c16; ++i) {
        c16c[2 * i] = static_cast<char>(c16[i] >> 8);
        c16c[2 * i + 1] = static_cast<char>(c16[i]);
      }
      const char* c_c16cp = c16c + (c16p - c16) * 2;
      assert(f32_16.in(mbs32_16, c16c, c_c16cp, c_c16cp, &c32, &c32 + 1,
                       c32p) == F32_8::ok);
      if (c32x < 0x10000)
        assert(c_c16cp - c16c == 2);
      else
        assert(c_c16cp - c16c == 4);
      assert(c32p - &c32 == 1);
      assert(c32 == c32x);

#if TEST_STD_VER > 17
      assert(f32_8t.out(mbs32_8t, &c32x, &c32x + 1, c_c32p, c8t, c8t + 4,
                        c8tp) == F32_8T::ok);
      assert(c_c32p - &c32x == 1);
      if (c32x < 0x80)
        assert(c8tp - c8t == 1);
      else if (c32x < 0x800)
        assert(c8tp - c8t == 2);
      else if (c32x < 0x10000)
        assert(c8tp - c8t == 3);
      else
        assert(c8tp - c8t == 4);
      c_c8tp = c8tp;
      assert(f16_8t.in(mbs16_8t, c8t, c_c8tp, c_c8tp, c16, c16 + 2, c16p) ==
             F16_8T::ok);
      if (c32x < 0x80)
        assert(c_c8tp - c8t == 1);
      else if (c32x < 0x800)
        assert(c_c8tp - c8t == 2);
      else if (c32x < 0x10000)
        assert(c_c8tp - c8t == 3);
      else
        assert(c_c8tp - c8t == 4);
      if (c32x < 0x10000)
        assert(c16p - c16 == 1);
      else
        assert(c16p - c16 == 2);
      c_c16p = c16p;
      assert(f16_8t.out(mbs16_8t, c16, c_c16p, c_c16p, c8t, c8t + 4, c8tp) ==
             F16_8T::ok);
      if (c32x < 0x10000)
        assert(c_c16p - c16 == 1);
      else
        assert(c_c16p - c16 == 2);
      if (c32x < 0x80)
        assert(c8tp - c8t == 1);
      else if (c32x < 0x800)
        assert(c8tp - c8t == 2);
      else if (c32x < 0x10000)
        assert(c8tp - c8t == 3);
      else
        assert(c8tp - c8t == 4);
      c_c8tp = c8tp;
      assert(f32_8t.in(mbs32_8t, c8t, c_c8tp, c_c8tp, &c32, &c32 + 1, c32p) ==
             F32_8T::ok);
      if (c32x < 0x80)
        assert(c_c8tp - c8t == 1);
      else if (c32x < 0x800)
        assert(c_c8tp - c8t == 2);
      else if (c32x < 0x10000)
        assert(c_c8tp - c8t == 3);
      else
        assert(c_c8tp - c8t == 4);
      assert(c32p - &c32 == 1);
      assert(c32 == c32x);
#endif
    }
  }

  return 0;
}
