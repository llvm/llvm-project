// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fms-extensions -verify=signed,expected %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fms-extensions -fno-signed-char -verify=unsigned,expected %s

#ifdef __SIZEOF_INT8__
static_assert(sizeof(0i8) == __SIZEOF_INT8__, "");

constexpr int f(char) { return 1; }
constexpr int f(signed char) { return 2; }

static_assert(f(0i8) == 1, "");
#endif
#ifdef __SIZEOF_INT16__
static_assert(sizeof(0i16) == __SIZEOF_INT16__, "");
#endif
#ifdef __SIZEOF_INT32__
static_assert(sizeof(0i32) == __SIZEOF_INT32__, "");
#endif
#ifdef __SIZEOF_INT64__
static_assert(sizeof(0i64) == __SIZEOF_INT64__, "");
#endif

namespace gh212504 {
  static_assert(1234i8 == -46, ""); // unsigned-error {{static assertion failed due to requirement '210i8 == -46':}}
  static_assert(1234i8 == 210, ""); // signed-error {{static assertion failed due to requirement '-46i8 == 210':}}
  static_assert(1234ui8 == 210, "");
  static_assert(18446744073709551615i8, "");
  static_assert(18446744073709551616i8 == 0, ""); // expected-error {{integer literal is too large to be represented in any integer type}}
}
