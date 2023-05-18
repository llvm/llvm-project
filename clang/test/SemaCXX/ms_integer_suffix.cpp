// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fms-extensions -verify %s
// expected-no-diagnostics

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
