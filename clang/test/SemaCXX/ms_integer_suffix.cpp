// RUN: %clang_cc1 -std=c++11 -fsyntax-only -fms-extensions -verify %s

#ifdef __SIZEOF_INT8__
static_assert(sizeof(0i8) == __SIZEOF_INT8__, "");

constexpr int f(char) { return 1; }
constexpr int f(signed char) { return 2; }

static_assert(f(0i8) == 1, "");

constexpr auto v1 = 127i8;
constexpr auto v2 = 128i8; // expected-error {{integer literal is too large to be represented in any integer type}}
constexpr auto v3 = 255i8; // expected-error {{integer literal is too large to be represented in any integer type}}
constexpr auto v4 = 256i8; // expected-error {{integer literal is too large to be represented in any integer type}}
constexpr auto v5 = -128i8; // expected-error {{integer literal is too large to be represented in any integer type}}
constexpr auto v6 = -255i8; // expected-error {{integer literal is too large to be represented in any integer type}}
static_assert(-255i8 == -255, ""); // expected-error {{integer literal is too large to be represented in any integer type}}

constexpr auto v7 = 0xFFi8;
constexpr auto v8 = 255ui8;

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
