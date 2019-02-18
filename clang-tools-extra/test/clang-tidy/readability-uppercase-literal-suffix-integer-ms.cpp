// RUN: %check_clang_tidy %s readability-uppercase-literal-suffix %t -- -- -target x86_64-pc-linux-gnu -I %S -fms-extensions
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-tidy %t.cpp -checks='-*,readability-uppercase-literal-suffix' -fix -- -target x86_64-pc-linux-gnu -I %S -fms-extensions
// RUN: clang-tidy %t.cpp -checks='-*,readability-uppercase-literal-suffix' -warnings-as-errors='-*,readability-uppercase-literal-suffix' -- -target x86_64-pc-linux-gnu -I %S -fms-extensions

#include "readability-uppercase-literal-suffix.h"

void integer_suffix() {
  static constexpr auto v0 = __LINE__; // synthetic
  static_assert(v0 == 9 || v0 == 5, "");

  static constexpr auto v1 = __cplusplus; // synthetic, long

  static constexpr auto v2 = 1; // no literal
  static_assert(is_same<decltype(v2), const int>::value, "");
  static_assert(v2 == 1, "");

  // i32

  static constexpr auto v3 = 1i32;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: integer literal has suffix 'i32', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v3 = 1i32;
  // CHECK-MESSAGES-NEXT: ^~
  // CHECK-MESSAGES-NEXT: {{^ *}}I32{{$}}
  // CHECK-FIXES: static constexpr auto v3 = 1I32;
  static_assert(is_same<decltype(v3), const int>::value, "");
  static_assert(v3 == 1I32, "");

  static constexpr auto v4 = 1I32; // OK.
  static_assert(is_same<decltype(v4), const int>::value, "");
  static_assert(v4 == 1I32, "");

  // i64

  static constexpr auto v5 = 1i64;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: integer literal has suffix 'i64', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v5 = 1i64;
  // CHECK-MESSAGES-NEXT: ^~
  // CHECK-MESSAGES-NEXT: {{^ *}}I64{{$}}
  // CHECK-FIXES: static constexpr auto v5 = 1I64;
  static_assert(is_same<decltype(v5), const long int>::value, "");
  static_assert(v5 == 1I64, "");

  static constexpr auto v6 = 1I64; // OK.
  static_assert(is_same<decltype(v6), const long int>::value, "");
  static_assert(v6 == 1I64, "");

  // i16

  static constexpr auto v7 = 1i16;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: integer literal has suffix 'i16', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v7 = 1i16;
  // CHECK-MESSAGES-NEXT: ^~
  // CHECK-MESSAGES-NEXT: {{^ *}}I16{{$}}
  // CHECK-FIXES: static constexpr auto v7 = 1I16;
  static_assert(is_same<decltype(v7), const short>::value, "");
  static_assert(v7 == 1I16, "");

  static constexpr auto v8 = 1I16; // OK.
  static_assert(is_same<decltype(v8), const short>::value, "");
  static_assert(v8 == 1I16, "");

  // i8

  static constexpr auto v9 = 1i8;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: integer literal has suffix 'i8', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v9 = 1i8;
  // CHECK-MESSAGES-NEXT: ^~
  // CHECK-MESSAGES-NEXT: {{^ *}}I8{{$}}
  // CHECK-FIXES: static constexpr auto v9 = 1I8;
  static_assert(is_same<decltype(v9), const char>::value, "");
  static_assert(v9 == 1I8, "");

  static constexpr auto v10 = 1I8; // OK.
  static_assert(is_same<decltype(v10), const char>::value, "");
  static_assert(v10 == 1I8, "");
}
