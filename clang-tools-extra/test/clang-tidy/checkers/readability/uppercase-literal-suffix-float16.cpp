// RUN: %check_clang_tidy %s readability-uppercase-literal-suffix %t -- -- -target aarch64-linux-gnu -I %S

#include "uppercase-literal-suffix.h"

void float16_normal_literals() {
  // _Float16

  static constexpr auto v14 = 1.f16;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: floating point literal has suffix 'f16', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v14 = 1.f16;
  // CHECK-MESSAGES-NEXT: ^ ~
  // CHECK-MESSAGES-NEXT: {{^ *}}F16{{$}}
  // CHECK-FIXES: static constexpr auto v14 = 1.F16;
  static_assert(is_same<decltype(v14), const _Float16>::value, "");
  static_assert(v14 == 1.F16, "");

  static constexpr auto v15 = 1.e0f16;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: floating point literal has suffix 'f16', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v15 = 1.e0f16;
  // CHECK-MESSAGES-NEXT: ^ ~
  // CHECK-MESSAGES-NEXT: {{^ *}}F16{{$}}
  // CHECK-FIXES: static constexpr auto v15 = 1.e0F16;
  static_assert(is_same<decltype(v15), const _Float16>::value, "");
  static_assert(v15 == 1.F16, "");

  static constexpr auto v16 = 1.F16; // OK.
  static_assert(is_same<decltype(v16), const _Float16>::value, "");
  static_assert(v16 == 1.F16, "");

  static constexpr auto v17 = 1.e0F16; // OK.
  static_assert(is_same<decltype(v17), const _Float16>::value, "");
  static_assert(v17 == 1.F16, "");
}

void float16_hexadecimal_literals() {
// _Float16

  static constexpr auto v13 = 0xfp0f16;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: floating point literal has suffix 'f16', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v13 = 0xfp0f16;
  // CHECK-MESSAGES-NEXT: ^    ~
  // CHECK-MESSAGES-NEXT: {{^ *}}F16{{$}}
  // CHECK-FIXES: static constexpr auto v13 = 0xfp0F16;
  static_assert(is_same<decltype(v13), const _Float16>::value, "");
  static_assert(v13 == 0xfp0F16, "");

  static constexpr auto v14 = 0xfp0F16; // OK.
  static_assert(is_same<decltype(v14), const _Float16>::value, "");
  static_assert(v14 == 0xfp0F16, "");

}
