// RUN: %check_clang_tidy %s readability-uppercase-literal-suffix %t -- -- -target x86_64-pc-linux-gnu -I %S
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-tidy %t.cpp -checks='-*,readability-uppercase-literal-suffix' -fix -- -target x86_64-pc-linux-gnu -I %S
// RUN: clang-tidy %t.cpp -checks='-*,readability-uppercase-literal-suffix' -warnings-as-errors='-*,readability-uppercase-literal-suffix' -- -target x86_64-pc-linux-gnu -I %S

#include "readability-uppercase-literal-suffix.h"

void floating_point_suffix() {
  static constexpr auto v0 = 1.; // no literal
  static_assert(is_same<decltype(v0), const double>::value, "");
  static_assert(v0 == 1., "");

  static constexpr auto v1 = 1.e0; // no literal
  static_assert(is_same<decltype(v1), const double>::value, "");
  static_assert(v1 == 1., "");

  // Float

  static constexpr auto v2 = 1.f;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'f', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v2 = 1.f;
  // CHECK-MESSAGES-NEXT: ^ ~
  // CHECK-MESSAGES-NEXT: {{^ *}}F{{$}}
  // CHECK-FIXES: static constexpr auto v2 = 1.F;
  static_assert(is_same<decltype(v2), const float>::value, "");
  static_assert(v2 == 1.0F, "");

  static constexpr auto v3 = 1.e0f;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'f', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v3 = 1.e0f;
  // CHECK-MESSAGES-NEXT: ^ ~
  // CHECK-MESSAGES-NEXT: {{^ *}}F{{$}}
  // CHECK-FIXES: static constexpr auto v3 = 1.e0F;
  static_assert(is_same<decltype(v3), const float>::value, "");
  static_assert(v3 == 1.0F, "");

  static constexpr auto v4 = 1.F; // OK.
  static_assert(is_same<decltype(v4), const float>::value, "");
  static_assert(v4 == 1.0F, "");

  static constexpr auto v5 = 1.e0F; // OK.
  static_assert(is_same<decltype(v5), const float>::value, "");
  static_assert(v5 == 1.0F, "");

  // Long double

  static constexpr auto v6 = 1.l;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'l', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v6 = 1.l;
  // CHECK-MESSAGES-NEXT: ^ ~
  // CHECK-MESSAGES-NEXT: {{^ *}}L{{$}}
  // CHECK-FIXES: static constexpr auto v6 = 1.L;
  static_assert(is_same<decltype(v6), const long double>::value, "");
  static_assert(v6 == 1., "");

  static constexpr auto v7 = 1.e0l;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'l', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v7 = 1.e0l;
  // CHECK-MESSAGES-NEXT: ^ ~
  // CHECK-MESSAGES-NEXT: {{^ *}}L{{$}}
  // CHECK-FIXES: static constexpr auto v7 = 1.e0L;
  static_assert(is_same<decltype(v7), const long double>::value, "");
  static_assert(v7 == 1., "");

  static constexpr auto v8 = 1.L; // OK.
  static_assert(is_same<decltype(v8), const long double>::value, "");
  static_assert(v8 == 1., "");

  static constexpr auto v9 = 1.e0L; // OK.
  static_assert(is_same<decltype(v9), const long double>::value, "");
  static_assert(v9 == 1., "");

  // __float128

  static constexpr auto v10 = 1.q;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: floating point literal has suffix 'q', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v10 = 1.q;
  // CHECK-MESSAGES-NEXT: ^ ~
  // CHECK-MESSAGES-NEXT: {{^ *}}Q{{$}}
  // CHECK-FIXES: static constexpr auto v10 = 1.Q;
  static_assert(is_same<decltype(v10), const __float128>::value, "");
  static_assert(v10 == 1., "");

  static constexpr auto v11 = 1.e0q;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: floating point literal has suffix 'q', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v11 = 1.e0q;
  // CHECK-MESSAGES-NEXT: ^ ~
  // CHECK-MESSAGES-NEXT: {{^ *}}Q{{$}}
  // CHECK-FIXES: static constexpr auto v11 = 1.e0Q;
  static_assert(is_same<decltype(v11), const __float128>::value, "");
  static_assert(v11 == 1., "");

  static constexpr auto v12 = 1.Q; // OK.
  static_assert(is_same<decltype(v12), const __float128>::value, "");
  static_assert(v12 == 1., "");

  static constexpr auto v13 = 1.e0Q; // OK.
  static_assert(is_same<decltype(v13), const __float128>::value, "");
  static_assert(v13 == 1., "");
}

void floating_point_complex_suffix() {
  // _Complex, I

  static constexpr auto v14 = 1.i;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: floating point literal has suffix 'i', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v14 = 1.i;
  // CHECK-MESSAGES-NEXT: ^ ~
  // CHECK-MESSAGES-NEXT: {{^ *}}I{{$}}
  // CHECK-FIXES: static constexpr auto v14 = 1.I;
  static_assert(is_same<decltype(v14), const _Complex double>::value, "");
  static_assert(v14 == 1.I, "");

  static constexpr auto v15 = 1.e0i;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: floating point literal has suffix 'i', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v15 = 1.e0i;
  // CHECK-MESSAGES-NEXT: ^ ~
  // CHECK-MESSAGES-NEXT: {{^ *}}I{{$}}
  // CHECK-FIXES: static constexpr auto v15 = 1.e0I;
  static_assert(is_same<decltype(v15), const _Complex double>::value, "");
  static_assert(v15 == 1.I, "");

  static constexpr auto v16 = 1.I; // OK.
  static_assert(is_same<decltype(v16), const _Complex double>::value, "");
  static_assert(v16 == 1.I, "");

  static constexpr auto v17 = 1.e0I; // OK.
  static_assert(is_same<decltype(v17), const _Complex double>::value, "");
  static_assert(v17 == 1.I, "");

  // _Complex, J

  static constexpr auto v18 = 1.j;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: floating point literal has suffix 'j', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v18 = 1.j;
  // CHECK-MESSAGES-NEXT: ^ ~
  // CHECK-MESSAGES-NEXT: {{^ *}}J{{$}}
  // CHECK-FIXES: static constexpr auto v18 = 1.J;
  static_assert(is_same<decltype(v18), const _Complex double>::value, "");
  static_assert(v18 == 1.J, "");

  static constexpr auto v19 = 1.e0j;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: floating point literal has suffix 'j', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v19 = 1.e0j;
  // CHECK-MESSAGES-NEXT: ^ ~
  // CHECK-MESSAGES-NEXT: {{^ *}}J{{$}}
  // CHECK-FIXES: static constexpr auto v19 = 1.e0J;
  static_assert(is_same<decltype(v19), const _Complex double>::value, "");
  static_assert(v19 == 1.J, "");

  static constexpr auto v20 = 1.J; // OK.
  static_assert(is_same<decltype(v20), const _Complex double>::value, "");
  static_assert(v20 == 1.J, "");

  static constexpr auto v21 = 1.e0J; // OK.
  static_assert(is_same<decltype(v21), const _Complex double>::value, "");
  static_assert(v21 == 1.J, "");
}

void macros() {
#define PASSTHROUGH(X) X
  static constexpr auto m0 = PASSTHROUGH(1.f);
  // CHECK-MESSAGES: :[[@LINE-1]]:42: warning: floating point literal has suffix 'f', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto m0 = PASSTHROUGH(1.f);
  // CHECK-MESSAGES-NEXT: ^ ~
  // CHECK-MESSAGES-NEXT: {{^ *}}F{{$}}
  // CHECK-FIXES: static constexpr auto m0 = PASSTHROUGH(1.F);
  static_assert(is_same<decltype(m0), const float>::value, "");
  static_assert(m0 == 1.0F, "");
}
