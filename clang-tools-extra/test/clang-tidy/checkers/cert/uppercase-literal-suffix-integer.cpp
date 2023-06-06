// RUN: %check_clang_tidy %s cert-dcl16-c %t -- -- -I %clang_tidy_headers
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-tidy %t.cpp -checks='-*,cert-dcl16-c' -fix -- -I %clang_tidy_headers
// RUN: clang-tidy %t.cpp -checks='-*,cert-dcl16-c' -warnings-as-errors='-*,cert-dcl16-c' -- -I %clang_tidy_headers

#include "integral_constant.h"

void integer_suffix() {
  static constexpr auto v0 = __LINE__; // synthetic
  static_assert(v0 == 9 || v0 == 5, "");

  static constexpr auto v1 = __cplusplus; // synthetic, long

  static constexpr auto v2 = 1; // no literal
  static_assert(is_same<decltype(v2), const int>::value, "");
  static_assert(v2 == 1, "");

  // Unsigned

  static constexpr auto v3 = 1u;
  static_assert(is_same<decltype(v3), const unsigned int>::value, "");
  static_assert(v3 == 1, "");

  static constexpr auto v4 = 1U; // OK.
  static_assert(is_same<decltype(v4), const unsigned int>::value, "");
  static_assert(v4 == 1, "");

  // Long

  static constexpr auto v5 = 1l;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: integer literal has suffix 'l', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v5 = 1l;
  // CHECK-MESSAGES-NEXT: ^~
  // CHECK-MESSAGES-NEXT: {{^ *| *}}L{{$}}
  // CHECK-FIXES: static constexpr auto v5 = 1L;
  static_assert(is_same<decltype(v5), const long>::value, "");
  static_assert(v5 == 1, "");

  static constexpr auto v6 = 1L; // OK.
  static_assert(is_same<decltype(v6), const long>::value, "");
  static_assert(v6 == 1, "");

  // Long Long

  static constexpr auto v7 = 1ll;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: integer literal has suffix 'll', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v7 = 1ll;
  // CHECK-MESSAGES-NEXT: ^~~
  // CHECK-MESSAGES-NEXT: {{^ *| *}}LL{{$}}
  // CHECK-FIXES: static constexpr auto v7 = 1LL;
  static_assert(is_same<decltype(v7), const long long>::value, "");
  static_assert(v7 == 1, "");

  static constexpr auto v8 = 1LL; // OK.
  static_assert(is_same<decltype(v8), const long long>::value, "");
  static_assert(v8 == 1, "");

  // Unsigned Long

  static constexpr auto v9 = 1ul;
  static_assert(is_same<decltype(v9), const unsigned long>::value, "");
  static_assert(v9 == 1, "");

  static constexpr auto v10 = 1uL;
  static_assert(is_same<decltype(v10), const unsigned long>::value, "");
  static_assert(v10 == 1, "");

  static constexpr auto v11 = 1Ul;
  static_assert(is_same<decltype(v11), const unsigned long>::value, "");
  static_assert(v11 == 1, "");

  static constexpr auto v12 = 1UL; // OK.
  static_assert(is_same<decltype(v12), const unsigned long>::value, "");
  static_assert(v12 == 1, "");

  // Long Unsigned

  static constexpr auto v13 = 1lu;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: integer literal has suffix 'lu', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v13 = 1lu;
  // CHECK-MESSAGES-NEXT: ^~~
  // CHECK-MESSAGES-NEXT: {{^ *| *}}LU{{$}}
  // CHECK-FIXES: static constexpr auto v13 = 1LU;
  static_assert(is_same<decltype(v13), const unsigned long>::value, "");
  static_assert(v13 == 1, "");

  static constexpr auto v14 = 1Lu;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: integer literal has suffix 'Lu', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v14 = 1Lu;
  // CHECK-MESSAGES-NEXT: ^~~
  // CHECK-MESSAGES-NEXT: {{^ *| *}}LU{{$}}
  // CHECK-FIXES: static constexpr auto v14 = 1LU;
  static_assert(is_same<decltype(v14), const unsigned long>::value, "");
  static_assert(v14 == 1, "");

  static constexpr auto v15 = 1lU;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: integer literal has suffix 'lU', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v15 = 1lU;
  // CHECK-MESSAGES-NEXT: ^~~
  // CHECK-MESSAGES-NEXT: {{^ *| *}}LU{{$}}
  // CHECK-FIXES: static constexpr auto v15 = 1LU;
  static_assert(is_same<decltype(v15), const unsigned long>::value, "");
  static_assert(v15 == 1, "");

  static constexpr auto v16 = 1LU; // OK.
  static_assert(is_same<decltype(v16), const unsigned long>::value, "");
  static_assert(v16 == 1, "");

  // Unsigned Long Long

  static constexpr auto v17 = 1ull;
  static_assert(is_same<decltype(v17), const unsigned long long>::value, "");
  static_assert(v17 == 1, "");

  static constexpr auto v18 = 1uLL;
  static_assert(is_same<decltype(v18), const unsigned long long>::value, "");
  static_assert(v18 == 1, "");

  static constexpr auto v19 = 1Ull;
  static_assert(is_same<decltype(v19), const unsigned long long>::value, "");
  static_assert(v19 == 1, "");

  static constexpr auto v20 = 1ULL; // OK.
  static_assert(is_same<decltype(v20), const unsigned long long>::value, "");
  static_assert(v20 == 1, "");

  // Long Long Unsigned

  static constexpr auto v21 = 1llu;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: integer literal has suffix 'llu', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v21 = 1llu;
  // CHECK-MESSAGES-NEXT: ^~~~
  // CHECK-MESSAGES-NEXT: {{^ *| *}}LLU{{$}}
  // CHECK-FIXES: static constexpr auto v21 = 1LLU;
  static_assert(is_same<decltype(v21), const unsigned long long>::value, "");
  static_assert(v21 == 1, "");

  static constexpr auto v22 = 1LLu;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: integer literal has suffix 'LLu', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v22 = 1LLu;
  // CHECK-MESSAGES-NEXT: ^~~~
  // CHECK-MESSAGES-NEXT: {{^ *| *}}LLU{{$}}
  // CHECK-FIXES: static constexpr auto v22 = 1LLU;
  static_assert(is_same<decltype(v22), const unsigned long long>::value, "");
  static_assert(v22 == 1, "");

  static constexpr auto v23 = 1llU;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: integer literal has suffix 'llU', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v23 = 1llU;
  // CHECK-MESSAGES-NEXT: ^~~~
  // CHECK-MESSAGES-NEXT: {{^ *| *}}LLU{{$}}
  // CHECK-FIXES: static constexpr auto v23 = 1LLU;
  static_assert(is_same<decltype(v23), const unsigned long long>::value, "");
  static_assert(v23 == 1, "");

  static constexpr auto v24 = 1LLU; // OK.
  static_assert(is_same<decltype(v24), const unsigned long long>::value, "");
  static_assert(v24 == 1, "");
}
