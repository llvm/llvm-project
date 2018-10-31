// RUN: %check_clang_tidy %s readability-uppercase-literal-suffix %t -- -config="{CheckOptions: [{key: readability-uppercase-literal-suffix.NewSuffixes, value: 'L;uL'}]}" -- -I %S
// RUN: grep -Ev "// *[A-Z-]+:" %s > %t.cpp
// RUN: clang-tidy %t.cpp -checks='-*,readability-uppercase-literal-suffix' -fix -config="{CheckOptions: [{key: readability-uppercase-literal-suffix.NewSuffixes, value: 'L;uL'}]}" -- -I %S
// RUN: clang-tidy %t.cpp -checks='-*,readability-uppercase-literal-suffix' -warnings-as-errors='-*,readability-uppercase-literal-suffix' -config="{CheckOptions: [{key: readability-uppercase-literal-suffix.NewSuffixes, value: 'L;uL'}]}" -- -I %S

#include "readability-uppercase-literal-suffix.h"

void integer_suffix() {
  // Unsigned

  static constexpr auto v3 = 1u; // OK.
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
  // CHECK-MESSAGES-NEXT: {{^ *}}L{{$}}
  // CHECK-FIXES: static constexpr auto v5 = 1L;
  static_assert(is_same<decltype(v5), const long>::value, "");
  static_assert(v5 == 1, "");

  static constexpr auto v6 = 1L; // OK.
  static_assert(is_same<decltype(v6), const long>::value, "");
  static_assert(v6 == 1, "");

  // Long Long

  static constexpr auto v7 = 1ll; // OK.
  static_assert(is_same<decltype(v7), const long long>::value, "");
  static_assert(v7 == 1, "");

  static constexpr auto v8 = 1LL; // OK.
  static_assert(is_same<decltype(v8), const long long>::value, "");
  static_assert(v8 == 1, "");

  // Unsigned Long

  static constexpr auto v9 = 1ul;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: integer literal has suffix 'ul', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v9 = 1ul;
  // CHECK-MESSAGES-NEXT: ^~~
  // CHECK-MESSAGES-NEXT: {{^ *}}uL{{$}}
  // CHECK-FIXES: static constexpr auto v9 = 1uL;
  static_assert(is_same<decltype(v9), const unsigned long>::value, "");
  static_assert(v9 == 1, "");

  static constexpr auto v10 = 1uL; // OK.
  static_assert(is_same<decltype(v10), const unsigned long>::value, "");
  static_assert(v10 == 1, "");

  static constexpr auto v11 = 1Ul;
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: integer literal has suffix 'Ul', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v11 = 1Ul;
  // CHECK-MESSAGES-NEXT: ^~~
  // CHECK-MESSAGES-NEXT: {{^ *}}uL{{$}}
  // CHECK-FIXES: static constexpr auto v11 = 1uL;
  static_assert(is_same<decltype(v11), const unsigned long>::value, "");
  static_assert(v11 == 1, "");

  static constexpr auto v12 = 1UL; // OK.
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: integer literal has suffix 'UL', which is not uppercase
  // CHECK-MESSAGES-NEXT: static constexpr auto v12 = 1UL;
  // CHECK-MESSAGES-NEXT: ^~~
  // CHECK-MESSAGES-NEXT: {{^ *}}uL{{$}}
  // CHECK-FIXES: static constexpr auto v12 = 1uL;
  static_assert(is_same<decltype(v12), const unsigned long>::value, "");
  static_assert(v12 == 1, "");

  // Long Unsigned

  static constexpr auto v13 = 1lu; // OK.
  static_assert(is_same<decltype(v13), const unsigned long>::value, "");
  static_assert(v13 == 1, "");

  static constexpr auto v14 = 1Lu; // OK.
  static_assert(is_same<decltype(v14), const unsigned long>::value, "");
  static_assert(v14 == 1, "");

  static constexpr auto v15 = 1lU; // OK.
  static_assert(is_same<decltype(v15), const unsigned long>::value, "");
  static_assert(v15 == 1, "");

  static constexpr auto v16 = 1LU; // OK.
  static_assert(is_same<decltype(v16), const unsigned long>::value, "");
  static_assert(v16 == 1, "");

  // Unsigned Long Long

  static constexpr auto v17 = 1ull; // OK.
  static_assert(is_same<decltype(v17), const unsigned long long>::value, "");
  static_assert(v17 == 1, "");

  static constexpr auto v18 = 1uLL; // OK.
  static_assert(is_same<decltype(v18), const unsigned long long>::value, "");
  static_assert(v18 == 1, "");

  static constexpr auto v19 = 1Ull; // OK.
  static_assert(is_same<decltype(v19), const unsigned long long>::value, "");
  static_assert(v19 == 1, "");

  static constexpr auto v20 = 1ULL; // OK.
  static_assert(is_same<decltype(v20), const unsigned long long>::value, "");
  static_assert(v20 == 1, "");

  // Long Long Unsigned

  static constexpr auto v21 = 1llu; // OK.
  static_assert(is_same<decltype(v21), const unsigned long long>::value, "");
  static_assert(v21 == 1, "");

  static constexpr auto v22 = 1LLu; // OK.
  static_assert(is_same<decltype(v22), const unsigned long long>::value, "");
  static_assert(v22 == 1, "");

  static constexpr auto v23 = 1llU; // OK.
  static_assert(is_same<decltype(v23), const unsigned long long>::value, "");
  static_assert(v23 == 1, "");

  static constexpr auto v24 = 1LLU; // OK.
  static_assert(is_same<decltype(v24), const unsigned long long>::value, "");
  static_assert(v24 == 1, "");
}
