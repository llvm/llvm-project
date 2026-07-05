// TODO: When Clang adds support for decimal floating point types, enable these tests by:
//    1. Removing all the #if 0 + #endif guards.
//    2. Removing all occurrences of the string "DISABLED-" in this file.
//    3. Deleting this message.

// RUN: %check_clang_tidy -std=c23-or-later %s readability-uppercase-literal-suffix %t

void bit_precise_literal_suffix() {
  // _BitInt()

  static constexpr auto v1 = 1wb;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: integer literal has suffix 'wb', which is not uppercase
  // CHECK-FIXES: static constexpr auto v1 = 1WB;
  static_assert(v1 == 1WB);

  static constexpr auto v2 = 1WB; // OK.
  static_assert(v2 == 1WB);

  // _BitInt() Unsigned

  static constexpr auto v3 = 1wbu;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: integer literal has suffix 'wbu', which is not uppercase
  // CHECK-FIXES: static constexpr auto v3 = 1WBU;
  static_assert(v3 == 1WBU);

  static constexpr auto v4 = 1WBu;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: integer literal has suffix 'WBu', which is not uppercase
  // CHECK-FIXES: static constexpr auto v4 = 1WBU;
  static_assert(v4 == 1WBU);

  static constexpr auto v5 = 1wbU;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: integer literal has suffix 'wbU', which is not uppercase
  // CHECK-FIXES: static constexpr auto v5 = 1WBU;
  static_assert(v5 == 1WBU);

  static constexpr auto v6 = 1WBU; // OK.
  static_assert(v6 == 1WBU);

  // Unsigned _BitInt()

  static constexpr auto v7 = 1uwb;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: integer literal has suffix 'uwb', which is not uppercase
  // CHECK-FIXES: static constexpr auto v7 = 1UWB;
  static_assert(v7 == 1UWB);

  static constexpr auto v8 = 1uWB;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: integer literal has suffix 'uWB', which is not uppercase
  // CHECK-FIXES: static constexpr auto v8 = 1UWB;
  static_assert(v8 == 1UWB);

  static constexpr auto v9 = 1Uwb;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: integer literal has suffix 'Uwb', which is not uppercase
  // CHECK-FIXES: static constexpr auto v9 = 1UWB;
  static_assert(v9 == 1UWB);

  static constexpr auto v10 = 1UWB; // OK.
  static_assert(v10 == 1UWB);
}

void decimal_floating_point_suffix() {
  // _Decimal32

#if 0
  static constexpr auto v1 = 1.df;
  // DISABLED-CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'df', which is not uppercase
  // DISABLED-CHECK-FIXES: static constexpr auto v1 = 1.DF;
  static_assert(v1 == 1.DF);

  static constexpr auto v2 = 1.e0df;
  // DISABLED-CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'df', which is not uppercase
  // DISABLED-CHECK-FIXES: static constexpr auto v2 = 1.e0DF;
  static_assert(v2 == 1.DF);

  static constexpr auto v3 = 1.DF; // OK.
  static_assert(v3 == 1.DF);

  static constexpr auto v4 = 1.e0DF; // OK.
  static_assert(v4 == 1.DF);
#endif

  // _Decimal64

#if 0
  static constexpr auto v5 = 1.dd;
  // DISABLED-CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'dd', which is not uppercase
  // DISABLED-CHECK-FIXES: static constexpr auto v5 = 1.DD;
  static_assert(v5 == 1.DD);

  static constexpr auto v6 = 1.e0dd;
  // DISABLED-CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'dd', which is not uppercase
  // DISABLED-CHECK-FIXES: static constexpr auto v6 = 1.e0DD;
  static_assert(v6 == 1.DD);

  static constexpr auto v7 = 1.DD; // OK.
  static_assert(v7 == 1.DD);

  static constexpr auto v8 = 1.e0DD; // OK.
  static_assert(v8 == 1.DD);
#endif

  // _Decimal128

#if 0
  static constexpr auto v9 = 1.dl;
  // DISABLED-CHECK-MESSAGES: :[[@LINE-1]]:30: warning: floating point literal has suffix 'dl', which is not uppercase
  // DISABLED-CHECK-FIXES: static constexpr auto v9 = 1.DL;
  static_assert(v9 == 1.DL);

  static constexpr auto v10 = 1.e0dl;
  // DISABLED-CHECK-MESSAGES: :[[@LINE-1]]:31: warning: floating point literal has suffix 'dl', which is not uppercase
  // DISABLED-CHECK-FIXES: static constexpr auto v10 = 1.e0DL;
  static_assert(v10 == 1.DL);

  static constexpr auto v11 = 1.DL; // OK.
  static_assert(v11 == 1.DL);

  static constexpr auto v12 = 1.e0DL; // OK.
  static_assert(v12 == 1.DL);
#endif
}
