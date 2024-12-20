// RUN: %check_clang_tidy %s readability-implicit-bool-conversion %t -- -- -std=c23
// RUN: %check_clang_tidy -check-suffix=TO-BOOL %s readability-implicit-bool-conversion %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-implicit-bool-conversion.CheckConversionsToBool: true, \
// RUN:         readability-implicit-bool-conversion.CheckConversionsFromBool: true \
// RUN:     }}' -- -std=c23
// RUN: %check_clang_tidy -check-suffix=FROM-BOOL %s readability-implicit-bool-conversion %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-implicit-bool-conversion.CheckConversionsToBool: true, \
// RUN:         readability-implicit-bool-conversion.CheckConversionsFromBool: true \
// RUN:     }}' -- -std=c23
// RUN: %check_clang_tidy -check-suffix=TO-BOOL,FROM-BOOL %s readability-implicit-bool-conversion %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-implicit-bool-conversion.CheckConversionsToBool: true, \
// RUN:         readability-implicit-bool-conversion.CheckConversionsFromBool: true \
// RUN:     }}' -- -std=c23


// ==========================================================
// Test Case: Conversions to bool (CheckConversionsToBool=true)
// ==========================================================
void TestConversionsToBool() {
  int x = 42;
  if (x) // CHECK-MESSAGES-TO-BOOL: :[[@LINE]]:8: warning: implicit conversion 'int' -> 'bool'
    (void)0;

  float f = 3.14;
  if (f) // CHECK-MESSAGES-TO-BOOL: :[[@LINE]]:8: warning: implicit conversion 'float' -> 'bool'
    (void)0;

  int *p = nullptr;
  if (p) // CHECK-MESSAGES-TO-BOOL: :[[@LINE]]:8: warning: implicit conversion 'int *' -> 'bool'
    (void)0;

  // Pointer-to-member
  struct S {
    int member;
  };
  int S::*ptr = nullptr;
  if (ptr) // CHECK-MESSAGES-TO-BOOL: :[[@LINE]]:8: warning: implicit conversion 'int S::*' -> 'bool'
    (void)0;
}

// ==========================================================
// Test Case: Conversions from bool (CheckConversionsFromBool=true)
// ==========================================================
void TestConversionsFromBool() {
  bool b = true;

  int x = b; // CHECK-MESSAGES-FROM-BOOL: :[[@LINE]]:12: warning: implicit conversion 'bool' -> 'int'
  float f = b; // CHECK-MESSAGES-FROM-BOOL: :[[@LINE]]:12: warning: implicit conversion 'bool' -> 'float'
}

// ==========================================================
// Test Case: Mixed Configurations (ToBool=true, FromBool=true)
// ==========================================================
void TestMixedConfig() {
  int x = 42;
  if (x) // CHECK-MESSAGES-TO-BOOL: :[[@LINE]]:8: warning: implicit conversion 'int' -> 'bool'
    (void)0;

  bool b = true;
  int y = b; // CHECK-MESSAGES-FROM-BOOL: :[[@LINE]]:12: warning: implicit conversion 'bool' -> 'int'
}

// ==========================================================
// Test Case: No Diagnostics (ToBool=false, FromBool=false)
// ==========================================================
void TestNoDiagnostics() {
  int x = 42;
  if (x) // No warning: CheckConversionsToBool=false
    (void)0;

  bool b = true;
  int y = b; // No warning: CheckConversionsFromBool=false
}

// ==========================================================
// Test Case: Edge Cases and Complex Expressions
// ==========================================================
void TestEdgeCases() {
  bool b = true;

  // Nested implicit casts
  int x = (b ? 1 : 0); // CHECK-MESSAGES-FROM-BOOL: :[[@LINE]]:13: warning: implicit conversion 'bool' -> 'int'

  // Function returns implicit bool
  auto ReturnBool = []() -> bool { return true; };
  int y = ReturnBool(); // CHECK-MESSAGES-FROM-BOOL: :[[@LINE]]:12: warning: implicit conversion 'bool' -> 'int'

  // Explicit casts (no diagnostics)
  int z = static_cast<int>(b); // No warning: explicit cast
}
