// RUN: %check_clang_tidy %s readability-implicit-bool-conversion %t

// RUN: %check_clang_tidy %s readability-implicit-bool-conversion %t \
// RUN: -- -config='{CheckOptions: [{key: readability-implicit-bool-conversion.CheckConversionsToBool, value: false}, {key: readability-implicit-bool-conversion.CheckConversionsFromBool, value: true}]}'

// RUN: %check_clang_tidy %s readability-implicit-bool-conversion %t \
// RUN: -- -config='{CheckOptions: [{key: readability-implicit-bool-conversion.CheckConversionsToBool, value: true}, {key: readability-implicit-bool-conversion.CheckConversionsFromBool, value: false}]}'

// RUN: %check_clang_tidy %s readability-implicit-bool-conversion %t \
// RUN: -- -config='{CheckOptions: [{key: readability-implicit-bool-conversion.CheckConversionsToBool, value: false}, {key: readability-implicit-bool-conversion.CheckConversionsFromBool, value: false}]}'

// ==========================================================
// Test Case: Conversions to bool (CheckConversionsToBool=true)
// ==========================================================
void TestConversionsToBool() {
  int x = 42;
  if (x) // CHECK-MESSAGES: :[[@LINE]]:8: warning: implicit conversion 'int' -> 'bool'
    (void)0;

  float f = 3.14;
  if (f) // CHECK-MESSAGES: :[[@LINE]]:8: warning: implicit conversion 'float' -> 'bool'
    (void)0;

  int *p = nullptr;
  if (p) // CHECK-MESSAGES: :[[@LINE]]:8: warning: implicit conversion 'int *' -> 'bool'
    (void)0;

  // Pointer-to-member
  struct S {
    int member;
  };
  int S::*ptr = nullptr;
  if (ptr) // CHECK-MESSAGES: :[[@LINE]]:8: warning: implicit conversion 'int S::*' -> 'bool'
    (void)0;
}

// ==========================================================
// Test Case: Conversions from bool (CheckConversionsFromBool=true)
// ==========================================================
void TestConversionsFromBool() {
  bool b = true;

  int x = b; // CHECK-MESSAGES: :[[@LINE]]:12: warning: implicit conversion 'bool' -> 'int'
  float f = b; // CHECK-MESSAGES: :[[@LINE]]:12: warning: implicit conversion 'bool' -> 'float'
}

// ==========================================================
// Test Case: Mixed Configurations (ToBool=false, FromBool=true)
// ==========================================================
void TestMixedConfig() {
  int x = 42;
  if (x) // No warning: CheckConversionsToBool=false
    (void)0;

  bool b = true;
  int y = b; // CHECK-MESSAGES: :[[@LINE]]:12: warning: implicit conversion 'bool' -> 'int'
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
  int x = (b ? 1 : 0); // CHECK-MESSAGES: :[[@LINE]]:12: warning: implicit conversion 'bool' -> 'int'

  // Function returns implicit bool
  auto ReturnBool = []() -> bool { return true; };
  int y = ReturnBool(); // CHECK-MESSAGES: :[[@LINE]]:12: warning: implicit conversion 'bool' -> 'int'

  // Explicit casts (no diagnostics)
  int z = static_cast<int>(b); // No warning: explicit cast
}
