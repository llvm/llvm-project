// RUN: %check_clang_tidy %s readability-implicit-bool-conversion %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-implicit-bool-conversion.CheckConversionsToBool: true, \
// RUN:         readability-implicit-bool-conversion.CheckConversionsFromBool: true \
// RUN:     }}' -- -std=c23

void TestImplicitBoolConversion() {
  int intValue = 10;
  if (intValue) // CHECK-MESSAGES: :[[@LINE]]:7: warning: implicit conversion 'int' -> 'bool' [readability-implicit-bool-conversion]
                // CHECK-FIXES: if (intValue != 0)
    (void)0;

  bool boolValue = true;
  int newIntValue = boolValue; // CHECK-MESSAGES: :[[@LINE]]:21: warning: implicit conversion 'bool' -> 'int' [readability-implicit-bool-conversion]
                               // CHECK-FIXES: int newIntValue = static_cast<int>(boolValue);
}
