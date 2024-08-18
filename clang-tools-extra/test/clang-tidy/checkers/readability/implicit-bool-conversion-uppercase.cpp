// RUN: %check_clang_tidy %s readability-implicit-bool-conversion,readability-uppercase-literal-suffix %t

bool implicitConversionToBoolInReturnValue() {
  float floating = 1.0F;
  return floating;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: implicit conversion 'float' -> 'bool'
  // CHECK-FIXES: return floating != 0.0F;
}

void functionTakingUnsignedLong(unsigned long);
void functionTakingFloat(float);

void implicitConversionFromBoolLiterals() {
  functionTakingUnsignedLong(false);
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: implicit conversion 'bool' -> 'unsigned long'
  // CHECK-FIXES: functionTakingUnsignedLong(0U);

  functionTakingFloat(false);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: implicit conversion 'bool' -> 'float'
  // CHECK-FIXES: functionTakingFloat(0.0F);
}
