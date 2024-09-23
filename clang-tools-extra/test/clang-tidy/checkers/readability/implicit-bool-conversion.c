// RUN: %check_clang_tidy %s readability-implicit-bool-conversion %t -- -- -std=c23
// RUN: %check_clang_tidy -check-suffix=UPPER-CASE %s readability-implicit-bool-conversion %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-implicit-bool-conversion.UseUpperCaseLiteralSuffix: true \
// RUN:     }}' -- -std=c23

#undef NULL
#define NULL 0L

void functionTakingBool(bool);
void functionTakingInt(int);
void functionTakingUnsignedLong(unsigned long);
void functionTakingChar(char);
void functionTakingFloat(float);
void functionTakingDouble(double);
void functionTakingSignedChar(signed char);


////////// Implicit conversion from bool.

void implicitConversionFromBoolSimpleCases() {
  bool boolean = true;

  functionTakingBool(boolean);

  functionTakingInt(boolean);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: implicit conversion 'bool' -> 'int' [readability-implicit-bool-conversion]
  // CHECK-FIXES: functionTakingInt((int)boolean);

  functionTakingUnsignedLong(boolean);
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: implicit conversion 'bool' -> 'unsigned long'
  // CHECK-FIXES: functionTakingUnsignedLong((unsigned long)boolean);

  functionTakingChar(boolean);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'bool' -> 'char'
  // CHECK-FIXES: functionTakingChar((char)boolean);

  functionTakingFloat(boolean);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: implicit conversion 'bool' -> 'float'
  // CHECK-FIXES: functionTakingFloat((float)boolean);

  functionTakingDouble(boolean);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'bool' -> 'double'
  // CHECK-FIXES: functionTakingDouble((double)boolean);
}

float implicitConversionFromBoolInReturnValue() {
  bool boolean = false;
  return boolean;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: implicit conversion 'bool' -> 'float'
  // CHECK-FIXES: return (float)boolean;
}

void implicitConversionFromBoolInSingleBoolExpressions(bool b1, bool b2) {
  bool boolean = true;
  boolean = b1 ^ b2;
  boolean |= !b1 || !b2;
  boolean &= b1;

  int integer = boolean - 3;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: implicit conversion 'bool' -> 'int'
  // CHECK-FIXES: int integer = (int)boolean - 3;

  float floating = boolean / 0.3f;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: implicit conversion 'bool' -> 'float'
  // CHECK-FIXES: float floating = (float)boolean / 0.3f;

  char character = boolean;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: implicit conversion 'bool' -> 'char'
  // CHECK-FIXES: char character = (char)boolean;
}

void implicitConversionFromBoolInComplexBoolExpressions() {
  bool boolean = true;
  bool anotherBoolean = false;

  int integer = boolean && anotherBoolean;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: implicit conversion 'bool' -> 'int'
  // CHECK-MESSAGES: :[[@LINE-2]]:28: warning: implicit conversion 'bool' -> 'int'
  // CHECK-FIXES: int integer = (int)boolean && (int)anotherBoolean;

  float floating = (boolean || anotherBoolean) * 0.3f;
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: implicit conversion 'bool' -> 'int'
  // CHECK-MESSAGES: :[[@LINE-2]]:32: warning: implicit conversion 'bool' -> 'int'
  // CHECK-FIXES: float floating = ((int)boolean || (int)anotherBoolean) * 0.3f;

  double doubleFloating = (boolean && (anotherBoolean || boolean)) * 0.3;
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: implicit conversion 'bool' -> 'int'
  // CHECK-MESSAGES: :[[@LINE-2]]:40: warning: implicit conversion 'bool' -> 'int'
  // CHECK-MESSAGES: :[[@LINE-3]]:58: warning: implicit conversion 'bool' -> 'int'
  // CHECK-FIXES: double doubleFloating = ((int)boolean && ((int)anotherBoolean || (int)boolean)) * 0.3;
}

void implicitConversionFromBoolLiterals() {
  functionTakingInt(true);
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: implicit conversion 'bool' -> 'int'
  // CHECK-FIXES: functionTakingInt(1);

  functionTakingUnsignedLong(false);
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: implicit conversion 'bool' -> 'unsigned long'
  // CHECK-FIXES: functionTakingUnsignedLong(0u);
  // CHECK-FIXES-UPPER-CASE: functionTakingUnsignedLong(0U);

  functionTakingSignedChar(true);
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: implicit conversion 'bool' -> 'signed char'
  // CHECK-FIXES: functionTakingSignedChar(1);

  functionTakingFloat(false);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: implicit conversion 'bool' -> 'float'
  // CHECK-FIXES: functionTakingFloat(0.0f);
  // CHECK-FIXES-UPPER-CASE: functionTakingFloat(0.0F);

  functionTakingDouble(true);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'bool' -> 'double'
  // CHECK-FIXES: functionTakingDouble(1.0);
}

void implicitConversionFromBoolInComparisons() {
  bool boolean = true;
  int integer = 0;

  functionTakingBool(boolean == integer);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'bool' -> 'int'
  // CHECK-FIXES: functionTakingBool((int)boolean == integer);

  functionTakingBool(integer != boolean);
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: implicit conversion 'bool' -> 'int'
  // CHECK-FIXES: functionTakingBool(integer != (int)boolean);
}

void ignoreBoolComparisons() {
  bool boolean = true;
  bool anotherBoolean = false;

  functionTakingBool(boolean == anotherBoolean);
  functionTakingBool(boolean != anotherBoolean);
}

void ignoreExplicitCastsFromBool() {
  bool boolean = true;

  int integer = (int)boolean + 3;
  float floating = (float)boolean * 0.3f;
  char character = (char)boolean;
}

void ignoreImplicitConversionFromBoolInMacroExpansions() {
  bool boolean = true;

  #define CAST_FROM_BOOL_IN_MACRO_BODY boolean + 3
  int integerFromMacroBody = CAST_FROM_BOOL_IN_MACRO_BODY;

  #define CAST_FROM_BOOL_IN_MACRO_ARGUMENT(x) x + 3
  int integerFromMacroArgument = CAST_FROM_BOOL_IN_MACRO_ARGUMENT(boolean);
}

////////// Implicit conversions to bool.

void implicitConversionToBoolSimpleCases() {
  int integer = 10;
  functionTakingBool(integer);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: functionTakingBool(integer != 0);

  unsigned long unsignedLong = 10;
  functionTakingBool(unsignedLong);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'unsigned long' -> 'bool'
  // CHECK-FIXES: functionTakingBool(unsignedLong != 0u);
  // CHECK-FIXES-UPPER-CASE: functionTakingBool(unsignedLong != 0U);

  float floating = 0.0f;
  functionTakingBool(floating);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'float' -> 'bool'
  // CHECK-FIXES: functionTakingBool(floating != 0.0f);
  // CHECK-FIXES-UPPER-CASE: functionTakingBool(floating != 0.0F);

  double doubleFloating = 1.0f;
  functionTakingBool(doubleFloating);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'double' -> 'bool'
  // CHECK-FIXES: functionTakingBool(doubleFloating != 0.0);

  signed char character = 'a';
  functionTakingBool(character);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'signed char' -> 'bool'
  // CHECK-FIXES: functionTakingBool(character != 0);

  int* pointer = nullptr;
  functionTakingBool(pointer);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'int *' -> 'bool'
  // CHECK-FIXES: functionTakingBool(pointer != nullptr);
}

void implicitConversionToBoolInSingleExpressions() {
  int integer = 10;
  bool boolComingFromInt;
  boolComingFromInt = integer;
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: boolComingFromInt = (integer != 0);

  float floating = 10.0f;
  bool boolComingFromFloat;
  boolComingFromFloat = floating;
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: implicit conversion 'float' -> 'bool'
  // CHECK-FIXES: boolComingFromFloat = (floating != 0.0f);
  // CHECK-FIXES-UPPER-CASE: boolComingFromFloat = (floating != 0.0F);

  signed char character = 'a';
  bool boolComingFromChar;
  boolComingFromChar = character;
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'signed char' -> 'bool'
  // CHECK-FIXES: boolComingFromChar = (character != 0);

  int* pointer = nullptr;
  bool boolComingFromPointer;
  boolComingFromPointer = pointer;
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: implicit conversion 'int *' -> 'bool'
  // CHECK-FIXES: boolComingFromPointer = (pointer != nullptr);
}

void implicitConversionToBoolInComplexExpressions() {
  bool boolean = true;

  int integer = 10;
  int anotherInteger = 20;
  bool boolComingFromInteger;
  boolComingFromInteger = integer + anotherInteger;
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: boolComingFromInteger = ((integer + anotherInteger) != 0);
}

void implicitConversionInNegationExpressions() {
  int integer = 10;
  bool boolComingFromNegatedInt;
  boolComingFromNegatedInt = !integer;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: boolComingFromNegatedInt = ((!integer) != 0);
}

bool implicitConversionToBoolInReturnValue() {
  float floating = 1.0f;
  return floating;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: implicit conversion 'float' -> 'bool'
  // CHECK-FIXES: return floating != 0.0f;
}

void implicitConversionToBoolFromLiterals() {
  functionTakingBool(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: functionTakingBool(false);

  functionTakingBool(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: functionTakingBool(true);

  functionTakingBool(2ul);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'unsigned long' -> 'bool'
  // CHECK-FIXES: functionTakingBool(true);

  functionTakingBool(0.0f);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'float' -> 'bool'
  // CHECK-FIXES: functionTakingBool(false);

  functionTakingBool(1.0f);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'float' -> 'bool'
  // CHECK-FIXES: functionTakingBool(true);

  functionTakingBool(2.0);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'double' -> 'bool'
  // CHECK-FIXES: functionTakingBool(true);

  functionTakingBool('\0');
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: functionTakingBool(false);

  functionTakingBool('a');
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: functionTakingBool(true);

  functionTakingBool("");
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'char *' -> 'bool'
  // CHECK-FIXES: functionTakingBool(true);

  functionTakingBool("abc");
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'char *' -> 'bool'
  // CHECK-FIXES: functionTakingBool(true);

  functionTakingBool(NULL);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'long' -> 'bool'
  // CHECK-FIXES: functionTakingBool(false);
}

void implicitConversionToBoolFromUnaryMinusAndZeroLiterals() {
  functionTakingBool(-0);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: functionTakingBool((-0) != 0);

  functionTakingBool(-0.0f);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'float' -> 'bool'
  // CHECK-FIXES: functionTakingBool((-0.0f) != 0.0f);
  // CHECK-FIXES-UPPER-CASE: functionTakingBool((-0.0f) != 0.0F);

  functionTakingBool(-0.0);
  // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: implicit conversion 'double' -> 'bool'
  // CHECK-FIXES: functionTakingBool((-0.0) != 0.0);
}

void ignoreExplicitCastsToBool() {
  int integer = 10;
  bool boolComingFromInt = (bool)integer;

  float floating = 10.0f;
  bool boolComingFromFloat = (bool)floating;

  char character = 'a';
  bool boolComingFromChar = (bool)character;

  int* pointer = nullptr;
  bool booleanComingFromPointer = (bool)pointer;
}

void ignoreImplicitConversionToBoolInMacroExpansions() {
  int integer = 3;

  #define CAST_TO_BOOL_IN_MACRO_BODY integer && false
  bool boolFromMacroBody = CAST_TO_BOOL_IN_MACRO_BODY;

  #define CAST_TO_BOOL_IN_MACRO_ARGUMENT(x) x || true
  bool boolFromMacroArgument = CAST_TO_BOOL_IN_MACRO_ARGUMENT(integer);
}

int implicitConversionReturnInt()
{
    return true;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: implicit conversion 'bool' -> 'int'
    // CHECK-FIXES: return 1
}

int implicitConversionReturnIntWithParens()
{
    return (true);
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: implicit conversion 'bool' -> 'int'
    // CHECK-FIXES: return 1
}

bool implicitConversionReturnBool()
{
    return 1;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: implicit conversion 'int' -> 'bool'
    // CHECK-FIXES: return true
}

bool implicitConversionReturnBoolWithParens()
{
    return (1);
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: implicit conversion 'int' -> 'bool'
    // CHECK-FIXES: return true
}

int keepCompactReturnInC_PR71848() {
  bool foo = false;
  return( foo );
// CHECK-MESSAGES: :[[@LINE-1]]:9: warning: implicit conversion 'bool' -> 'int' [readability-implicit-bool-conversion]
// CHECK-FIXES: return(int)( foo );
}
