// RUN: %check_clang_tidy %s readability-implicit-bool-cast %t

// We need NULL macro, but some buildbots don't like including <cstddef> header
// This is a portable way of getting it to work
#undef NULL
#define NULL 0L

template<typename T>
void functionTaking(T);

struct Struct {
  int member;
};


////////// Implicit cast from bool.

void implicitCastFromBoolSimpleCases() {
  bool boolean = true;

  functionTaking<bool>(boolean);

  functionTaking<int>(boolean);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: implicit cast bool -> 'int' [readability-implicit-bool-cast]
  // CHECK-FIXES: functionTaking<int>(static_cast<int>(boolean));

  functionTaking<unsigned long>(boolean);
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: implicit cast bool -> 'unsigned long'
  // CHECK-FIXES: functionTaking<unsigned long>(static_cast<unsigned long>(boolean));

  functionTaking<char>(boolean);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast bool -> 'char'
  // CHECK-FIXES: functionTaking<char>(static_cast<char>(boolean));

  functionTaking<float>(boolean);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: implicit cast bool -> 'float'
  // CHECK-FIXES: functionTaking<float>(static_cast<float>(boolean));

  functionTaking<double>(boolean);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: implicit cast bool -> 'double'
  // CHECK-FIXES: functionTaking<double>(static_cast<double>(boolean));
}

float implicitCastFromBoolInReturnValue() {
  bool boolean = false;
  return boolean;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: implicit cast bool -> 'float'
  // CHECK-FIXES: return static_cast<float>(boolean);
}

void implicitCastFromBoolInSingleBoolExpressions(bool b1, bool b2) {
  bool boolean = true;
  boolean = b1 ^ b2;
  boolean = b1 && b2;
  boolean |= !b1 || !b2;
  boolean &= b1;
  boolean = b1 == true;
  boolean = b2 != false;

  int integer = boolean - 3;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: implicit cast bool -> 'int'
  // CHECK-FIXES: int integer = static_cast<int>(boolean) - 3;

  float floating = boolean / 0.3f;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: implicit cast bool -> 'float'
  // CHECK-FIXES: float floating = static_cast<float>(boolean) / 0.3f;

  char character = boolean;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: implicit cast bool -> 'char'
  // CHECK-FIXES: char character = static_cast<char>(boolean);
}

void implicitCastFromBoollInComplexBoolExpressions() {
  bool boolean = true;
  bool anotherBoolean = false;

  int integer = boolean && anotherBoolean;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: implicit cast bool -> 'int'
  // CHECK-FIXES: int integer = static_cast<int>(boolean && anotherBoolean);

  unsigned long unsignedLong = (! boolean) + 4ul;
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: implicit cast bool -> 'unsigned long'
  // CHECK-FIXES: unsigned long unsignedLong = static_cast<unsigned long>(! boolean) + 4ul;

  float floating = (boolean || anotherBoolean) * 0.3f;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: implicit cast bool -> 'float'
  // CHECK-FIXES: float floating = static_cast<float>(boolean || anotherBoolean) * 0.3f;

  double doubleFloating = (boolean && (anotherBoolean || boolean)) * 0.3;
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: implicit cast bool -> 'double'
  // CHECK-FIXES: double doubleFloating = static_cast<double>(boolean && (anotherBoolean || boolean)) * 0.3;
}

void implicitCastFromBoolLiterals() {
  functionTaking<int>(true);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: implicit cast bool -> 'int'
  // CHECK-FIXES: functionTaking<int>(1);

  functionTaking<unsigned long>(false);
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: implicit cast bool -> 'unsigned long'
  // CHECK-FIXES: functionTaking<unsigned long>(0u);

  functionTaking<signed char>(true);
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: implicit cast bool -> 'signed char'
  // CHECK-FIXES: functionTaking<signed char>(1);

  functionTaking<float>(false);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: implicit cast bool -> 'float'
  // CHECK-FIXES: functionTaking<float>(0.0f);

  functionTaking<double>(true);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: implicit cast bool -> 'double'
  // CHECK-FIXES: functionTaking<double>(1.0);
}

void implicitCastFromBoolInComparisons() {
  bool boolean = true;
  int integer = 0;

  functionTaking<bool>(boolean == integer);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast bool -> 'int'
  // CHECK-FIXES: functionTaking<bool>(static_cast<int>(boolean) == integer);

  functionTaking<bool>(integer != boolean);
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: implicit cast bool -> 'int'
  // CHECK-FIXES: functionTaking<bool>(integer != static_cast<int>(boolean));
}

void ignoreBoolComparisons() {
  bool boolean = true;
  bool anotherBoolean = false;

  functionTaking<bool>(boolean == anotherBoolean);
  functionTaking<bool>(boolean != anotherBoolean);
}

void ignoreExplicitCastsFromBool() {
  bool boolean = true;

  int integer = static_cast<int>(boolean) + 3;
  float floating = static_cast<float>(boolean) * 0.3f;
  char character = static_cast<char>(boolean);
}

void ignoreImplicitCastFromBoolInMacroExpansions() {
  bool boolean = true;

  #define CAST_FROM_BOOL_IN_MACRO_BODY boolean + 3
  int integerFromMacroBody = CAST_FROM_BOOL_IN_MACRO_BODY;

  #define CAST_FROM_BOOL_IN_MACRO_ARGUMENT(x) x + 3
  int integerFromMacroArgument = CAST_FROM_BOOL_IN_MACRO_ARGUMENT(boolean);
}

namespace ignoreImplicitCastFromBoolInTemplateInstantiations {

template<typename T>
void templateFunction() {
  bool boolean = true;
  T uknownType = boolean + 3;
}

void useOfTemplateFunction() {
  templateFunction<int>();
}

} // namespace ignoreImplicitCastFromBoolInTemplateInstantiations

////////// Implicit cast to bool.

void implicitCastToBoolSimpleCases() {
  int integer = 10;
  functionTaking<bool>(integer);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'int' -> bool
  // CHECK-FIXES: functionTaking<bool>(integer != 0);

  unsigned long unsignedLong = 10;
  functionTaking<bool>(unsignedLong);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'unsigned long' -> bool
  // CHECK-FIXES: functionTaking<bool>(unsignedLong != 0u);

  float floating = 0.0f;
  functionTaking<bool>(floating);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'float' -> bool
  // CHECK-FIXES: functionTaking<bool>(floating != 0.0f);

  double doubleFloating = 1.0f;
  functionTaking<bool>(doubleFloating);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'double' -> bool
  // CHECK-FIXES: functionTaking<bool>(doubleFloating != 0.0);

  signed char character = 'a';
  functionTaking<bool>(character);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'signed char' -> bool
  // CHECK-FIXES: functionTaking<bool>(character != 0);

  int* pointer = nullptr;
  functionTaking<bool>(pointer);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'int *' -> bool
  // CHECK-FIXES: functionTaking<bool>(pointer != nullptr);

  auto pointerToMember = &Struct::member;
  functionTaking<bool>(pointerToMember);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'int Struct::*' -> bool
  // CHECK-FIXES: functionTaking<bool>(pointerToMember != nullptr);
}

void implicitCastToBoolInSingleExpressions() {
  int integer = 10;
  bool boolComingFromInt = integer;
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: implicit cast 'int' -> bool
  // CHECK-FIXES: bool boolComingFromInt = integer != 0;

  float floating = 10.0f;
  bool boolComingFromFloat = floating;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: implicit cast 'float' -> bool
  // CHECK-FIXES: bool boolComingFromFloat = floating != 0.0f;

  signed char character = 'a';
  bool boolComingFromChar = character;
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: implicit cast 'signed char' -> bool
  // CHECK-FIXES: bool boolComingFromChar = character != 0;

  int* pointer = nullptr;
  bool boolComingFromPointer = pointer;
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: implicit cast 'int *' -> bool
  // CHECK-FIXES: bool boolComingFromPointer = pointer != nullptr;
}

void implicitCastToBoolInComplexExpressions() {
  bool boolean = true;

  int integer = 10;
  int anotherInteger = 20;
  bool boolComingFromInteger = integer + anotherInteger;
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: implicit cast 'int' -> bool
  // CHECK-FIXES: bool boolComingFromInteger = (integer + anotherInteger) != 0;

  float floating = 0.2f;
  bool boolComingFromFloating = floating - 0.3f || boolean;
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: implicit cast 'float' -> bool
  // CHECK-FIXES: bool boolComingFromFloating = ((floating - 0.3f) != 0.0f) || boolean;

  double doubleFloating = 0.3;
  bool boolComingFromDoubleFloating = (doubleFloating - 0.4) && boolean;
  // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: implicit cast 'double' -> bool
  // CHECK-FIXES: bool boolComingFromDoubleFloating = ((doubleFloating - 0.4) != 0.0) && boolean;
}

void implicitCastInNegationExpressions() {
  int integer = 10;
  bool boolComingFromNegatedInt = !integer;
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: implicit cast 'int' -> bool
  // CHECK-FIXES: bool boolComingFromNegatedInt = integer == 0;

  float floating = 10.0f;
  bool boolComingFromNegatedFloat = ! floating;
  // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: implicit cast 'float' -> bool
  // CHECK-FIXES: bool boolComingFromNegatedFloat = floating == 0.0f;

  signed char character = 'a';
  bool boolComingFromNegatedChar = (! character);
  // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: implicit cast 'signed char' -> bool
  // CHECK-FIXES: bool boolComingFromNegatedChar = (character == 0);

  int* pointer = nullptr;
  bool boolComingFromNegatedPointer = not pointer;
  // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: implicit cast 'int *' -> bool
  // CHECK-FIXES: bool boolComingFromNegatedPointer = pointer == nullptr;
}

void implicitCastToBoolInControlStatements() {
  int integer = 10;
  if (integer) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: implicit cast 'int' -> bool
  // CHECK-FIXES: if (integer != 0) {}

  long int longInteger = 0.2f;
  for (;longInteger;) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: implicit cast 'long' -> bool
  // CHECK-FIXES: for (;longInteger != 0;) {}

  float floating = 0.3f;
  while (floating) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: implicit cast 'float' -> bool
  // CHECK-FIXES: while (floating != 0.0f) {}

  double doubleFloating = 0.4;
  do {} while (doubleFloating);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: implicit cast 'double' -> bool
  // CHECK-FIXES: do {} while (doubleFloating != 0.0);
}

bool implicitCastToBoolInReturnValue() {
  float floating = 1.0f;
  return floating;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: implicit cast 'float' -> bool
  // CHECK-FIXES: return floating != 0.0f;
}

void implicitCastToBoolFromLiterals() {
  functionTaking<bool>(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'int' -> bool
  // CHECK-FIXES: functionTaking<bool>(false);

  functionTaking<bool>(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'int' -> bool
  // CHECK-FIXES: functionTaking<bool>(true);

  functionTaking<bool>(2ul);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'unsigned long' -> bool
  // CHECK-FIXES: functionTaking<bool>(true);


  functionTaking<bool>(0.0f);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'float' -> bool
  // CHECK-FIXES: functionTaking<bool>(false);

  functionTaking<bool>(1.0f);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'float' -> bool
  // CHECK-FIXES: functionTaking<bool>(true);

  functionTaking<bool>(2.0);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'double' -> bool
  // CHECK-FIXES: functionTaking<bool>(true);


  functionTaking<bool>('\0');
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'char' -> bool
  // CHECK-FIXES: functionTaking<bool>(false);

  functionTaking<bool>('a');
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'char' -> bool
  // CHECK-FIXES: functionTaking<bool>(true);


  functionTaking<bool>("");
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'const char *' -> bool
  // CHECK-FIXES: functionTaking<bool>(true);

  functionTaking<bool>("abc");
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'const char *' -> bool
  // CHECK-FIXES: functionTaking<bool>(true);

  functionTaking<bool>(NULL);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'long' -> bool
  // CHECK-FIXES: functionTaking<bool>(false);
}

void implicitCastToBoolFromUnaryMinusAndZeroLiterals() {
  functionTaking<bool>(-0);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'int' -> bool
  // CHECK-FIXES: functionTaking<bool>((-0) != 0);

  functionTaking<bool>(-0.0f);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'float' -> bool
  // CHECK-FIXES: functionTaking<bool>((-0.0f) != 0.0f);

  functionTaking<bool>(-0.0);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'double' -> bool
  // CHECK-FIXES: functionTaking<bool>((-0.0) != 0.0);
}

void implicitCastToBoolInWithOverloadedOperators() {
  struct UserStruct {
    int operator()(int x) { return x; }
    int operator+(int y) { return y; }
  };

  UserStruct s;

  functionTaking<bool>(s(0));
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'int' -> bool
  // CHECK-FIXES: functionTaking<bool>(s(0) != 0);

  functionTaking<bool>(s + 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit cast 'int' -> bool
  // CHECK-FIXES: functionTaking<bool>((s + 2) != 0);
}

int functionReturningInt();
int* functionReturningPointer();

void ignoreImplicitCastToBoolWhenDeclaringVariableInControlStatements() {
  if (int integer = functionReturningInt()) {}

  while (int* pointer = functionReturningPointer()) {}
}

void ignoreExplicitCastsToBool() {
  int integer = 10;
  bool boolComingFromInt = static_cast<bool>(integer);

  float floating = 10.0f;
  bool boolComingFromFloat = static_cast<bool>(floating);

  char character = 'a';
  bool boolComingFromChar = static_cast<bool>(character);

  int* pointer = nullptr;
  bool booleanComingFromPointer = static_cast<bool>(pointer);
}

void ignoreImplicitCastToBoolInMacroExpansions() {
  int integer = 3;

  #define CAST_TO_BOOL_IN_MACRO_BODY integer && false
  bool boolFromMacroBody = CAST_TO_BOOL_IN_MACRO_BODY;

  #define CAST_TO_BOOL_IN_MACRO_ARGUMENT(x) x || true
  bool boolFromMacroArgument = CAST_TO_BOOL_IN_MACRO_ARGUMENT(integer);
}

namespace ignoreImplicitCastToBoolInTemplateInstantiations {

template<typename T>
void templateFunction() {
  T unknownType = 0;
  bool boolean = unknownType;
}

void useOfTemplateFunction() {
  templateFunction<int>();
}

} // namespace ignoreImplicitCastToBoolInTemplateInstantiations

namespace ignoreUserDefinedConversionOperator {

struct StructWithUserConversion {
  operator bool();
};

void useOfUserConversion() {
  StructWithUserConversion structure;
  functionTaking<bool>(structure);
}

} // namespace ignoreUserDefinedConversionOperator
