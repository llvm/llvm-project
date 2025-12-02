// RUN: %check_clang_tidy %s readability-implicit-bool-conversion %t
// RUN: %check_clang_tidy -check-suffix=UPPER-CASE %s readability-implicit-bool-conversion %t -- \
// RUN:     -config='{CheckOptions: { \
// RUN:         readability-implicit-bool-conversion.UseUpperCaseLiteralSuffix: true \
// RUN:     }}'

// We need NULL macro, but some buildbots don't like including <cstddef> header
// This is a portable way of getting it to work
#undef NULL
#define NULL 0L

template<typename T>
void functionTaking(T);

struct Struct {
  int member;
};


////////// Implicit conversion from bool.

void implicitConversionFromBoolSimpleCases() {
  bool boolean = true;

  functionTaking<bool>(boolean);

  functionTaking<int>(boolean);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: implicit conversion 'bool' -> 'int' [readability-implicit-bool-conversion]
  // CHECK-FIXES: functionTaking<int>(static_cast<int>(boolean));

  functionTaking<unsigned long>(boolean);
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: implicit conversion 'bool' -> 'unsigned long'
  // CHECK-FIXES: functionTaking<unsigned long>(static_cast<unsigned long>(boolean));

  functionTaking<char>(boolean);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'bool' -> 'char'
  // CHECK-FIXES: functionTaking<char>(static_cast<char>(boolean));

  functionTaking<float>(boolean);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: implicit conversion 'bool' -> 'float'
  // CHECK-FIXES: functionTaking<float>(static_cast<float>(boolean));

  functionTaking<double>(boolean);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: implicit conversion 'bool' -> 'double'
  // CHECK-FIXES: functionTaking<double>(static_cast<double>(boolean));
}

float implicitConversionFromBoolInReturnValue() {
  bool boolean = false;
  return boolean;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: implicit conversion 'bool' -> 'float'
  // CHECK-FIXES: return static_cast<float>(boolean);
}

void implicitConversionFromBoolInSingleBoolExpressions(bool b1, bool b2) {
  bool boolean = true;
  boolean = b1 ^ b2;
  boolean = b1 && b2;
  boolean |= !b1 || !b2;
  boolean &= b1;
  boolean = b1 == true;
  boolean = b2 != false;

  int integer = boolean - 3;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: implicit conversion 'bool' -> 'int'
  // CHECK-FIXES: int integer = static_cast<int>(boolean) - 3;

  float floating = boolean / 0.3f;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: implicit conversion 'bool' -> 'float'
  // CHECK-FIXES: float floating = static_cast<float>(boolean) / 0.3f;

  char character = boolean;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: implicit conversion 'bool' -> 'char'
  // CHECK-FIXES: char character = static_cast<char>(boolean);
}

void implicitConversionFromBoollInComplexBoolExpressions() {
  bool boolean = true;
  bool anotherBoolean = false;

  int integer = boolean && anotherBoolean;
  // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: implicit conversion 'bool' -> 'int'
  // CHECK-FIXES: int integer = static_cast<int>(boolean && anotherBoolean);

  unsigned long unsignedLong = (! boolean) + 4ul;
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: implicit conversion 'bool' -> 'unsigned long'
  // CHECK-FIXES: unsigned long unsignedLong = static_cast<unsigned long>(! boolean) + 4ul;

  float floating = (boolean || anotherBoolean) * 0.3f;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: implicit conversion 'bool' -> 'float'
  // CHECK-FIXES: float floating = static_cast<float>(boolean || anotherBoolean) * 0.3f;

  double doubleFloating = (boolean && (anotherBoolean || boolean)) * 0.3;
  // CHECK-MESSAGES: :[[@LINE-1]]:27: warning: implicit conversion 'bool' -> 'double'
  // CHECK-FIXES: double doubleFloating = static_cast<double>(boolean && (anotherBoolean || boolean)) * 0.3;
}

void implicitConversionFromBoolLiterals() {
  functionTaking<int>(true);
  // CHECK-MESSAGES: :[[@LINE-1]]:23: warning: implicit conversion 'bool' -> 'int'
  // CHECK-FIXES: functionTaking<int>(1);

  functionTaking<unsigned long>(false);
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: implicit conversion 'bool' -> 'unsigned long'
  // CHECK-FIXES: functionTaking<unsigned long>(0u);
  // CHECK-FIXES-UPPER-CASE: functionTaking<unsigned long>(0U);

  functionTaking<signed char>(true);
  // CHECK-MESSAGES: :[[@LINE-1]]:31: warning: implicit conversion 'bool' -> 'signed char'
  // CHECK-FIXES: functionTaking<signed char>(1);

  functionTaking<float>(false);
  // CHECK-MESSAGES: :[[@LINE-1]]:25: warning: implicit conversion 'bool' -> 'float'
  // CHECK-FIXES: functionTaking<float>(0.0f);
  // CHECK-FIXES-UPPER-CASE: functionTaking<float>(0.0F);

  functionTaking<double>(true);
  // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: implicit conversion 'bool' -> 'double'
  // CHECK-FIXES: functionTaking<double>(1.0);
}

void implicitConversionFromBoolInComparisons() {
  bool boolean = true;
  int integer = 0;

  functionTaking<bool>(boolean == integer);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'bool' -> 'int'
  // CHECK-FIXES: functionTaking<bool>(static_cast<int>(boolean) == integer);

  functionTaking<bool>(integer != boolean);
  // CHECK-MESSAGES: :[[@LINE-1]]:35: warning: implicit conversion 'bool' -> 'int'
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

void ignoreImplicitConversionFromBoolInMacroExpansions() {
  bool boolean = true;

  #define CAST_FROM_BOOL_IN_MACRO_BODY boolean + 3
  int integerFromMacroBody = CAST_FROM_BOOL_IN_MACRO_BODY;

  #define CAST_FROM_BOOL_IN_MACRO_ARGUMENT(x) x + 3
  int integerFromMacroArgument = CAST_FROM_BOOL_IN_MACRO_ARGUMENT(boolean);
}

namespace ignoreImplicitConversionFromBoolInTemplateInstantiations {

template<typename T>
void templateFunction() {
  bool boolean = true;
  T uknownType = boolean + 3;
}

void useOfTemplateFunction() {
  templateFunction<int>();
}

} // namespace ignoreImplicitConversionFromBoolInTemplateInstantiations

////////// Implicit conversions to bool.

void implicitConversionToBoolSimpleCases() {
  int integer = 10;
  functionTaking<bool>(integer);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(integer != 0);

  unsigned long unsignedLong = 10;
  functionTaking<bool>(unsignedLong);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'unsigned long' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(unsignedLong != 0u);
  // CHECK-FIXES-UPPER-CASE: functionTaking<bool>(unsignedLong != 0U);

  float floating = 0.0f;
  functionTaking<bool>(floating);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'float' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(floating != 0.0f);
  // CHECK-FIXES-UPPER-CASE: functionTaking<bool>(floating != 0.0F);

  double doubleFloating = 1.0f;
  functionTaking<bool>(doubleFloating);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'double' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(doubleFloating != 0.0);

  signed char character = 'a';
  functionTaking<bool>(character);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'signed char' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(character != 0);

  int* pointer = nullptr;
  functionTaking<bool>(pointer);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'int *' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(pointer != nullptr);

  auto pointerToMember = &Struct::member;
  functionTaking<bool>(pointerToMember);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'int Struct::*' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(pointerToMember != nullptr);
}

void implicitConversionToBoolInSingleExpressions() {
  int integer = 10;
  bool boolComingFromInt = integer;
  // CHECK-MESSAGES: :[[@LINE-1]]:28: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: bool boolComingFromInt = integer != 0;

  float floating = 10.0f;
  bool boolComingFromFloat = floating;
  // CHECK-MESSAGES: :[[@LINE-1]]:30: warning: implicit conversion 'float' -> 'bool'
  // CHECK-FIXES: bool boolComingFromFloat = floating != 0.0f;
  // CHECK-FIXES-UPPER-CASE: bool boolComingFromFloat = floating != 0.0F;

  signed char character = 'a';
  bool boolComingFromChar = character;
  // CHECK-MESSAGES: :[[@LINE-1]]:29: warning: implicit conversion 'signed char' -> 'bool'
  // CHECK-FIXES: bool boolComingFromChar = character != 0;

  int* pointer = nullptr;
  bool boolComingFromPointer = pointer;
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: implicit conversion 'int *' -> 'bool'
  // CHECK-FIXES: bool boolComingFromPointer = pointer != nullptr;
}

void implicitConversionToBoolInComplexExpressions() {
  bool boolean = true;

  int integer = 10;
  int anotherInteger = 20;
  bool boolComingFromInteger = integer + anotherInteger;
  // CHECK-MESSAGES: :[[@LINE-1]]:32: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: bool boolComingFromInteger = (integer + anotherInteger) != 0;

  float floating = 0.2f;
  bool boolComingFromFloating = floating - 0.3f || boolean;
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: implicit conversion 'float' -> 'bool'
  // CHECK-FIXES: bool boolComingFromFloating = ((floating - 0.3f) != 0.0f) || boolean;
  // CHECK-FIXES-UPPER-CASE: bool boolComingFromFloating = ((floating - 0.3f) != 0.0F) || boolean;

  double doubleFloating = 0.3;
  bool boolComingFromDoubleFloating = (doubleFloating - 0.4) && boolean;
  // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: implicit conversion 'double' -> 'bool'
  // CHECK-FIXES: bool boolComingFromDoubleFloating = ((doubleFloating - 0.4) != 0.0) && boolean;
}

void implicitConversionInNegationExpressions() {
  int integer = 10;
  bool boolComingFromNegatedInt = !integer;
  // CHECK-MESSAGES: :[[@LINE-1]]:36: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: bool boolComingFromNegatedInt = integer == 0;

  float floating = 10.0f;
  bool boolComingFromNegatedFloat = ! floating;
  // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: implicit conversion 'float' -> 'bool'
  // CHECK-FIXES: bool boolComingFromNegatedFloat = floating == 0.0f;
  // CHECK-FIXES-UPPER-CASE: bool boolComingFromNegatedFloat = floating == 0.0F;

  signed char character = 'a';
  bool boolComingFromNegatedChar = (! character);
  // CHECK-MESSAGES: :[[@LINE-1]]:39: warning: implicit conversion 'signed char' -> 'bool'
  // CHECK-FIXES: bool boolComingFromNegatedChar = (character == 0);

  int* pointer = nullptr;
  bool boolComingFromNegatedPointer = not pointer;
  // CHECK-MESSAGES: :[[@LINE-1]]:43: warning: implicit conversion 'int *' -> 'bool'
  // CHECK-FIXES: bool boolComingFromNegatedPointer = pointer == nullptr;
}

void implicitConversionToBoolInControlStatements() {
  int integer = 10;
  if (integer) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: if (integer != 0) {}

  long int longInteger = 0.2f;
  for (;longInteger;) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: implicit conversion 'long' -> 'bool'
  // CHECK-FIXES: for (;longInteger != 0;) {}

  float floating = 0.3f;
  while (floating) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: implicit conversion 'float' -> 'bool'
  // CHECK-FIXES: while (floating != 0.0f) {}
  // CHECK-FIXES-UPPER-CASE: while (floating != 0.0F) {}

  double doubleFloating = 0.4;
  do {} while (doubleFloating);
  // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: implicit conversion 'double' -> 'bool'
  // CHECK-FIXES: do {} while (doubleFloating != 0.0);
}

bool implicitConversionToBoolInReturnValue() {
  float floating = 1.0f;
  return floating;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: implicit conversion 'float' -> 'bool'
  // CHECK-FIXES: return floating != 0.0f;
  // CHECK-FIXES-UPPER-CASE: return floating != 0.0F;
}

void implicitConversionToBoolFromLiterals() {
  functionTaking<bool>(0);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(false);

  functionTaking<bool>(1);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(true);

  functionTaking<bool>(2ul);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'unsigned long' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(true);


  functionTaking<bool>(0.0f);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'float' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(false);

  functionTaking<bool>(1.0f);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'float' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(true);

  functionTaking<bool>(2.0);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'double' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(true);


  functionTaking<bool>('\0');
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'char' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(false);

  functionTaking<bool>('a');
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'char' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(true);


  functionTaking<bool>("");
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'const char *' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(true);

  functionTaking<bool>("abc");
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'const char *' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(true);

  functionTaking<bool>(NULL);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'long' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(false);
}

void implicitConversionToBoolFromUnaryMinusAndZeroLiterals() {
  functionTaking<bool>(-0);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>((-0) != 0);

  functionTaking<bool>(-0.0f);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'float' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>((-0.0f) != 0.0f);
  // CHECK-FIXES-UPPER-CASE: functionTaking<bool>((-0.0f) != 0.0F);

  functionTaking<bool>(-0.0);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'double' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>((-0.0) != 0.0);
}

void implicitConversionToBoolInWithOverloadedOperators() {
  struct UserStruct {
    int operator()(int x) { return x; }
    int operator+(int y) { return y; }
  };

  UserStruct s;

  functionTaking<bool>(s(0));
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(s(0) != 0);

  functionTaking<bool>(s + 2);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>((s + 2) != 0);
}

int functionReturningInt();
int* functionReturningPointer();

void ignoreImplicitConversionToBoolWhenDeclaringVariableInControlStatements() {
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

void ignoreImplicitConversionToBoolInMacroExpansions() {
  int integer = 3;

  #define CAST_TO_BOOL_IN_MACRO_BODY integer && false
  bool boolFromMacroBody = CAST_TO_BOOL_IN_MACRO_BODY;

  #define CAST_TO_BOOL_IN_MACRO_ARGUMENT(x) x || true
  bool boolFromMacroArgument = CAST_TO_BOOL_IN_MACRO_ARGUMENT(integer);
}

namespace ignoreImplicitConversionToBoolInTemplateInstantiations {

template<typename T>
void templateFunction() {
  T unknownType = 0;
  bool boolean = unknownType;
}

void useOfTemplateFunction() {
  templateFunction<int>();
}

} // namespace ignoreImplicitConversionToBoolInTemplateInstantiations

namespace ignoreUserDefinedConversionOperator {

struct StructWithUserConversion {
  operator bool();
};

void useOfUserConversion() {
  StructWithUserConversion structure;
  functionTaking<bool>(structure);
}

} // namespace ignoreUserDefinedConversionOperator

namespace ignore_1bit_bitfields {

struct S {
  int a;
  int b : 1;
  int c : 2;

  S(bool a, bool b, bool c) : a(a), b(b), c(c) {}
  // CHECK-MESSAGES: :[[@LINE-1]]:33: warning: implicit conversion 'bool' -> 'int'
  // CHECK-MESSAGES: :[[@LINE-2]]:45: warning: implicit conversion 'bool' -> 'int'
  // CHECK-FIXES: S(bool a, bool b, bool c) : a(static_cast<int>(a)), b(b), c(static_cast<int>(c)) {}
};

void f(S& s) {
  functionTaking<bool>(s.a);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(s.a != 0);
  functionTaking<bool>(s.b);
  // CHECK-FIXES: functionTaking<bool>(s.b);
  s.a = true;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: implicit conversion 'bool' -> 'int'
  // CHECK-FIXES: s.a = 1;
  s.b = true;
  // CHECK-FIXES: s.b = true;
  s.c = true;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: implicit conversion 'bool' -> 'int'
  // CHECK-FIXES: s.c = 1;
  functionTaking<bool>(s.c);
  // CHECK-MESSAGES: :[[@LINE-1]]:24: warning: implicit conversion 'int' -> 'bool'
  // CHECK-FIXES: functionTaking<bool>(s.c != 0);
}

} // namespace ignore_1bit_bitfields

int implicitConversionReturnInt()
{
    return true;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: implicit conversion 'bool' -> 'int'
    // CHECK-FIXES: return 1;
}

int implicitConversionReturnIntWithParens()
{
    return (true);
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: implicit conversion 'bool' -> 'int'
    // CHECK-FIXES: return 1;
}


bool implicitConversionReturnBool()
{
    return 1;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: implicit conversion 'int' -> 'bool'
    // CHECK-FIXES: return true;
}

bool implicitConversionReturnBoolWithParens()
{
    return (1);
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: implicit conversion 'int' -> 'bool'
    // CHECK-FIXES: return true;
}


namespace PR47000 {
  int to_int(bool x) { return int{x}; }

  using IntType = int;
  int to_int2(bool x) { return IntType{x}; }
}

namespace PR71867 {
  bool foo(bool x) {
    return x ? 1 : false;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: implicit conversion 'int' -> 'bool'
    // CHECK-MESSAGES: :[[@LINE-2]]:20: warning: implicit conversion 'bool' -> 'int'
    // CHECK-FIXES: return (x ? 1 : 0) != 0;
  }

  bool boo(bool x) {
    return x ? true : 0;
    // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: implicit conversion 'int' -> 'bool'
    // CHECK-MESSAGES: :[[@LINE-2]]:16: warning: implicit conversion 'bool' -> 'int'
    // CHECK-FIXES: return (x ? 1 : 0) != 0;
  }
}

namespace PR71848 {
  int fun() {
    bool foo = false;
    return( foo );
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: implicit conversion 'bool' -> 'int' [readability-implicit-bool-conversion]
// CHECK-FIXES: return static_cast<int>( foo );
  }
}

namespace PR161318 {  
  int AddParenOutsideOfCompoundAssignOp() {
    int val = -1;
    while(val >>= 7) {
    // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: implicit conversion 'int' -> 'bool' [readability-implicit-bool-conversion]
    // CHECK-FIXES: while((val >>= 7) != 0) {
    }
  }
}
