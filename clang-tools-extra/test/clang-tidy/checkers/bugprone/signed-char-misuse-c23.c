// RUN: %check_clang_tidy -std=c23-or-later %s bugprone-signed-char-misuse %t

///////////////////////////////////////////////////////////////////
/// Test cases correctly caught by the check.

int SimpleVarDeclaration() {
  signed char CCharacter = -5;
  int NCharacter = CCharacter;
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: 'signed char' to 'int' conversion; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]

  return NCharacter;
}

int SimpleAssignment() {
  signed char CCharacter = -5;
  int NCharacter;
  NCharacter = CCharacter;
  // CHECK-MESSAGES: [[@LINE-1]]:16: warning: 'signed char' to 'int' conversion; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]

  return NCharacter;
}

int NegativeConstValue() {
  const signed char CCharacter = -5;
  int NCharacter = CCharacter;
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: 'signed char' to 'int' conversion; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]

  return NCharacter;
}

int CharPointer(signed char *CCharacter) {
  int NCharacter = *CCharacter;
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: 'signed char' to 'int' conversion; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]

  return NCharacter;
}

int SignedUnsignedCharEquality(signed char SCharacter) {
  unsigned char USCharacter = 'a';
  if (SCharacter == USCharacter) // CHECK-MESSAGES: [[@LINE]]:7: warning: comparison between 'signed char' and 'unsigned char' [bugprone-signed-char-misuse]
    return 1;
  return 0;
}

int SignedUnsignedCharIneqiality(signed char SCharacter) {
  unsigned char USCharacter = 'a';
  if (SCharacter != USCharacter) // CHECK-MESSAGES: [[@LINE]]:7: warning: comparison between 'signed char' and 'unsigned char' [bugprone-signed-char-misuse]
    return 1;
  return 0;
}

int CompareWithNonAsciiConstant(unsigned char USCharacter) {
  const signed char SCharacter = -5;
  if (USCharacter == SCharacter) // CHECK-MESSAGES: [[@LINE]]:7: warning: comparison between 'signed char' and 'unsigned char' [bugprone-signed-char-misuse]
    return 1;
  return 0;
}

int CompareWithUnsignedNonAsciiConstant(signed char SCharacter) {
  const unsigned char USCharacter = 128;
  if (USCharacter == SCharacter) // CHECK-MESSAGES: [[@LINE]]:7: warning: comparison between 'signed char' and 'unsigned char' [bugprone-signed-char-misuse]
    return 1;
  return 0;
}

///////////////////////////////////////////////////////////////////
/// Test cases correctly ignored by the check.

// Enum with a fixed underlying type of signed char.
enum EType1 : signed char {
  EType1_M128 = -128,
  EType1_1 = 1,
};

enum EType1 es1_1 = EType1_M128;
enum EType1 es1_2 = EType1_1;
enum EType1 es1_3 = -128;

void assign(enum EType1 *es) {
    *es = EType1_M128;
}

// Type aliased enum with a fixed underlying type of signed char.
typedef signed char int8_t;
typedef enum EType2 : int8_t {
  EType2_M128 = -128,
  EType2_1 = 1,
} EType2_t;

EType2_t es2_1 = EType2_M128;
EType2_t es2_2 = EType2_1;
EType2_t es2_3 = -128;

// Enum with a fixed underlying type of unsigned char.
enum EType3 : unsigned char {
  EType3_1 = 1,
  EType3_128 = 128,
};

enum EType3 es3_1 = EType3_1;
enum EType3 es3_2 = EType3_128;
enum EType3 es3_3 = 128;


int UnsignedCharCast() {
  unsigned char CCharacter = 'a';
  int NCharacter = CCharacter;

  return NCharacter;
}

int PositiveConstValue() {
  const signed char CCharacter = 5;
  int NCharacter = CCharacter;

  return NCharacter;
}

// signed char -> integer cast is not the direct child of declaration expression.
int DescendantCast() {
  signed char CCharacter = 'a';
  int NCharacter = 10 + CCharacter;

  return NCharacter;
}

// signed char -> integer cast is not the direct child of assignment expression.
int DescendantCastAssignment() {
  signed char CCharacter = 'a';
  int NCharacter;
  NCharacter = 10 + CCharacter;

  return NCharacter;
}

// bool is an integer type in clang; make sure to ignore it.
bool BoolVarDeclaration() {
  signed char CCharacter = 'a';
  bool BCharacter = CCharacter == 'b';

  return BCharacter;
}

// bool is an integer type in clang; make sure to ignore it.
bool BoolAssignment() {
  signed char CCharacter = 'a';
  bool BCharacter;
  BCharacter = CCharacter == 'b';

  return BCharacter;
}

// char is an integer type in clang; make sure to ignore it.
unsigned char CharToCharCast() {
  signed char SCCharacter = 'a';
  unsigned char USCharacter;
  USCharacter = SCCharacter;

  return USCharacter;
}

int SameCharTypeComparison(signed char SCharacter) {
  signed char SCharacter2 = 'a';
  if (SCharacter == SCharacter2)
    return 1;
  return 0;
}

int SameCharTypeComparison2(unsigned char USCharacter) {
  unsigned char USCharacter2 = 'a';
  if (USCharacter == USCharacter2)
    return 1;
  return 0;
}

int CharIntComparison(signed char SCharacter) {
  int ICharacter = 10;
  if (SCharacter == ICharacter)
    return 1;
  return 0;
}

int CompareWithAsciiLiteral(unsigned char USCharacter) {
  if (USCharacter == 'x')
    return 1;
  return 0;
}

int CompareWithAsciiConstant(unsigned char USCharacter) {
  const signed char SCharacter = 'a';
  if (USCharacter == SCharacter)
    return 1;
  return 0;
}

int CompareWithUnsignedAsciiConstant(signed char SCharacter) {
  const unsigned char USCharacter = 'a';
  if (USCharacter == SCharacter)
    return 1;
  return 0;
}
