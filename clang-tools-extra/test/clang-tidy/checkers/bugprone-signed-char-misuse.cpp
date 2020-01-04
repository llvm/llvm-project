// RUN: %check_clang_tidy %s bugprone-signed-char-misuse %t

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

int CStyleCast() {
  signed char CCharacter = -5;
  int NCharacter;
  NCharacter = (int)CCharacter;
  // CHECK-MESSAGES: [[@LINE-1]]:21: warning: 'signed char' to 'int' conversion; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]

  return NCharacter;
}

int StaticCast() {
  signed char CCharacter = -5;
  int NCharacter;
  NCharacter = static_cast<int>(CCharacter);
  // CHECK-MESSAGES: [[@LINE-1]]:33: warning: 'signed char' to 'int' conversion; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]

  return NCharacter;
}

int FunctionalCast() {
  signed char CCharacter = -5;
  int NCharacter;
  NCharacter = int(CCharacter);
  // CHECK-MESSAGES: [[@LINE-1]]:20: warning: 'signed char' to 'int' conversion; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]

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

///////////////////////////////////////////////////////////////////
/// Test cases correctly ignored by the check.

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

// singed char -> integer cast is not the direct child of declaration expression.
int DescendantCast() {
  signed char CCharacter = 'a';
  int NCharacter = 10 + CCharacter;

  return NCharacter;
}

// singed char -> integer cast is not the direct child of assignment expression.
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
