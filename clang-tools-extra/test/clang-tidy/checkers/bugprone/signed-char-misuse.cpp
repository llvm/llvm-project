// RUN: %check_clang_tidy %s bugprone-signed-char-misuse %t

///////////////////////////////////////////////////////////////////
/// Test cases correctly caught by the check.

typedef __SIZE_TYPE__ size_t;

namespace std {
template <typename T, size_t N>
struct array {
  T &operator[](size_t n);
  T &at(size_t n);
};
} // namespace std

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

int SignedCharCArraySubscript(signed char SCharacter) {
  int Array[3] = {1, 2, 3};

  return Array[static_cast<unsigned int>(SCharacter)]; // CHECK-MESSAGES: [[@LINE]]:42: warning: 'signed char' to 'unsigned int' conversion in array subscript; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]
}

int SignedCharSTDArraySubscript(std::array<int, 3> Array, signed char SCharacter) {
  return Array[static_cast<unsigned int>(SCharacter)]; // CHECK-MESSAGES: [[@LINE]]:42: warning: 'signed char' to 'unsigned int' conversion in array subscript; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]
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

int FixComparisonWithSignedCharCast(signed char SCharacter) {
  unsigned char USCharacter = 'a';
  if (SCharacter == static_cast<signed char>(USCharacter))
    return 1;
  return 0;
}

int FixComparisonWithUnSignedCharCast(signed char SCharacter) {
  unsigned char USCharacter = 'a';
  if (static_cast<unsigned char>(SCharacter) == USCharacter)
    return 1;
  return 0;
}

// Make sure we don't catch other type of char comparison.
int SameCharTypeComparison(signed char SCharacter) {
  signed char SCharacter2 = 'a';
  if (SCharacter == SCharacter2)
    return 1;
  return 0;
}

// Make sure we don't catch other type of char comparison.
int SameCharTypeComparison2(unsigned char USCharacter) {
  unsigned char USCharacter2 = 'a';
  if (USCharacter == USCharacter2)
    return 1;
  return 0;
}

// Make sure we don't catch integer - char comparison.
int CharIntComparison(signed char SCharacter) {
  int ICharacter = 10;
  if (SCharacter == ICharacter)
    return 1;
  return 0;
}

int CompareWithAsciiLiteral(unsigned char USCharacter) {
  if (USCharacter == 'x') // no warning
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

int UnsignedCharCArraySubscript(unsigned char USCharacter) {
  int Array[3] = {1, 2, 3};

  return Array[static_cast<unsigned int>(USCharacter)];
}

int CastedCArraySubscript(signed char SCharacter) {
  int Array[3] = {1, 2, 3};

  return Array[static_cast<unsigned char>(SCharacter)];
}

int UnsignedCharSTDArraySubscript(std::array<int, 3> Array, unsigned char USCharacter) {
  return Array[static_cast<unsigned int>(USCharacter)];
}

int CastedSTDArraySubscript(std::array<int, 3> Array, signed char SCharacter) {
  return Array[static_cast<unsigned char>(SCharacter)];
}
