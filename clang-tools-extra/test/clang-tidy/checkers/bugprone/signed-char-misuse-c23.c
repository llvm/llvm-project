// RUN: %check_clang_tidy %s bugprone-signed-char-misuse %t -- -- -std=c23

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

enum EType1 : signed char {
  EType1_M128 = -128,
  EType1_1 = 1,
};

enum EType1 es1_1 = EType1_M128;
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: 'signed char' to 'enum EType1' conversion; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]
enum EType1 es1_2 = EType1_1;
enum EType1 es1_3 = -128;

void assign(enum EType1 *es) {
    *es = EType1_M128;
// CHECK-MESSAGES: :[[@LINE-1]]:11: warning: 'signed char' to 'enum EType1' conversion; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]
}


typedef signed char int8_t;
typedef enum EType2 : int8_t {
  EType2_M128 = -128,
  EType2_1 = 1,
} EType2_t;

EType2_t es2_1 = EType2_M128;
// CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'signed char' to 'EType2_t' (aka 'enum EType2') conversion; consider casting to 'unsigned char' first. [bugprone-signed-char-misuse]
EType2_t es2_2 = EType2_1;
EType2_t es2_3 = -128;
