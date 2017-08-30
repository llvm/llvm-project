// RUN: %check_clang_tidy %s hicpp-signed-bitwise %t -- -- -std=c++11 -target x86_64-unknown-unknown

// These could cause false positives and should not be considered.
struct StreamClass {
};
StreamClass &operator<<(StreamClass &os, unsigned int i) {
  return os;
}
StreamClass &operator<<(StreamClass &os, int i) {
  return os;
}
StreamClass &operator>>(StreamClass &os, unsigned int i) {
  return os;
}
StreamClass &operator>>(StreamClass &os, int i) {
  return os;
}
struct AnotherStream {
  AnotherStream &operator<<(unsigned char c) { return *this; }
  AnotherStream &operator<<(char c) { return *this; }

  AnotherStream &operator>>(unsigned char c) { return *this; }
  AnotherStream &operator>>(char c) { return *this; }
};

void binary_bitwise() {
  int SValue = 42;
  int SResult;

  unsigned int UValue = 42;
  unsigned int UResult;

  SResult = SValue & 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  SResult = SValue & -1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  SResult = SValue & SValue;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator

  UResult = SValue & 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  UResult = SValue & -1;
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator

  UResult = UValue & 1u;     // Ok
  UResult = UValue & UValue; // Ok

  unsigned char UByte1 = 0u;
  unsigned char UByte2 = 16u;
  char SByte1 = 0;
  char SByte2 = 16;

  UByte1 = UByte1 & UByte2; // Ok
  UByte1 = SByte1 & UByte2;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use of a signed integer operand with a binary bitwise operator
  UByte1 = SByte1 & SByte2;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use of a signed integer operand with a binary bitwise operator
  SByte1 = SByte1 & SByte2;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use of a signed integer operand with a binary bitwise operator

  // More complex expressions.
  UResult = UValue & (SByte1 + (SByte1 | SByte2));
  // CHECK-MESSAGES: :[[@LINE-1]]:13: warning: use of a signed integer operand with a binary bitwise operator
  // CHECK-MESSAGES: :[[@LINE-2]]:33: warning: use of a signed integer operand with a binary bitwise operator

  // The rest is to demonstrate functionality but all operators are matched equally.
  // Therefore functionality is the same for all binary operations.
  UByte1 = UByte1 | UByte2; // Ok
  UByte1 = UByte1 | SByte2;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use of a signed integer operand with a binary bitwise operator

  UByte1 = UByte1 ^ UByte2; // Ok
  UByte1 = UByte1 ^ SByte2;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use of a signed integer operand with a binary bitwise operator

  UByte1 = UByte1 >> UByte2; // Ok
  UByte1 = UByte1 >> SByte2;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use of a signed integer operand with a binary bitwise operator

  UByte1 = UByte1 << UByte2; // Ok
  UByte1 = UByte1 << SByte2;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use of a signed integer operand with a binary bitwise operator

  int SignedInt1 = 1 << 12;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use of a signed integer operand with a binary bitwise operator
  int SignedInt2 = 1u << 12;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use of a signed integer operand with a binary bitwise operator
}

void f1(unsigned char c) {}
void f2(char c) {}
void f3(int c) {}

void unary_bitwise() {
  unsigned char UByte1 = 0u;
  char SByte1 = 0;

  UByte1 = ~UByte1; // Ok
  SByte1 = ~UByte1;
  SByte1 = ~SByte1;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use of a signed integer operand with a unary bitwise operator
  UByte1 = ~SByte1;
  // CHECK-MESSAGES: :[[@LINE-1]]:12: warning: use of a signed integer operand with a unary bitwise operator

  unsigned int UInt = 0u;
  int SInt = 0;

  f1(~UByte1); // Ok
  f1(~SByte1);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use of a signed integer operand with a unary bitwise operator
  f1(~UInt);
  f1(~SInt);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use of a signed integer operand with a unary bitwise operator
  f2(~UByte1);
  f2(~SByte1);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use of a signed integer operand with a unary bitwise operator
  f2(~UInt);
  f2(~SInt);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use of a signed integer operand with a unary bitwise operator
  f3(~UByte1); // Ok
  f3(~SByte1);
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: use of a signed integer operand with a unary bitwise operator
}

/// HICPP uses these examples to demonstrate the rule.
void standard_examples() {
  int i = 3;
  unsigned int k = 0u;

  int r = i << -1; // Emits -Wshift-count-negative from clang
  // CHECK-MESSAGES: :[[@LINE-1]]:11: warning: use of a signed integer operand with a binary bitwise operator
  r = i << 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use of a signed integer operand with a binary bitwise operator

  r = -1 >> -1; // Emits -Wshift-count-negative from clang
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use of a signed integer operand with a binary bitwise operator
  r = -1 >> 1;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use of a signed integer operand with a binary bitwise operator

  r = -1 >> i;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use of a signed integer operand with a binary bitwise operator
  r = -1 >> -i;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use of a signed integer operand with a binary bitwise operator

  r = ~0;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use of a signed integer operand with a unary bitwise operator
  r = ~0u; // Ok
  k = ~k;  // Ok

  unsigned int u = (-1) & 2u;
  // CHECK-MESSAGES: :[[@LINE-1]]:20: warning: use of a signed integer operand with a binary bitwise operator
  u = (-1) | 1u;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use of a signed integer operand with a binary bitwise operator
  u = (-1) ^ 1u;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: use of a signed integer operand with a binary bitwise operator
}

void streams_should_work() {
  StreamClass s;
  s << 1u; // Ok
  s << 1;  // Ok
  s >> 1;  // Ok
  s >> 1u; // Ok

  AnotherStream as;
  unsigned char uc = 1u;
  char sc = 1;
  as << uc; // Ok
  as << sc; // Ok
  as >> uc; // Ok
  as >> sc; // Ok
}

enum OldEnum {
  ValueOne,
  ValueTwo,
};

enum OldSigned : int {
  IntOne,
  IntTwo,
};

void classicEnums() {
  OldEnum e1 = ValueOne, e2 = ValueTwo;
  int e3;                   // Using the enum type, results in an error.
  e3 = ValueOne | ValueTwo; // Ok
  e3 = ValueOne & ValueTwo; // Ok
  e3 = ValueOne ^ ValueTwo; // Ok
  e3 = e1 | e2;             // Ok
  e3 = e1 & e2;             // Ok
  e3 = e1 ^ e2;             // Ok

  OldSigned s1 = IntOne, s2 = IntTwo;
  int s3;
  s3 = IntOne | IntTwo; // Signed
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: use of a signed integer operand with a binary bitwise operator
  s3 = IntOne & IntTwo; // Signed
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: use of a signed integer operand with a binary bitwise operator
  s3 = IntOne ^ IntTwo; // Signed
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: use of a signed integer operand with a binary bitwise operator
  s3 = s1 | s2; // Signed
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: use of a signed integer operand with a binary bitwise operator
  s3 = s1 & s2; // Signed
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: use of a signed integer operand with a binary bitwise operator
  s3 = s1 ^ s2; // Signed
  // CHECK-MESSAGES: [[@LINE-1]]:8: warning: use of a signed integer operand with a binary bitwise operator
}

enum EnumConstruction {
  one = 1,
  two = 2,
  test1 = 1 << 12,
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: use of a signed integer operand with a binary bitwise operator
  test2 = one << two,
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: use of a signed integer operand with a binary bitwise operator
  test3 = 1u << 12,
  // CHECK-MESSAGES: [[@LINE-1]]:11: warning: use of a signed integer operand with a binary bitwise operator
};
