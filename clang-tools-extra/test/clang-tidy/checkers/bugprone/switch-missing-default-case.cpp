// RUN: %check_clang_tidy %s bugprone-switch-missing-default-case %t -- -- -fno-delayed-template-parsing

typedef int MyInt;
enum EnumType { eE2 };
typedef EnumType MyEnum;

void positive() {
  int I1 = 0;
  // CHECK-MESSAGES: [[@LINE+1]]:3: warning: switching on non-enum value without default case may not cover all cases [bugprone-switch-missing-default-case]
  switch (I1) {
  case 0:
    break;
  }

  MyInt I2 = 0;
  // CHECK-MESSAGES: [[@LINE+1]]:3: warning: switching on non-enum value without default case may not cover all cases [bugprone-switch-missing-default-case]
  switch (I2) {
  case 0:
    break;
  }

  int getValue(void);
  // CHECK-MESSAGES: [[@LINE+1]]:3: warning: switching on non-enum value without default case may not cover all cases [bugprone-switch-missing-default-case]
  switch (getValue()) {
  case 0:
    break;
  }
}

void negative() {
  enum E { eE1 };
  E E1 = eE1;
  switch (E1) { // no-warning
  case eE1:
    break;
  }

  MyEnum E2 = eE2;
  switch (E2) { // no-warning
  case eE2:
    break;
  }

  int I1 = 0;
  switch (I1) { // no-warning
  case 0:
    break;
  default:
    break;
  }

  MyInt I2 = 0;
  switch (I2) { // no-warning
  case 0:
    break;
  default:
    break;
  }

  int getValue(void);
  switch (getValue()) { // no-warning
  case 0:
    break;
    default:
    break;
  }
}

template<typename T>
void testTemplate(T Value) {
  switch (Value) {
    case 0:
      break;
  }
}

void exampleUsage() {
  testTemplate(5);
  testTemplate(EnumType::eE2);
}
