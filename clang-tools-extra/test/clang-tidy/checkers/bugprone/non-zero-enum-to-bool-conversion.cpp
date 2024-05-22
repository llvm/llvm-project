// RUN: %check_clang_tidy -std=c++98-or-later %s bugprone-non-zero-enum-to-bool-conversion %t -- \
// RUN:   -config="{CheckOptions: {bugprone-non-zero-enum-to-bool-conversion.EnumIgnoreList: '::without::issue::IgnoredEnum;IgnoredSecondEnum'}}"

namespace with::issue {

typedef enum EStatus {
  SUCCESS       = 1,
  FAILURE       = 2,
  INVALID_PARAM = 3,
  UNKNOWN       = 4
} Status;

bool testEnumConversion(EStatus value) {
  // CHECK-MESSAGES: :[[@LINE+1]]:10: warning: conversion of 'EStatus' into 'bool' will always return 'true', enum doesn't have a zero-value enumerator [bugprone-non-zero-enum-to-bool-conversion]
  return value;
}

bool testTypedefConversion(Status value) {
  // CHECK-MESSAGES: :[[@LINE+1]]:10: warning: conversion of 'EStatus' into 'bool' will always return 'true', enum doesn't have a zero-value enumerator [bugprone-non-zero-enum-to-bool-conversion]
  return value;
}

bool testExplicitConversion(EStatus value) {
  // CHECK-MESSAGES: :[[@LINE+1]]:28: warning: conversion of 'EStatus' into 'bool' will always return 'true', enum doesn't have a zero-value enumerator [bugprone-non-zero-enum-to-bool-conversion]
  return static_cast<bool>(value);
}

bool testInIfConversion(EStatus value) {
  // CHECK-MESSAGES: :[[@LINE+1]]:7: warning: conversion of 'EStatus' into 'bool' will always return 'true', enum doesn't have a zero-value enumerator [bugprone-non-zero-enum-to-bool-conversion]
  if (value) {
    return false;
  }
  return true;
}

bool testWithNegation(EStatus value) {
  // CHECK-MESSAGES: :[[@LINE+1]]:14: warning: conversion of 'EStatus' into 'bool' will always return 'true', enum doesn't have a zero-value enumerator [bugprone-non-zero-enum-to-bool-conversion]
  return not value;
}

}

namespace without::issue {

enum StatusWithZero {
  UNK  = 0,
  OK   = 1,
  NOT_OK = 2
};

bool testEnumConversion(StatusWithZero value) {
  return value;
}

enum WithDefault {
  Value0,
  Value1
};

bool testEnumConversion(WithDefault value) {
  return value;
}

enum WithNegative : int {
  Nen2 = -2,
  Nen1,
  Nen0
};

bool testEnumConversion(WithNegative value) {
  return value;
}

enum EStatus {
  SUCCESS = 1,
  FAILURE,
  INVALID_PARAM,
  UNKNOWN
};

bool explicitCompare(EStatus value) {
  return value == SUCCESS;
}

bool explicitBitUsage1(EStatus value) {
  return (value & SUCCESS);
}

bool explicitBitUsage2(EStatus value) {
  return (value | SUCCESS);
}

bool testEnumeratorCompare() {
  return SUCCESS;
}

enum IgnoredEnum {
  IGNORED_VALUE_1 = 1,
  IGNORED_VALUE_2
};

enum IgnoredSecondEnum {
  IGNORED_SECOND_VALUE_1 = 1,
  IGNORED_SECOND_VALUE_2
};

bool testIgnored(IgnoredEnum value) {
  return value;
}

bool testIgnored(IgnoredSecondEnum value) {
  return value;
}

enum CustomOperatorEnum {
    E0 = 0x1,
    E1 = 0x2,
    E2 = 0x4
};

CustomOperatorEnum operator&(CustomOperatorEnum a, CustomOperatorEnum b) { return static_cast<CustomOperatorEnum>(a & b); }

void testCustomOperator(CustomOperatorEnum e) {
    if (e & E1) {}
}

}
