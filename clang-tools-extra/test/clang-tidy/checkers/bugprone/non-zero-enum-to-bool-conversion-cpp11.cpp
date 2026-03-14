// RUN: %check_clang_tidy -std=c++11-or-later %s bugprone-non-zero-enum-to-bool-conversion %t

namespace with::issue {

enum class EStatusC : char {
  SUCCESS       = 1,
  FAILURE       = 2,
  INVALID_PARAM = 3,
  UNKNOWN       = 4
};

bool testEnumConversion(EStatusC value) {
  // CHECK-MESSAGES: :[[@LINE+1]]:10: warning: conversion of 'EStatusC' into 'bool' will always return 'true', enum doesn't have a zero-value enumerator [bugprone-non-zero-enum-to-bool-conversion]
  return static_cast<bool>(value);
}

enum class EStatusS : short {
  SUCCESS       = 1,
  FAILURE       = 2,
  INVALID_PARAM = 3,
  UNKNOWN       = 4
};

bool testEnumConversion(EStatusS value) {
  // CHECK-MESSAGES: :[[@LINE+1]]:10: warning: conversion of 'EStatusS' into 'bool' will always return 'true', enum doesn't have a zero-value enumerator [bugprone-non-zero-enum-to-bool-conversion]
  return static_cast<bool>(value);
}


enum class EStatusI : int {
  SUCCESS       = 1,
  FAILURE       = 2,
  INVALID_PARAM = 3,
  UNKNOWN       = 4
};

bool testEnumConversion(EStatusI value) {
  // CHECK-MESSAGES: :[[@LINE+1]]:10: warning: conversion of 'EStatusI' into 'bool' will always return 'true', enum doesn't have a zero-value enumerator [bugprone-non-zero-enum-to-bool-conversion]
  return static_cast<bool>(value);
}

enum class EStatus {
  SUCCESS       = 1,
  FAILURE       = 2,
  INVALID_PARAM = 3,
  UNKNOWN       = 4
};

bool testEnumConversion(EStatus value) {
  // CHECK-MESSAGES: :[[@LINE+1]]:10: warning: conversion of 'EStatus' into 'bool' will always return 'true', enum doesn't have a zero-value enumerator [bugprone-non-zero-enum-to-bool-conversion]
  return static_cast<bool>(value);
}

namespace enum_int {

enum EResult : int {
  OK = 1,
  NOT_OK
};

bool testEnumConversion(const EResult& value) {
  // CHECK-MESSAGES: :[[@LINE+1]]:10: warning: conversion of 'EResult' into 'bool' will always return 'true', enum doesn't have a zero-value enumerator [bugprone-non-zero-enum-to-bool-conversion]
  return value;
}

}

namespace enum_short {

enum EResult : short {
  OK = 1,
  NOT_OK
};

bool testEnumConversion(const EResult& value) {
  // CHECK-MESSAGES: :[[@LINE+1]]:10: warning: conversion of 'EResult' into 'bool' will always return 'true', enum doesn't have a zero-value enumerator [bugprone-non-zero-enum-to-bool-conversion]
  return value;
}

}

namespace enum_char {

enum EResult : char {
  OK = 1,
  NOT_OK
};

bool testEnumConversion(const EResult& value) {
  // CHECK-MESSAGES: :[[@LINE+1]]:10: warning: conversion of 'EResult' into 'bool' will always return 'true', enum doesn't have a zero-value enumerator [bugprone-non-zero-enum-to-bool-conversion]
  return value;
}

}

namespace enum_default {

enum EResult {
  OK = 1,
  NOT_OK
};

bool testEnumConversion(const EResult& value) {
  // CHECK-MESSAGES: :[[@LINE+1]]:10: warning: conversion of 'EResult' into 'bool' will always return 'true', enum doesn't have a zero-value enumerator [bugprone-non-zero-enum-to-bool-conversion]
  return value;
}

}

}
