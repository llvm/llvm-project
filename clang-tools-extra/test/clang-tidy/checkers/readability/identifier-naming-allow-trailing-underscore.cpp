// RUN: %check_clang_tidy -std=c++14-or-later %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.LocalVariableCase: lower_case, \
// RUN:     readability-identifier-naming.ParameterCase: UPPER_CASE, \
// RUN:     readability-identifier-naming.FunctionCase: CamelCase, \
// RUN:     readability-identifier-naming.LambdaCaptureCase: camel_Snake_Back, \
// RUN:     readability-identifier-naming.MemberCase: lower_case, \
// RUN:     readability-identifier-naming.MemberSuffix: '_impl', \
// RUN:   }}'

// RUN: %check_clang_tidy -std=c++14-or-later -check-suffixes=ALLOWED %s \
// RUN:   readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.AllowTrailingUnderscore: true, \
// RUN:     readability-identifier-naming.LocalVariableCase: lower_case, \
// RUN:     readability-identifier-naming.ParameterCase: UPPER_CASE, \
// RUN:     readability-identifier-naming.FunctionCase: CamelCase, \
// RUN:     readability-identifier-naming.LambdaCaptureCase: camel_Snake_Back, \
// RUN:     readability-identifier-naming.MemberCase: lower_case, \
// RUN:     readability-identifier-naming.MemberSuffix: '_impl', \
// RUN:   }}'

void Positive(int TRANSLATOR) {
  int translator_ = TRANSLATOR;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for local variable 'translator_' [readability-identifier-naming]
  // CHECK-FIXES: int translator = TRANSLATOR;
}

void WrongCaseKeepsUnderscore() {
  int Bad_Name_;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for local variable 'Bad_Name_'
  // CHECK-MESSAGES-ALLOWED: :[[@LINE-2]]:7: warning: invalid case style for local variable 'Bad_Name_'
  // CHECK-FIXES: int bad_name;
  // CHECK-FIXES-ALLOWED: int bad_name_;
}

void ExtraUnderscoresRejected() {
  int bad__;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for local variable 'bad__'
  // CHECK-MESSAGES-ALLOWED: :[[@LINE-2]]:7: warning: invalid case style for local variable 'bad__'
  // CHECK-FIXES: int bad;
  // CHECK-FIXES-ALLOWED: int bad_;
}

void LeadingUnderscoreRejected() {
  int _bad;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for local variable '_bad'
  // CHECK-MESSAGES-ALLOWED: :[[@LINE-2]]:7: warning: invalid case style for local variable '_bad'
  // CHECK-FIXES: int bad;
  // CHECK-FIXES-ALLOWED: int bad;
}

void TakesParam(int VALUE_) {
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: invalid case style for parameter 'VALUE_'
  // CHECK-FIXES: void TakesParam(int VALUE) {
}

void Helper_() {}
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: invalid case style for function 'Helper_'
// CHECK-FIXES: void Helper() {}

void LambdaCapture(int VALUE) {
  auto lambda = [value_Snake_ = VALUE]() { return value_Snake_; };
  // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: invalid case style for lambda capture 'value_Snake_'
  // CHECK-FIXES: auto lambda = [value_Snake = VALUE]() { return value_Snake; };
}

struct MemberSuffix {
  int foo_impl_;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for member 'foo_impl_'
  // CHECK-FIXES: int foo_impl_impl;
};
