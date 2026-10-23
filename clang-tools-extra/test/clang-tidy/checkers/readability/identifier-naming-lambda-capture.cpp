// RUN: %check_clang_tidy -std=c++14-or-later %s readability-identifier-naming %t -- \
// RUN:   -config='{CheckOptions: { \
// RUN:     readability-identifier-naming.LambdaCaptureCase: CamelCase, \
// RUN:     readability-identifier-naming.LambdaCapturePrefix: 'c_', \
// RUN:     readability-identifier-naming.LocalVariableCase: lower_case, \
// RUN:   }}'

void goodInitCapture() {
  int local_variable = 0;
  auto lambda = [c_LocalVariable = local_variable]() {
    return c_LocalVariable;
  };
  (void)lambda();
}

void badInitCapture() {
  int local_variable = 0;
  auto lambda = [captured_value = local_variable]() {
    return captured_value;
  };
  // CHECK-MESSAGES: :[[@LINE-3]]:18: warning: invalid case style for lambda capture 'captured_value' [readability-identifier-naming]
  // CHECK-FIXES: auto lambda = [c_CapturedValue = local_variable]() {
  // CHECK-FIXES-NEXT: return c_CapturedValue;
  (void)lambda();
}

// Simple (non-init) explicit captures reuse the *same* VarDecl as the
// outer declaration, so they must keep following LocalVariable's style,
// not LambdaCapture, and must not gain the 'c_' prefix.
void simpleCapturesUseLocalVariableStyle() {
  int local_variable = 0;
  auto by_copy = [local_variable]() { return local_variable; };
  auto by_ref = [&local_variable]() { return local_variable; };
  (void)by_copy();
  (void)by_ref();
}

void badLocalVariableCapturedSimply() {
  int LocalVariable = 0;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: invalid case style for local variable 'LocalVariable' [readability-identifier-naming]
  // CHECK-FIXES: int local_variable = 0;
  auto lambda = [LocalVariable]() { return LocalVariable; };
  // CHECK-FIXES: auto lambda = [local_variable]() { return local_variable; };
  (void)lambda();
}
