// RUN: %check_clang_tidy %s readability-identifier-length %t \
// RUN: -config='{CheckOptions: {readability-identifier-length.MinimumVariableNameLength: 5}}' \
// RUN: -- -fexceptions

void doIt();

void test() {
  int valu = 5;
  // CHECK-MESSAGES: :[[@LINE-1]]:7: warning: variable name 'valu' is too short, expected at least 5 characters [readability-identifier-length]
  int value = 6; // 5 chars, ok
}
