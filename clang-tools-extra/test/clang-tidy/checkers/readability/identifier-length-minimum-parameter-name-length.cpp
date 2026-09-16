// RUN: %check_clang_tidy %s readability-identifier-length %t \
// RUN: -config='{CheckOptions: {readability-identifier-length.MinimumParameterNameLength: 5}}' \
// RUN: -- -fexceptions

void test(int data)
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: parameter name 'data' is too short, expected at least 5 characters [readability-identifier-length]
{
  (void)data;
}
