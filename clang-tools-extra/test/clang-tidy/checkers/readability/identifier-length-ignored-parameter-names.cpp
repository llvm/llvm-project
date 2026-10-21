// RUN: %check_clang_tidy %s readability-identifier-length %t \
// RUN: -config='{CheckOptions: {readability-identifier-length.IgnoredParameterNames: "^[ab]$"}}' \
// RUN: -- -fexceptions

void test(int a, int b, int c)
// CHECK-MESSAGES: :[[@LINE-1]]:29: warning: parameter name 'c' is too short, expected at least 3 characters [readability-identifier-length]
{
  (void)a;
  (void)b;
  (void)c;
}
