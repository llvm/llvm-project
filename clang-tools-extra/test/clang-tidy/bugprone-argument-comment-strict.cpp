// RUN: %check_clang_tidy %s bugprone-argument-comment %t -- \
// RUN:   -config="{CheckOptions: [{key: StrictMode, value: 1}]}" --

void f(int _with_underscores_);
void g(int x_);
void ignores_underscores() {
  f(/*With_Underscores=*/0);
// CHECK-MESSAGES: [[@LINE-1]]:5: warning: argument name 'With_Underscores' in comment does not match parameter name '_with_underscores_'
// CHECK-FIXES: f(/*_with_underscores_=*/0);
  f(/*with_underscores=*/1);
// CHECK-MESSAGES: [[@LINE-1]]:5: warning: argument name 'with_underscores' in comment does not match parameter name '_with_underscores_'
// CHECK-FIXES: f(/*_with_underscores_=*/1);
  f(/*_With_Underscores_=*/2);
// CHECK-MESSAGES: [[@LINE-1]]:5: warning: argument name '_With_Underscores_' in comment does not match parameter name '_with_underscores_'
// CHECK-FIXES: f(/*_with_underscores_=*/2);
  g(/*X=*/3);
// CHECK-MESSAGES: [[@LINE-1]]:5: warning: argument name 'X' in comment does not match parameter name 'x_'
// CHECK-FIXES: g(/*x_=*/3);
}
