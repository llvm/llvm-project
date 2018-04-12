// RUN: %check_clang_tidy %s readability-function-size %t -- -config='{CheckOptions: [{key: readability-function-size.LineThreshold, value: 0}, {key: readability-function-size.StatementThreshold, value: 0}, {key: readability-function-size.BranchThreshold, value: 0}, {key: readability-function-size.ParameterThreshold, value: 5}, {key: readability-function-size.NestingThreshold, value: 2}, {key: readability-function-size.VariableThreshold, value: 1}]}' -- -std=c++17

void structured_bindings() {
  int a[2] = {1, 2};
  auto [x, y] = a;
}
// CHECK-MESSAGES: :[[@LINE-4]]:6: warning: function 'structured_bindings' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-5]]:6: note: 3 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-6]]:6: note: 2 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-7]]:6: note: 3 variables (threshold 1)

#define SWAP(x, y) ({auto& [x0, x1] = x;  __typeof__(x) t = {x0, x1}; auto& [y0, y1] = y; auto& [t0, t1] = t; x0 = y0; x1 = y1; y0 = t0; y1 = t1; })
void variables_13() {
  int a[2] = {1, 2};
  int b[2] = {3, 4};
  SWAP(a, b);
}
// CHECK-MESSAGES: :[[@LINE-5]]:6: warning: function 'variables_13' exceeds recommended size/complexity thresholds [readability-function-size]
// CHECK-MESSAGES: :[[@LINE-6]]:6: note: 4 lines including whitespace and comments (threshold 0)
// CHECK-MESSAGES: :[[@LINE-7]]:6: note: 11 statements (threshold 0)
// CHECK-MESSAGES: :[[@LINE-8]]:6: note: 2 variables (threshold 1)
