// RUN: %check_clang_tidy %s performance-type-promotion-in-math-fn %t -- \
// RUN:   -config="{CheckOptions: {performance-type-promotion-in-math-fn.IncludeStyle: 'google'}}"

// CHECK-FIXES: #include <cmath>

double acos(double);

void check() {
  float a;
  acos(a);
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: call to 'acos' promotes float to double [performance-type-promotion-in-math-fn]
  // CHECK-FIXES: std::acos(a);
}
