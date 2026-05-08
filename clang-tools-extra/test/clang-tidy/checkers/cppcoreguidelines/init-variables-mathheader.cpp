// RUN: %check_clang_tidy %s cppcoreguidelines-init-variables %t -- \
// RUN:   -config="{CheckOptions: { \
// RUN:     cppcoreguidelines-init-variables.IncludeStyle: 'google', \
// RUN:     cppcoreguidelines-init-variables.MathHeader: '<cmath>' \
// RUN:   }}" -- -fexceptions

// CHECK-FIXES: #include <cmath>

void init_unit_tests() {
  float f;
  // CHECK-MESSAGES: :[[@LINE-1]]:9: warning: variable 'f' is not initialized [cppcoreguidelines-init-variables]
  // CHECK-FIXES: float f = NAN;
}
