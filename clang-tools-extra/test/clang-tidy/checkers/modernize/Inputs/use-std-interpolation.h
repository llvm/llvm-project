#ifndef CLANG_TIDY_TEST_USE_STD_INTERPOLATION_H
#define CLANG_TIDY_TEST_USE_STD_INTERPOLATION_H

#include <type_traits>
// CHECK-FIXES: #include <cmath>
// CHECK-FIXES-NEXT: #include <numeric>
// CHECK-FIXES-NEXT: #include <type_traits>

inline double header_midpoint(double a, double b) {
  return (a + b) / 2;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::midpoint' instead of manual midpoint calculation
  // CHECK-FIXES: return std::midpoint(a, b);
}
inline double header_interpolation(double a, double b, double t) {
  return a + (b - a) * t;
  // CHECK-MESSAGES: :[[@LINE-1]]:10: warning: use 'std::lerp' instead of manual linear interpolation
  // CHECK-FIXES: return std::lerp(a, b, t);
}
#endif
