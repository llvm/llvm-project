// RUN: rm -rf %t
// RUN: mkdir %t
// RUN: cp %S/Inputs/performance-unnecessary-value-param/header.h %t/header.h
// RUN: %check_clang_tidy %s performance-unnecessary-value-param %t/temp -- -- -std=c++11 -I %t
// RUN: diff %t/header.h %S/Inputs/performance-unnecessary-value-param/header-fixed.h

#include "header.h"



int f1(int n, ABC v1, ABC v2) {
  // CHECK-MESSAGES: [[@LINE-1]]:19: warning: the parameter 'v1' is copied for each invocation but only used as a const reference; consider making it a const reference [performance-unnecessary-value-param]
  // CHECK-MESSAGES: [[@LINE-2]]:27: warning: the parameter 'v2' is copied for each invocation but only used as a const reference; consider making it a const reference [performance-unnecessary-value-param]
  // CHECK-FIXES: int f1(int n, const ABC& v1, const ABC& v2) {
  return v1.get(n) + v2.get(n);
}
int f2(int n, ABC v2) {
  // CHECK-MESSAGES: [[@LINE-1]]:19: warning: the parameter 'v2' is copied for each invocation but only used as a const reference; consider making it a const reference [performance-unnecessary-value-param]
  // CHECK-FIXES: int f2(int n, const ABC& v2) {
}
