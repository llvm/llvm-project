// RUN: %check_clang_tidy -std=c++11-or-later %s readability-trailing-comma %t

enum E0 {};

enum E1 { A };
// CHECK-MESSAGES: :[[@LINE-1]]:12: warning: enum should have a trailing comma
// CHECK-FIXES: enum E1 { A, };

enum E2 { B, C };
// CHECK-MESSAGES: :[[@LINE-1]]:15: warning: enum should have a trailing comma
// CHECK-FIXES: enum E2 { B, C, };

// Init lists: default InitListThreshold=3
void f() {
  int a[] = {1};       // No warning - only 1 element
  int b[] = {1, 2};    // No warning - only 2 elements
  int c[] = {1, 2, 3};
  // CHECK-MESSAGES: :[[@LINE-1]]:21: warning: initializer list should have a trailing comma
  // CHECK-FIXES: int c[] = {1, 2, 3,};
}
