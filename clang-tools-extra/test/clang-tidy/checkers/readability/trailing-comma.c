// RUN: %check_clang_tidy %s readability-trailing-comma %t

enum Color {
  Red,
  Green,
  Blue
};
// CHECK-MESSAGES: :[[@LINE-2]]:7: warning: enum should have a trailing comma
// CHECK-FIXES: enum Color {
// CHECK-FIXES-NEXT:   Red,
// CHECK-FIXES-NEXT:   Green,
// CHECK-FIXES-NEXT:   Blue,
// CHECK-FIXES-NEXT: };

enum SingleLine { A, B, C };
enum SingleLine2 { X1, Y1, };

struct Point {
  int x;
  int y;
};

struct Point p = {
  10,
  20
};
// CHECK-MESSAGES: :[[@LINE-2]]:5: warning: initializer list should have a trailing comma
// CHECK-FIXES: struct Point p = {
// CHECK-FIXES-NEXT:   10,
// CHECK-FIXES-NEXT:   20,
// CHECK-FIXES-NEXT: };

struct Point p2 = {
  .x = 1,
  .y = 2
};
// CHECK-MESSAGES: :[[@LINE-2]]:9: warning: initializer list should have a trailing comma
// CHECK-FIXES: struct Point p2 = {
// CHECK-FIXES-NEXT:   .x = 1,
// CHECK-FIXES-NEXT:   .y = 2,
// CHECK-FIXES-NEXT: };

int arr2[5] = {
  [0] = 1
};
// CHECK-MESSAGES: :[[@LINE-2]]:10: warning: initializer list should have a trailing comma
// CHECK-FIXES: int arr2[5] = {
// CHECK-FIXES-NEXT:   [0] = 1,
// CHECK-FIXES-NEXT: };

int multiArr[] = {
  1
};
// CHECK-MESSAGES: :[[@LINE-2]]:4: warning: initializer list should have a trailing comma
// CHECK-FIXES: int multiArr[] = {
// CHECK-FIXES-NEXT:   1,
// CHECK-FIXES-NEXT: };

int arr[] = {1, 2, 3};
struct Point p3 = {10, 20};
struct Point p4 = {.x = 1, .y = 2};
int matrix[2][2] = {{1, 2}, {3, 4}};

enum WithComma {
  X,
  Y,
};
