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
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: enum should not have a trailing comma
// CHECK-FIXES: enum SingleLine2 { X1, Y1 };
enum SingleVal { VAL1 };
enum SingleVal2 { VAL2, };
// CHECK-MESSAGES: :[[@LINE-1]]:23: warning: enum should not have a trailing comma
// CHECK-FIXES: enum SingleVal2 { VAL2 };
enum SingleVal3 {
  VAL3
};
// CHECK-MESSAGES: :[[@LINE-2]]:7: warning: enum should have a trailing comma
// CHECK-FIXES: enum SingleVal3 {
// CHECK-FIXES-NEXT:   VAL3,
// CHECK-FIXES-NEXT: };

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
int single1[1] = {42};
int single2[1] = {42,};
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: initializer list should not have a trailing comma
// CHECK-FIXES: int single2[1] = {42};
int single3[1] = {
  42
};
// CHECK-MESSAGES: :[[@LINE-2]]:5: warning: initializer list should have a trailing comma
// CHECK-FIXES: int single3[1] = {
// CHECK-FIXES-NEXT:   42,
// CHECK-FIXES-NEXT: };
int single4[1] = {
  42,
};
int empty1[] = {};
struct Nested { int a[2]; int b; };
struct Nested nest1 = {{1}, 2};
struct Nested nest2 = {{1,}, 2};
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: initializer list should not have a trailing comma
// CHECK-FIXES: struct Nested nest2 = {{[{][{]}}1}, 2};
struct Nested nest3 = {
  {1},
  2
};
// CHECK-MESSAGES: :[[@LINE-2]]:4: warning: initializer list should have a trailing comma
// CHECK-FIXES: struct Nested nest3 = {
// CHECK-FIXES-NEXT:   {1},
// CHECK-FIXES-NEXT:   2,
// CHECK-FIXES-NEXT: };

struct Point singleDesig1 = {.x = 10};
struct Point singleDesig2 = {.x = 10,};
// CHECK-MESSAGES: :[[@LINE-1]]:37: warning: initializer list should not have a trailing comma
// CHECK-FIXES: struct Point singleDesig2 = {.x = 10};
struct Point singleDesig3 = {};
struct Point singleDesig4 = {
  .x = 10
};
// CHECK-MESSAGES: :[[@LINE-2]]:10: warning: initializer list should have a trailing comma
// CHECK-FIXES: struct Point singleDesig4 = {
// CHECK-FIXES-NEXT:   .x = 10,
// CHECK-FIXES-NEXT: };
