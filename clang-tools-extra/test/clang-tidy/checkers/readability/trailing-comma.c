// RUN: %check_clang_tidy %s readability-trailing-comma %t -- \
// RUN:   -config='{CheckOptions: {readability-trailing-comma.InitListThreshold: 1}}'

enum Color { Red, Green, Blue };
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: enum should have a trailing comma
// CHECK-FIXES: enum Color { Red, Green, Blue, };

enum SingleValue { Only };
// CHECK-MESSAGES: :[[@LINE-1]]:24: warning: enum should have a trailing comma
// CHECK-FIXES: enum SingleValue { Only, };

int arr[] = {1, 2, 3};
// CHECK-MESSAGES: :[[@LINE-1]]:21: warning: initializer list should have a trailing comma
// CHECK-FIXES: int arr[] = {1, 2, 3,};

struct Point {
  int x;
  int y;
};

struct Point p = {10, 20};
// CHECK-MESSAGES: :[[@LINE-1]]:25: warning: initializer list should have a trailing comma
// CHECK-FIXES: struct Point p = {10, 20,};

struct Point p2 = {.x = 1, .y = 2};
// CHECK-MESSAGES: :[[@LINE-1]]:34: warning: initializer list should have a trailing comma
// CHECK-FIXES: struct Point p2 = {.x = 1, .y = 2,};

struct Point p3 = {.x = 5, 10};
// CHECK-MESSAGES: :[[@LINE-1]]:30: warning: initializer list should have a trailing comma
// CHECK-FIXES: struct Point p3 = {.x = 5, 10,};

int arr2[5] = {[0] = 1, [4] = 5};
// CHECK-MESSAGES: :[[@LINE-1]]:32: warning: initializer list should have a trailing comma
// CHECK-FIXES: int arr2[5] = {[0] = 1, [4] = 5,};

int matrix[2][2] = {{1, 2}, {3, 4}};
// CHECK-MESSAGES: :[[@LINE-1]]:26: warning: initializer list should have a trailing comma
// CHECK-MESSAGES: :[[@LINE-2]]:34: warning: initializer list should have a trailing comma
// CHECK-MESSAGES: :[[@LINE-3]]:35: warning: initializer list should have a trailing comma
