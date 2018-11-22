// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s -O2 | FileCheck %s

int a = 42;

inline int bcp(int x) {
  return __builtin_constant_p(x);
}

/* --- Compound literals */

struct foo { int x, y; };

struct foo test0(int expr) {
  // CHECK: define i64 @test0(i32 %expr)
  // CHECK: call i1 @llvm.is.constant.i32(i32 %expr)
  struct foo f = (struct foo){ __builtin_constant_p(expr), 42 };
  return f;
}

/* --- Pointer types */

inline int test1_i(int *x) {
  return *x;
}

int test1() {
  // CHECK: define i32 @test1
  // CHECK: add nsw i32 %0, -13
  // CHECK-NEXT: call i1 @llvm.is.constant.i32(i32 %sub)
  return bcp(test1_i(&a) - 13);
}

int test2() {
  // CHECK: define i32 @test2
  // CHECK: ret i32 0
  return __builtin_constant_p(&a - 13);
}

inline int test3_i(int *x) {
  return 42;
}

int test3() {
  // CHECK: define i32 @test3
  // CHECK: ret i32 1
  return bcp(test3_i(&a) - 13);
}

/* --- Aggregate types */

int b[] = {1, 2, 3};

int test4() {
  // CHECK: define i32 @test4
  // CHECK: ret i32 0
  return __builtin_constant_p(b);
}

const char test5_c[] = {1, 2, 3, 0};

int test5() {
  // CHECK: define i32 @test5
  // CHECK: ret i32 0
  return __builtin_constant_p(test5_c);
}

inline char test6_i(const char *x) {
  return x[1];
}

int test6() {
  // CHECK: define i32 @test6
  // CHECK: ret i32 0
  return __builtin_constant_p(test6_i(test5_c));
}

/* --- Non-constant global variables */

int test7() {
  // CHECK: define i32 @test7
  // CHECK: call i1 @llvm.is.constant.i32(i32 %0)
  return bcp(a);
}

/* --- Constant global variables */

const int c = 42;

int test8() {
  // CHECK: define i32 @test8
  // CHECK: ret i32 1
  return bcp(c);
}

/* --- Array types */

int arr[] = { 1, 2, 3 };
const int c_arr[] = { 1, 2, 3 };

int test9() {
  // CHECK: define i32 @test9
  // CHECK: call i1 @llvm.is.constant.i32(i32 %0)
  return __builtin_constant_p(arr[2]);
}

int test10() {
  // CHECK: define i32 @test10
  // CHECK: ret i32 1
  return __builtin_constant_p(c_arr[2]);
}

int test11() {
  // CHECK: define i32 @test11
  // CHECK: ret i32 0
  return __builtin_constant_p(c_arr);
}

/* --- Function pointers */

int test12() {
  // CHECK: define i32 @test12
  // CHECK: ret i32 0
  return __builtin_constant_p(&test10);
}

int test13() {
  // CHECK: define i32 @test13
  // CHECK: ret i32 1
  return __builtin_constant_p(&test10 != 0);
}
