// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -Wno-unreachable-code -Werror -triple x86_64-linux-gnu -emit-llvm -o - %s | FileCheck %s

int val = 42;
int& test1() {
  return throw val, val;
}

int test2() {
  return val ? throw val : val;
}

// rdar://problem/8608801
void test3() {
  throw false;
}

// PR10582
int test4() {
  return 1 ? throw val : val;
}

// PR15923
int test5(bool x, bool y, int z) {
  return (x ? throw 1 : y) ? z : throw 2;
}
// CHECK: define i32 @_Z5test5bbi(
// CHECK: br i1
//
// x.true:
// CHECK: call void @__cxa_throw(
// CHECK-NEXT: unreachable
//
// x.false:
// CHECK: br i1
//
// y.true:
// CHECK: load i32*
// CHECK: br label
//
// y.false:
// CHECK: call void @__cxa_throw(
// CHECK-NEXT: unreachable
//
// end:
// CHECK: ret i32
