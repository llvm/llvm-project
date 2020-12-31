// RUN: %clang_cc1 -triple x86_64-apple-darwin %s -emit-llvm -o - | FileCheck %s

int* foo(int** a, int* b, int* c) {
return __sync_val_compare_and_swap (a, b, c);
}
// CHECK-LABEL: define{{.*}} i32* @foo
// CHECK: cmpxchg 

int foo2(int** a, int* b, int* c) {
return __sync_bool_compare_and_swap (a, b, c);
}
// CHECK-LABEL: define{{.*}} i32 @foo2
// CHECK: cmpxchg

int* foo3(int** a, int b) {
  return __sync_fetch_and_add (a, b);
}
// CHECK-LABEL: define{{.*}} i32* @foo3
// CHECK: atomicrmw add


int* foo4(int** a, int b) {
  return __sync_fetch_and_sub (a, b);
}
// CHECK-LABEL: define{{.*}} i32* @foo4
// CHECK: atomicrmw sub


int* foo5(int** a, int* b) {
  return __sync_lock_test_and_set (a, b);
}
// CHECK-LABEL: define{{.*}} i32* @foo5
// CHECK: atomicrmw xchg


int* foo6(int** a, int*** b) {
  return __sync_lock_test_and_set (a, b);
}
// CHECK-LABEL: define{{.*}} i32* @foo6
// CHECK: atomicrmw xchg
