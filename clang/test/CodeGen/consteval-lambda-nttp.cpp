// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -emit-llvm -o - %s | FileCheck %s

consteval int foo() {
  return 42;
}

template <auto Fn>
int bar() {
  return Fn();
}

int test() {
  return bar<[] { return foo(); }>();
}

// CHECK-NOT: @_Z3foov
// CHECK: define{{.*}} i32 @_Z4testv
// CHECK-NOT: @_Z3foov
