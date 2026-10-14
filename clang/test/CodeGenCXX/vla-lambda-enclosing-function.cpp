// RUN: %clang_cc1 -triple x86_64-linux-gnu -std=c++20 -Wno-vla-cxx-extension -emit-llvm -o - %s | FileCheck %s

// A lambda that declares a variable of a VLA type from the enclosing function
// captures the array bound.

int foo();

// CHECK-LABEL: define {{.*}} @_Z9by_copy_tv(
// CHECK: [[SIZE:%.*]] = call noundef i32 @_Z3foov()
// CHECK: [[EXT:%.*]] = zext i32 [[SIZE]] to i64
// CHECK: store i64 [[EXT]], ptr {{%.*}}
// CHECK-LABEL: define internal {{.*}} @"_ZZ9by_copy_tvENK3$_0clEv"(
// CHECK: [[BOUND:%[0-9]+]] = load i64, ptr {{%[0-9]+}}
// CHECK: alloca i32, i64 [[BOUND]]
int by_copy_t() {
  using X = int[foo()];
  return [] { X x; return (int)sizeof(x); }();
}

// CHECK-LABEL: define {{.*}} @_Z8nested_tv(
// CHECK-LABEL: define internal {{.*}} @"_ZZZ8nested_tvENK3$_0clEvENKUlvE_clEv"(
// CHECK: [[BOUND:%[0-9]+]] = load i64, ptr {{%[0-9]+}}
// CHECK: alloca i32, i64 [[BOUND]]
int nested_t() {
  using X = int[foo()];
  return [&] { return [&] { X x; return (int)sizeof(x); }(); }();
}
