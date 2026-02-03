// RUN: %clang_cc1 -triple x86_64-linux-gnu -emit-llvm %s -o - | llvm-cxxfilt | FileCheck %s

struct S {
  void *a();
};

// CHECK-LABEL: @S::a()
// CHECK: call ptr @llvm.stackaddress.p0()
void *S::a() {
  return __builtin_stack_address();
}

// CHECK-LABEL: define {{[^@]+}} @two()
// CHECK: call {{[^@]+}} @"two()::$_0::operator()() const"
//
// CHECK-LABEL: define {{[^@]+}} @"two()::$_0::operator()() const"
// CHECK: [[PTR:%.*]] = call ptr @llvm.stackaddress.p0()
// CHECK: ret ptr [[PTR]]
void *two() {
  auto l = []() { return __builtin_stack_address(); };
  return l();
}
