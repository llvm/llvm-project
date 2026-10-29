// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -emit-llvm %s -triple x86_64-unknown-linux-gnu -o - | FileCheck %s

// musttail in noexcept functions should work when the callee is also noexcept.

int NoexceptCallee(int) noexcept;
int ThrowingFunc(int);

// CHECK-LABEL: define {{.*}} @_Z12TestNoexcepti(
// CHECK: musttail call {{.*}} @_Z14NoexceptCalleei(
// CHECK-NEXT: ret i32
int TestNoexcept(int x) noexcept {
  [[clang::musttail]] return NoexceptCallee(x);
}

// Noexcept caller with regular call to non-noexcept, then musttail to noexcept.
// CHECK-LABEL: define {{.*}} @_Z21TestMixedCallNoexcepti(
// CHECK: invoke {{.*}} @_Z12ThrowingFunci(
// CHECK-NEXT: to label %{{.*}} unwind label %terminate.lpad
// CHECK: musttail call {{.*}} @_Z14NoexceptCalleei(
// CHECK-NEXT: ret i32
// CHECK: terminate.lpad:
// CHECK: call void @__clang_call_terminate(
int TestMixedCallNoexcept(int x) noexcept {
  int y = ThrowingFunc(x);
  [[clang::musttail]] return NoexceptCallee(y);
}

// Noexcept caller musttails to itself (recursive).
// CHECK-LABEL: define {{.*}} @_Z13TestRecursivei(
// CHECK: musttail call {{.*}} @_Z13TestRecursivei(
// CHECK-NEXT: ret i32
int TestRecursive(int x) noexcept {
  [[clang::musttail]] return TestRecursive(x);
}
