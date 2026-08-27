// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 %s -emit-llvm -O0 -verify -o - \
// RUN:   -triple=amdgcn-amd-amdhsa | FileCheck %s

// Test that [[clang::amdgpu_av("none")]] on non-atomic statements emits a
// warning and does NOT produce !mmra metadata.

// CHECK-LABEL: define {{.*}} @_Z16test_plain_storePii(
// CHECK-NOT: !mmra
// CHECK: ret void
void test_plain_store(int *p, int val) {
  [[clang::amdgpu_av("none")]] *p = val; // expected-warning {{'clang::amdgpu_av' attribute only applies to atomic expressions or Clang atomic builtins}}
}

// CHECK-LABEL: define {{.*}} @_Z15test_plain_callv(
// CHECK-NOT: !mmra
// CHECK: ret void
extern void foo();
void test_plain_call() {
  [[clang::amdgpu_av("none")]] foo(); // expected-warning {{'clang::amdgpu_av' attribute only applies to atomic expressions or Clang atomic builtins}}
}

// CHECK-LABEL: define {{.*}} @_Z18test_for_with_atomPi(
// CHECK-NOT: !mmra
// CHECK: ret void
void test_for_with_atom(int *p) {
  // The attribute on a for loop should warn even if the body contains atomics.
  [[clang::amdgpu_av("none")]] for (;;) { // expected-warning {{'clang::amdgpu_av' attribute can only be applied to an expression}}
    __atomic_fetch_add(p, 1, __ATOMIC_SEQ_CST);
    break;
  }
}

// The attribute on an if statement should warn even if the condition is atomic.
// CHECK-LABEL: define {{.*}} @_Z20test_if_atomic_condnPi(
// CHECK-NOT: !mmra
// CHECK: ret void
void test_if_atomic_condn(int *p) {
  [[clang::amdgpu_av("none")]] if (__atomic_load_n(p, __ATOMIC_ACQUIRE)) { // expected-warning {{'clang::amdgpu_av' attribute can only be applied to an expression}}
  }
}

// CHECK-NOT: amdgcn-av
