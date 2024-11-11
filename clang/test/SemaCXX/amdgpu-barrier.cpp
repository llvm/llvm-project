// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -fsyntax-only -verify -std=gnu++11 -triple amdgcn -Wno-unused-value %s

void foo() {
  int n = 100;
  __amdgpu_named_workgroup_barrier_t v = 0; // expected-error {{cannot initialize a variable of type '__amdgpu_named_workgroup_barrier_t' with an rvalue of type 'int'}}
  static_cast<__amdgpu_named_workgroup_barrier_t>(n); // expected-error {{static_cast from 'int' to '__amdgpu_named_workgroup_barrier_t' is not allowed}}
  dynamic_cast<__amdgpu_named_workgroup_barrier_t>(n); // expected-error {{invalid target type '__amdgpu_named_workgroup_barrier_t' for dynamic_cast; target type must be a reference or pointer type to a defined class}}
  reinterpret_cast<__amdgpu_named_workgroup_barrier_t>(n); // expected-error {{reinterpret_cast from 'int' to '__amdgpu_named_workgroup_barrier_t' is not allowed}}
  int c(v); // expected-error {{cannot initialize a variable of type 'int' with an lvalue of type '__amdgpu_named_workgroup_barrier_t'}}
  __amdgpu_named_workgroup_barrier_t k;
  int *ip = (int *)k; // expected-error {{cannot cast from type '__amdgpu_named_workgroup_barrier_t' to pointer type 'int *'}}
  void *vp = (void *)k; // expected-error {{cannot cast from type '__amdgpu_named_workgroup_barrier_t' to pointer type 'void *'}}
}

static_assert(sizeof(__amdgpu_named_workgroup_barrier_t) == 16, "wrong size");
static_assert(alignof(__amdgpu_named_workgroup_barrier_t) == 4, "wrong alignment");
