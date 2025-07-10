// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=amdgcn-amd-amdhsa -triple amdgcn-amd-amdhsa -fopenmp-is-target-device -Wno-unused-value %s

void foo() {
#pragma omp target
  {
    int n = 100;
    __amdgpu_buffer_rsrc_t v = 0; // expected-error {{cannot initialize a variable of type '__amdgpu_buffer_rsrc_t' with an rvalue of type 'int'}}
    static_cast<__amdgpu_buffer_rsrc_t>(n); // expected-error {{static_cast from 'int' to '__amdgpu_buffer_rsrc_t' is not allowed}}
    dynamic_cast<__amdgpu_buffer_rsrc_t>(n); // expected-error {{invalid target type '__amdgpu_buffer_rsrc_t' for dynamic_cast; target type must be a reference or pointer type to a defined class}}
    reinterpret_cast<__amdgpu_buffer_rsrc_t>(n); // expected-error {{reinterpret_cast from 'int' to '__amdgpu_buffer_rsrc_t' is not allowed}}
    int c(v); // expected-error {{cannot initialize a variable of type 'int' with an lvalue of type '__amdgpu_buffer_rsrc_t'}}
    __amdgpu_buffer_rsrc_t k;
    int *ip = (int *)k; // expected-error {{cannot cast from type '__amdgpu_buffer_rsrc_t' to pointer type 'int *'}}
    void *vp = (void *)k; // expected-error {{cannot cast from type '__amdgpu_buffer_rsrc_t' to pointer type 'void *'}}
  }
 }
