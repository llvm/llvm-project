// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple amdgcn-amd-amdhsa -fsyntax-only -verify=host -xhip %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fsyntax-only -fcuda-is-device -verify=dev -xhip %s

// dev-no-diagnostics

#include "Inputs/cuda.h"

__global__ void kernel() {                         
  __attribute__((address_space(0))) void *mem_ptr;
  (void)__builtin_amdgcn_is_shared(mem_ptr);
}

template<typename T>
__global__ void template_kernel(T *p) {                         
  __attribute__((address_space(0))) void *mem_ptr;
  (void)__builtin_amdgcn_is_shared(mem_ptr);
}

void hfun() {
  __attribute__((address_space(0))) void *mem_ptr;
  (void)__builtin_amdgcn_is_shared(mem_ptr); // host-error {{reference to __device__ function '__builtin_amdgcn_is_shared' in __host__ function}}
}

template<typename T>
void template_hfun(T *p) {
  __attribute__((address_space(0))) void *mem_ptr;
  (void)__builtin_amdgcn_is_shared(mem_ptr); // host-error {{reference to __device__ function '__builtin_amdgcn_is_shared' in __host__ function}}
}


int main() {
  int *p;
  kernel<<<1,1>>>();
  template_kernel<<<1,1>>>(p);
  template_hfun(p); // host-note {{called by 'main'}}
}
