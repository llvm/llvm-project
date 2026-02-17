// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1250 -fsyntax-only -fcuda-is-device -verify %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1251 -fsyntax-only -fcuda-is-device -verify %s

#define __device__ __attribute__((device))

typedef int    v2i   __attribute__((ext_vector_type(2)));
typedef int    v4i   __attribute__((ext_vector_type(4)));

template<unsigned AO>
__device__ void template_cooperative_atomic_store_32x4B(int* gaddr, int val) {
  // expected-warning@+1 {{memory order argument to atomic operation is invalid}}
  __builtin_amdgcn_cooperative_atomic_store_32x4B(gaddr, val, AO, "agent");
}

__device__ void test_amdgcn_cooperative_atomic_store_32x4B_invalid_ao(int* gaddr, int val)
{
  // expected-note@+1 {{in instantiation of function template specialization 'template_cooperative_atomic_store_32x4B<42U>' requested here}}
  template_cooperative_atomic_store_32x4B<42>(gaddr, val);
}

template<unsigned AO>
__device__ int template_cooperative_atomic_load_32x4B(int* gaddr) {
  // expected-warning@+1 {{memory order argument to atomic operation is invalid}}
  return __builtin_amdgcn_cooperative_atomic_load_32x4B(gaddr, AO, "");
}

__device__ void test_amdgcn_cooperative_atomic_load_32x4B_invalid_ao(int* addr, int *out)
{
  // expected-note@+1 {{in instantiation of function template specialization 'template_cooperative_atomic_load_32x4B<42U>' requested here}}
  *out = template_cooperative_atomic_load_32x4B<42>(addr);
}
