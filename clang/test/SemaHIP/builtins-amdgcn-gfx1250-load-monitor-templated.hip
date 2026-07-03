// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -target-cpu gfx1250 -fsyntax-only -fcuda-is-device -verify %s

#define __device__ __attribute__((device))

typedef int    v4i   __attribute__((ext_vector_type(4)));

template<unsigned AO, unsigned Scope>
__device__ v4i templated_amdgcn_flat_load_monitor_b128(v4i* inptr)
{
  // expected-error@+2 {{synchronization scope argument to atomic operation is invalid}}
  // expected-warning@+1 {{memory order argument to atomic operation is invalid}}
  return __builtin_amdgcn_flat_load_monitor_b128(inptr, AO, Scope);
}

__device__ void test_amdgcn_flat_load_monitor_b128_invalid_ao_from_template(v4i* inptr, v4i *out)
{
  // expected-note@+1 {{in instantiation of function template specialization 'templated_amdgcn_flat_load_monitor_b128<42U, 0U>' requested here}}
  *out = templated_amdgcn_flat_load_monitor_b128<42, __MEMORY_SCOPE_SYSTEM>(inptr);
}

__device__ void test_amdgcn_flat_load_monitor_b128_invalid_sc_from_template(v4i* inptr, v4i *out)
{
  // expected-note@+1 {{in instantiation of function template specialization 'templated_amdgcn_flat_load_monitor_b128<5U, 42U>' requested here}}
  *out = templated_amdgcn_flat_load_monitor_b128<__ATOMIC_SEQ_CST, 42>(inptr);
}
