// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -cl-std=CL2.0 -triple amdgcn-unknown-unknown -target-cpu gfx1300 -target-feature -wavefrontsize64 \
// RUN: -emit-llvm -verify -o - %s

typedef int    v2i   __attribute__((ext_vector_type(2)));
typedef int    v4i   __attribute__((ext_vector_type(4)));

__attribute__((address_space(11))) void*
test_amdgcn_map_shared_rank(__local void* ptr, int rank, __attribute__((address_space(11))) const void* ptrsr,
                            __attribute__((address_space(11))) const int* gaddr, __attribute__((address_space(11))) const v2i* gaddr2,
                            __attribute__((address_space(11))) const v4i* gaddr4, local int* laddr, local v2i* laddr2, local v4i* laddr4)
{
  __builtin_amdgcn_query_shared_rank(ptrsr); // expected-error{{'__builtin_amdgcn_query_shared_rank' needs target feature gfx13-insts,wavefrontsize32}}
  __builtin_amdgcn_dds_load_async_to_lds_b32(gaddr, laddr, 16, 0); // expected-error{{'__builtin_amdgcn_dds_load_async_to_lds_b32' needs target feature gfx13-insts,wavefrontsize32}}
  __builtin_amdgcn_dds_load_async_to_lds_b64(gaddr2, laddr2, 16, 0); // expected-error{{'__builtin_amdgcn_dds_load_async_to_lds_b64' needs target feature gfx13-insts,wavefrontsize32}}
  __builtin_amdgcn_dds_load_async_to_lds_b128(gaddr4, laddr4, 16, 0); // expected-error{{'__builtin_amdgcn_dds_load_async_to_lds_b128' needs target feature gfx13-insts,wavefrontsize32}}
  return __builtin_amdgcn_map_shared_rank(ptr, rank); // expected-error{{'__builtin_amdgcn_map_shared_rank' needs target feature gfx13-insts,wavefrontsize32}}
}

