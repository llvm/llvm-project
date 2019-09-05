// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu tahiti -verify -S -o - %s

void test_flat_address_space_builtins(int* ptr)
{
  (void)__builtin_amdgcn_is_shared(ptr); // expected-error {{'__builtin_amdgcn_is_shared' needs target feature flat-address-space}}
  (void)__builtin_amdgcn_is_private(ptr); // expected-error {{'__builtin_amdgcn_is_private' needs target feature flat-address-space}}
}
