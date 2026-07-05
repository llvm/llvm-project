// REQUIRES: amdgpu-registered-target
// RUN: %clang_cc1 -triple amdgcn-- -target-cpu tahiti -verify -S -o - %s

// expected-no-diagnostics

// Make sure no warning is produced on due to dead "flat-address-space" feature.
__attribute__((target("flat-address-space")))
void test_flat_address_space_builtins(int* ptr)
{
  (void)__builtin_amdgcn_is_shared(ptr);
  (void)__builtin_amdgcn_is_private(ptr);
}
