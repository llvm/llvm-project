// RUN: %clang_cc1 -triple amdgpu11.00-amd-amdhsa -fsyntax-only -verify -x c %s
// RUN: %clang_cc1 -triple amdgpu11.00-amd-amdhsa -fsyntax-only -verify -x c++ %s
// RUN: %clang_cc1 -triple amdgpu11.00-amd-amdhsa -fcuda-is-device -x hip \
// RUN:   -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple amdgpu11.00-amd-amdhsa -cl-std=CL1.2 -fsyntax-only \
// RUN:   -verify -x cl %s
// RUN: %clang_cc1 -triple amdgpu11.00-amd-amdhsa -cl-std=CL2.0 -fsyntax-only \
// RUN:   -verify -x cl %s
// RUN: %clang_cc1 -triple amdgpu11.00-amd-amdhsa -cl-std=CL3.0 -fsyntax-only \
// RUN:   -verify -x cl %s

// expected-no-diagnostics

#if defined(__HIP__)
__attribute__((device))
#endif
void test(void *flat_ptr,
          void __attribute__((address_space(0))) *flat_addrspace,
          void __attribute__((address_space(1))) *global_addrspace,
          void __attribute__((address_space(3))) *local_addrspace,
          void __attribute__((address_space(5))) *private_addrspace,
          void __attribute__((opencl_global)) *global_ptr,
          void __attribute__((opencl_local)) *local_ptr,
          void __attribute__((opencl_private)) *private_ptr,
          void __attribute__((opencl_constant)) *constant_ptr,
          void __attribute__((opencl_generic)) *generic_ptr) {
  (void)__builtin_amdgcn_is_shared(flat_ptr);
  (void)__builtin_amdgcn_is_private(flat_ptr);
  (void)__builtin_amdgcn_is_shared(flat_addrspace);
  (void)__builtin_amdgcn_is_private(flat_addrspace);
  (void)__builtin_amdgcn_is_shared(global_addrspace);
  (void)__builtin_amdgcn_is_private(global_addrspace);
  (void)__builtin_amdgcn_is_shared(local_addrspace);
  (void)__builtin_amdgcn_is_private(local_addrspace);
  (void)__builtin_amdgcn_is_shared(private_addrspace);
  (void)__builtin_amdgcn_is_private(private_addrspace);
  (void)__builtin_amdgcn_is_shared(global_ptr);
  (void)__builtin_amdgcn_is_private(global_ptr);
  (void)__builtin_amdgcn_is_shared(local_ptr);
  (void)__builtin_amdgcn_is_private(local_ptr);
  (void)__builtin_amdgcn_is_shared(private_ptr);
  (void)__builtin_amdgcn_is_private(private_ptr);
  (void)__builtin_amdgcn_is_shared(constant_ptr);
  (void)__builtin_amdgcn_is_private(constant_ptr);
  (void)__builtin_amdgcn_is_shared(generic_ptr);
  (void)__builtin_amdgcn_is_private(generic_ptr);
}
