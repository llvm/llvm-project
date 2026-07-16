// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -x hip -fsyntax-only -verify=noaux %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple amdgcn-amd-amdhsa -x hip -fsyntax-only -verify=aux %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip -fsyntax-only -verify=device %s

// noaux-no-diagnostics
// aux-no-diagnostics
// device-no-diagnostics

#include "Inputs/cuda.h"

__device__ int device_var;
__device__ int device_array[4];
__device__ int *device_ptr;
__constant__ int constant_var;
__device__ const int const_device_var = 1;

static_assert(__builtin_pointee_address_space(&device_var) ==
              __CLANG_ADDRESS_SPACE_HIP_DEVICE);
static_assert(__builtin_pointee_address_space(device_array) ==
              __CLANG_ADDRESS_SPACE_HIP_DEVICE);
static_assert(__builtin_pointee_address_space(&device_array) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__builtin_pointee_address_space(device_array + 1) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__builtin_pointee_address_space(&constant_var) ==
              __CLANG_ADDRESS_SPACE_HIP_CONSTANT);
static_assert(__builtin_pointee_address_space(&const_device_var) ==
              __CLANG_ADDRESS_SPACE_HIP_CONSTANT);
static_assert(__builtin_pointee_address_space((int *)&constant_var) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__builtin_pointee_address_space(device_ptr) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);

void host_queries() {
  (void)__builtin_pointee_address_space(&device_var);
  (void)__builtin_pointee_address_space(device_array);
  (void)__builtin_pointee_address_space(&device_array);
  (void)__builtin_pointee_address_space(device_array + 1);
  (void)__builtin_pointee_address_space(&constant_var);
  (void)__builtin_pointee_address_space(&const_device_var);
  (void)__builtin_pointee_address_space((int *)&constant_var);
  (void)__builtin_pointee_address_space(device_ptr);
}
