// RUN: %clang_cc1 -emit-llvm -x hip %s -o - -triple x86_64-linux-gnu \
// RUN:   | FileCheck -check-prefixes=CHECK,HOST %s
// RUN: %clang_cc1 -emit-llvm -x hip %s -o - -triple amdgcn-amd-amdhsa -fcuda-is-device \
// RUN:   | FileCheck -check-prefixes=CHECK,DEV %s

#include "Inputs/cuda.h"

// CHECK: %class.anon = type { ptr, float, ptr, ptr }
// CHECK: %class.anon.0 = type { ptr, float, ptr, ptr }
// CHECK: %class.anon.1 = type { ptr, ptr, ptr }
// CHECK: %class.anon.2 = type { ptr, float, ptr, ptr }

// HOST: call void @_ZN8DevByVal21__device_stub__kernelIZNS_4testEPKfS2_PfEUljE_EEvT_(ptr noundef byval(%class.anon)
// DEV: define amdgpu_kernel void @_ZN8DevByVal6kernelIZNS_4testEPKfS2_PfEUljE_EEvT_(ptr addrspace(4) noundef byref(%class.anon)

// Only the device function passes arugments by value.
namespace DevByVal {
__device__ float fun(float x, float y) {
  return x;
}

float fun(const float &x, const float &y) {
  return x;
}

template<typename F>
void __global__ kernel(F f)
{
  f(1);
}

void test(float const * fl, float const * A, float * Vf)
{
  float constexpr small(1.0e-25);

  auto lambda = [=] __device__ __host__ (unsigned int n) {
    float const value = fun(small, fl[0]);
    Vf[0] = value * A[0];
  };
  kernel<<<1, 1>>>(lambda);
}
}

// HOST: call void @_ZN9HostByVal21__device_stub__kernelIZNS_4testEPKfS2_PfEUljE_EEvT_(ptr noundef byval(%class.anon.0)
// DEV: define amdgpu_kernel void @_ZN9HostByVal6kernelIZNS_4testEPKfS2_PfEUljE_EEvT_(ptr addrspace(4) noundef byref(%class.anon.0)

// Only the host function passes arugments by value.
namespace HostByVal {
float fun(float x, float y) {
  return x;
}

__device__ float fun(const float &x, const float &y) {
  return x;
}

template<typename F>
void __global__ kernel(F f)
{
  f(1);
}

void test(float const * fl, float const * A, float * Vf)
{
  float constexpr small(1.0e-25);

  auto lambda = [=] __device__ __host__ (unsigned int n) {
    float const value = fun(small, fl[0]);
    Vf[0] = value * A[0];
  };
  kernel<<<1, 1>>>(lambda);
}
}

// HOST: call void @_ZN9BothByVal21__device_stub__kernelIZNS_4testEPKfS2_PfEUljE_EEvT_(ptr noundef byval(%class.anon.1)
// DEV: define amdgpu_kernel void @_ZN9BothByVal6kernelIZNS_4testEPKfS2_PfEUljE_EEvT_(ptr addrspace(4) noundef byref(%class.anon.1)

// Both the host and device functions pass arugments by value.
namespace BothByVal {
float fun(float x, float y) {
  return x;
}

__device__ float fun(float x, float y) {
  return x;
}

template<typename F>
void __global__ kernel(F f)
{
  f(1);
}

void test(float const * fl, float const * A, float * Vf)
{
  float constexpr small(1.0e-25);

  auto lambda = [=] __device__ __host__ (unsigned int n) {
    float const value = fun(small, fl[0]);
    Vf[0] = value * A[0];
  };
  kernel<<<1, 1>>>(lambda);
}
}

// HOST: call void @_ZN12NeitherByVal21__device_stub__kernelIZNS_4testEPKfS2_PfEUljE_EEvT_(ptr noundef byval(%class.anon.2)
// DEV: define amdgpu_kernel void @_ZN12NeitherByVal6kernelIZNS_4testEPKfS2_PfEUljE_EEvT_(ptr addrspace(4) noundef byref(%class.anon.2)

// Neither the host nor device function passes arugments by value.
namespace NeitherByVal {
float fun(const float& x, const float& y) {
  return x;
}

__device__ float fun(const float& x, const float& y) {
  return x;
}

template<typename F>
void __global__ kernel(F f)
{
  f(1);
}

void test(float const * fl, float const * A, float * Vf)
{
  float constexpr small(1.0e-25);

  auto lambda = [=] __device__ __host__ (unsigned int n) {
    float const value = fun(small, fl[0]);
    Vf[0] = value * A[0];
  };
  kernel<<<1, 1>>>(lambda);
}
}
