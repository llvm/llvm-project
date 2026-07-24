// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -std=c++20 -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,CUDA %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip -std=c++20 -emit-llvm -o - %s | FileCheck --check-prefixes=CHECK,HIP %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple amdgcn-amd-amdhsa -x hip -std=c++20 -DHOST_TEST -emit-llvm -o - %s | FileCheck --check-prefix=HOST %s

#include "Inputs/cuda.h"

__device__ int device_var;
__device__ int device_array[4];
__device__ int *device_ptr;
__constant__ int constant_var;
__device__ const int const_device_var = 1;

#if defined(__HIP__)
#define EXPECTED_DEVICE_ADDRESS_SPACE __CLANG_ADDRESS_SPACE_HIP_DEVICE
#define EXPECTED_CONSTANT_ADDRESS_SPACE __CLANG_ADDRESS_SPACE_HIP_CONSTANT
#define EXPECTED_SHARED_ADDRESS_SPACE __CLANG_ADDRESS_SPACE_HIP_SHARED
#else
#define EXPECTED_DEVICE_ADDRESS_SPACE __CLANG_ADDRESS_SPACE_CUDA_DEVICE
#define EXPECTED_CONSTANT_ADDRESS_SPACE __CLANG_ADDRESS_SPACE_CUDA_CONSTANT
#define EXPECTED_SHARED_ADDRESS_SPACE __CLANG_ADDRESS_SPACE_CUDA_SHARED
#endif

static_assert(__addrspaceof(device_ptr) == EXPECTED_DEVICE_ADDRESS_SPACE);
static_assert(__addrspaceof(*device_ptr) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__addrspaceof((device_ptr)) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__addrspaceof((device_var)) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);

#ifdef HOST_TEST

template <class T> consteval int host_consteval_address_space(T *p) {
  return __addrspaceof(*p);
}

template <class T> constexpr int host_constexpr_address_space(T *p) {
  return __addrspaceof(*p);
}

static_assert(__addrspaceof(device_var) ==
              EXPECTED_DEVICE_ADDRESS_SPACE);
static_assert(__addrspaceof(constant_var) ==
              EXPECTED_CONSTANT_ADDRESS_SPACE);
static_assert(__addrspaceof(const_device_var) ==
              EXPECTED_CONSTANT_ADDRESS_SPACE);
static_assert(__addrspaceof(device_array) ==
              EXPECTED_DEVICE_ADDRESS_SPACE);
static_assert(__addrspaceof(*&device_array) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__addrspaceof(*(device_array + 1)) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(host_consteval_address_space(&device_var) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(host_consteval_address_space(&constant_var) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(host_consteval_address_space(&const_device_var) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(host_consteval_address_space((int *)&constant_var) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(host_constexpr_address_space(&constant_var) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);

extern "C" int test_host_device_var() {
  return __addrspaceof(device_var);
}

// HOST-LABEL: define{{.*}} i32 @test_host_device_var(
// HOST: ret i32 23

extern "C" int test_host_constant_var() {
  return __addrspaceof(constant_var);
}

// HOST-LABEL: define{{.*}} i32 @test_host_constant_var(
// HOST: ret i32 24

extern "C" int test_host_const_device_var() {
  return __addrspaceof(const_device_var);
}

// HOST-LABEL: define{{.*}} i32 @test_host_const_device_var(
// HOST: ret i32 24

extern "C" int test_host_device_array_address() {
  return __addrspaceof(*&device_array);
}

// HOST-LABEL: define{{.*}} i32 @test_host_device_array_address(
// HOST: ret i32 0

#else

template <class T> consteval int consteval_address_space(T *p) {
  return __addrspaceof(*p);
}

template <class T> constexpr int constexpr_address_space(T *p) {
  return __addrspaceof(*p);
}

template <int AS> struct AddressSpaceSpecialization;
template <>
struct AddressSpaceSpecialization<__CLANG_ADDRESS_SPACE_DEFAULT> {
  static constexpr int value = __CLANG_ADDRESS_SPACE_DEFAULT;
};
template <> struct AddressSpaceSpecialization<EXPECTED_DEVICE_ADDRESS_SPACE> {
  static constexpr int value = EXPECTED_DEVICE_ADDRESS_SPACE;
};
template <> struct AddressSpaceSpecialization<EXPECTED_SHARED_ADDRESS_SPACE> {
  static constexpr int value = EXPECTED_SHARED_ADDRESS_SPACE;
};
template <>
struct AddressSpaceSpecialization<EXPECTED_CONSTANT_ADDRESS_SPACE> {
  static constexpr int value = EXPECTED_CONSTANT_ADDRESS_SPACE;
};

static_assert(__addrspaceof(*(int *)&constant_var) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(consteval_address_space(&device_var) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(consteval_address_space(&constant_var) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(consteval_address_space(&const_device_var) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(consteval_address_space((int *)&constant_var) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(constexpr_address_space(&constant_var) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__addrspaceof(device_array) ==
              EXPECTED_DEVICE_ADDRESS_SPACE);
static_assert(__addrspaceof(*&device_array) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__addrspaceof(*(device_array + 1)) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(__addrspaceof(*device_ptr) ==
              __CLANG_ADDRESS_SPACE_DEFAULT);
static_assert(
    AddressSpaceSpecialization<
      __addrspaceof(device_var)>::value ==
      EXPECTED_DEVICE_ADDRESS_SPACE);
static_assert(
    AddressSpaceSpecialization<
      __addrspaceof(constant_var)>::value ==
      EXPECTED_CONSTANT_ADDRESS_SPACE);
static_assert(
    AddressSpaceSpecialization<
        consteval_address_space(&constant_var)>::value ==
    __CLANG_ADDRESS_SPACE_DEFAULT);

extern "C" __device__ int test_generic_pointer(int *p) {
  return __addrspaceof(*p);
}

// CHECK-LABEL: define{{.*}} i32 @test_generic_pointer(
// CHECK: ret i32 0

extern "C" __device__ int test_shared_local() {
  __shared__ int shared_var;
  __shared__ int shared_array[4];
  static_assert(__addrspaceof(shared_var) ==
                EXPECTED_SHARED_ADDRESS_SPACE);
  static_assert(__addrspaceof(*(char *)&shared_var) ==
                __CLANG_ADDRESS_SPACE_DEFAULT);
  static_assert(__addrspaceof(*(&shared_var + 1)) ==
                __CLANG_ADDRESS_SPACE_DEFAULT);
  static_assert(__addrspaceof(shared_array) ==
                EXPECTED_SHARED_ADDRESS_SPACE);
  static_assert(__addrspaceof(*&shared_array) ==
                __CLANG_ADDRESS_SPACE_DEFAULT);
  static_assert(__addrspaceof(shared_array[0]) ==
                __CLANG_ADDRESS_SPACE_DEFAULT);
  static_assert(__addrspaceof(*(shared_array + 1)) ==
                __CLANG_ADDRESS_SPACE_DEFAULT);
  static_assert(consteval_address_space(&shared_var) ==
                __CLANG_ADDRESS_SPACE_DEFAULT);
  static_assert(consteval_address_space(shared_array) ==
                __CLANG_ADDRESS_SPACE_DEFAULT);
  static_assert(constexpr_address_space(&shared_var) ==
                __CLANG_ADDRESS_SPACE_DEFAULT);
  static_assert(
      AddressSpaceSpecialization<
      __addrspaceof(shared_var)>::value ==
      EXPECTED_SHARED_ADDRESS_SPACE);
  return __addrspaceof(shared_var);
}

// CHECK-LABEL: define{{.*}} i32 @test_shared_local(
// CUDA: ret i32 8
// HIP: ret i32 25

extern "C" __device__ int test_device_var() {
  return __addrspaceof(device_var);
}

// CHECK-LABEL: define{{.*}} i32 @test_device_var(
// CUDA: ret i32 6
// HIP: ret i32 23

extern "C" __device__ int test_device_array() {
  return __addrspaceof(device_array);
}

// CHECK-LABEL: define{{.*}} i32 @test_device_array(
// CUDA: ret i32 6
// HIP: ret i32 23

extern "C" __device__ int test_device_array_address() {
  return __addrspaceof(*&device_array);
}

// CHECK-LABEL: define{{.*}} i32 @test_device_array_address(
// CHECK: ret i32 0

extern "C" __device__ int test_device_array_arithmetic() {
  return __addrspaceof(*(device_array + 1));
}

// CHECK-LABEL: define{{.*}} i32 @test_device_array_arithmetic(
// CHECK: ret i32 0

extern "C" __device__ int test_device_pointer_value() {
  return __addrspaceof(*device_ptr);
}

// CHECK-LABEL: define{{.*}} i32 @test_device_pointer_value(
// CHECK: ret i32 0

extern "C" __device__ int test_constant_var() {
  return __addrspaceof(constant_var);
}

// CHECK-LABEL: define{{.*}} i32 @test_constant_var(
// CUDA: ret i32 7
// HIP: ret i32 24

extern "C" __device__ int test_const_device_var() {
  return __addrspaceof(const_device_var);
}

// CHECK-LABEL: define{{.*}} i32 @test_const_device_var(
// CUDA: ret i32 7
// HIP: ret i32 24

extern "C" __device__ int test_explicit_cast() {
  return __addrspaceof(*(int *)&constant_var);
}

// CHECK-LABEL: define{{.*}} i32 @test_explicit_cast(
// CHECK: ret i32 0

extern "C" __device__ int
test_target_address_space_3(int __attribute__((address_space(3))) *p) {
  return __addrspaceof(*p);
}

// CHECK-LABEL: define{{.*}} i32 @test_target_address_space_3(
// CHECK: ret i32 16777219

extern "C" __device__ int
test_target_address_space_4(int __attribute__((address_space(4))) *p) {
  return __addrspaceof(*p);
}

// CHECK-LABEL: define{{.*}} i32 @test_target_address_space_4(
// CHECK: ret i32 16777220

#endif
