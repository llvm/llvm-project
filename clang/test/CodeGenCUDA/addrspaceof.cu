// RUN: %clang_cc1 -triple nvptx64-nvidia-cuda -fcuda-is-device -std=c++20 -emit-llvm -o - %s | FileCheck --check-prefix=CHECK %s
// RUN: %clang_cc1 -triple amdgcn-amd-amdhsa -fcuda-is-device -x hip -std=c++20 -emit-llvm -o - %s | FileCheck --check-prefix=CHECK %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -aux-triple amdgcn-amd-amdhsa -x hip -std=c++20 -DHOST_TEST -emit-llvm -o - %s | FileCheck --check-prefix=HOST %s

#include "Inputs/cuda.h"

__device__ int device_var;
__device__ int device_array[4];
__device__ int *device_ptr;
__constant__ int constant_var;
__device__ const int const_device_var = 1;
extern __device__ const int extern_const_device_var;

static_assert(__addrspaceof(device_ptr) == __ADDRSPACE_GLOBAL);
static_assert(__addrspaceof(*device_ptr) ==
              __ADDRSPACE_DEFAULT);
static_assert(__addrspaceof((device_ptr)) ==
              __ADDRSPACE_DEFAULT);
static_assert(__addrspaceof((device_var)) ==
              __ADDRSPACE_DEFAULT);
static_assert(__addrspaceof(extern_const_device_var) ==
              __ADDRSPACE_GLOBAL);

#ifdef HOST_TEST

template <class T> consteval int host_consteval_address_space(T *p) {
  return __addrspaceof(*p);
}

template <class T> constexpr int host_constexpr_address_space(T *p) {
  return __addrspaceof(*p);
}

static_assert(__addrspaceof(device_var) ==
              __ADDRSPACE_GLOBAL);
static_assert(__addrspaceof(constant_var) ==
              __ADDRSPACE_CONSTANT);
static_assert(__addrspaceof(const_device_var) ==
              __ADDRSPACE_CONSTANT);
static_assert(__addrspaceof(device_array) ==
              __ADDRSPACE_GLOBAL);
static_assert(__addrspaceof(*&device_array) ==
              __ADDRSPACE_DEFAULT);
static_assert(__addrspaceof(*(device_array + 1)) ==
              __ADDRSPACE_DEFAULT);
static_assert(host_consteval_address_space(&device_var) ==
              __ADDRSPACE_DEFAULT);
static_assert(host_consteval_address_space(&constant_var) ==
              __ADDRSPACE_DEFAULT);
static_assert(host_consteval_address_space(&const_device_var) ==
              __ADDRSPACE_DEFAULT);
static_assert(host_consteval_address_space((int *)&constant_var) ==
              __ADDRSPACE_DEFAULT);
static_assert(host_constexpr_address_space(&constant_var) ==
              __ADDRSPACE_DEFAULT);

extern "C" int test_host_device_var() {
  return __addrspaceof(device_var);
}

// HOST-LABEL: define{{.*}} i32 @test_host_device_var(
// HOST: ret i32 1

extern "C" int test_host_constant_var() {
  return __addrspaceof(constant_var);
}

// HOST-LABEL: define{{.*}} i32 @test_host_constant_var(
// HOST: ret i32 3

extern "C" int test_host_const_device_var() {
  return __addrspaceof(const_device_var);
}

// HOST-LABEL: define{{.*}} i32 @test_host_const_device_var(
// HOST: ret i32 3

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
struct AddressSpaceSpecialization<__ADDRSPACE_DEFAULT> {
  static constexpr int value = __ADDRSPACE_DEFAULT;
};
template <> struct AddressSpaceSpecialization<__ADDRSPACE_GLOBAL> {
  static constexpr int value = __ADDRSPACE_GLOBAL;
};
template <> struct AddressSpaceSpecialization<__ADDRSPACE_LOCAL> {
  static constexpr int value = __ADDRSPACE_LOCAL;
};
template <>
struct AddressSpaceSpecialization<__ADDRSPACE_CONSTANT> {
  static constexpr int value = __ADDRSPACE_CONSTANT;
};

static_assert(__addrspaceof(*(int *)&constant_var) ==
              __ADDRSPACE_DEFAULT);
static_assert(consteval_address_space(&device_var) ==
              __ADDRSPACE_DEFAULT);
static_assert(consteval_address_space(&constant_var) ==
              __ADDRSPACE_DEFAULT);
static_assert(consteval_address_space(&const_device_var) ==
              __ADDRSPACE_DEFAULT);
static_assert(consteval_address_space((int *)&constant_var) ==
              __ADDRSPACE_DEFAULT);
static_assert(constexpr_address_space(&constant_var) ==
              __ADDRSPACE_DEFAULT);
static_assert(__addrspaceof(device_array) ==
              __ADDRSPACE_GLOBAL);
static_assert(__addrspaceof(*&device_array) ==
              __ADDRSPACE_DEFAULT);
static_assert(__addrspaceof(*(device_array + 1)) ==
              __ADDRSPACE_DEFAULT);
static_assert(
    AddressSpaceSpecialization<
      __addrspaceof(device_var)>::value ==
      __ADDRSPACE_GLOBAL);
static_assert(
    AddressSpaceSpecialization<
      __addrspaceof(constant_var)>::value ==
      __ADDRSPACE_CONSTANT);
static_assert(
    AddressSpaceSpecialization<
        consteval_address_space(&constant_var)>::value ==
    __ADDRSPACE_DEFAULT);

extern "C" __device__ int test_generic_pointer(int *p) {
  return __addrspaceof(*p);
}

// CHECK-LABEL: define{{.*}} i32 @test_generic_pointer(
// CHECK: ret i32 0

extern "C" __device__ int test_shared_local() {
  __shared__ int shared_var;
  __shared__ int shared_array[4];
#define SELECT_SAME(X) _Generic(X, int : X)
  static_assert(__addrspaceof(decltype(shared_var)) == __ADDRSPACE_DEFAULT);
  static_assert(__addrspaceof(SELECT_SAME(shared_var)) ==
                __ADDRSPACE_DEFAULT);
#undef SELECT_SAME
  static_assert(__addrspaceof(shared_var) ==
                __ADDRSPACE_LOCAL);
  static_assert(__addrspaceof(*(char *)&shared_var) ==
                __ADDRSPACE_DEFAULT);
  static_assert(__addrspaceof(*(&shared_var + 1)) ==
                __ADDRSPACE_DEFAULT);
  static_assert(__addrspaceof(shared_array) ==
                __ADDRSPACE_LOCAL);
  static_assert(__addrspaceof(*&shared_array) ==
                __ADDRSPACE_DEFAULT);
  static_assert(__addrspaceof(shared_array[0]) ==
                __ADDRSPACE_DEFAULT);
  static_assert(__addrspaceof(*(shared_array + 1)) ==
                __ADDRSPACE_DEFAULT);
  static_assert(consteval_address_space(&shared_var) ==
                __ADDRSPACE_DEFAULT);
  static_assert(consteval_address_space(shared_array) ==
                __ADDRSPACE_DEFAULT);
  static_assert(constexpr_address_space(&shared_var) ==
                __ADDRSPACE_DEFAULT);
  static_assert(
      AddressSpaceSpecialization<
      __addrspaceof(shared_var)>::value ==
      __ADDRSPACE_LOCAL);
  return __addrspaceof(shared_var);
}

// CHECK-LABEL: define{{.*}} i32 @test_shared_local(
// CHECK: ret i32 2

extern "C" __device__ int test_device_var() {
  return __addrspaceof(device_var);
}

// CHECK-LABEL: define{{.*}} i32 @test_device_var(
// CHECK: ret i32 1

extern "C" __device__ int test_device_array() {
  return __addrspaceof(device_array);
}

// CHECK-LABEL: define{{.*}} i32 @test_device_array(
// CHECK: ret i32 1

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
// CHECK: ret i32 3

extern "C" __device__ int test_const_device_var() {
  return __addrspaceof(const_device_var);
}

// CHECK-LABEL: define{{.*}} i32 @test_const_device_var(
// CHECK: ret i32 3

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
