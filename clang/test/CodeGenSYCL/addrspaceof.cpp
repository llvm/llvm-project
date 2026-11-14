// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -Wno-deprecated-attributes -emit-llvm -o - %s | FileCheck %s

[[clang::sycl_external]] int
global_device_as(__attribute__((opencl_global_device)) int *p) {
  static_assert(__addrspaceof(*p) ==
                __ADDRSPACE_GLOBAL);
  return __addrspaceof(*p);
}

// CHECK-LABEL: define{{.*}} i32 @_Z16global_device_asPU3AS5i(
// CHECK-SAME: ptr addrspace(5)
// CHECK: ret i32 1

[[clang::sycl_external]] int
global_host_as(__attribute__((opencl_global_host)) int *p) {
  static_assert(__addrspaceof(*p) ==
                __ADDRSPACE_GLOBAL);
  return __addrspaceof(*p);
}

// CHECK-LABEL: define{{.*}} i32 @_Z14global_host_asPU3AS6i(
// CHECK-SAME: ptr addrspace(6)
// CHECK: ret i32 1

[[clang::sycl_external]] int
generic_as(int [[clang::sycl_generic]] *p) {
  static_assert(__addrspaceof(int [[clang::sycl_generic]]) ==
                __ADDRSPACE_GENERIC);
  static_assert(__addrspaceof(*p) == __ADDRSPACE_GENERIC);
  return __addrspaceof(*p);
}

// CHECK-LABEL: define{{.*}} i32 @_Z10generic_asPU3AS4i(
// CHECK-SAME: ptr addrspace(4)
// CHECK: ret i32 5

[[clang::sycl_external]] int
constant_as(int [[clang::sycl_constant]] *p) {
  static_assert(__addrspaceof(int [[clang::sycl_constant]]) ==
                __ADDRSPACE_CONSTANT);
  static_assert(__addrspaceof(*p) == __ADDRSPACE_CONSTANT);
  return __addrspaceof(*p);
}

// CHECK-LABEL: define{{.*}} i32 @_Z11constant_asPU3AS2i(
// CHECK-SAME: ptr addrspace(2)
// CHECK: ret i32 3
