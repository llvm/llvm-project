// RUN: %clang_cc1 -triple spir64 -fsycl-is-device -Wno-deprecated-attributes -emit-llvm -o - %s | FileCheck %s

[[clang::sycl_external]] int
global_device_as(__attribute__((opencl_global_device)) int *p) {
  static_assert(__addrspaceof(*p) ==
                __CLANG_ADDRESS_SPACE_SYCL_GLOBAL);
  return __addrspaceof(*p);
}

// CHECK-LABEL: define{{.*}} i32 @_Z16global_device_asPU3AS5i(
// CHECK-SAME: ptr addrspace(5)
// CHECK: ret i32 9

[[clang::sycl_external]] int
global_host_as(__attribute__((opencl_global_host)) int *p) {
  static_assert(__addrspaceof(*p) ==
                __CLANG_ADDRESS_SPACE_SYCL_GLOBAL);
  return __addrspaceof(*p);
}

// CHECK-LABEL: define{{.*}} i32 @_Z14global_host_asPU3AS6i(
// CHECK-SAME: ptr addrspace(6)
// CHECK: ret i32 9
