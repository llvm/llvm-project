// RUN: %clang_cc1 %s -E -dM | FileCheck %s
// RUN: %clang_cc1 %s -fsycl-is-device -E -dM | FileCheck --check-prefix=CHECK-SYCL %s

// CHECK-NOT:#define __SYCL_DEVICE_ONLY__ 1
// CHECK-SYCL:#define __SYCL_DEVICE_ONLY__ 1
