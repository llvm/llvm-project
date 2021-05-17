// RUN: %clang_cc1 %s -E -dM | FileCheck %s
// RUN: %clang_cc1 %s -fsycl-is-device -sycl-std=2017 -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD %s
// RUN: %clang_cc1 %s -fsycl-is-host -sycl-std=2017 -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD %s
// RUN: %clang_cc1 %s -fsycl-is-device -sycl-std=1.2.1 -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD %s
// RUN: %clang_cc1 %s -fsycl-is-device -E -dM | FileCheck --check-prefixes=CHECK-SYCL %s

// CHECK-NOT:#define __SYCL_DEVICE_ONLY__ 1
// CHECK-NOT:#define CL_SYCL_LANGUAGE_VERSION 121
// CHECK-SYCL-STD:#define CL_SYCL_LANGUAGE_VERSION 121
// CHECK-SYCL:#define __SYCL_DEVICE_ONLY__ 1
