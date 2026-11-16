// RUN: %clang_cc1 %s -E -dM | FileCheck %s
// RUN: %clang_cc1 %s -fsycl-is-device -sycl-std=2017 -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD %s
// RUN: %clang_cc1 %s -fsycl-is-host -sycl-std=2017 -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD %s
// RUN: %clang_cc1 %s -fsycl-is-host -sycl-std=2020 -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD-2020 %s
// RUN: %clang_cc1 %s -fsycl-is-host -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD-2020 %s
// RUN: %clang_cc1 %s -fsycl-is-device -sycl-std=1.2.1 -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD %s
// RUN: %clang_cc1 %s -fsycl-is-device -sycl-std=2020 -E -dM | FileCheck --check-prefix=CHECK-SYCL-STD-2020 %s
// RUN: %clang_cc1 %s -fsycl-is-device -E -dM | FileCheck --check-prefixes=CHECK-SYCL %s

/// __CLANG_RDC__ tells a SYCL implementation whether device symbols may be
/// referenced across translation units. It is predefined in both the device and
/// the host pass, and only when relocatable device code is enabled.
// RUN: %clang_cc1 %s -fsycl-is-device -fgpu-rdc -E -dM | FileCheck --check-prefix=CHECK-RDC %s
// RUN: %clang_cc1 %s -fsycl-is-host -fgpu-rdc -E -dM | FileCheck --check-prefix=CHECK-RDC %s
// RUN: %clang_cc1 %s -fsycl-is-device -E -dM | FileCheck --check-prefix=CHECK-NO-RDC %s
// RUN: %clang_cc1 %s -fsycl-is-host -E -dM | FileCheck --check-prefix=CHECK-NO-RDC %s

// CHECK-NOT:#define __SYCL_DEVICE_ONLY__ 1
// CHECK-NOT:#define CL_SYCL_LANGUAGE_VERSION 121
// CHECK-SYCL-STD:#define CL_SYCL_LANGUAGE_VERSION 121
// CHECK-SYCL-STD-2020:#define SYCL_LANGUAGE_VERSION 202012
// CHECK-SYCL:#define __SYCL_DEVICE_ONLY__ 1
// CHECK-RDC:#define __CLANG_RDC__ 1
// CHECK-NO-RDC-NOT:#define __CLANG_RDC__
