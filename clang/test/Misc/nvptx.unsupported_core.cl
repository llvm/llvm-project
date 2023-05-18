// RUN: %clang_cc1 -cl-std=CL2.0 -triple nvptx-unknown-unknown -Wpedantic-core-features %s 2> %t
// RUN: FileCheck --check-prefixes=CHECK-C < %t %s
// RUN: %clang_cc1 -cl-std=CLC++ -triple nvptx-unknown-unknown -Wpedantic-core-features %s 2> %t
// RUN: FileCheck --check-prefixes=CHECK-CPP < %t %s

// CHECK-C: cl_khr_3d_image_writes is a core feature in OpenCL C version 2.0 but not supported on this target
// CHECK-CPP: cl_khr_3d_image_writes is a core feature in C++ for OpenCL version 1.0 but not supported on this target
