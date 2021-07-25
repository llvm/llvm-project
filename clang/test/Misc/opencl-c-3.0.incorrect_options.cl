// RUN: not %clang_cc1 -cl-std=CL3.0 -triple spir-unknown-unknown -cl-ext=-__opencl_c_fp64,+cl_khr_fp64 %s 2>&1 | FileCheck -check-prefix=CHECK-FP64 %s
// RUN: not %clang_cc1 -cl-std=CL3.0 -triple spir-unknown-unknown -cl-ext=+__opencl_c_fp64,-cl_khr_fp64 %s 2>&1 | FileCheck -check-prefix=CHECK-FP64 %s
// RUN: not %clang_cc1 -cl-std=CL3.0 -triple spir-unknown-unknown -cl-ext=+__opencl_c_read_write_images,-__opencl_c_images %s 2>&1 | FileCheck -check-prefix=CHECK-READ-WRITE-IMAGES %s

// CHECK-FP64: error: options cl_khr_fp64 and __opencl_c_fp64 are set to different values
// CHECK-READ-WRITE-IMAGES: error: feature __opencl_c_read_write_images requires support of __opencl_c_images feature
