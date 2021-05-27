// RUN: not %clang_cc1 -cl-std=CL3.0 -triple spir-unknown-unknown -cl-ext=-__opencl_c_fp64,+cl_khr_fp64 %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -cl-std=CL3.0 -triple spir-unknown-unknown -cl-ext=+__opencl_c_fp64,-cl_khr_fp64 %s 2>&1 | FileCheck %s

// CHECK: error: options cl_khr_fp64 and __opencl_c_fp64 are set to different values
