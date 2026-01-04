// RUN: %clang_cc1 -verify -triple spir-unknown-unknown -cl-std=CL3.0 -cl-ext=-__opencl_c_fp64,-cl_khr_fp64 %s
// RUN: %clang_cc1 -verify -triple spir-unknown-unknown -cl-std=clc++2021 -cl-ext=-__opencl_c_fp64,-cl_khr_fp64 %s

#if __opencl_c_ext_fp64_global_atomic_add != 0
#error "Incorrectly defined __opencl_c_ext_fp64_global_atomic_add"
#endif
#if __opencl_c_ext_fp64_local_atomic_add != 0
#error "Incorrectly defined __opencl_c_ext_fp64_local_atomic_add"
#endif
#if __opencl_c_ext_fp64_global_atomic_min_max != 0
#error "Incorrectly defined __opencl_c_ext_fp64_global_atomic_min_max"
#endif
#if __opencl_c_ext_fp64_local_atomic_min_max != 0
#error "Incorrectly defined __opencl_c_ext_fp64_local_atomic_min_max"
#endif

// expected-no-diagnostics
