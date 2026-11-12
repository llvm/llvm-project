// RUN: %clang_cc1 -no-enable-noundef-analysis -cl-std=CL1.2 -cl-ext=-+cl_khr_fp64 -triple spir-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s | FileCheck -check-prefixes=FP64,ALL %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -cl-std=CL1.2 -cl-ext=-cl_khr_fp64 -triple spir-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s | FileCheck -check-prefixes=NOFP64,ALL %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -cl-std=CL3.0 -cl-ext=+__opencl_c_fp64,+cl_khr_fp64 -triple spir-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s | FileCheck -check-prefixes=FP64,ALL %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -cl-std=CL3.0 -cl-ext=-__opencl_c_fp64,-cl_khr_fp64 -triple spir-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s | FileCheck -check-prefixes=NOFP64,ALL %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -cl-std=clc++2021 -cl-ext=+__opencl_c_fp64,+cl_khr_fp64 -triple spir-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s | FileCheck -check-prefixes=FP64,ALL %s
// RUN: %clang_cc1 -no-enable-noundef-analysis -cl-std=clc++2021 -cl-ext=-__opencl_c_fp64,-cl_khr_fp64 -triple spir-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s | FileCheck -check-prefixes=NOFP64,ALL %s

typedef __attribute__((ext_vector_type(2))) float float2;
typedef __attribute__((ext_vector_type(2))) half half2;

#if defined(cl_khr_fp64) || defined(__opencl_c_fp64)
typedef __attribute__((ext_vector_type(2))) double double2;
#endif

int printf(__constant const char* st, ...) __attribute__((format(printf, 1, 2)));


// ALL-LABEL: @test_printf_float2(
// FP64: %call = call spir_func i32 (ptr addrspace(2), ...) @{{.*}}printf{{.*}}(ptr addrspace(2) @.str, <2 x float> %0)

// NOFP64:  call spir_func i32 (ptr addrspace(2), ...) @{{.*}}printf{{.*}}(ptr addrspace(2) @.str, <2 x float> %0)
kernel void test_printf_float2(float2 arg) {
  printf("%v2hlf", arg);
}

// ALL-LABEL: @test_printf_half2(
// FP64:  %call = call spir_func i32 (ptr addrspace(2), ...) @{{.*}}printf{{.*}}(ptr addrspace(2) @.str.1, <2 x half> %0)

// NOFP64:  %call = call spir_func i32 (ptr addrspace(2), ...) @{{.*}}printf{{.*}}(ptr addrspace(2) @.str.1, <2 x half> %0)
kernel void test_printf_half2(half2 arg) {
  printf("%v2hf", arg);
}

#if defined(cl_khr_fp64) || defined(__opencl_c_fp64)
// FP64-LABEL: @test_printf_double2(
// FP64: call spir_func i32 (ptr addrspace(2), ...) @{{.*}}printf{{.*}}(ptr addrspace(2) @.str.2, <2 x double> %0)
kernel void test_printf_double2(double2 arg) {
  printf("%v2lf", arg);
}
#endif
