// RUN: %clang_cc1 -cl-std=CL1.2 -cl-ext=-+cl_khr_fp64 -triple spir-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s | FileCheck -check-prefixes=FP64,ALL %s
// RUN: %clang_cc1 -cl-std=CL1.2 -cl-ext=-cl_khr_fp64 -triple spir-unknown-unknown -disable-llvm-passes -emit-llvm -o - %s | FileCheck -check-prefixes=NOFP64,ALL %s

typedef __attribute__((ext_vector_type(2))) float float2;
typedef __attribute__((ext_vector_type(2))) half half2;

#ifdef cl_khr_fp64
typedef __attribute__((ext_vector_type(2))) double double2;
#endif

int printf(__constant const char* st, ...) __attribute__((format(printf, 1, 2)));


// ALL-LABEL: @test_printf_float2(
// FP64: %conv = fpext <2 x float> %0 to <2 x double>
// FP64: %call = call spir_func i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([5 x i8], [5 x i8] addrspace(2)* @.str, i32 0, i32 0), <2 x double> %conv)

// NOFP64:  call spir_func i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([5 x i8], [5 x i8] addrspace(2)* @.str, i32 0, i32 0), <2 x float> %0)
kernel void test_printf_float2(float2 arg) {
  printf("%v2f", arg);
}

// ALL-LABEL: @test_printf_half2(
// FP64: %conv = fpext <2 x half> %0 to <2 x double>
// FP64:  %call = call spir_func i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([5 x i8], [5 x i8] addrspace(2)* @.str, i32 0, i32 0), <2 x double> %conv) #2

// NOFP64: %conv = fpext <2 x half> %0 to <2 x float>
// NOFP64:  %call = call spir_func i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([5 x i8], [5 x i8] addrspace(2)* @.str, i32 0, i32 0), <2 x float> %conv) #2
kernel void test_printf_half2(half2 arg) {
  printf("%v2f", arg);
}

#ifdef cl_khr_fp64
// FP64-LABEL: @test_printf_double2(
// FP64: call spir_func i32 (i8 addrspace(2)*, ...) @printf(i8 addrspace(2)* getelementptr inbounds ([5 x i8], [5 x i8] addrspace(2)* @.str, i32 0, i32 0), <2 x double> %0) #2
kernel void test_printf_double2(double2 arg) {
  printf("%v2f", arg);
}
#endif
