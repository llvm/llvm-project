// RUN: %clang_cc1 -fclangir -no-enable-noundef-analysis -cl-std=CL1.2 -cl-ext=-+cl_khr_fp64 -triple spirv64-unknown-unknown -disable-llvm-passes -emit-cir -fno-clangir-call-conv-lowering -o %t.12fp64.cir %s
// RUN: FileCheck -input-file=%t.12fp64.cir -check-prefixes=CIR-FP64,CIR-ALL %s
// RUN: %clang_cc1 -fclangir -no-enable-noundef-analysis -cl-std=CL1.2 -cl-ext=-cl_khr_fp64 -triple spirv64-unknown-unknown -disable-llvm-passes -emit-cir -fno-clangir-call-conv-lowering -o %t.12nofp64.cir %s
// RUN: FileCheck -input-file=%t.12nofp64.cir -check-prefixes=CIR-NOFP64,CIR-ALL %s
// RUN: %clang_cc1 -fclangir -no-enable-noundef-analysis -cl-std=CL3.0 -cl-ext=+__opencl_c_fp64,+cl_khr_fp64 -triple spirv64-unknown-unknown -disable-llvm-passes -emit-cir -fno-clangir-call-conv-lowering -o %t.30fp64.cir %s
// RUN: FileCheck -input-file=%t.30fp64.cir -check-prefixes=CIR-FP64,CIR-ALL %s
// RUN: %clang_cc1 -fclangir -no-enable-noundef-analysis -cl-std=CL3.0 -cl-ext=-__opencl_c_fp64,-cl_khr_fp64 -triple spirv64-unknown-unknown -disable-llvm-passes -emit-cir -fno-clangir-call-conv-lowering -o %t.30nofp64.cir %s
// RUN: FileCheck -input-file=%t.30nofp64.cir -check-prefixes=CIR-NOFP64,CIR-ALL %s
// RUN: %clang_cc1 -fclangir -no-enable-noundef-analysis -cl-std=CL1.2 -cl-ext=-+cl_khr_fp64 -triple spirv64-unknown-unknown -disable-llvm-passes -emit-llvm -fno-clangir-call-conv-lowering -o %t.12fp64.ll %s
// RUN: FileCheck -input-file=%t.12fp64.ll -check-prefixes=LLVM-FP64,LLVM-ALL %s
// RUN: %clang_cc1 -fclangir -no-enable-noundef-analysis -cl-std=CL1.2 -cl-ext=-cl_khr_fp64 -triple spirv64-unknown-unknown -disable-llvm-passes -emit-llvm -fno-clangir-call-conv-lowering -o %t.12nofp64.ll %s
// RUN: FileCheck -input-file=%t.12nofp64.ll -check-prefixes=LLVM-NOFP64,LLVM-ALL %s
// RUN: %clang_cc1 -fclangir -no-enable-noundef-analysis -cl-std=CL3.0 -cl-ext=+__opencl_c_fp64,+cl_khr_fp64 -triple spirv64-unknown-unknown -disable-llvm-passes -emit-llvm -fno-clangir-call-conv-lowering -o %t.30fp64.ll %s
// RUN: FileCheck -input-file=%t.30fp64.ll -check-prefixes=LLVM-FP64,LLVM-ALL %s
// RUN: %clang_cc1 -fclangir -no-enable-noundef-analysis -cl-std=CL3.0 -cl-ext=-__opencl_c_fp64,-cl_khr_fp64 -triple spirv64-unknown-unknown -disable-llvm-passes -emit-llvm -fno-clangir-call-conv-lowering -o %t.30nofp64.ll %s
// RUN: FileCheck -input-file=%t.30nofp64.ll -check-prefixes=LLVM-NOFP64,LLVM-ALL %s

typedef __attribute__((ext_vector_type(2))) float float2;
typedef __attribute__((ext_vector_type(2))) half half2;

#if defined(cl_khr_fp64) || defined(__opencl_c_fp64)
typedef __attribute__((ext_vector_type(2))) double double2;
#endif

int printf(__constant const char* st, ...) __attribute__((format(printf, 1, 2)));

kernel void test_printf_float2(float2 arg) {
  printf("%v2hlf", arg);
}
// CIR-ALL-LABEL: @test_printf_float2(
// CIR-FP64: %{{.+}} = cir.call @printf(%{{.+}}, %{{.+}}) : (!cir.ptr<!s8i, addrspace(offload_constant)>, !cir.vector<!cir.float x 2>) -> !s32i cc(spir_function)
// CIR-NOFP64:%{{.+}} = cir.call @printf(%{{.+}}, %{{.+}}) : (!cir.ptr<!s8i, addrspace(offload_constant)>, !cir.vector<!cir.float x 2>) -> !s32i cc(spir_function)
// LLVM-ALL-LABEL: @test_printf_float2(
// LLVM-FP64: %{{.+}} = call spir_func i32 (ptr addrspace(2), ...) @{{.*}}printf{{.*}}(ptr addrspace(2) @.str, <2 x float> %{{.*}})
// LLVM-NOFP64:  call spir_func i32 (ptr addrspace(2), ...) @{{.*}}printf{{.*}}(ptr addrspace(2) @.str, <2 x float> %{{.*}})

kernel void test_printf_half2(half2 arg) {
  printf("%v2hf", arg);
}
// CIR-ALL-LABEL: @test_printf_half2(
// CIR-FP64: %{{.+}} = cir.call @printf(%{{.+}}, %{{.+}}) : (!cir.ptr<!s8i, addrspace(offload_constant)>, !cir.vector<!cir.f16 x 2>) -> !s32i cc(spir_function)
// CIR-NOFP64:%{{.+}} = cir.call @printf(%{{.+}}, %{{.+}}) : (!cir.ptr<!s8i, addrspace(offload_constant)>, !cir.vector<!cir.f16 x 2>) -> !s32i cc(spir_function)
// LLVM-ALL-LABEL: @test_printf_half2(
// LLVM-FP64:  %{{.+}} = call spir_func i32 (ptr addrspace(2), ...) @{{.*}}printf{{.*}}(ptr addrspace(2) @.str.1, <2 x half> %{{.*}})
// LLVM-NOFP64:  %{{.+}} = call spir_func i32 (ptr addrspace(2), ...) @{{.*}}printf{{.*}}(ptr addrspace(2) @.str.1, <2 x half> %{{.*}})

#if defined(cl_khr_fp64) || defined(__opencl_c_fp64)
kernel void test_printf_double2(double2 arg) {
  printf("%v2lf", arg);
}
// CIR-FP64-LABEL: @test_printf_double2(
// CIR-FP64: %{{.+}} = cir.call @printf(%{{.+}}, %{{.+}}) : (!cir.ptr<!s8i, addrspace(offload_constant)>, !cir.vector<!cir.double x 2>) -> !s32i cc(spir_function)
// LLVM-FP64-LABEL: @test_printf_double2(
// LLVM-FP64: call spir_func i32 (ptr addrspace(2), ...) @{{.*}}printf{{.*}}(ptr addrspace(2) @.str.2, <2 x double> %{{.*}})
#endif
