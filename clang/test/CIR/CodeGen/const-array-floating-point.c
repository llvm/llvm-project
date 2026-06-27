// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

long double ld_arr[3] = {1.0, -2.0, 0.0};
long double ld_zero[4] = {0};
__float128 q_arr[3] = {1.0, -2.0, 0.0};
_Float16 h_arr[3] = {1.0, -2.0, 0.0};
__bf16 bf_arr[3] = {1.0, -2.0, 0.0};
float f_arr[3] = {1.0, -2.0, 0.0};
double d_arr[3] = {1.0, -2.0, 0.0};

// CIR-DAG: cir.global external @ld_arr = #cir.const_array<[#cir.fp<1.000000e+00> : !cir.long_double<!cir.f80>, #cir.fp<-2.000000e+00> : !cir.long_double<!cir.f80>, #cir.fp<0.000000e+00> : !cir.long_double<!cir.f80>]> : !cir.array<!cir.long_double<!cir.f80> x 3>
// CIR-DAG: cir.global external @ld_zero = #cir.zero : !cir.array<!cir.long_double<!cir.f80> x 4>
// CIR-DAG: cir.global external @q_arr = #cir.const_array<[#cir.fp<1.000000e+00> : !cir.f128, #cir.fp<-2.000000e+00> : !cir.f128, #cir.fp<0.000000e+00> : !cir.f128]> : !cir.array<!cir.f128 x 3>
// CIR-DAG: cir.global external @h_arr = #cir.const_array<[#cir.fp<1.000000e+00> : !cir.f16, #cir.fp<-2.000000e+00> : !cir.f16, #cir.fp<0.000000e+00> : !cir.f16]> : !cir.array<!cir.f16 x 3>
// CIR-DAG: cir.global external @bf_arr = #cir.const_array<[#cir.fp<1.000000e+00> : !cir.bf16, #cir.fp<-2.000000e+00> : !cir.bf16, #cir.fp<0.000000e+00> : !cir.bf16]> : !cir.array<!cir.bf16 x 3>
// CIR-DAG: cir.global external @f_arr = #cir.const_array<[#cir.fp<1.000000e+00> : !cir.float, #cir.fp<-2.000000e+00> : !cir.float, #cir.fp<0.000000e+00> : !cir.float]> : !cir.array<!cir.float x 3>
// CIR-DAG: cir.global external @d_arr = #cir.const_array<[#cir.fp<1.000000e+00> : !cir.double, #cir.fp<-2.000000e+00> : !cir.double, #cir.fp<0.000000e+00> : !cir.double]> : !cir.array<!cir.double x 3>

// LLVM-DAG: @ld_arr = global [3 x x86_fp80] [x86_fp80 1.000000e+00, x86_fp80 -2.000000e+00, x86_fp80 0.000000e+00]
// LLVM-DAG: @ld_zero = global [4 x x86_fp80] zeroinitializer
// LLVM-DAG: @q_arr = global [3 x fp128] [fp128 1.000000e+00, fp128 -2.000000e+00, fp128 0.000000e+00]
// LLVM-DAG: @h_arr = global [3 x half] [half 1.000000e+00, half -2.000000e+00, half 0.000000e+00]
// LLVM-DAG: @bf_arr = global [3 x bfloat] [bfloat 1.000000e+00, bfloat -2.000000e+00, bfloat 0.000000e+00]
// LLVM-DAG: @f_arr = global [3 x float] [float 1.000000e+00, float -2.000000e+00, float 0.000000e+00]
// LLVM-DAG: @d_arr = global [3 x double] [double 1.000000e+00, double -2.000000e+00, double 0.000000e+00]
