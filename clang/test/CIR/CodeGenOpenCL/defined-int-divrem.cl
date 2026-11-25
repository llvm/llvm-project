// OpenCL C v3.0 s6.5.1: integer division and remainder by zero, and INT_MIN /
// -1, give an unspecified value. ClangIR implements -fdefined-int-divrem (the
// OpenCL default) by replacing the divisor by 1 for these operands.

// RUN: %clang_cc1 -triple x86_64-unknown-unknown -cl-std=CL3.0 -O0 -fclangir -emit-cir %s -o - | FileCheck %s --check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -cl-std=CL3.0 -O0 -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -cl-std=clc++2021 -O0 -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefixes=LLVM,CLCXX
// RUN: %clang_cc1 -triple spirv64-unknown-unknown -cl-std=CL3.0 -O0 -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -x c -O0 -fdefined-int-divrem -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -cl-std=CL3.0 -O0 -fno-defined-int-divrem -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=OFF
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -x c -O0 -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=OFF

typedef int int4 __attribute__((ext_vector_type(4)));
typedef unsigned int uint4 __attribute__((ext_vector_type(4)));

#ifdef __cplusplus
extern "C" {
#endif

// CIR-LABEL: cir.func {{.*}} @sdiv(
// CIR:         %[[NMIN:.*]] = cir.cmp ne %[[A:.*]], %{{.*}} : !s32i
// CIR:         %[[NM1:.*]] = cir.cmp ne %[[B:.*]], %{{.*}} : !s32i
// CIR:         %[[D1:.*]] = cir.select if %[[NMIN]] then %[[B]] else %[[ONE:.*]] :
// CIR:         %[[D2:.*]] = cir.select if %[[NM1]] then %[[B]] else %[[D1]] :
// CIR:         %[[NZ:.*]] = cir.cmp ne %[[B]], %{{.*}} : !s32i
// CIR:         %[[D:.*]] = cir.select if %[[NZ]] then %[[D2]] else %[[ONE]] :
// CIR:         cir.div %[[A]], %[[D]] : !s32i
// LLVM-LABEL: @sdiv(
// LLVM:         %[[NMIN:.*]] = icmp ne i32 %[[A:.*]], -2147483648
// LLVM:         %[[NM1:.*]] = icmp ne i32 %[[B:.*]], -1
// LLVM:         %[[D1:.*]] = select i1 %[[NMIN]], i32 %[[B]], i32 1
// LLVM:         %[[D2:.*]] = select i1 %[[NM1]], i32 %[[B]], i32 %[[D1]]
// LLVM:         %[[NZ:.*]] = icmp ne i32 %[[B]], 0
// LLVM:         %[[D:.*]] = select i1 %[[NZ]], i32 %[[D2]], i32 1
// LLVM:         sdiv i32 %[[A]], %[[D]]
// OFF-LABEL: @sdiv(
// OFF-NOT:     select
// OFF:         sdiv i32
int sdiv(int a, int b) { return a / b; }

// LLVM-LABEL: @srem(
// LLVM:         %[[NZ:.*]] = icmp ne i32 %[[B:.*]], 0
// LLVM-NEXT:    %[[D:.*]] = select i1 %[[NZ]], i32 %{{.*}}, i32 1
// LLVM-NEXT:    srem i32 %{{.*}}, %[[D]]
// OFF-LABEL: @srem(
// OFF-NOT:     select
// OFF:         srem i32
int srem(int a, int b) { return a % b; }

// CIR-LABEL: cir.func {{.*}} @udiv(
// CIR:         %[[ONE:.*]] = cir.const #cir.int<1> : !u32i
// CIR:         %[[NZ:.*]] = cir.cmp ne %[[B:.*]], %{{.*}} : !u32i
// CIR-NEXT:    %[[D:.*]] = cir.select if %[[NZ]] then %[[B]] else %[[ONE]]
// CIR-NEXT:    cir.div %{{.*}}, %[[D]] : !u32i
// LLVM-LABEL: @udiv(
// LLVM:         %[[NZ:.*]] = icmp ne i32 %[[B:.*]], 0
// LLVM-NEXT:    %[[D:.*]] = select i1 %[[NZ]], i32 %[[B]], i32 1
// LLVM-NEXT:    udiv i32 %{{.*}}, %[[D]]
unsigned udiv(unsigned a, unsigned b) { return a / b; }

// LLVM-LABEL: @urem_assign(
// LLVM:         %[[NZ:.*]] = icmp ne i64 %[[B:.*]], 0
// LLVM-NEXT:    %[[D:.*]] = select i1 %[[NZ]], i64 %[[B]], i64 1
// LLVM-NEXT:    urem i64 %{{.*}}, %[[D]]
unsigned long urem_assign(unsigned long a, unsigned long b) {
  a %= b;
  return a;
}

// CIR-LABEL: cir.func {{.*}} @vdiv(
// CIR:         %[[NMIN:.*]] = cir.vec.cmp(ne, %[[A:.*]], %{{.*}}) : !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>
// CIR:         %[[NM1:.*]] = cir.vec.cmp(ne, %[[B:.*]], %{{.*}}) : !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>
// CIR:         %[[D1:.*]] = cir.vec.ternary(%[[NMIN]], %[[B]], %[[ONE:.*]])
// CIR:         %[[D2:.*]] = cir.vec.ternary(%[[NM1]], %[[B]], %[[D1]])
// CIR:         %[[NZ:.*]] = cir.vec.cmp(ne, %[[B]], %{{.*}}) : !cir.vector<4 x !s32i>, !cir.vector<4 x !s32i>
// CIR:         %[[D:.*]] = cir.vec.ternary(%[[NZ]], %[[D2]], %[[ONE]])
// CIR:         cir.div %[[A]], %[[D]] : !cir.vector<4 x !s32i>
// LLVM-LABEL: @vdiv(
// LLVM:         icmp ne <4 x i32> %[[A:.*]], splat (i32 -2147483648)
// LLVM:         icmp ne <4 x i32> %[[B:.*]], splat (i32 -1)
// LLVM:         icmp ne <4 x i32> %[[B]], zeroinitializer
// LLVM:         %[[D:.*]] = select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> splat (i32 1)
// LLVM-NEXT:    sdiv <4 x i32> %[[A]], %[[D]]
// OFF-LABEL: @vdiv(
// OFF-NOT:     select
// OFF:         sdiv <4 x i32>
int4 vdiv(int4 a, int4 b) { return a / b; }

// LLVM-LABEL: @vurem(
// LLVM:         %[[D:.*]] = select <4 x i1> %{{.*}}, <4 x i32> %[[B:.*]], <4 x i32> splat (i32 1)
// LLVM-NEXT:    urem <4 x i32> %{{.*}}, %[[D]]
uint4 vurem(uint4 a, uint4 b) { return a % b; }

// CIR-LABEL: cir.func {{.*}} @sdiv_const(
// CIR-NOT:     cir.select
// CIR:         cir.div %{{.*}}, %{{.*}} : !s32i
// LLVM-LABEL: @sdiv_const(
// LLVM-NOT:    select
// LLVM:        sdiv i32 %{{.*}}, 4
int sdiv_const(int a) { return a / 4; }

// CIR-LABEL: cir.func {{.*}} @udiv_allones(
// CIR-NOT:     cir.select
// CIR:         cir.div %{{.*}}, %{{.*}} : !u32i
// LLVM-LABEL: @udiv_allones(
// LLVM-NOT:    select
// LLVM:        udiv i32 %{{.*}}, -1
unsigned udiv_allones(unsigned a) { return a / 0xffffffffu; }

// CIR-LABEL: cir.func {{.*}} @sdiv_minus_one(
// CIR:         cir.select
// LLVM-LABEL: @sdiv_minus_one(
// LLVM:        select i1
// LLVM:        sdiv i32
int sdiv_minus_one(int a) { return a / -1; }

// CIR-LABEL: cir.func {{.*}} @vdiv_splat(
// CIR-NOT:     cir.vec.ternary
// CIR:         cir.div %{{.*}}, %{{.*}} : !cir.vector<4 x !s32i>
// LLVM-LABEL: @vdiv_splat(
// LLVM-NOT:    select
// LLVM:        sdiv <4 x i32> %{{.*}}, splat (i32 4)
int4 vdiv_splat(int4 a) { return a / 4; }

// CIR-LABEL: cir.func {{.*}} @vdiv_zero(
// CIR:         cir.vec.ternary
// LLVM-LABEL: @vdiv_zero(
// LLVM:        select <4 x i1>
// LLVM:        sdiv <4 x i32>
int4 vdiv_zero(int4 a) { return a / (int4)(0); }

#ifdef __cplusplus
// CLCXX-LABEL: @vdiv_const_vec(
// CLCXX-NOT:    select
// CLCXX:        sdiv <4 x i32> %{{.*}}, <i32 2, i32 3, i32 4, i32 5>
int4 vdiv_const_vec(int4 a) { return a / int4{2, 3, 4, 5}; }

// CLCXX-LABEL: @vdiv_mixed_zero_vec(
// CLCXX:        select <4 x i1>
// CLCXX:        sdiv <4 x i32>
int4 vdiv_mixed_zero_vec(int4 a) { return a / int4{2, 0, 4, 5}; }
}
#endif

#ifdef __OPENCL_C_VERSION__
// The kernels of CodeGenOpenCL/defined-int-divrem-O2.cl: without the defined
// lowering, the optimizer removes the stores to out[1].
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -cl-std=CL3.0 -O2 -Wno-division-by-zero -fclangir -emit-llvm %s -o - | FileCheck %s --check-prefix=O2

// O2-LABEL: define {{.*}} void @divisor_checked_after(
// O2:         store i32 7,
// O2:         ret void
__kernel void divisor_checked_after(__global int *out, int a, int b) {
  int q = a / b;
  if (b == 0)
    out[1] = 7;
  out[0] = q;
}

// O2-LABEL: define {{.*}} void @constant_zero_divisor(
// O2:         store i32 11, ptr
// O2:         [[P1:%.*]] = getelementptr {{.*}}i8, ptr %{{.*}}, i64 4
// O2:         store i32 {{.*}}, ptr [[P1]]
// O2:         [[P2:%.*]] = getelementptr {{.*}}i8, ptr %{{.*}}, i64 8
// O2:         store i32 22, ptr [[P2]]
// O2:         ret void
__kernel void constant_zero_divisor(__global int *out) {
  out[0] = 11;
  int v = 5 / 0;
  out[1] = v == 0 ? 33 : 44;
  out[2] = 22;
}
#endif
