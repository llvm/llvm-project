// Test host codegen.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

template <typename tx, typename ty>
struct TT {
  tx X;
  ty Y;
};
#pragma omp declare target
int ga = 5;
#pragma omp end declare target

// CHECK-DAG:  [[TT:%.+]] = type { i64, i8 }
// CHECK-DAG:  [[TTII:%.+]] = type { i32, i32 }
// CHECK-DAG:  [[S1:%.+]] = type { double }

// TCHECK-DAG:  [[TT:%.+]] = type { i64, i8 }
// TCHECK-DAG:  [[TTII:%.+]] = type { i32, i32 }
// TCHECK-DAG:  [[S1:%.+]] = type { double }

// CHECK-DAG:  [[SIZET:@.+]] = private unnamed_addr constant [3 x i{{32|64}}] [i[[SZ:32|64]] 4, i{{64|32}} {{8|4}}, i[[SZ:32|64]] 4]
// CHECK-DAG:  [[MAPT:@.+]] = private unnamed_addr constant [3 x i64] [i64 288, i64 49, i64 288]
// CHECK-DAG:  [[SIZET2:@.+]] = private unnamed_addr constant [9 x i64] [i64 2, i64 40, i64 {{4|8}}, i64 0, i64 400, i64 {{4|8}}, i64 {{4|8}}, i64 0, i64 {{12|16}}]
// CHECK-DAG:  [[MAPT2:@.+]] = private unnamed_addr constant [9 x i64] [i64 288, i64 161, i64 800, i64 161, i64 161, i64 800, i64 800, i64 161, i64 161]
// CHECK-DAG:  [[SIZET3:@.+]] = private unnamed_addr constant [2 x i{{32|64}}] [i{{32|64}} 0, i{{32|64}} 8]
// CHECK-DAG:  [[MAPT3:@.+]] = private unnamed_addr constant [2 x i64] [i64 32, i64 161]
// CHECK-DAG:  [[SIZET4:@.+]] = private unnamed_addr constant [5 x i64] [i64 8, i64 4, i64 {{4|8}}, i64 {{4|8}}, i64 0]
// CHECK-DAG:  [[MAPT4:@.+]] = private unnamed_addr constant [5 x i64] [i64 547, i64 288, i64 800, i64 800, i64 161]
// CHECK-DAG:  [[SIZET5:@.+]] = private unnamed_addr constant [3 x i{{32|64}}] [i[[SZ]] 4, i[[SZ]] 1, i[[SZ]] 40]
// CHECK-DAG:  [[MAPT5:@.+]] = private unnamed_addr constant [3 x i64] [i64 288, i64 288, i64 161]
// CHECK-DAG:  [[SIZET6:@.+]] = private unnamed_addr constant [2 x i{{32|64}}] [i[[SZ]] 4, i[[SZ]] 40]
// CHECK-DAG:  [[MAPT6:@.+]] = private unnamed_addr constant [2 x i64] [i64 288, i64 161]

// CHECK: define {{.*}}[[FOO:@.+]](
int foo(int n, double *ptr) {
  int a = 0;
  short aa = 0;
  float b[10];
  float bn[n];
  double c[5][10];
  double cn[5][n];
  TT<long long, char> d;
  const TT<int, int> e = {n, n};
  int *p __attribute__ ((aligned (64))) = &a;

#pragma omp target firstprivate(a, p, ga)
  {
  }

  // a is passed by value to tgt_target
  // CHECK:  [[N_ADDR:%.+]] = alloca i{{[0-9]+}},
  // CHECK:  [[PTR_ADDR:%.+]] = alloca double*,
  // CHECK:  [[A:%.+]] = alloca i{{[0-9]+}},
  // CHECK:  [[A2:%.+]] = alloca i{{[0-9]+}},
  // CHECK:  [[B:%.+]] = alloca [10 x float],
  // CHECK:  [[SSTACK:%.+]] = alloca i8*,
  // CHECK:  [[C:%.+]] = alloca [5 x [10 x double]],
  // CHECK:  [[D:%.+]] = alloca [[TT]],
  // CHECK:  [[FP_E:%.+]] = alloca [[TTII]],
  // CHECK:  [[P:%.+]] = alloca i32*, align 64
  // CHECK:  [[ACAST:%.+]] = alloca i{{[0-9]+}},
  // CHECK:  [[BASE_PTR_ARR:%.+]] = alloca [3 x i8*],
  // CHECK:  [[PTR_ARR:%.+]] = alloca [3 x i8*],
  // CHECK:  [[A2CAST:%.+]] = alloca i{{[0-9]+}},
  // CHECK:  [[BASE_PTR_ARR2:%.+]] = alloca [9 x i8*],
  // CHECK:  [[PTR_ARR2:%.+]] = alloca [9 x i8*],
  // CHECK:  [[SIZET2:%.+]] = alloca [9 x i{{[0-9]+}}],
  // CHECK:  [[BASE_PTR_ARR3:%.+]] = alloca [2 x i8*],
  // CHECK:  [[PTR_ARR3:%.+]] = alloca [2 x i8*],
  // CHECK:  [[N_ADDR_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[N_ADDR]],
  // CHECK-64:  [[N_EXT:%.+]] = zext i{{[0-9]+}} [[N_ADDR_VAL]] to i{{[0-9]+}}
  // CHECK:  [[SSAVE_RET:%.+]] = call i8* @llvm.stacksave()
  // CHECK:  store i8* [[SSAVE_RET]], i8** [[SSTACK]],
  // CHECK-64:  [[BN_VLA:%.+]] = alloca float, i{{[0-9]+}} [[N_EXT]],
  // CHECK-32:  [[BN_VLA:%.+]] = alloca float, i{{[0-9]+}} [[N_ADDR_VAL]],
  // CHECK:  [[N_ADDR_VAL2:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[N_ADDR]],
  // CHECK-64:  [[N_EXT2:%.+]] = zext i{{[0-9]+}} [[N_ADDR_VAL2]] to i{{[0-9]+}}
  // CHECK-64:  [[CN_SIZE:%.+]] = mul{{.+}} i{{[0-9]+}} 5, [[N_EXT2]]
  // CHECK-32:  [[CN_SIZE:%.+]] = mul{{.+}} i{{[0-9]+}} 5, [[N_ADDR_VAL2]]
  // CHECK:  [[CN_VLA:%.+]] = alloca double, i{{[0-9]+}} [[CN_SIZE]],
  // CHECK:  [[AVAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A]],
  // CHECK-64:  [[CONV:%.+]] = bitcast i{{[0-9]+}}* [[ACAST]] to i{{[0-9]+}}*
  // CHECK-64:  store i{{[0-9]+}} [[AVAL]], i{{[0-9]+}}* [[CONV]],
  // CHECK-32:  store i{{[0-9]+}} [[AVAL]], i{{[0-9]+}}* [[ACAST]],
  // CHECK:  [[ACAST_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[ACAST]],
  // CHECK:  [[P_PTR:%.+]] = load i32*, i32** [[P]], align 64
  // CHECK:  [[BASE_PTR_GEP:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BASE_PTR_ARR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK:  [[ACAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTR_GEP]] to i{{[0-9]+}}*
  // CHECK:  store i{{[0-9]+}} [[ACAST_VAL]], i{{[0-9]+}}* [[ACAST_TOPTR]],
  // CHECK:  [[PTR_GEP:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PTR_ARR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK:  [[ACAST_TOPTR2:%.+]] = bitcast i8** [[PTR_GEP]] to i{{[0-9]+}}*
  // CHECK:  store i{{[0-9]+}} [[ACAST_VAL]], i{{[0-9]+}}* [[ACAST_TOPTR2]],
  // CHECK:  [[BASE_PTR_GEP:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BASE_PTR_ARR]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK:  [[PCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTR_GEP]] to i32***
  // CHECK:  store i32** [[P]], i32*** [[PCAST_TOPTR]],
  // CHECK:  [[PTR_GEP:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PTR_ARR]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK:  [[PCAST_TOPTR2:%.+]] = bitcast i8** [[PTR_GEP]] to i32**
  // CHECK:  store i32* [[P_PTR]], i32** [[PCAST_TOPTR2]],
  // CHECK:  [[BASE_PTR_GEP:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BASE_PTR_ARR]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
  // CHECK:  [[PCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTR_GEP]] to i{{64|32}}*
  // CHECK:  store i{{64|32}} [[GA_VAL:%.*]], i{{64|32}}* [[PCAST_TOPTR]],
  // CHECK:  [[PTR_GEP:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PTR_ARR]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
  // CHECK:  [[PCAST_TOPTR2:%.+]] = bitcast i8** [[PTR_GEP]] to i{{64|32}}*
  // CHECK:  store i{{64|32}} [[GA_VAL]], i{{64|32}}* [[PCAST_TOPTR2]],
  // CHECK:  [[BASE_PTR_GEP_ARG:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BASE_PTR_ARR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK:  [[PTR_GEP_ARG:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PTR_ARR]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK:  {{.+}} = call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, {{.+}}, i32 3, i8** [[BASE_PTR_GEP_ARG]], i8** [[PTR_GEP_ARG]], i[[SZ]]* getelementptr inbounds ([3 x i[[SZ]]], [3 x i[[SZ]]]* [[SIZET]], i32 0, i32 0), i64* getelementptr inbounds ([3 x i64], [3 x i64]* [[MAPT]], i32 0, i32 0), i8** null, i8** null)

  // TCHECK:  define weak_odr void @__omp_offloading_{{.+}}(i{{[0-9]+}} noundef [[A_IN:%.+]], i32** noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[P_IN:%.+]], i{{[0-9]+}} noundef [[GA_IN:%.+]])
  // TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[P_ADDR:%.+]] = alloca i32**,
  // TCHECK:  [[GA_ADDR:%.+]] = alloca i{{64|32}},
  // TCHECK:  [[P_PRIV:%.+]] = alloca i32*,
  // TCHECK-NOT: alloca i{{[0-9]+}}
  // TCHECK:  store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[A_ADDR]],
  // TCHECK:  store i32** [[P_IN]], i32*** [[P_ADDR]],
  // TCHECK:  store i{{[0-9]+}} [[GA_IN]], i{{[0-9]+}}* [[GA_ADDR]],
  // TCHECK-NOT: store i{{[0-9]+}} %
  // TCHECK:  ret void

#pragma omp target firstprivate(aa, b, bn, c, cn, d)
  {
    aa += 1;
    b[2] = 1.0;
    bn[3] = 1.0;
    c[1][2] = 1.0;
    cn[1][3] = 1.0;
    d.X = 1;
    d.Y = 1;
  }

  // CHECK:  [[A2VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A2]],
  // CHECK:  [[A2CASTCONV:%.+]] = bitcast i{{[0-9]+}}* [[A2CAST]] to i{{[0-9]+}}*
  // CHECK:  store i{{[0-9]+}} [[A2VAL]], i{{[0-9]+}}* [[A2CASTCONV]],
  // CHECK:  [[A2CAST_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[A2CAST]],
  // CHECK-64:  [[BN_SIZE:%.+]] = mul{{.+}} i{{[0-9]+}} [[N_EXT]], 4
  // CHECK-32:  [[BN_SZ_SIZE:%.+]] = mul{{.+}} i{{[0-9]+}} [[N_ADDR_VAL]], 4
  // CHECK-32:  [[BN_SIZE:%.+]] = sext i32 [[BN_SZ_SIZE]] to i64
  // CHECK-64:  [[CN_SIZE_1:%.+]] = mul{{.+}} i{{[0-9]+}} 5, [[N_EXT2]]
  // CHECK-32:  [[CN_SIZE_1:%.+]] = mul{{.+}} i{{[0-9]+}} 5, [[N_ADDR_VAL2]]
  // CHECK-64:  [[CN_SIZE_2:%.+]] = mul{{.+}} i{{[0-9]+}} [[CN_SIZE_1]], 8
  // CHECK-32:  [[CN_SZ_SIZE_2:%.+]] = mul{{.+}} i{{[0-9]+}} [[CN_SIZE_1]], 8
  // CHECK-32:  [[CN_SIZE_2:%.+]] = sext i32 [[CN_SZ_SIZE_2]] to i64

  // firstprivate(aa) --> base_ptr = aa, ptr = aa, size = 2 (short)
  // CHECK:  [[BASE_PTR_GEP2_0:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BASE_PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK:  [[ACAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTR_GEP2_0]] to i{{[0-9]+}}*
  // CHECK:  store i{{[0-9]+}} [[A2CAST_VAL]], i{{[0-9]+}}* [[ACAST_TOPTR]],
  // CHECK:  [[PTR_GEP2_0:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK:  [[ACAST_TOPTR:%.+]] = bitcast i8** [[PTR_GEP2_0]] to i{{[0-9]+}}*
  // CHECK:  store i{{[0-9]+}} [[A2CAST_VAL]], i{{[0-9]+}}* [[ACAST_TOPTR]],

  // firstprivate(b): base_ptr = &b[0], ptr = &b[0], size = 40 (sizeof(float)*10)
  // CHECK:  [[BASE_PTR_GEP2_1:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BASE_PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTR_GEP2_1]] to [10 x float]**
  // CHECK:  store [10 x float]* [[B]], [10 x float]** [[BCAST_TOPTR]],
  // CHECK:  [[PTR_GEP2_1:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[PTR_GEP2_1]] to [10 x float]**
  // CHECK:  store [10 x float]* [[B]], [10 x float]** [[BCAST_TOPTR]],

  // firstprivate(bn), 2 entries, n and bn: (1) base_ptr = n, ptr = n, size = 8 ; (2) base_ptr = &c[0], ptr = &c[0], size = n*sizeof(float)
  // CHECK:  [[BASE_PTR_GEP2_2:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BASE_PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTR_GEP2_2]] to i{{[0-9]+}}*
  // CHECK-64:  store i{{[0-9]+}} [[N_EXT]], i{{[0-9]+}}* [[BCAST_TOPTR]],
  // CHECK-32:  store i{{[0-9]+}} [[N_ADDR_VAL]], i{{[0-9]+}}* [[BCAST_TOPTR]],
  // CHECK:  [[PTR_GEP2_2:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[PTR_GEP2_2]] to i{{[0-9]+}}*
  // CHECK-64:  store i{{[0-9]+}} [[N_EXT]], i{{[0-9]+}}* [[BCAST_TOPTR]],
  // CHECK-32:  store i{{[0-9]+}} [[N_ADDR_VAL]], i{{[0-9]+}}* [[BCAST_TOPTR]],
  // CHECK:  [[BASE_PTR_GEP2_3:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BASE_PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTR_GEP2_3]] to float**
  // CHECK:  store float* [[BN_VLA]], float** [[BCAST_TOPTR]],
  // CHECK: [[SIZE_GEPBN_3:%.+]] = getelementptr inbounds [9 x i{{[0-9]+}}], [9 x i{{[0-9]+}}]* [[SIZET2]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
  // CHECK:  store i{{[0-9]+}} [[BN_SIZE]], i{{[0-9]+}}* [[SIZE_GEPBN_3]]

  // firstprivate(c): base_ptr = &c[0], ptr = &c[0], size = 400 (5*10*sizeof(double))
  // CHECK:  [[BASE_PTR_GEP2_4:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BASE_PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 4
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTR_GEP2_4]] to [5 x [10 x double]]**
  // CHECK:  store [5 x [10 x double]]* [[C]], [5 x [10 x double]]** [[BCAST_TOPTR]],
  // CHECK:  [[PTR_GEP2_4:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 4
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[PTR_GEP2_4]] to [5 x [10 x double]]**
  // CHECK:  store [5 x [10 x double]]* [[C]], [5 x [10 x double]]** [[BCAST_TOPTR]],

  // firstprivate(cn), 3 entries, 5, n, cn: (1) base_ptr = 5, ptr = 5, size = 8; (2) (1) base_ptr = n, ptr = n, size = 8; (3) base_ptr = &cn[0], ptr = &cn[0], size = 5*n*sizeof(double)
  // CHECK:  [[BASE_PTR_GEP2_5:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BASE_PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 5
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTR_GEP2_5]] to i{{[0-9]+}}*
  // CHECK:  store i{{[0-9]+}} 5, i{{[0-9]+}}* [[BCAST_TOPTR]],
  // CHECK:  [[PTR_GEP2_5:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 5
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[PTR_GEP2_5]] to i{{[0-9]+}}*
  // CHECK:  store i{{[0-9]+}} 5, i{{[0-9]+}}* [[BCAST_TOPTR]],
  // CHECK:  [[BASE_PTR_GEP2_6:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BASE_PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 6
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTR_GEP2_6]] to i{{[0-9]+}}*
  // CHECK-64:  store i{{[0-9]+}} [[N_EXT2]], i{{[0-9]+}}* [[BCAST_TOPTR]],
  // CHECK-32:  store i{{[0-9]+}} [[N_ADDR_VAL2]], i{{[0-9]+}}* [[BCAST_TOPTR]],
  // CHECK:  [[PTR_GEP2_6:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 6
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[PTR_GEP2_6]] to i{{[0-9]+}}*
  // CHECK-64:  store i{{[0-9]+}} [[N_EXT2]], i{{[0-9]+}}* [[BCAST_TOPTR]],
  // CHECK-32:  store i{{[0-9]+}} [[N_ADDR_VAL2]], i{{[0-9]+}}* [[BCAST_TOPTR]],
  // CHECK:  [[BASE_PTR_GEP2_7:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BASE_PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 7
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTR_GEP2_7]] to double**
  // CHECK:  store double* [[CN_VLA]], double** [[BCAST_TOPTR]],
  // CHECK:  [[PTR_GEP2_7:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 7
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[PTR_GEP2_7]] to double**
  // CHECK:  store double* [[CN_VLA]], double** [[BCAST_TOPTR]],
  // CHECK:  [[SIZE_GEPCN_7:%.+]] = getelementptr inbounds [9 x i{{[0-9]+}}], [9 x i{{[0-9]+}}]* [[SIZET2]], i{{[0-9]+}} 0, i{{[0-9]+}} 7
  // CHECK:  store i{{[0-9]+}} [[CN_SIZE_2]], i{{[0-9]+}}* [[SIZE_GEPCN_7]],

  // firstprivate(d): base_ptr = &d, ptr = &d, size = 16
  // CHECK:  [[BASE_PTR_GEP2_8:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BASE_PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 8
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTR_GEP2_8]] to [[TT]]**
  // CHECK:  store [[TT]]* [[D]], [[TT]]** [[BCAST_TOPTR]],
  // CHECK:  [[PTR_GEP2_8:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 8
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[PTR_GEP2_8]] to [[TT]]**
  // CHECK:  store [[TT]]* [[D]], [[TT]]** [[BCAST_TOPTR]],

  // CHECK:  [[BASE_PTR_GEP_ARG2:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[BASE_PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK:  [[PTR_GEP_ARG2:%.+]] = getelementptr inbounds [9 x i8*], [9 x i8*]* [[PTR_ARR2]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK:  [[SIZES_ARG2:%.+]] = getelementptr inbounds [9 x i[[SZ]]], [9 x i[[SZ]]]* [[SIZET2]],  i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK: {{.+}} = call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, {{.+}}, i32 9, i8** [[BASE_PTR_GEP_ARG2]], i8** [[PTR_GEP_ARG2]], i[[SZ]]* [[SIZES_ARG2]], i64* getelementptr inbounds ([9 x i64], [9 x i64]* [[MAPT2]], i32 0, i32 0), i8** null, i8** null)

  // make sure that firstprivate variables are generated in all cases and that we use those instances for operations inside the
  // target region
  // TCHECK:  define {{.*}}void @__omp_offloading_{{.+}}(i{{[0-9]+}} noundef [[A2_IN:%.+]], [10 x float]* {{.+}} [[B_IN:%.+]], i{{[0-9]+}} noundef [[BN_SZ:%.+]], float* {{.+}} [[BN_IN:%.+]], [5 x [10 x double]]* {{.+}} [[C_IN:%.+]], i{{[0-9]+}} noundef [[CN_SZ1:%.+]], i{{[0-9]+}} noundef [[CN_SZ2:%.+]], double* {{.+}} [[CN_IN:%.+]], [[TT]]* {{.+}} [[D_IN:%.+]])
  // TCHECK:  [[A2_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[B_ADDR:%.+]] = alloca [10 x float]*,
  // TCHECK:  [[VLA_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[BN_ADDR:%.+]] = alloca float*,
  // TCHECK:  [[C_ADDR:%.+]] = alloca [5 x [10 x double]]*,
  // TCHECK:  [[VLA_ADDR2:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[VLA_ADDR4:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[CN_ADDR:%.+]] = alloca double*,
  // TCHECK:  [[D_ADDR:%.+]] = alloca [[TT]]*,
  // TCHECK-NOT: alloca i{{[0-9]+}},
  // TCHECK:  [[B_PRIV:%.+]] = alloca [10 x float],
  // TCHECK:  [[SSTACK:%.+]] = alloca i8*,
  // TCHECK:  [[C_PRIV:%.+]] = alloca [5 x [10 x double]],
  // TCHECK:  [[D_PRIV:%.+]] = alloca [[TT]],
  // TCHECK:  store i{{[0-9]+}} [[A2_IN]], i{{[0-9]+}}* [[A2_ADDR]],
  // TCHECK:  store [10 x float]* [[B_IN]], [10 x float]** [[B_ADDR]],
  // TCHECK:  store i{{[0-9]+}} [[BN_SZ]], i{{[0-9]+}}* [[VLA_ADDR]],
  // TCHECK:  store float* [[BN_IN]], float** [[BN_ADDR]],
  // TCHECK:  store [5 x [10 x double]]* [[C_IN]], [5 x [10 x double]]** [[C_ADDR]],
  // TCHECK:  store i{{[0-9]+}} [[CN_SZ1]], i{{[0-9]+}}* [[VLA_ADDR2]],
  // TCHECK:  store i{{[0-9]+}} [[CN_SZ2]], i{{[0-9]+}}* [[VLA_ADDR4]],
  // TCHECK:  store double* [[CN_IN]], double** [[CN_ADDR]],
  // TCHECK:  store [[TT]]* [[D_IN]], [[TT]]** [[D_ADDR]],
  // TCHECK:  [[CONV_A2ADDR:%.+]] = bitcast i{{[0-9]+}}* [[A2_ADDR]] to i{{[0-9]+}}*
  // TCHECK:  [[B_ADDR_REF:%.+]] = load [10 x float]*, [10 x float]** [[B_ADDR]],
  // TCHECK:  [[BN_SZ_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[VLA_ADDR]],
  // TCHECK:  [[BN_ADDR_REF:%.+]] = load float*, float** [[BN_ADDR]],
  // TCHECK:  [[C_ADDR_REF:%.+]] = load [5 x [10 x double]]*, [5 x [10 x double]]** [[C_ADDR]],
  // TCHECK:  [[CN_SZ1_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[VLA_ADDR2]],
  // TCHECK:  [[CN_SZ2_VAL:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[VLA_ADDR4]],
  // TCHECK:  [[CN_ADDR_REF:%.+]] = load double*, double** [[CN_ADDR]],
  // TCHECK:  [[D_ADDR_REF:%.+]] = load [[TT]]*, [[TT]]** [[D_ADDR]],

  // firstprivate(aa): a_priv = a_in
  // TCHECK-NOT:  store i{{[0-9]+}} %

  //  firstprivate(b): memcpy(b_priv,b_in)
  // TCHECK:  [[B_PRIV_BCAST:%.+]] = bitcast [10 x float]* [[B_PRIV]] to i8*
  // TCHECK:  [[B_ADDR_REF_BCAST:%.+]] = bitcast [10 x float]* [[B_ADDR_REF]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[B_PRIV_BCAST]], i8* align {{[0-9]+}} [[B_ADDR_REF_BCAST]], {{.+}})

  // TCHECK:  [[RET_STACK:%.+]] = call i8* @llvm.stacksave()
  // TCHECK:  store i8* [[RET_STACK]], i8** [[SSTACK]],

  // firstprivate(bn)
  // TCHECK:  [[BN_PRIV:%.+]] = alloca float, i{{[0-9]+}} [[BN_SZ_VAL]],
  // TCHECK:  [[BN_COPY_SZ:%.+]] = mul{{.+}} i{{[0-9]+}} [[BN_SZ_VAL]], 4
  // TCHECK:  [[BN_PRIV__BCAST:%.+]] = bitcast float* [[BN_PRIV]] to i8*
  // TCHECK:  [[BN_REF_IN_BCAST:%.+]] = bitcast float* [[BN_ADDR_REF]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[BN_PRIV__BCAST]], i8* align {{[0-9]+}} [[BN_REF_IN_BCAST]], i{{[0-9]+}} [[BN_COPY_SZ]],{{.+}})

  // firstprivate(c)
  // TCHECK:  [[C_PRIV_BCAST:%.+]] = bitcast [5 x [10 x double]]* [[C_PRIV]] to i8*
  // TCHECK:  [[C_IN_BCAST:%.+]] = bitcast [5 x [10 x double]]* [[C_ADDR_REF]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[C_PRIV_BCAST]], i8* align {{[0-9]+}} [[C_IN_BCAST]],{{.+}})

  // firstprivate(cn)
  // TCHECK:  [[CN_SZ:%.+]] = mul{{.+}} i{{[0-9]+}} [[CN_SZ1_VAL]], [[CN_SZ2_VAL]]
  // TCHECK:  [[CN_PRIV:%.+]] = alloca double, i{{[0-9]+}} [[CN_SZ]],
  // TCHECK:  [[CN_SZ2:%.+]] = mul{{.+}} i{{[0-9]+}} [[CN_SZ1_VAL]], [[CN_SZ2_VAL]]
  // TCHECK:  [[CN_SZ2_CPY:%.+]] = mul{{.+}} i{{[0-9]+}} [[CN_SZ2]], 8
  // TCHECK:  [[CN_PRIV_BCAST:%.+]] = bitcast double* [[CN_PRIV]] to i8*
  // TCHECK:  [[CN_IN_BCAST:%.+]] = bitcast double* [[CN_ADDR_REF]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[CN_PRIV_BCAST]], i8* align {{[0-9]+}} [[CN_IN_BCAST]], i{{[0-9]+}} [[CN_SZ2_CPY]],{{.+}})

  // firstprivate(d)
  // TCHECK:  [[D_PRIV_BCAST:%.+]] = bitcast [[TT]]* [[D_PRIV]] to i8*
  // TCHECK:  [[D_IN_BCAST:%.+]] = bitcast [[TT]]* [[D_ADDR_REF]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[D_PRIV_BCAST]], i8* align {{[0-9]+}} [[D_IN_BCAST]],{{.+}})

#pragma omp target firstprivate(ptr, e)
  {
    ptr[0] = e.X;
    ptr[0]++;
  }
  // CHECK:  [[PTR_ADDR_REF:%.+]] = load double*, double** [[PTR_ADDR]],

  // CHECK:  [[BASE_PTR_GEP3_0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BASE_PTR_ARR3]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTR_GEP3_0]] to double**
  // CHECK:  store double* [[PTR_ADDR_REF]], double** [[BCAST_TOPTR]],
  // CHECK:  [[PTR_GEP3_0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[PTR_ARR3]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[PTR_GEP3_0]] to double**
  // CHECK:  store double* [[PTR_ADDR_REF]], double** [[BCAST_TOPTR]],
  // CHECK:  [[BASE_PTR_GEP3_1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BASE_PTR_ARR3]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTR_GEP3_1]] to [[TTII]]**
  // CHECK:  store [[TTII]]* [[FP_E]], [[TTII]]** [[BCAST_TOPTR]],
  // CHECK:  [[PTR_GEP3_1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[PTR_ARR3]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[PTR_GEP3_1]] to [[TTII]]**
  // CHECK:  store [[TTII]]* [[FP_E]], [[TTII]]** [[BCAST_TOPTR]],

  // CHECK:  [[BASE_PTR_GEP_ARG3:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BASE_PTR_ARR3]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK:  [[PTR_GEP_ARG3:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[PTR_ARR3]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK: {{.+}} = call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, {{.+}}, i32 2, i8** [[BASE_PTR_GEP_ARG3]], i8** [[PTR_GEP_ARG3]], i[[SZ]]* getelementptr inbounds ([2 x i[[SZ]]], [2 x i[[SZ]]]* [[SIZET3]], i32 0, i32 0), i64* getelementptr inbounds ([2 x i64], [2 x i64]* [[MAPT3]], i32 0, i32 0), i8** null, i8** null)

  // TCHECK:  define weak_odr void @__omp_offloading_{{.+}}(double* noundef [[PTR_IN:%.+]], [[TTII]]* noundef nonnull align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[E:%.+]])
  // TCHECK:  [[PTR_ADDR:%.+]] = alloca double*,
  // TCHECK-NOT: alloca [[TTII]],
  // TCHECK-NOT: alloca double*,
  // TCHECK:  store double* [[PTR_IN]], double** [[PTR_ADDR]],
  // TCHECK-NOT: store double* %

  return a;
}

template <typename tx>
tx ftemplate(int n) {
  tx a = 0;
  tx b[10];

#pragma omp target firstprivate(a, b)
  {
    a += 1;
    b[2] += 1;
  }

  return a;
}

static int fstatic(int n) {
  int a = 0;
  char aaa = 0;
  int b[10];

#pragma omp target firstprivate(a, aaa, b)
  {
    a += 1;
    aaa += 1;
    b[2] += 1;
  }

  return a;
}

// TCHECK: define weak_odr void @__omp_offloading_{{.+}}(i{{[0-9]+}} noundef [[A_IN:%.+]], i{{[0-9]+}} noundef [[A3_IN:%.+]], [10 x i{{[0-9]+}}]*{{.+}} [[B_IN:%.+]])
// TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[A3_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[B_ADDR:%.+]] = alloca [10 x i{{[0-9]+}}]*,
// TCHECK-NOT: alloca i{{[0-9]+}},
// TCHECK:  [[B_PRIV:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK:  store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[A_ADDR]],
// TCHECK:  store i{{[0-9]+}} [[A3_IN]], i{{[0-9]+}}* [[A3_ADDR]],
// TCHECK:  store [10 x i{{[0-9]+}}]* [[B_IN]], [10 x i{{[0-9]+}}]** [[B_ADDR]],
// TCHECK-64:  [[A_CONV:%.+]] = bitcast i{{[0-9]+}}* [[A_ADDR]] to i{{[0-9]+}}*
// TCHECK:  [[A3_CONV:%.+]] = bitcast i{{[0-9]+}}* [[A3_ADDR]] to i8*
// TCHECK:  [[B_ADDR_REF:%.+]] = load [10 x i{{[0-9]+}}]*, [10 x i{{[0-9]+}}]** [[B_ADDR]],

// firstprivate(a): a_priv = a_in

// firstprivate(aaa)
// TCHECK-NOT:  store i{{[0-9]+}} %

// firstprivate(b)
// TCHECK:  [[B_PRIV_BCAST:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[B_PRIV]] to i8*
// TCHECK:  [[B_IN_BCAST:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[B_ADDR_REF]] to i8*
// TCHECK:  call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[B_PRIV_BCAST]], i8* align {{[0-9]+}} [[B_IN_BCAST]],{{.+}})

// TCHECK:  ret void

struct S1 {
  double a;

  int r1(int n) {
    int b = n + 1;
    short int c[2][n];

#pragma omp target firstprivate(b, c)
    {
      this->a = (double)b + 1.5;
      c[1][1] = ++a;
    }

    return c[1][1] + (int)b;
  }

  // on the host side, we first generate r1, then the static function and the template above
  // CHECK:  define{{.+}} i32 {{.+}}([[S1]]* {{.+}}, i{{[0-9]+}} {{.+}})
  // CHECK:  [[BASE_PTRS4:%.+]] = alloca [5 x i8*],
  // CHECK:  [[PTRS4:%.+]] = alloca [5 x i8*],
  // CHECK:  [[SIZET4:%.+]] = alloca [5 x i{{[0-9]+}}],

  // map(this: this ptr is implicitly captured (not firstprivate matter)
  // CHECK:  [[BP0:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BASE_PTRS4]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK:  [[CBP0:%.+]] = bitcast i8** [[BP0]] to %struct.S1**
  // CHECK:  store %struct.S1* [[THIS:%.+]], %struct.S1** [[CBP0]],
  // CHECK:  [[P0:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS4]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK:  [[CP0:%.+]] = bitcast i8** [[P0]] to double**
  // CHECK:  store double* [[A:%.+]], double** [[CP0]],

  // firstprivate(b): base_ptr = b, ptr = b, size = 4 (pass by-value)
  // CHECK:  [[BASE_PTRS_GEP4_1:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BASE_PTRS4]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTRS_GEP4_1]] to i{{[0-9]+}}*
  // CHECK:  store i{{[0-9]+}} [[B_CAST:%.+]], i{{[0-9]+}}* [[BCAST_TOPTR]],
  // CHECK:  [[PTRS_GEP4_1:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS4]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[PTRS_GEP4_1]] to i{{[0-9]+}}*
  // CHECK:  store i{{[0-9]+}} [[B_CAST]], i{{[0-9]+}}* [[BCAST_TOPTR]],

  // firstprivate(c), 3 entries: 2, n, c
  // CHECK:  [[BASE_PTRS_GEP4_2:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BASE_PTRS4]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTRS_GEP4_2]] to i{{[0-9]+}}*
  // CHECK:  store i{{[0-9]+}} 2, i{{[0-9]+}}* [[BCAST_TOPTR]],
  // CHECK:  [[PTRS_GEP4_2:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS4]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[PTRS_GEP4_2]] to i{{[0-9]+}}*
  // CHECK:  store i{{[0-9]+}} 2, i{{[0-9]+}}* [[BCAST_TOPTR]],
  // CHECK:  [[BASE_PTRS_GEP4_3:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BASE_PTRS4]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTRS_GEP4_3]] to i{{[0-9]+}}*
  // CHECK:  store i{{[0-9]+}} [[N:%.+]], i{{[0-9]+}}* [[BCAST_TOPTR]],
  // CHECK:  [[PTRS_GEP4_3:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS4]], i{{[0-9]+}} 0, i{{[0-9]+}} 3
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[PTRS_GEP4_3]] to i{{[0-9]+}}*
  // CHECK:  store i{{[0-9]+}} [[N]], i{{[0-9]+}}* [[BCAST_TOPTR]],
  // CHECK:  [[BASE_PTRS_GEP4_4:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BASE_PTRS4]], i{{[0-9]+}} 0, i{{[0-9]+}} 4
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTRS_GEP4_4]] to i{{[0-9]+}}**
  // CHECK:  store i{{[0-9]+}}* [[B:%.+]], i{{[0-9]+}}** [[BCAST_TOPTR]],
  // CHECK:  [[PTRS_GEP4_4:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[PTRS4]], i{{[0-9]+}} 0, i{{[0-9]+}} 4
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[PTRS_GEP4_4]] to i{{[0-9]+}}**
  // CHECK:  store i{{[0-9]+}}* [[B]], i{{[0-9]+}}** [[BCAST_TOPTR]],
  // CHECK:  [[SIZES_GEP4_4:%.+]] = getelementptr inbounds [5 x i{{[0-9]+}}], [5 x i{{[0-9]+}}]* [[SIZET4]], i{{[0-9]+}} 0, i{{[0-9]+}} 4
  // CHECK:  store i{{[0-9]+}} [[B_SIZE:%.+]], i{{[0-9]+}}* [[SIZES_GEP4_4]],

  // only check that we use the map types stored in the global variable
  // CHECK:  call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, {{.+}}, i32 5, i8** {{.+}}, i8** {{.+}}, i{{[0-9]+}}* {{.+}}, i64* getelementptr inbounds ([5 x i64], [5 x i64]* [[MAPT4]], i32 0, i32 0), i8** null, i8** null)

  // TCHECK: define weak_odr void @__omp_offloading_{{.+}}([[S1]]* noundef [[TH:%.+]], i{{[0-9]+}} noundef [[B_IN:%.+]], i{{[0-9]+}} noundef [[VLA:%.+]], i{{[0-9]+}} noundef [[VLA1:%.+]], i{{[0-9]+}}{{.+}} [[C_IN:%.+]])
  // TCHECK:  [[TH_ADDR:%.+]] = alloca [[S1]]*,
  // TCHECK:  [[B_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[VLA_ADDR:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[VLA_ADDR2:%.+]] = alloca i{{[0-9]+}},
  // TCHECK:  [[C_ADDR:%.+]] = alloca i{{[0-9]+}}*,
  // TCHECK-NOT: alloca i{{[0-9]+}},
  // TCHECK:  [[SSTACK:%.+]] = alloca i8*,

  // TCHECK:  store [[S1]]* [[TH]], [[S1]]** [[TH_ADDR]],
  // TCHECK:  store i{{[0-9]+}} [[B_IN]], i{{[0-9]+}}* [[B_ADDR]],
  // TCHECK:  store i{{[0-9]+}} [[VLA]], i{{[0-9]+}}* [[VLA_ADDR]],
  // TCHECK:  store i{{[0-9]+}} [[VLA1]], i{{[0-9]+}}* [[VLA_ADDR2]],
  // TCHECK:  store i{{[0-9]+}}* [[C_IN]], i{{[0-9]+}}** [[C_ADDR]],
  // TCHECK:  [[TH_ADDR_REF:%.+]] = load [[S1]]*, [[S1]]** [[TH_ADDR]],
  // TCHECK-64:  [[B_ADDR_CONV:%.+]] = bitcast i{{[0-9]+}}* [[B_ADDR]] to i{{[0-9]+}}*
  // TCHECK:  [[VLA_ADDR_REF:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[VLA_ADDR]],
  // TCHECK:  [[VLA_ADDR_REF2:%.+]] = load i{{[0-9]+}}, i{{[0-9]+}}* [[VLA_ADDR2]],
  // TCHECK:  [[C_ADDR_REF:%.+]] = load i{{[0-9]+}}*, i{{[0-9]+}}** [[C_ADDR]],

  // firstprivate(b)
  // TCHECK-NOT:  store i{{[0-9]+}} %

  // TCHECK:  [[RET_STACK:%.+]] = call i8* @llvm.stacksave()
  // TCHECK:  store i8* [[RET_STACK:%.+]], i8** [[SSTACK]],

  // firstprivate(c)
  // TCHECK:  [[C_SZ:%.+]] = mul{{.+}} i{{[0-9]+}} [[VLA_ADDR_REF]], [[VLA_ADDR_REF2]]
  // TCHECK:  [[C_PRIV:%.+]] = alloca i{{[0-9]+}}, i{{[0-9]+}} [[C_SZ]],
  // TCHECK:  [[C_SZ2:%.+]] = mul{{.+}} i{{[0-9]+}} [[VLA_ADDR_REF]], [[VLA_ADDR_REF2]]
  // TCHECK:  [[C_SZ_CPY:%.+]] = mul{{.+}} i{{[0-9]+}} [[C_SZ2]],  2
  // TCHECK:  [[C_PRIV_BCAST:%.+]] = bitcast i{{[0-9]+}}* [[C_PRIV]] to i8*
  // TCHECK:  [[C_IN_BCAST:%.+]] = bitcast i{{[0-9]+}}* [[C_ADDR_REF]] to i8*
  // TCHECK:  call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[C_PRIV_BCAST]], i8* align {{[0-9]+}} [[C_IN_BCAST]],{{.+}})

  // finish
  // TCHECK: [[RELOAD_SSTACK:%.+]] = load i8*, i8** [[SSTACK]],
  // TCHECK: call void @llvm.stackrestore(i8* [[RELOAD_SSTACK]])
  // TCHECK: ret void

  // static host function
  // CHECK:  define{{.+}} i32 {{.+}}(i{{[0-9]+}} {{.+}})
  // CHECK:  [[BASE_PTRS5:%.+]] = alloca [3 x i8*],
  // CHECK:  [[PTRS5:%.+]] = alloca [3 x i8*],

  // firstprivate(a): by value
  // CHECK:  [[BASE_PTRS_GEP5_0:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BASE_PTRS5]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTRS_GEP5_0]] to i{{[0-9]+}}*
  // CHECK:  store i{{[0-9]+}} [[A_CAST:%.+]], i{{[0-9]+}}* [[BCAST_TOPTR]],
  // CHECK:  [[PTRS_GEP5_0:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PTRS5]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[PTRS_GEP5_0]] to i{{[0-9]+}}*
  // CHECK:  store i{{[0-9]+}} [[A_CAST]], i{{[0-9]+}}* [[BCAST_TOPTR]],

  // firstprivate(aaa): by value
  // CHECK:  [[BASE_PTRS_GEP5_1:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BASE_PTRS5]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTRS_GEP5_1]] to i{{[0-9]+}}*
  // CHECK:  store i{{[0-9]+}} [[A3_CAST:%.+]], i{{[0-9]+}}* [[BCAST_TOPTR]],
  // CHECK:  [[PTRS_GEP5_1:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PTRS5]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[PTRS_GEP5_1]] to i{{[0-9]+}}*
  // CHECK:  store i{{[0-9]+}} [[A3_CAST]], i{{[0-9]+}}* [[BCAST_TOPTR]],

  // firstprivate(b): base_ptr = &b[0], ptr= &b[0]
  // CHECK:  [[BASE_PTRS_GEP5_2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BASE_PTRS5]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTRS_GEP5_2]] to [10 x i{{[0-9]+}}]**
  // CHECK:  store [10 x i{{[0-9]+}}]* [[B:%.+]], [10 x i{{[0-9]+}}]** [[BCAST_TOPTR]],
  // CHECK:  [[PTRS_GEP5_2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PTRS5]], i{{[0-9]+}} 0, i{{[0-9]+}} 2
  // CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[PTRS_GEP5_2]] to [10 x i{{[0-9]+}}]**
  // CHECK:  store [10 x i{{[0-9]+}}]* [[B]], [10 x i{{[0-9]+}}]** [[BCAST_TOPTR]],

  // only check that the right sizes and map types are used
  // CHECK:  call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, {{.+}}, i32 3, i8** {{.+}}, i8** {{.+}}, i[[SZ]]* getelementptr inbounds ([3 x i[[SZ]]], [3 x i[[SZ]]]* [[SIZET5]], i32 0, i32 0), i64* getelementptr inbounds ([3 x i64], [3 x i64]* [[MAPT5]], i32 0, i32 0), i8** null, i8** null)
};

int bar(int n, double *ptr) {
  int a = 0;
  a += foo(n, ptr);
  S1 S;
  a += S.r1(n);
  a += fstatic(n);
  a += ftemplate<int>(n);

  return a;
}

// template host and device

// CHECK:  define{{.+}} i32 {{.+}}(i{{[0-9]+}} {{.+}})
// CHECK:  [[BASE_PTRS6:%.+]] = alloca [2 x i8*],
// CHECK:  [[PTRS6:%.+]] = alloca [2 x i8*],

// firstprivate(a): by value
// CHECK:  [[BASE_PTRS_GEP6_0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BASE_PTRS6]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTRS_GEP6_0]] to i{{[0-9]+}}*
// CHECK:  store i{{[0-9]+}} [[AT_CAST:%.+]], i{{[0-9]+}}* [[BCAST_TOPTR]],
// CHECK:  [[PTRS_GEP6_0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[PTRS6]], i{{[0-9]+}} 0, i{{[0-9]+}} 0
// CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[PTRS_GEP6_0]] to i{{[0-9]+}}*
// CHECK:  store i{{[0-9]+}} [[AT_CAST]], i{{[0-9]+}}* [[BCAST_TOPTR]],

// firstprivate(b): pointer
// CHECK:  [[BASE_PTRS_GEP6_1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BASE_PTRS6]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[BASE_PTRS_GEP6_1]] to [10 x i{{[0-9]+}}]**
// CHECK:  store [10 x i{{[0-9]+}}]* [[B:%.+]], [10 x i{{[0-9]+}}]** [[BCAST_TOPTR]],
// CHECK:  [[PTRS_GEP6_1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[PTRS6]], i{{[0-9]+}} 0, i{{[0-9]+}} 1
// CHECK:  [[BCAST_TOPTR:%.+]] = bitcast i8** [[PTRS_GEP6_1]] to [10 x i{{[0-9]+}}]**
// CHECK:  store [10 x i{{[0-9]+}}]* [[B]], [10 x i{{[0-9]+}}]** [[BCAST_TOPTR]],

// CHECK:  call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, {{.+}}, i32 2, i8** {{.+}}, i8** {{.+}}, i[[SZ]]* getelementptr inbounds ([2 x i[[SZ]]], [2 x i[[SZ]]]* [[SIZET6]], i32 0, i32 0), i64* getelementptr inbounds ([2 x i64], [2 x i64]* [[MAPT6]], i32 0, i32 0), i8** null, i8** null)

// TCHECK: define weak_odr void @__omp_offloading_{{.+}}(i{{[0-9]+}} noundef [[A_IN:%.+]], [10 x i{{[0-9]+}}]*{{.+}} [[B_IN:%.+]])
// TCHECK:  [[A_ADDR:%.+]] = alloca i{{[0-9]+}},
// TCHECK:  [[B_ADDR:%.+]] = alloca [10 x i{{[0-9]+}}]*,
// TCHECK-NOT: alloca i{{[0-9]+}},
// TCHECK:  [[B_PRIV:%.+]] = alloca [10 x i{{[0-9]+}}],
// TCHECK:  store i{{[0-9]+}} [[A_IN]], i{{[0-9]+}}* [[A_ADDR]],
// TCHECK:  store [10 x i{{[0-9]+}}]* [[B_IN]], [10 x i{{[0-9]+}}]** [[B_ADDR]],
// TCHECK-64:  [[A_ADDR_CONV:%.+]] = bitcast i{{[0-9]+}}* [[A_ADDR]] to i{{[0-9]+}}*
// TCHECK:  [[B_ADDR_REF:%.+]] = load [10 x i{{[0-9]+}}]*, [10 x i{{[0-9]+}}]** [[B_ADDR]],

// firstprivate(a)
// TCHECK-NOT:  store i{{[0-9]+}} %

// firstprivate(b)
// TCHECK:  [[B_PRIV_BCAST:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[B_PRIV]] to i8*
// TCHECK:  [[B_IN_BCAST:%.+]] = bitcast [10 x i{{[0-9]+}}]* [[B_ADDR_REF]] to i8*
// TCHECK:  call void @llvm.memcpy.{{.+}}(i8* align {{[0-9]+}} [[B_PRIV_BCAST]], i8* align {{[0-9]+}} [[B_IN_BCAST]],{{.+}})

// TCHECK: ret void

#endif
