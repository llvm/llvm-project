// Test host codegen.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix TCHECK --check-prefix TCHECK-32
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix TCHECK --check-prefix TCHECK-32

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// Test host codegen.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CHECK --check-prefix CHECK-64
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CHECK --check-prefix CHECK-32
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CHECK --check-prefix CHECK-32

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix TCHECK --check-prefix TCHECK-32
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix TCHECK --check-prefix TCHECK-32

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-DAG: %struct.ident_t = type { i32, i32, i32, i32, i8* }
// CHECK-DAG: [[STR:@.+]] = private unnamed_addr constant [23 x i8] c";unknown;unknown;0;0;;\00"
// CHECK-DAG: [[DEF_LOC:@.+]] = private unnamed_addr constant %struct.ident_t { i32 0, i32 2, i32 0, i32 0, i8* getelementptr inbounds ([23 x i8], [23 x i8]* [[STR]], i32 0, i32 0) }

// CHECK-DAG: [[TT:%.+]] = type { i64, i8 }
// CHECK-DAG: [[S1:%.+]] = type { double }
// CHECK-DAG: [[ENTTY:%.+]] = type { i8*, i8*, i[[SZ:32|64]], i32, i32 }

// TCHECK: [[ENTTY:%.+]] = type { i8*, i8*, i{{32|64}}, i32, i32 }

// We have 8 target regions, but only 7 that actually will generate offloading
// code, only 6 will have mapped arguments, and only 4 have all-constant map
// sizes.

// CHECK-DAG: [[SIZET:@.+]] = private unnamed_addr constant [3 x i64] [i64 2, i64 4, i64 4]
// CHECK-DAG: [[MAPT:@.+]] = private unnamed_addr constant [3 x i64] [i64 288, i64 800, i64 800]
// CHECK-DAG: [[SIZET2:@.+]] = private unnamed_addr constant [1 x i64] [i64 2]
// CHECK-DAG: [[MAPT2:@.+]] = private unnamed_addr constant [1 x i64] [i64 800]
// CHECK-DAG: [[SIZET3:@.+]] = private unnamed_addr constant [2 x i64] [i64 4, i64 2]
// CHECK-DAG: [[MAPT3:@.+]] = private unnamed_addr constant [2 x i64] [i64 800, i64 800]
// CHECK-DAG: [[MAPT4:@.+]] = private unnamed_addr constant [10 x i64] [i64 800, i64 547, i64 800, i64 547, i64 547, i64 800, i64 800, i64 547, i64 547, i64 800]
// CHECK-DAG: [[MAPT5:@.+]] = private unnamed_addr constant [6 x i64] [i64 32, i64 281474976711171, i64 800, i64 800, i64 800, i64 547]
// CHECK-DAG: [[SIZET6:@.+]] = private unnamed_addr constant [5 x i64] [i64 4, i64 4, i64 2, i64 1, i64 40]
// CHECK-DAG: [[MAPT6:@.+]] = private unnamed_addr constant [5 x i64] [i64 800, i64 800, i64 800, i64 800, i64 547]
// CHECK-DAG: [[SIZET7:@.+]] = private unnamed_addr constant [3 x i64] [i64 4, i64 2, i64 40]
// CHECK-DAG: [[MAPT7:@.+]] = private unnamed_addr constant [3 x i64] [i64 800, i64 800, i64 547]
// CHECK-DAG: @{{.*}} = weak constant i8 0
// CHECK-DAG: @{{.*}} = weak constant i8 0
// CHECK-DAG: @{{.*}} = weak constant i8 0
// CHECK-DAG: @{{.*}} = weak constant i8 0
// CHECK-DAG: @{{.*}} = weak constant i8 0
// CHECK-DAG: @{{.*}} = weak constant i8 0
// CHECK-DAG: @{{.*}} = weak constant i8 0

// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK-NOT: @{{.+}} = weak constant [[ENTTY]]

// Check target registration is registered as a Ctor.
// CHECK: appending global [1 x { i32, void ()*, i8* }] [{ i32, void ()*, i8* } { i32 0, void ()* @.omp_offloading.requires_reg, i8* null }]


template<typename tx, typename ty>
struct TT{
  tx X;
  ty Y;
};

int global;

// CHECK: define {{.*}}[[FOO:@.+]](
int foo(int n) {
  int a = 0;
  short aa = 0;
  float b[10];
  float bn[n];
  double c[5][10];
  double cn[5][n];
  TT<long long, char> d;

  // CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams_nowait_mapper(i64 -1, i8* @{{[^,]+}}, i32 3, i8** [[BP:%[^,]+]], i8** [[P:%[^,]+]], i64* getelementptr inbounds ([3 x i64], [3 x i64]* [[SIZET]], i32 0, i32 0), i64* getelementptr inbounds ([3 x i64], [3 x i64]* [[MAPT]], i32 0, i32 0), i8** null, i32 {{[^,]+}}, i32 {{[^)]+}})
  // CHECK-DAG:   [[BP]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BPR:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[P]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PR:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BPR]], i32 0, i32 [[IDX0:[0-9]+]]
  // CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PR]], i32 0, i32 [[IDX0]]
  // CHECK-DAG:   [[CBPADDR0:%.+]] = bitcast i8** [[BPADDR0]] to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR0:%.+]] = bitcast i8** [[PADDR0]] to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[CBPADDR0]]
  // CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[CPADDR0]]
  // CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BPR]], i32 0, i32 [[IDX1:[0-9]+]]
  // CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PR]], i32 0, i32 [[IDX1]]
  // CHECK-DAG:   [[CBPADDR1:%.+]] = bitcast i8** [[BPADDR1]] to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR1:%.+]] = bitcast i8** [[PADDR1]] to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[CBPADDR1]]
  // CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[CPADDR1]]
  // CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BPR]], i32 0, i32 [[IDX1:[0-9]+]]
  // CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[PR]], i32 0, i32 [[IDX1]]
  // CHECK-DAG:   [[CBPADDR2:%.+]] = bitcast i8** [[BPADDR2]] to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR2:%.+]] = bitcast i8** [[PADDR2]] to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[CBPADDR2]]
  // CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[CPADDR2]]
  // CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT0:@.+]](i[[SZ]] {{[^,]+}}, i[[SZ]] {{[^,]+}}, i[[SZ]] {{[^)]+}})
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  #pragma omp target teams distribute num_teams(a) thread_limit(a) firstprivate(aa) nowait
  for (int i = 0; i < 10; ++i) {
  }

  // CHECK:       call void [[HVT1:@.+]](i[[SZ]] {{[^,]+}})
  #pragma omp target teams distribute if(target: 0)
  for (int i = 0; i < 10; ++i) {
    a += 1;
  }

  // CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams_mapper(i64 -1, i8* @{{[^,]+}}, i32 1, i8** [[BP:%[^,]+]], i8** [[P:%[^,]+]], i64* getelementptr inbounds ([1 x i64], [1 x i64]* [[SIZET2]], i32 0, i32 0), i64* getelementptr inbounds ([1 x i64], [1 x i64]* [[MAPT2]], i32 0, i32 0), i8** null, i32 0, i32 0)
  // CHECK-DAG:   [[BP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BPR:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[P]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[PR:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BPR]], i32 0, i32 [[IDX0:[0-9]+]]
  // CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[PR]], i32 0, i32 [[IDX0]]
  // CHECK-DAG:   [[CBPADDR0:%.+]] = bitcast i8** [[BPADDR0]] to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR0:%.+]] = bitcast i8** [[PADDR0]] to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[CBPADDR0]]
  // CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[CPADDR0]]

  // CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT2:@.+]](i[[SZ]] {{[^,]+}})
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  #pragma omp target teams distribute if(target: 1)
  for (int i = 0; i < 10; ++i) {
    aa += 1;
  }

  // CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 10
  // CHECK:       br i1 [[IF]], label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
  // CHECK:       [[IFTHEN]]
  // CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams_mapper(i64 -1, i8* @{{[^,]+}}, i32 2, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i64* getelementptr inbounds ([2 x i64], [2 x i64]* [[SIZET3]], i32 0, i32 0), i64* getelementptr inbounds ([2 x i64], [2 x i64]* [[MAPT3]], i32 0, i32 0), i8** null, i32 0, i32 0)
  // CHECK-DAG:   [[BPR]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BP:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[PR]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[P:%[^,]+]], i32 0, i32 0

  // CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BP]], i32 0, i32 0
  // CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[P]], i32 0, i32 0
  // CHECK-DAG:   [[CBPADDR0:%.+]] = bitcast i8** [[BPADDR0]] to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR0:%.+]] = bitcast i8** [[PADDR0]] to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[CBPADDR0]]
  // CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[CPADDR0]]

  // CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BP]], i32 0, i32 1
  // CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[P]], i32 0, i32 1
  // CHECK-DAG:   [[CBPADDR1:%.+]] = bitcast i8** [[BPADDR1]] to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR1:%.+]] = bitcast i8** [[PADDR1]] to i[[SZ]]*
  // CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[CBPADDR1]]
  // CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[CPADDR1]]
  // CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT3:@.+]]({{[^,]+}}, {{[^,]+}})
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  // CHECK-NEXT:  br label %[[IFEND:.+]]
  // CHECK:       [[IFELSE]]
  // CHECK:       call void [[HVT3]]({{[^,]+}}, {{[^,]+}})
  // CHECK-NEXT:  br label %[[IFEND]]
  // CHECK:       [[IFEND]]
  #pragma omp target teams distribute if(target: n>10)
  for (int i = 0; i < 10; ++i) {
    a += 1;
    aa += 1;
  }

  // We capture 3 VLA sizes in this target region
  // CHECK:          load i32, i32* %
  // CHECK-64:       [[A_VAL:%.+]] = load i32, i32* %{{.+}},
  // CHECK-64:       [[A_ADDR:%.+]] = bitcast i[[SZ]]* [[A_CADDR:%.+]] to i32*
  // CHECK-64:       store i32 [[A_VAL]], i32* [[A_ADDR]],
  // CHECK-64:       [[A_CVAL:%.+]] = load i[[SZ]], i[[SZ]]* [[A_CADDR]],

  // CHECK-32:       [[A_VAL:%.+]] = load i32, i32* %{{.+}},
  // CHECK-32:       store i32 [[A_VAL]], i32* [[A_CADDR:%.+]],
  // CHECK-32:       [[A_CVAL:%.+]] = load i[[SZ]], i[[SZ]]* [[A_CADDR]],

  // CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 20
  // CHECK:       br i1 [[IF]], label %[[TRY:[^,]+]], label %[[FAIL:[^,]+]]
  // CHECK:       [[TRY]]
  // CHECK-64:    [[BNSIZE:%.+]] = mul nuw i64 [[VLA0:%.+]], 4
  // CHECK-32:    [[BNSZSIZE:%.+]] = mul nuw i32 [[VLA0:%.+]], 4
  // CHECK-32:    [[BNSIZE:%.+]] = sext i32 [[BNSZSIZE]] to i64
  // CHECK:       [[CNELEMSIZE2:%.+]] = mul nuw i[[SZ]] 5, [[VLA1:%.+]]
  // CHECK-64:    [[CNSIZE:%.+]] = mul nuw i64 [[CNELEMSIZE2]], 8
  // CHECK-32:    [[CNSZSIZE:%.+]] = mul nuw i32 [[CNELEMSIZE2]], 8
  // CHECK-32:    [[CNSIZE:%.+]] = sext i32 [[CNSZSIZE]] to i64

  // CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams_mapper(i64 -1, i8* @{{[^,]+}}, i32 10, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i64* [[SR:%[^,]+]], i64* getelementptr inbounds ([10 x i64], [10 x i64]* [[MAPT4]], i32 0, i32 0), i8** null, i32 0, i32 0)
  // CHECK-DAG:   [[BPR]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[BP:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[PR]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[P:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[SR]] = getelementptr inbounds [10 x i64], [10 x i64]* [[S:%[^,]+]], i32 0, i32 0

  // CHECK-DAG:   [[SADDR0:%.+]] = getelementptr inbounds [10 x i64], [10 x i64]* [[S]], i32 0, i32 [[IDX0:[0-9]+]]
  // CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[BP]], i32 0, i32 [[IDX0]]
  // CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[P]], i32 0, i32 [[IDX0]]
  // CHECK-DAG:   [[SADDR1:%.+]] = getelementptr inbounds [10 x i64], [10 x i64]* [[S]], i32 0, i32 [[IDX1:[0-9]+]]
  // CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[BP]], i32 0, i32 [[IDX1]]
  // CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[P]], i32 0, i32 [[IDX1]]
  // CHECK-DAG:   [[SADDR2:%.+]] = getelementptr inbounds [10 x i64], [10 x i64]* [[S]], i32 0, i32 [[IDX2:[0-9]+]]
  // CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[BP]], i32 0, i32 [[IDX2]]
  // CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[P]], i32 0, i32 [[IDX2]]
  // CHECK-DAG:   [[SADDR3:%.+]] = getelementptr inbounds [10 x i64], [10 x i64]* [[S]], i32 0, i32 [[IDX3:[0-9]+]]
  // CHECK-DAG:   [[BPADDR3:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[BP]], i32 0, i32 [[IDX3]]
  // CHECK-DAG:   [[PADDR3:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[P]], i32 0, i32 [[IDX3]]
  // CHECK-DAG:   [[SADDR4:%.+]] = getelementptr inbounds [10 x i64], [10 x i64]* [[S]], i32 0, i32 [[IDX4:[0-9]+]]
  // CHECK-DAG:   [[BPADDR4:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[BP]], i32 0, i32 [[IDX4]]
  // CHECK-DAG:   [[PADDR4:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[P]], i32 0, i32 [[IDX4]]
  // CHECK-DAG:   [[SADDR5:%.+]] = getelementptr inbounds [10 x i64], [10 x i64]* [[S]], i32 0, i32 [[IDX5:[0-9]+]]
  // CHECK-DAG:   [[BPADDR5:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[BP]], i32 0, i32 [[IDX5]]
  // CHECK-DAG:   [[PADDR5:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[P]], i32 0, i32 [[IDX5]]
  // CHECK-DAG:   [[SADDR6:%.+]] = getelementptr inbounds [10 x i64], [10 x i64]* [[S]], i32 0, i32 [[IDX6:[0-9]+]]
  // CHECK-DAG:   [[BPADDR6:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[BP]], i32 0, i32 [[IDX6]]
  // CHECK-DAG:   [[PADDR6:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[P]], i32 0, i32 [[IDX6]]
  // CHECK-DAG:   [[SADDR7:%.+]] = getelementptr inbounds [10 x i64], [10 x i64]* [[S]], i32 0, i32 [[IDX7:[0-9]+]]
  // CHECK-DAG:   [[BPADDR7:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[BP]], i32 0, i32 [[IDX7]]
  // CHECK-DAG:   [[PADDR7:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[P]], i32 0, i32 [[IDX7]]
  // CHECK-DAG:   [[SADDR8:%.+]] = getelementptr inbounds [10 x i64], [10 x i64]* [[S]], i32 0, i32 [[IDX8:[0-9]+]]
  // CHECK-DAG:   [[BPADDR8:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[BP]], i32 0, i32 [[IDX8]]
  // CHECK-DAG:   [[PADDR8:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[P]], i32 0, i32 [[IDX8]]
  // CHECK-DAG:   [[SADDR9:%.+]] = getelementptr inbounds [10 x i64], [10 x i64]* [[S]], i32 0, i32 [[IDX9:[0-9]+]]
  // CHECK-DAG:   [[BPADDR9:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[BP]], i32 0, i32 [[IDX9]]
  // CHECK-DAG:   [[PADDR9:%.+]] = getelementptr inbounds [10 x i8*], [10 x i8*]* [[P]], i32 0, i32 [[IDX9]]

  // The names below are not necessarily consistent with the names used for the
  // addresses above as some are repeated.
  // CHECK-DAG:   store i[[SZ]] [[VLA0]], i[[SZ]]* [[CBPADDR0:%.+]],
  // CHECK-DAG:   store i[[SZ]] [[VLA0]], i[[SZ]]* [[CPADDR0:%.+]],
  // CHECK-DAG:   [[CBPADDR0]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR0]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
  // CHECK-DAG:   store i64 {{4|8}}, i64* {{%[^,]+}}

  // CHECK-DAG:   store i[[SZ]] [[VLA1]], i[[SZ]]* [[CBPADDR1:%.+]],
  // CHECK-DAG:   store i[[SZ]] [[VLA1]], i[[SZ]]* [[CPADDR1:%.+]],
  // CHECK-DAG:   [[CBPADDR1]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR1]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
  // CHECK-DAG:   store i64 {{4|8}}, i64* {{%[^,]+}}

  // CHECK-DAG:   store i[[SZ]] 5, i[[SZ]]* [[CBPADDR2:%.+]],
  // CHECK-DAG:   store i[[SZ]] 5, i[[SZ]]* [[CPADDR2:%.+]],
  // CHECK-DAG:   [[CBPADDR2]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR2]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
  // CHECK-DAG:   store i64 {{4|8}}, i64* {{%[^,]+}}

  // CHECK-DAG:   store i[[SZ]] [[A_CVAL]], i[[SZ]]* [[CBPADDR3:%.+]],
  // CHECK-DAG:   store i[[SZ]] [[A_CVAL]], i[[SZ]]* [[CPADDR3:%.+]],
  // CHECK-DAG:   [[CBPADDR3]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR3]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
  // CHECK-DAG:   store i64 4, i64* {{%[^,]+}}

  // CHECK-DAG:   store [10 x float]* %{{.+}}, [10 x float]** [[CBPADDR4:%.+]],
  // CHECK-DAG:   store [10 x float]* %{{.+}}, [10 x float]** [[CPADDR4:%.+]],
  // CHECK-DAG:   [[CBPADDR4]] = bitcast i8** {{%[^,]+}} to [10 x float]**
  // CHECK-DAG:   [[CPADDR4]] = bitcast i8** {{%[^,]+}} to [10 x float]**
  // CHECK-DAG:   store i64 40, i64* {{%[^,]+}}

  // CHECK-DAG:   store float* %{{.+}}, float** [[CBPADDR5:%.+]],
  // CHECK-DAG:   store float* %{{.+}}, float** [[CPADDR5:%.+]],
  // CHECK-DAG:   [[CBPADDR5]] = bitcast i8** {{%[^,]+}} to  float**
  // CHECK-DAG:   [[CPADDR5]] = bitcast i8** {{%[^,]+}} to float**
  // CHECK-DAG:   store i64 [[BNSIZE]], i64* {{%[^,]+}}

  // CHECK-DAG:   store [5 x [10 x double]]* %{{.+}}, [5 x [10 x double]]** [[CBPADDR6:%.+]],
  // CHECK-DAG:   store [5 x [10 x double]]* %{{.+}}, [5 x [10 x double]]** [[CPADDR6:%.+]],
  // CHECK-DAG:   [[CBPADDR6]] = bitcast i8** {{%[^,]+}} to  [5 x [10 x double]]**
  // CHECK-DAG:   [[CPADDR6]] = bitcast i8** {{%[^,]+}} to [5 x [10 x double]]**
  // CHECK-DAG:   store i64 400, i64* {{%[^,]+}}

  // CHECK-DAG:   store double* %{{.+}}, double** [[CBPADDR7:%.+]],
  // CHECK-DAG:   store double* %{{.+}}, double** [[CPADDR7:%.+]],
  // CHECK-DAG:   [[CBPADDR7]] = bitcast i8** {{%[^,]+}} to  double**
  // CHECK-DAG:   [[CPADDR7]] = bitcast i8** {{%[^,]+}} to double**
  // CHECK-DAG:   store i64 [[CNSIZE]], i64* {{%[^,]+}}

  // CHECK-DAG:   store [[TT]]* %{{.+}}, [[TT]]** [[CBPADDR8:%.+]],
  // CHECK-DAG:   store [[TT]]* %{{.+}}, [[TT]]** [[CPADDR8:%.+]],
  // CHECK-DAG:   [[CBPADDR8]] = bitcast i8** {{%[^,]+}} to  [[TT]]**
  // CHECK-DAG:   [[CPADDR8]] = bitcast i8** {{%[^,]+}} to [[TT]]**
  // CHECK-DAG:   store i64 {{12|16}}, i64* {{%[^,]+}}

  // CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[CBPADDR9:%.+]],
  // CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[CPADDR9:%.+]],
  // CHECK-DAG:   [[CBPADDR9]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
  // CHECK-DAG:   [[CPADDR9]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
  // CHECK-DAG:   store i64 {{4|8}}, i64* {{%[^,]+}}

  // CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]

  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT4:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  #pragma omp target teams distribute if(target: n>20) dist_schedule(static, n)
  for (int i = 0; i < 10; ++i) {
    a += 1;
    b[2] += 1.0;
    bn[3] += 1.0;
    c[1][2] += 1.0;
    cn[1][3] += 1.0;
    d.X += 1;
    d.Y += 1;
  }

  return a;
}

// Check that the offloading functions are emitted and that the arguments are
// correct and loaded correctly for the target regions in foo().

// CHECK:       define internal void [[HVT0]](i[[SZ]] {{[^,]+}}, i[[SZ]] {{[^,]+}}, i[[SZ]] {{[^)]+}})
// CHECK:       call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(%struct.ident_t* [[DEF_LOC]], i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i[[SZ]])* [[OMP_OUTLINED:@.+]] to void (i32*, i32*, ...)*), i[[SZ]] {{[^)]+}})
//
//
// CHECK:       define internal {{.*}}void [[OMP_OUTLINED]](i32* noalias %.global_tid., i32* noalias %.bound_tid., i[[SZ]] {{[^)]+}})
// CHECK:       alloca i[[SZ]],
// CHECK:       bitcast i[[SZ]]* {{.+}} to i16*
// CHECK:       ret void
// CHECK-NEXT:  }


// CHECK:       define internal void [[HVT1]](i[[SZ]] %{{.+}})
// Create stack storage and store argument in there.
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       [[AA_CASTED:%.+]] = alloca i[[SZ]], align
// CHECK:       store i[[SZ]] %{{.+}}, i[[SZ]]* [[AA_ADDR]], align
// CHECK-64:    [[AA_CADDR:%.+]] = bitcast i[[SZ]]* [[AA_ADDR]] to i32*
// CHECK-64:    [[AA:%.+]] = load i32, i32* [[AA_CADDR]], align
// CHECK-32:    [[AA:%.+]] = load i32, i32* [[AA_ADDR]], align
// CHECK-64:    [[AA_C:%.+]] = bitcast i[[SZ]]* [[AA_CASTED]] to i32*
// CHECK-64:    store i32 [[AA]], i32* [[AA_C]], align
// CHECK-32:    store i32 [[AA]], i32* [[AA_CASTED]], align
// CHECK:       [[PARAM:%.+]] = load i[[SZ]], i[[SZ]]* [[AA_CASTED]], align
// CHECK:       call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(%struct.ident_t* [[DEF_LOC]], i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i[[SZ]])* [[OMP_OUTLINED1:@.+]] to void (i32*, i32*, ...)*), i[[SZ]] [[PARAM]])
//
//
// CHECK:       define internal {{.*}}void [[OMP_OUTLINED1]](i32* noalias %.global_tid., i32* noalias %.bound_tid., i[[SZ]] %{{.+}})
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       store i[[SZ]] %{{.+}}, i[[SZ]]* [[AA_ADDR]], align
// CHECK-64:    [[AA_CADDR:%.+]] = bitcast i[[SZ]]* [[AA_ADDR]] to i32*
// CHECK-64:    [[AA:%.+]] = load i32, i32* [[AA_CADDR]], align
// CHECK-32:    [[AA:%.+]] = load i32, i32* [[AA_ADDR]], align
// CHECK:       ret void
// CHECK-NEXT:  }

// CHECK:       define internal void [[HVT2]](i[[SZ]] %{{.+}})
// Create stack storage and store argument in there.
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       [[AA_CASTED:%.+]] = alloca i[[SZ]], align
// CHECK:       store i[[SZ]] %{{.+}}, i[[SZ]]* [[AA_ADDR]], align
// CHECK:       [[AA_CADDR:%.+]] = bitcast i[[SZ]]* [[AA_ADDR]] to i16*
// CHECK:       [[AA:%.+]] = load i16, i16* [[AA_CADDR]], align
// CHECK:       [[AA_C:%.+]] = bitcast i[[SZ]]* [[AA_CASTED]] to i16*
// CHECK:       store i16 [[AA]], i16* [[AA_C]], align
// CHECK:       [[PARAM:%.+]] = load i[[SZ]], i[[SZ]]* [[AA_CASTED]], align
// CHECK:       call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(%struct.ident_t* [[DEF_LOC]], i32 1, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i[[SZ]])* [[OMP_OUTLINED2:@.+]] to void (i32*, i32*, ...)*), i[[SZ]] [[PARAM]])
//
//
// CHECK:       define internal {{.*}}void [[OMP_OUTLINED2]](i32* noalias %.global_tid., i32* noalias %.bound_tid., i[[SZ]] %{{.+}})
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       store i[[SZ]] %{{.+}}, i[[SZ]]* [[AA_ADDR]], align
// CHECK:       [[AA_CADDR:%.+]] = bitcast i[[SZ]]* [[AA_ADDR]] to i16*
// CHECK:       [[AA:%.+]] = load i16, i16* [[AA_CADDR]], align
// CHECK:       ret void
// CHECK-NEXT:  }

// CHECK:       define internal void [[HVT3]]
// Create stack storage and store argument in there.
// CHECK:       [[A_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       [[A_CASTED:%.+]] = alloca i[[SZ]], align
// CHECK:       [[AA_CASTED:%.+]] = alloca i[[SZ]], align
// CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[A_ADDR]], align
// CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[AA_ADDR]], align
// CHECK-64-DAG:[[A_CADDR:%.+]] = bitcast i[[SZ]]* [[A_ADDR]] to i32*
// CHECK-DAG:   [[AA_CADDR:%.+]] = bitcast i[[SZ]]* [[AA_ADDR]] to i16*
// CHECK-64-DAG:[[A:%.+]] = load i32, i32* [[A_CADDR]], align
// CHECK-32-DAG:[[A:%.+]] = load i32, i32* [[A_ADDR]], align
// CHECK-64-DAG:[[A_C:%.+]] = bitcast i[[SZ]]* [[A_CASTED]] to i32*
// CHECK-64-DAG:store i32 [[A]], i32* [[A_C]], align
// CHECK-32-DAG:store i32 [[A]], i32* [[A_CASTED]], align
// CHECK-DAG:   [[AA:%.+]] = load i16, i16* [[AA_CADDR]], align
// CHECK-DAG:   [[AA_C:%.+]] = bitcast i[[SZ]]* [[AA_CASTED]] to i16*
// CHECK-DAG:   store i16 [[AA]], i16* [[AA_C]], align
// CHECK-DAG:   [[PARAM1:%.+]] = load i[[SZ]], i[[SZ]]* [[A_CASTED]], align
// CHECK-DAG:   [[PARAM2:%.+]] = load i[[SZ]], i[[SZ]]* [[AA_CASTED]], align
// CHECK-DAG:   call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(%struct.ident_t* [[DEF_LOC]], i32 2, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i[[SZ]], i[[SZ]])* [[OMP_OUTLINED3:@.+]] to void (i32*, i32*, ...)*), i[[SZ]] [[PARAM1]], i[[SZ]] [[PARAM2]])
//
//
// CHECK:       define internal {{.*}}void [[OMP_OUTLINED3]](i32* noalias %.global_tid., i32* noalias %.bound_tid., i[[SZ]] %{{.+}}, i[[SZ]] %{{.+}})
// CHECK:       [[A_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[A_ADDR]], align
// CHECK-DAG:   store i[[SZ]] %{{.+}}, i[[SZ]]* [[AA_ADDR]], align
// CHECK-64-DAG:[[A_CADDR:%.+]] = bitcast i[[SZ]]* [[A_ADDR]] to i32*
// CHECK-DAG:   [[AA_CADDR:%.+]] = bitcast i[[SZ]]* [[AA_ADDR]] to i16*
// CHECK:       ret void
// CHECK-NEXT:  }

// CHECK:       define internal void [[HVT4]]
// Create local storage for each capture.
// CHECK:       [[LOCAL_A:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_B:%.+]] = alloca [10 x float]*
// CHECK:       [[LOCAL_VLA1:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_BN:%.+]] = alloca float*
// CHECK:       [[LOCAL_C:%.+]] = alloca [5 x [10 x double]]*
// CHECK:       [[LOCAL_VLA2:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_VLA3:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_CN:%.+]] = alloca double*
// CHECK:       [[LOCAL_D:%.+]] = alloca [[TT]]*
// CHECK:       alloca i[[SZ]],
// CHECK:       [[LOCAL_A_CASTED:%.+]] = alloca i[[SZ]]
// CHECK-DAG:   store i[[SZ]] [[ARG_A:%.+]], i[[SZ]]* [[LOCAL_A]]
// CHECK-DAG:   store [10 x float]* [[ARG_B:%.+]], [10 x float]** [[LOCAL_B]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA1:%.+]], i[[SZ]]* [[LOCAL_VLA1]]
// CHECK-DAG:   store float* [[ARG_BN:%.+]], float** [[LOCAL_BN]]
// CHECK-DAG:   store [5 x [10 x double]]* [[ARG_C:%.+]], [5 x [10 x double]]** [[LOCAL_C]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA2:%.+]], i[[SZ]]* [[LOCAL_VLA2]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA3:%.+]], i[[SZ]]* [[LOCAL_VLA3]]
// CHECK-DAG:   store double* [[ARG_CN:%.+]], double** [[LOCAL_CN]]
// CHECK-DAG:   store [[TT]]* [[ARG_D:%.+]], [[TT]]** [[LOCAL_D]]

// CHECK-64-DAG:[[CONV_AP:%.+]] = bitcast i[[SZ]]* [[LOCAL_A]] to i32*
// CHECK-DAG:   [[REF_B:%.+]] = load [10 x float]*, [10 x float]** [[LOCAL_B]],
// CHECK-DAG:   [[VAL_VLA1:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA1]],
// CHECK-DAG:   [[REF_BN:%.+]] = load float*, float** [[LOCAL_BN]],
// CHECK-DAG:   [[REF_C:%.+]] = load [5 x [10 x double]]*, [5 x [10 x double]]** [[LOCAL_C]],
// CHECK-DAG:   [[VAL_VLA2:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA2]],
// CHECK-DAG:   [[VAL_VLA3:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA3]],
// CHECK-DAG:   [[REF_CN:%.+]] = load double*, double** [[LOCAL_CN]],
// CHECK-DAG:   [[REF_D:%.+]] = load [[TT]]*, [[TT]]** [[LOCAL_D]],

// CHECK-64-DAG:[[CONV_A:%.+]] = load i32, i32* [[CONV_AP]]
// CHECK-64-DAG:[[CONV:%.+]] = bitcast i[[SZ]]* [[LOCAL_A_CASTED]] to i32*
// CHECK-64-DAG:store i32 [[CONV_A]], i32* [[CONV]], align
// CHECK-32-DAG:[[LOCAL_AV:%.+]] = load i32, i32* [[LOCAL_A]]
// CHECK-32-DAG:store i32 [[LOCAL_AV]], i32* [[LOCAL_A_CASTED]], align
// CHECK-DAG:   [[REF_A:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_A_CASTED]],

// CHECK:       call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(%struct.ident_t* [[DEF_LOC]], i32 10, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i[[SZ]], [10 x float]*, i[[SZ]], float*, [5 x [10 x double]]*, i[[SZ]], i[[SZ]], double*, [[TT]]*, i[[SZ]])* [[OMP_OUTLINED4:@.+]] to void (i32*, i32*, ...)*), i[[SZ]] [[REF_A]], [10 x float]* [[REF_B]], i[[SZ]] [[VAL_VLA1]], float* [[REF_BN]], [5 x [10 x double]]* [[REF_C]], i[[SZ]] [[VAL_VLA2]], i[[SZ]] [[VAL_VLA3]], double* [[REF_CN]], [[TT]]* [[REF_D]], i[[SZ]] %{{.+}})
//
//
// CHECK:       define internal {{.*}}void [[OMP_OUTLINED4]](i32* noalias %.global_tid., i32* noalias %.bound_tid., i[[SZ]] %{{.+}}, [10 x float]* {{.+}}, i[[SZ]] %{{.+}}, float* {{.+}}, [5 x [10 x double]]* {{.+}}, i[[SZ]] %{{.+}}, i[[SZ]] %{{.+}}, double* {{.+}}, [[TT]]* {{.+}})
// To reduce complexity, we're only going as far as validating the signature of the outlined parallel function.

template<typename tx>
tx ftemplate(int n) {
  tx a = 0;
  short aa = 0;
  tx b[10];

  #pragma omp target teams distribute if(target: n>40)
  for (int i = 0; i < 10; ++i) {
    a += 1;
    aa += 1;
    b[2] += 1;
  }

  return a;
}

static
int fstatic(int n) {
  int a = 0;
  short aa = 0;
  char aaa = 0;
  int b[10];

  #pragma omp target teams distribute if(target: n>50)
  for (int i = a; i < n; ++i) {
    a += 1;
    aa += 1;
    aaa += 1;
    b[2] += 1;
  }

  return a;
}

struct S1 {
  double a;

  int r1(int n){
    int b = n+1;
    short int c[2][n];

    #pragma omp target teams distribute if(target: n>60)
    for (int i = 0; i < 10; ++i) {
      this->a = (double)b + 1.5;
      c[1][1] = ++a;
    }

    return c[1][1] + (int)b;
  }
};

// CHECK: define {{.*}}@{{.*}}bar{{.*}}
int bar(int n){
  int a = 0;

  // CHECK: call {{.*}}i32 [[FOO]](i32 {{.*}})
  a += foo(n);

  S1 S;
  // CHECK: call {{.*}}i32 [[FS1:@.+]]([[S1]]* {{.*}}, i32 {{.*}})
  a += S.r1(n);

  // CHECK: call {{.*}}i32 [[FSTATIC:@.+]](i32 {{.*}})
  a += fstatic(n);

  // CHECK: call {{.*}}i32 [[FTEMPLATE:@.+]](i32 {{.*}})
  a += ftemplate<int>(n);

  return a;
}

//
// CHECK: define {{.*}}[[FS1]]
//
// CHECK:          i8* @llvm.stacksave()
// CHECK-64:       [[B_ADDR:%.+]] = bitcast i[[SZ]]* [[B_CADDR:%.+]] to i32*
// CHECK-64:       store i32 %{{.+}}, i32* [[B_ADDR]],
// CHECK-64:       [[B_CVAL:%.+]] = load i[[SZ]], i[[SZ]]* [[B_CADDR]],

// CHECK-32:       store i32 %{{.+}}, i32* %__vla_expr
// CHECK-32:       store i32 %{{.+}}, i32* [[B_ADDR:%.+]],
// CHECK-32:       [[B_CVAL:%.+]] = load i[[SZ]], i[[SZ]]* [[B_ADDR]],

// CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 60
// CHECK:       br i1 [[IF]], label %[[TRY:[^,]+]], label %[[FAIL:[^,]+]]
// CHECK:       [[TRY]]
// We capture 2 VLA sizes in this target region
// CHECK:       [[CELEMSIZE2:%.+]] = mul nuw i[[SZ]] 2, [[VLA0:%.+]]
// CHECK-64:    [[CSIZE:%.+]] = mul nuw i[[SZ]] [[CELEMSIZE2]], 2
// CHECK-32:    [[CSSZSIZE:%.+]] = mul nuw i[[SZ]] [[CELEMSIZE2]], 2
// CHECK-32:    [[CSIZE:%.+]] = sext i32 [[CSSZSIZE]] to i64

// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams_mapper(i64 -1, i8* @{{[^,]+}}, i32 6, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i64* [[SR:%[^,]+]], i64* getelementptr inbounds ([6 x i64], [6 x i64]* [[MAPT5]], i32 0, i32 0), i8** null, i32 0, i32 0)
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BP:%.+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[P:%.+]], i32 0, i32 0
// CHECK-DAG:   [[SR]] = getelementptr inbounds [6 x i64], [6 x i64]* [[S:%.+]], i32 0, i32 0
// CHECK-DAG:   [[SADDR0:%.+]] = getelementptr inbounds [6 x i64], [6 x i64]* [[S]], i32 [[IDX0:[0-9]+]]
// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BP]], i32 [[IDX0]]
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[P]], i32 [[IDX0]]
// CHECK-DAG:   [[SADDR1:%.+]] = getelementptr inbounds [6 x i64], [6 x i64]* [[S]], i32 [[IDX1:[0-9]+]]
// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BP]], i32 [[IDX1]]
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[P]], i32 [[IDX1]]
// CHECK-DAG:   [[SADDR2:%.+]] = getelementptr inbounds [6 x i64], [6 x i64]* [[S]], i32 [[IDX2:[0-9]+]]
// CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BP]], i32 [[IDX2]]
// CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[P]], i32 [[IDX2]]
// CHECK-DAG:   [[SADDR3:%.+]] = getelementptr inbounds [6 x i64], [6 x i64]* [[S]], i32 [[IDX3:[0-9]+]]
// CHECK-DAG:   [[BPADDR3:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BP]], i32 [[IDX3]]
// CHECK-DAG:   [[PADDR3:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[P]], i32 [[IDX3]]
// CHECK-DAG:   [[SADDR4:%.+]] = getelementptr inbounds [6 x i64], [6 x i64]* [[S]], i32 [[IDX3:[0-9]+]]
// CHECK-DAG:   [[BPADDR4:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BP]], i32 [[IDX3]]
// CHECK-DAG:   [[PADDR4:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[P]], i32 [[IDX3]]
// CHECK-DAG:   [[SADDR5:%.+]] = getelementptr inbounds [6 x i64], [6 x i64]* [[S]], i32 [[IDX3:[0-9]+]]
// CHECK-DAG:   [[BPADDR5:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BP]], i32 [[IDX3]]
// CHECK-DAG:   [[PADDR5:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[P]], i32 [[IDX3]]

// The names below are not necessarily consistent with the names used for the
// addresses above as some are repeated.
// CHECK-DAG:   store i[[SZ]] [[VLA0]], i[[SZ]]* [[CBPADDR0:%.+]],
// CHECK-DAG:   store i[[SZ]] [[VLA0]], i[[SZ]]* [[CPADDR0:%.+]],
// CHECK-DAG:   [[CBPADDR0]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   [[CPADDR0]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   store i64 {{4|8}}, i64* {{%[^,]+}}

// CHECK-DAG:   store i[[SZ]] 2, i[[SZ]]* [[CBPADDR1:%.+]],
// CHECK-DAG:   store i[[SZ]] 2, i[[SZ]]* [[CPADDR1:%.+]],
// CHECK-DAG:   [[CBPADDR1]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   [[CPADDR1]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   store i64 {{4|8}}, i64* {{%[^,]+}}

// CHECK-DAG:   store i[[SZ]] [[B_CVAL]], i[[SZ]]* [[CBPADDR2:%.+]],
// CHECK-DAG:   store i[[SZ]] [[B_CVAL]], i[[SZ]]* [[CPADDR2:%.+]],
// CHECK-DAG:   [[CBPADDR2]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   [[CPADDR2]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   store i64 4, i64* {{%[^,]+}}

// CHECK-DAG:   store [[S1]]* %{{.+}}, [[S1]]** [[CBPADDR3:%.+]],
// CHECK-DAG:   store double* %{{.+}}, double** [[CPADDR3:%.+]],
// CHECK-DAG:   [[CBPADDR3]] = bitcast i8** {{%[^,]+}} to [[S1]]**
// CHECK-DAG:   [[CPADDR3]] = bitcast i8** {{%[^,]+}} to double**
// CHECK-DAG:   store i64 8, i64* {{%[^,]+}}

// CHECK-DAG:   store [[S1]]* %{{.+}}, [[S1]]** [[CBPADDR4:%.+]],
// CHECK-DAG:   store double* %{{.+}}, double** [[CPADDR4:%.+]],
// CHECK-DAG:   [[CBPADDR4]] = bitcast i8** {{%[^,]+}} to [[S1]]**
// CHECK-DAG:   [[CPADDR4]] = bitcast i8** {{%[^,]+}} to double**
// CHECK-DAG:   store i64 8, i64* {{%[^,]+}}

// CHECK-DAG:   store i16* %{{.+}}, i16** [[CBPADDR5:%.+]],
// CHECK-DAG:   store i16* %{{.+}}, i16** [[CPADDR5:%.+]],
// CHECK-DAG:   [[CBPADDR5]] = bitcast i8** {{%[^,]+}} to i16**
// CHECK-DAG:   [[CPADDR5]] = bitcast i8** {{%[^,]+}} to i16**
// CHECK-DAG:   store i64 [[CSIZE]], i64* {{%[^,]+}}

// CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]

// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT7:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]

//
// CHECK: define {{.*}}[[FSTATIC]]
//
// CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 50
// CHECK:       br i1 [[IF]], label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CHECK:       [[IFTHEN]]
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams_mapper(i64 -1, i8* @{{[^,]+}}, i32 5, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i64* getelementptr inbounds ([5 x i64], [5 x i64]* [[SIZET6]], i32 0, i32 0), i64* getelementptr inbounds ([5 x i64], [5 x i64]* [[MAPT6]], i32 0, i32 0), i8** null, i32 0, i32 0)
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BP:%.+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[P:%.+]], i32 0, i32 0

// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BP]], i32 0, i32 0
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[P]], i32 0, i32 0
// CHECK-DAG:   store i[[SZ]] [[VAL0:%.+]], i[[SZ]]* [[CBPADDR0:%.+]],
// CHECK-DAG:   store i[[SZ]] [[VAL0]], i[[SZ]]* [[CPADDR0:%.+]],
// CHECK-DAG:   [[CBPADDR0]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   [[CPADDR0]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*

// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BP]], i32 0, i32 1
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[P]], i32 0, i32 1
// CHECK-DAG:   store i[[SZ]] [[VAL1:%.+]], i[[SZ]]* [[CBPADDR1:%.+]],
// CHECK-DAG:   store i[[SZ]] [[VAL1]], i[[SZ]]* [[CPADDR1:%.+]],
// CHECK-DAG:   [[CBPADDR1]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   [[CPADDR1]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*

// CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BP]], i32 0, i32 2
// CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[P]], i32 0, i32 2
// CHECK-DAG:   store i[[SZ]] [[VAL2:%.+]], i[[SZ]]* [[CBPADDR2:%.+]],
// CHECK-DAG:   store i[[SZ]] [[VAL2]], i[[SZ]]* [[CPADDR2:%.+]],
// CHECK-DAG:   [[CBPADDR2]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   [[CPADDR2]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*

// CHECK-DAG:   [[BPADDR3:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BP]], i32 0, i32 3
// CHECK-DAG:   [[PADDR3:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[P]], i32 0, i32 3
// CHECK-DAG:   store i[[SZ]] [[VAL3:%.+]], i[[SZ]]* [[CBPADDR3:%.+]],
// CHECK-DAG:   store i[[SZ]] [[VAL3]], i[[SZ]]* [[CPADDR3:%.+]],
// CHECK-DAG:   [[CBPADDR3]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   [[CPADDR3]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*

// CHECK-DAG:   [[BPADDR4:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[BP]], i32 0, i32 4
// CHECK-DAG:   [[PADDR4:%.+]] = getelementptr inbounds [5 x i8*], [5 x i8*]* [[P]], i32 0, i32 4
// CHECK-DAG:   store [10 x i32]* %{{.+}}, [10 x i32]** [[CBPADDR4:%.+]],
// CHECK-DAG:   store [10 x i32]* %{{.+}}, [10 x i32]** [[CPADDR4:%.+]],
// CHECK-DAG:   [[CBPADDR4]] = bitcast i8** {{%[^,]+}} to [10 x i32]**
// CHECK-DAG:   [[CPADDR4]] = bitcast i8** {{%[^,]+}} to [10 x i32]**

// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT6:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
// CHECK-NEXT:  br label %[[IFEND:.+]]
// CHECK:       [[IFELSE]]
// CHECK:       call void [[HVT6]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[IFEND]]
// CHECK:       [[IFEND]]

//
// CHECK: define {{.*}}[[FTEMPLATE]]
//
// CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 40
// CHECK:       br i1 [[IF]], label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CHECK:       [[IFTHEN]]
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_teams_mapper(i64 -1, i8* @{{[^,]+}}, i32 3, i8** [[BPR:%[^,]+]], i8** [[PR:%[^,]+]], i64* getelementptr inbounds ([3 x i64], [3 x i64]* [[SIZET7]], i32 0, i32 0), i64* getelementptr inbounds ([3 x i64], [3 x i64]* [[MAPT7]], i32 0, i32 0), i8** null, i32 0, i32 0)
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BP:%.+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[P:%.+]], i32 0, i32 0

// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BP]], i32 0, i32 0
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[P]], i32 0, i32 0
// CHECK-DAG:   store i[[SZ]] [[VAL0:%.+]], i[[SZ]]* [[CBPADDR0:%.+]],
// CHECK-DAG:   store i[[SZ]] [[VAL0]], i[[SZ]]* [[CPADDR0:%.+]],
// CHECK-DAG:   [[CBPADDR0]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   [[CPADDR0]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*

// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BP]], i32 0, i32 1
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[P]], i32 0, i32 1
// CHECK-DAG:   store i[[SZ]] [[VAL1:%.+]], i[[SZ]]* [[CBPADDR1:%.+]],
// CHECK-DAG:   store i[[SZ]] [[VAL1]], i[[SZ]]* [[CPADDR1:%.+]],
// CHECK-DAG:   [[CBPADDR1]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   [[CPADDR1]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*

// CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BP]], i32 0, i32 2
// CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[P]], i32 0, i32 2
// CHECK-DAG:   store i[[SZ]] [[VAL2:%.+]], i[[SZ]]* [[CBPADDR2:%.+]],
// CHECK-DAG:   store i[[SZ]] [[VAL2]], i[[SZ]]* [[CPADDR2:%.+]],
// CHECK-DAG:   [[CBPADDR2]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*
// CHECK-DAG:   [[CPADDR2]] = bitcast i8** {{%[^,]+}} to i[[SZ]]*

// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT5:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
// CHECK-NEXT:  br label %[[IFEND:.+]]
// CHECK:       [[IFELSE]]
// CHECK:       call void [[HVT5]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[IFEND]]
// CHECK:       [[IFEND]]



// Check that the offloading functions are emitted and that the arguments are
// correct and loaded correctly for the target regions of the callees of bar().

// CHECK:       define internal void [[HVT7]]
// Create local storage for each capture.
// CHECK:       [[LOCAL_THIS:%.+]] = alloca [[S1]]*
// CHECK:       [[LOCAL_B:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_VLA1:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_VLA2:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_C:%.+]] = alloca i16*
// CHECK:       [[LOCAL_B_CASTED:%.+]] = alloca i[[SZ]]
// CHECK-DAG:   store [[S1]]* [[ARG_THIS:%.+]], [[S1]]** [[LOCAL_THIS]]
// CHECK-DAG:   store i[[SZ]] [[ARG_B:%.+]], i[[SZ]]* [[LOCAL_B]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA1:%.+]], i[[SZ]]* [[LOCAL_VLA1]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA2:%.+]], i[[SZ]]* [[LOCAL_VLA2]]
// CHECK-DAG:   store i16* [[ARG_C:%.+]], i16** [[LOCAL_C]]
// Store captures in the context.
// CHECK-DAG:   [[REF_THIS:%.+]] = load [[S1]]*, [[S1]]** [[LOCAL_THIS]],
// CHECK-64-DAG:[[CONV_BP:%.+]] = bitcast i[[SZ]]* [[LOCAL_B]] to i32*
// CHECK-DAG:   [[VAL_VLA1:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA1]],
// CHECK-DAG:   [[VAL_VLA2:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_VLA2]],
// CHECK-DAG:   [[REF_C:%.+]] = load i16*, i16** [[LOCAL_C]],

// CHECK-64-DAG:[[CONV_B:%.+]] = load i32, i32* [[CONV_BP]]
// CHECK-64-DAG:[[CONV:%.+]] = bitcast i[[SZ]]* [[LOCAL_B_CASTED]] to i32*
// CHECK-64-DAG:store i32 [[CONV_B]], i32* [[CONV]], align
// CHECK-32-DAG:[[LOCAL_BV:%.+]] = load i32, i32* [[LOCAL_B]]
// CHECK-32-DAG:store i32 [[LOCAL_BV]], i32* [[LOCAL_B_CASTED]], align
// CHECK-DAG:   [[REF_B:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_B_CASTED]],

// CHECK:       call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(%struct.ident_t* [[DEF_LOC]], i32 5, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, [[S1]]*, i[[SZ]], i[[SZ]], i[[SZ]], i16*)* [[OMP_OUTLINED5:@.+]] to void (i32*, i32*, ...)*), [[S1]]* [[REF_THIS]], i[[SZ]] [[REF_B]], i[[SZ]] [[VAL_VLA1]], i[[SZ]] [[VAL_VLA2]], i16* [[REF_C]])
//
//
// CHECK:       define internal {{.*}}void [[OMP_OUTLINED5]](i32* noalias %.global_tid., i32* noalias %.bound_tid., [[S1]]* %{{.+}}, i[[SZ]] %{{.+}}, i[[SZ]] %{{.+}}, i[[SZ]] %{{.+}}, i16* {{.+}})
// To reduce complexity, we're only going as far as validating the signature of the outlined parallel function.


// CHECK:       define internal void [[HVT6]]
// Create local storage for each capture.
// CHECK:       [[LOCAL_A:%.+]] = alloca i[[SZ]]
// CHECK:       alloca i[[SZ]],
// CHECK:       [[LOCAL_AA:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_AAA:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_B:%.+]] = alloca [10 x i32]*
// CHECK:       [[LOCAL_A_CASTED:%.+]] = alloca i[[SZ]]
// CHECK:       alloca i[[SZ]],
// CHECK:       [[LOCAL_AA_CASTED:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_AAA_CASTED:%.+]] = alloca i[[SZ]]
// CHECK-DAG:   store i[[SZ]] [[ARG_A:%.+]], i[[SZ]]* [[LOCAL_A]]
// CHECK-DAG:   store i[[SZ]] [[ARG_AA:%.+]], i[[SZ]]* [[LOCAL_AA]]
// CHECK-DAG:   store i[[SZ]] [[ARG_AAA:%.+]], i[[SZ]]* [[LOCAL_AAA]]
// CHECK-DAG:   store [10 x i32]* [[ARG_B:%.+]], [10 x i32]** [[LOCAL_B]]
// Store captures in the context.
// CHECK-64-DAG:[[CONV_AP:%.+]] = bitcast i[[SZ]]* [[LOCAL_A]] to i32*
// CHECK-DAG:   [[CONV_AAP:%.+]] = bitcast i[[SZ]]* [[LOCAL_AA]] to i16*
// CHECK-DAG:   [[CONV_AAAP:%.+]] = bitcast i[[SZ]]* [[LOCAL_AAA]] to i8*
// CHECK-DAG:   [[REF_B:%.+]] = load [10 x i32]*, [10 x i32]** [[LOCAL_B]],

// CHECK-64-DAG:[[CONV_A:%.+]] = load i32, i32* [[CONV_AP]]
// CHECK-64-DAG:[[CONV:%.+]] = bitcast i[[SZ]]* [[LOCAL_A_CASTED]] to i32*
// CHECK-64-DAG:store i32 [[CONV_A]], i32* [[CONV]], align
// CHECK-32-DAG:[[LOCAL_AV:%.+]] = load i32, i32* [[LOCAL_A]]
// CHECK-32-DAG:store i32 [[LOCAL_AV]], i32* [[LOCAL_A_CASTED]], align
// CHECK-DAG:   [[REF_A:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_A_CASTED]],

// CHECK-DAG:   [[CONV_AA:%.+]] = load i16, i16* [[CONV_AAP]]
// CHECK-DAG:   [[CONV:%.+]] = bitcast i[[SZ]]* [[LOCAL_AA_CASTED]] to i16*
// CHECK-DAG:   store i16 [[CONV_AA]], i16* [[CONV]], align
// CHECK-DAG:   [[REF_AA:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_AA_CASTED]],

// CHECK-DAG:   [[CONV_AAA:%.+]] = load i8, i8* [[CONV_AAAP]]
// CHECK-DAG:   [[CONV:%.+]] = bitcast i[[SZ]]* [[LOCAL_AAA_CASTED]] to i8*
// CHECK-DAG:   store i8 [[CONV_AAA]], i8* [[CONV]], align
// CHECK-DAG:   [[REF_AAA:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_AAA_CASTED]],

// CHECK:       call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(%struct.ident_t* [[DEF_LOC]], i32 5, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i[[SZ]], i[[SZ]], i[[SZ]], i[[SZ]], [10 x i32]*)* [[OMP_OUTLINED6:@.+]] to void (i32*, i32*, ...)*), i[[SZ]] [[REF_A]], i[[SZ]] {{.+}}, i[[SZ]] [[REF_AA]], i[[SZ]] [[REF_AAA]], [10 x i32]* [[REF_B]])
//
//
// CHECK:       define internal {{.*}}void [[OMP_OUTLINED6]](i32* noalias %.global_tid., i32* noalias %.bound_tid., i[[SZ]] %{{.+}}, i[[SZ]] %{{.+}}, i[[SZ]] %{{.+}}, [10 x i32]* {{.+}})
// To reduce complexity, we're only going as far as validating the signature of the outlined parallel function.

// CHECK:       define internal void [[HVT5]]
// Create local storage for each capture.
// CHECK:       [[LOCAL_A:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_AA:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_B:%.+]] = alloca [10 x i32]*
// CHECK:       [[LOCAL_A_CASTED:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_AA_CASTED:%.+]] = alloca i[[SZ]]
// CHECK-DAG:   store i[[SZ]] [[ARG_A:%.+]], i[[SZ]]* [[LOCAL_A]]
// CHECK-DAG:   store i[[SZ]] [[ARG_AA:%.+]], i[[SZ]]* [[LOCAL_AA]]
// CHECK-DAG:   store [10 x i32]* [[ARG_B:%.+]], [10 x i32]** [[LOCAL_B]]
// Store captures in the context.
// CHECK-64-DAG:[[CONV_AP:%.+]] = bitcast i[[SZ]]* [[LOCAL_A]] to i32*
// CHECK-DAG:   [[CONV_AAP:%.+]] = bitcast i[[SZ]]* [[LOCAL_AA]] to i16*
// CHECK-DAG:   [[REF_B:%.+]] = load [10 x i32]*, [10 x i32]** [[LOCAL_B]],

// CHECK-64-DAG:[[CONV_A:%.+]] = load i32, i32* [[CONV_AP]]
// CHECK-64-DAG:[[CONV:%.+]] = bitcast i[[SZ]]* [[LOCAL_A_CASTED]] to i32*
// CHECK-64-DAG:store i32 [[CONV_A]], i32* [[CONV]], align
// CHECK-32-DAG:[[LOCAL_AV:%.+]] = load i32, i32* [[LOCAL_A]]
// CHECK-32-DAG:store i32 [[LOCAL_AV]], i32* [[LOCAL_A_CASTED]], align
// CHECK-DAG:   [[REF_A:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_A_CASTED]],

// CHECK-DAG:   [[CONV_AA:%.+]] = load i16, i16* [[CONV_AAP]]
// CHECK-DAG:   [[CONV:%.+]] = bitcast i[[SZ]]* [[LOCAL_AA_CASTED]] to i16*
// CHECK-DAG:   store i16 [[CONV_AA]], i16* [[CONV]], align
// CHECK-DAG:   [[REF_AA:%.+]] = load i[[SZ]], i[[SZ]]* [[LOCAL_AA_CASTED]],

// CHECK:       call {{.*}}void (%struct.ident_t*, i32, void (i32*, i32*, ...)*, ...) @__kmpc_fork_teams(%struct.ident_t* [[DEF_LOC]], i32 3, void (i32*, i32*, ...)* bitcast (void (i32*, i32*, i[[SZ]], i[[SZ]], [10 x i32]*)* [[OMP_OUTLINED7:@.+]] to void (i32*, i32*, ...)*), i[[SZ]] [[REF_A]], i[[SZ]] [[REF_AA]], [10 x i32]* [[REF_B]])
//
//
// CHECK:       define internal {{.*}}void [[OMP_OUTLINED7]](i32* noalias %.global_tid., i32* noalias %.bound_tid., i[[SZ]] %{{.+}}, i[[SZ]] %{{.+}}, [10 x i32]* {{.+}})
// To reduce complexity, we're only going as far as validating the signature of the outlined parallel function.

#endif
