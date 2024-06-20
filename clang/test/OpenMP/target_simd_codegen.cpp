// Test host codegen.
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix OMP45
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix OMP45
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix OMP45
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix OMP45
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -DOMP5 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix OMP51
// RUN: %clang_cc1 -fopenmp -DOMP5 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -DOMP5 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix OMP51
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -DOMP5 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix OMP51
// RUN: %clang_cc1 -fopenmp -DOMP5 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -DOMP5 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix OMP51

// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -DOMP5 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -DOMP5 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -DOMP5 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -DOMP5 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -DOMP5 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -DOMP5 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64 --check-prefix TOMP45
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64 --check-prefix TOMP45
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32 --check-prefix TOMP45
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32 --check-prefix TOMP45
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -DOMP5 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -DOMP5 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64 --check-prefix TOMP51
// RUN: %clang_cc1 -fopenmp -DOMP5 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -DOMP5 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64 --check-prefix TOMP51
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -DOMP5 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp -DOMP5 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32 --check-prefix TOMP51
// RUN: %clang_cc1 -fopenmp -DOMP5 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -DOMP5 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32 --check-prefix TOMP51

// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -DOMP5 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -DOMP5 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -DOMP5 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -DOMP5 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -DOMP5 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -Wno-vla -fopenmp-simd -DOMP5 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -DOMP5 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -DOMP5 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-DAG: [[IDENT_T:%.+]] = type { i32, i32, i32, i32, ptr }
// CHECK-DAG: [[KMP_TASK_T_WITH_PRIVATES:%.+]] = type { [[KMP_TASK_T:%.+]] }
// CHECK-DAG: [[KMP_TASK_T]] = type { ptr, ptr, i32, %{{[^,]+}}, %{{[^,]+}} }
// CHECK-DAG: [[TT:%.+]] = type { i64, i8 }
// CHECK-DAG: [[S1:%.+]] = type { double }
// CHECK-DAG: [[ENTTY:%.+]] = type { ptr, ptr, i[[SZ:32|64]], i32, i32 }

// TCHECK: [[ENTTY:%.+]] = type { ptr, ptr, i{{32|64}}, i32, i32 }

// We have 8 target regions, but only 7 that actually will generate offloading
// code, only 6 will have mapped arguments, and only 4 have all-constant map
// sizes.

// CHECK-DAG: [[SIZET2:@.+]] = private unnamed_addr constant [3 x i64] [i64 2, i64 4, i64 4]
// CHECK-DAG: [[MAPT2:@.+]] = private unnamed_addr constant [3 x i64] [i64 800, i64 800, i64 800]
// CHECK-DAG: [[SIZET3:@.+]] = private unnamed_addr constant [2 x i64] [i64 4, i64 2]
// CHECK-DAG: [[MAPT3:@.+]] = private unnamed_addr constant [2 x i64] [i64 800, i64 800]
// CHECK-DAG: [[SIZET4:@.+]] = private unnamed_addr constant [9 x i64] [i64 4, i64 40, i64 {{8|4}}, i64 0, i64 400, i64 {{8|4}}, i64 {{8|4}}, i64 0, i64 {{16|12}}]
// CHECK-DAG: [[MAPT4:@.+]] = private unnamed_addr constant [9 x i64] [i64 800, i64 547, i64 800, i64 547, i64 547, i64 800, i64 800, i64 547, i64 547]
// CHECK-DAG: [[SIZET5:@.+]] = private unnamed_addr constant [3 x i64] [i64 4, i64 2, i64 40]
// CHECK-DAG: [[MAPT5:@.+]] = private unnamed_addr constant [3 x i64] [i64 800, i64 800, i64 547]
// CHECK-DAG: [[SIZET6:@.+]] = private unnamed_addr constant [4 x i64] [i64 4, i64 2, i64 1, i64 40]
// CHECK-DAG: [[MAPT6:@.+]] = private unnamed_addr constant [4 x i64] [i64 800, i64 800, i64 800, i64 547]
// OMP45-DAG: [[SIZET7:@.+]] = private unnamed_addr constant [5 x i64] [i64 {{8|4}}, i64 4, i64 {{8|4}}, i64 {{8|4}}, i64 0]
// OMP51-DAG: [[SIZET7:@.+]] = private unnamed_addr constant [6 x i64] [i64 {{8|4}}, i64 4, i64 {{8|4}}, i64 {{8|4}}, i64 0, i64 1]
// OMP45-DAG: [[MAPT7:@.+]] = private unnamed_addr constant [5 x i64] [i64 547, i64 800, i64 800, i64 800, i64 547]
// OMP51-DAG: [[MAPT7:@.+]] = private unnamed_addr constant [6 x i64] [i64 547, i64 800, i64 800, i64 800, i64 547, i64 800]
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

template<typename tx, typename ty>
struct TT{
  tx X;
  ty Y;
};

// CHECK-LABEL: get_val
long long get_val() { return 0; }

// CHECK: define {{.*}}[[FOO:@.+]](
int foo(int n) {
  int a = 0;
  short aa = 0;
  float b[10];
  float bn[n];
  double c[5][10];
  double cn[5][n];
  TT<long long, char> d;

  // CHECK-32:    [[TASK:%.+]] = call ptr @__kmpc_omp_target_task_alloc(ptr @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i32 20, i32 1, ptr [[OMP_TASK_ENTRY:@[^,]+]], i64 -1)
  // CHECK-64:    [[TASK:%.+]] = call ptr @__kmpc_omp_target_task_alloc(ptr @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i64 40, i64 1, ptr [[OMP_TASK_ENTRY:@[^,]+]], i64 -1)
  // CHECK:       call i32 @__kmpc_omp_task(ptr @{{[^,]+}}, i32 %{{[^,]+}}, ptr [[TASK]])
  #pragma omp target simd nowait
  for (int i = 3; i < 32; i += 5) {
  }

  // CHECK:       call void [[HVT1:@.+]](i[[SZ]] {{[^,]+}}, {{[^)]+}})
  long long k = get_val();
  #pragma omp target simd if(target: 0) linear(k : 3)
  for (int i = 10; i > 1; i--) {
    a += 1;
  }

  // CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 1, i32 1, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
  // CHECK-DAG:   [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
  // CHECK-DAG:   store ptr [[BP:%.+]], ptr [[BPARG]]
  // CHECK-DAG:   [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
  // CHECK-DAG:   store ptr [[P:%.+]], ptr [[PARG]]
  // CHECK-DAG:   [[BP]] = getelementptr inbounds [3 x ptr], ptr [[BPR:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[P]] = getelementptr inbounds [3 x ptr], ptr [[PR:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [3 x ptr], ptr [[BPR]], i32 0, i32 0
  // CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [3 x ptr], ptr [[PR]], i32 0, i32 0
  // CHECK-DAG:   store i[[SZ]] [[VAL0:%.+]], ptr [[BPADDR0]],
  // CHECK-DAG:   store i[[SZ]] [[VAL0]], ptr [[PADDR0]],
  // CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [3 x ptr], ptr [[BPR]], i32 0, i32 1
  // CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [3 x ptr], ptr [[PR]], i32 0, i32 1
  // CHECK-DAG:   store i[[SZ]] [[VAL1:%.+]], ptr [[BPADDR1]],
  // CHECK-DAG:   store i[[SZ]] [[VAL1]], ptr [[PADDR1]],
  // CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [3 x ptr], ptr [[BPR]], i32 0, i32 2
  // CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [3 x ptr], ptr [[PR]], i32 0, i32 2
  // CHECK-DAG:   store i[[SZ]] [[VAL2:%.+]], ptr [[BPADDR2]],
  // CHECK-DAG:   store i[[SZ]] [[VAL2]], ptr [[PADDR2]],

  // CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
  // CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
  // CHECK:       [[FAIL]]
  // CHECK:       call void [[HVT2:@.+]](i[[SZ]] {{[^,]+}}, i[[SZ]] {{[^)]+}})
  // CHECK-NEXT:  br label %[[END]]
  // CHECK:       [[END]]
  int lin = 12;
  #pragma omp target simd if(target: 1) linear(lin, a : get_val())
  for (unsigned long long it = 2000; it >= 600; it-=400) {
    aa += 1;
  }

  // CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 10
  // CHECK:       br i1 [[IF]], label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
  // CHECK:       [[IFTHEN]]
  // CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 1, i32 1, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
  // CHECK-DAG:   [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
  // CHECK-DAG:   store ptr [[BPR:%.+]], ptr [[BPARG]]
  // CHECK-DAG:   [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
  // CHECK-DAG:   store ptr [[PR:%.+]], ptr [[PARG]]
  // CHECK-DAG:   [[BPR]] = getelementptr inbounds [2 x ptr], ptr [[BP:%[^,]+]], i32 0, i32 0
  // CHECK-DAG:   [[PR]] = getelementptr inbounds [2 x ptr], ptr [[P:%[^,]+]], i32 0, i32 0

  // CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [2 x ptr], ptr [[BP]], i32 0, i32 0
  // CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [2 x ptr], ptr [[P]], i32 0, i32 0
  // CHECK-DAG:   store i[[SZ]] [[VAL0:%.+]], ptr [[BPADDR0]],
  // CHECK-DAG:   store i[[SZ]] [[VAL0]], ptr [[PADDR0]],

  // CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [2 x ptr], ptr [[BP]], i32 0, i32 1
  // CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [2 x ptr], ptr [[P]], i32 0, i32 1
  // CHECK-DAG:   store i[[SZ]] [[VAL1:%.+]], ptr [[BPADDR1]],
  // CHECK-DAG:   store i[[SZ]] [[VAL1]], ptr [[PADDR1]],
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

  #pragma omp target simd if(target: n>10)
  for (short it = 6; it <= 20; it-=-4) {
    a += 1;
    aa += 1;
  }

  // We capture 3 VLA sizes in this target region
  // CHECK-64:       [[A_VAL:%.+]] = load i32, ptr %{{.+}},
  // CHECK-64:       store i32 [[A_VAL]], ptr [[A_CADDR:%.+]],
  // CHECK-64:       [[A_CVAL:%.+]] = load i[[SZ]], ptr [[A_CADDR]],

  // CHECK-32:       [[A_VAL:%.+]] = load i32, ptr %{{.+}},
  // CHECK-32:       store i32 [[A_VAL]], ptr [[A_CADDR:%.+]],
  // CHECK-32:       [[A_CVAL:%.+]] = load i[[SZ]], ptr [[A_CADDR]],

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

// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 1, i32 1, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CHECK-DAG:   [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CHECK-DAG:   store ptr [[BPR:%.+]], ptr [[BPARG]]
// CHECK-DAG:   [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CHECK-DAG:   store ptr [[PR:%.+]], ptr [[PARG]]
// CHECK-DAG:   [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CHECK-DAG:   store ptr [[SR:%.+]], ptr [[SARG]]
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [9 x ptr], ptr [[BP:%[^,]+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [9 x ptr], ptr [[P:%[^,]+]], i32 0, i32 0
// CHECK-DAG:   [[SR]] = getelementptr inbounds [9 x i64], ptr [[S:%[^,]+]], i32 0, i32 0

// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [9 x ptr], ptr [[BP]], i32 0, i32 [[IDX0:[0-9]+]]
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [9 x ptr], ptr [[P]], i32 0, i32 [[IDX0]]
// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [9 x ptr], ptr [[BP]], i32 0, i32 [[IDX1:[0-9]+]]
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [9 x ptr], ptr [[P]], i32 0, i32 [[IDX1]]
// CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [9 x ptr], ptr [[BP]], i32 0, i32 [[IDX2:[0-9]+]]
// CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [9 x ptr], ptr [[P]], i32 0, i32 [[IDX2]]
// CHECK-DAG:   [[SADDR3:%.+]] = getelementptr inbounds [9 x i64], ptr [[S]], i32 0, i32 [[IDX3:[0-9]+]]
// CHECK-DAG:   [[BPADDR3:%.+]] = getelementptr inbounds [9 x ptr], ptr [[BP]], i32 0, i32 [[IDX3]]
// CHECK-DAG:   [[PADDR3:%.+]] = getelementptr inbounds [9 x ptr], ptr [[P]], i32 0, i32 [[IDX3]]
// CHECK-DAG:   [[BPADDR4:%.+]] = getelementptr inbounds [9 x ptr], ptr [[BP]], i32 0, i32 [[IDX4:[0-9]+]]
// CHECK-DAG:   [[PADDR4:%.+]] = getelementptr inbounds [9 x ptr], ptr [[P]], i32 0, i32 [[IDX4]]
// CHECK-DAG:   [[BPADDR5:%.+]] = getelementptr inbounds [9 x ptr], ptr [[BP]], i32 0, i32 [[IDX5:[0-9]+]]
// CHECK-DAG:   [[PADDR5:%.+]] = getelementptr inbounds [9 x ptr], ptr [[P]], i32 0, i32 [[IDX5]]
// CHECK-DAG:   [[BPADDR6:%.+]] = getelementptr inbounds [9 x ptr], ptr [[BP]], i32 0, i32 [[IDX6:[0-9]+]]
// CHECK-DAG:   [[PADDR6:%.+]] = getelementptr inbounds [9 x ptr], ptr [[P]], i32 0, i32 [[IDX6]]
// CHECK-DAG:   [[SADDR7:%.+]] = getelementptr inbounds [9 x i64], ptr [[S]], i32 0, i32 [[IDX7:[0-9]+]]
// CHECK-DAG:   [[BPADDR7:%.+]] = getelementptr inbounds [9 x ptr], ptr [[BP]], i32 0, i32 [[IDX7]]
// CHECK-DAG:   [[PADDR7:%.+]] = getelementptr inbounds [9 x ptr], ptr [[P]], i32 0, i32 [[IDX7]]
// CHECK-DAG:   [[BPADDR8:%.+]] = getelementptr inbounds [9 x ptr], ptr [[BP]], i32 0, i32 [[IDX8:[0-9]+]]
// CHECK-DAG:   [[PADDR8:%.+]] = getelementptr inbounds [9 x ptr], ptr [[P]], i32 0, i32 [[IDX8]]

// The names below are not necessarily consistent with the names used for the
// addresses above as some are repeated.
// CHECK-DAG:   store i[[SZ]] [[A_CVAL]], ptr {{%[^,]+}},
// CHECK-DAG:   store i[[SZ]] [[A_CVAL]], ptr {{%[^,]+}},

// CHECK-DAG:   store ptr %{{.+}}, ptr {{%[^,]+}},
// CHECK-DAG:   store ptr %{{.+}}, ptr {{%[^,]+}},

// CHECK-DAG:   store i[[SZ]] %{{.+}}, ptr {{%[^,]+}},
// CHECK-DAG:   store i[[SZ]] %{{.+}}, ptr {{%[^,]+}},

// CHECK-DAG:   store ptr %{{.+}}, ptr {{%[^,]+}},
// CHECK-DAG:   store ptr %{{.+}}, ptr {{%[^,]+}},
// CHECK-DAG:   store i64 [[BNSIZE]], ptr {{%[^,]+}}

// CHECK-DAG:   store ptr %{{.+}}, ptr {{%[^,]+}},
// CHECK-DAG:   store ptr %{{.+}}, ptr {{%[^,]+}},

// CHECK-DAG:   store i[[SZ]] 5, ptr {{%[^,]+}},
// CHECK-DAG:   store i[[SZ]] 5, ptr {{%[^,]+}},

// CHECK-DAG:   store i[[SZ]] %{{.+}}, ptr {{%[^,]+}},
// CHECK-DAG:   store i[[SZ]] %{{.+}}, ptr {{%[^,]+}},

// CHECK-DAG:   store ptr %{{.+}}, ptr {{%[^,]+}},
// CHECK-DAG:   store ptr %{{.+}}, ptr {{%[^,]+}},
// CHECK-DAG:   store i64 [[CNSIZE]], ptr {{%[^,]+}}

// CHECK-DAG:   store ptr %{{.+}}, ptr {{%[^,]+}},
// CHECK-DAG:   store ptr %{{.+}}, ptr {{%[^,]+}},

// CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]

// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT4:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
#pragma omp target simd if (target \
                            : n > 20)
  for (unsigned char it = 'z'; it >= 'a'; it+=-1) {
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

// CHECK:       define internal void [[HVT0:@.+]]()
// CHECK:       !llvm.loop
// CHECK:       ret void
// CHECK-NEXT:  }

// CHECK:       define internal {{.*}}i32 [[OMP_TASK_ENTRY]](i32 {{.*}}%0, ptr noalias noundef %1)
// CHECK:       [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 1, i32 1, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT0]]()
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]

// CHECK:       define internal void [[HVT1]](i[[SZ]] noundef %{{.+}}, {{.+}})
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       store i[[SZ]] %{{.+}}, ptr [[AA_ADDR]], align
// CHECK-64:    [[AA:%.+]] = load i32, ptr [[AA_ADDR]], align
// CHECK-32:    [[AA:%.+]] = load i32, ptr [[AA_ADDR]], align
// CHECK:       !llvm.access.group
// CHECK:       !llvm.loop
// CHECK:       ret void
// CHECK-NEXT:  }

// CHECK:       define internal void [[HVT2]](i[[SZ]] noundef %{{.+}}, i[[SZ]] noundef %{{.+}}, i[[SZ]] noundef %{{.+}})
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       store i[[SZ]] %{{.+}}, ptr [[AA_ADDR]], align
// CHECK:       [[AA:%.+]] = load i16, ptr [[AA_ADDR]], align
// CHECK:       !llvm.loop
// CHECK:       ret void
// CHECK-NEXT:  }

// CHECK:       define internal void [[HVT3]]
// CHECK:       [[A_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK-DAG:   store i[[SZ]] %{{.+}}, ptr [[A_ADDR]], align
// CHECK-DAG:   store i[[SZ]] %{{.+}}, ptr [[AA_ADDR]], align
// CHECK:       !llvm.loop
// CHECK:       ret void
// CHECK-NEXT:  }

// CHECK:       define internal void [[HVT4]]
// Create local storage for each capture.
// CHECK:       [[LOCAL_A:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_B:%.+]] = alloca ptr
// CHECK:       [[LOCAL_VLA1:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_BN:%.+]] = alloca ptr
// CHECK:       [[LOCAL_C:%.+]] = alloca ptr
// CHECK:       [[LOCAL_VLA2:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_VLA3:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_CN:%.+]] = alloca ptr
// CHECK:       [[LOCAL_D:%.+]] = alloca ptr
// CHECK-DAG:   store i[[SZ]] [[ARG_A:%.+]], ptr [[LOCAL_A]]
// CHECK-DAG:   store ptr [[ARG_B:%.+]], ptr [[LOCAL_B]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA1:%.+]], ptr [[LOCAL_VLA1]]
// CHECK-DAG:   store ptr [[ARG_BN:%.+]], ptr [[LOCAL_BN]]
// CHECK-DAG:   store ptr [[ARG_C:%.+]], ptr [[LOCAL_C]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA2:%.+]], ptr [[LOCAL_VLA2]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA3:%.+]], ptr [[LOCAL_VLA3]]
// CHECK-DAG:   store ptr [[ARG_CN:%.+]], ptr [[LOCAL_CN]]
// CHECK-DAG:   store ptr [[ARG_D:%.+]], ptr [[LOCAL_D]]

// CHECK-DAG:   [[REF_B:%.+]] = load ptr, ptr [[LOCAL_B]],
// CHECK-DAG:   [[VAL_VLA1:%.+]] = load i[[SZ]], ptr [[LOCAL_VLA1]],
// CHECK-DAG:   [[REF_BN:%.+]] = load ptr, ptr [[LOCAL_BN]],
// CHECK-DAG:   [[REF_C:%.+]] = load ptr, ptr [[LOCAL_C]],
// CHECK-DAG:   [[VAL_VLA2:%.+]] = load i[[SZ]], ptr [[LOCAL_VLA2]],
// CHECK-DAG:   [[VAL_VLA3:%.+]] = load i[[SZ]], ptr [[LOCAL_VLA3]],
// CHECK-DAG:   [[REF_CN:%.+]] = load ptr, ptr [[LOCAL_CN]],
// CHECK-DAG:   [[REF_D:%.+]] = load ptr, ptr [[LOCAL_D]],


template<typename tx>
tx ftemplate(int n) {
  tx a = 0;
  short aa = 0;
  tx b[10];

  #pragma omp target simd if(target: n>40)
  for (long long i = -10; i < 10; i += 3) {
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

  #pragma omp target simd if(target: n>50)
  for (unsigned i=100; i<10; i+=10) {
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

#ifdef OMP5
    #pragma omp target simd if(n>60) nontemporal(a) private(a)
#else
    #pragma omp target simd if(n>60) private(a)
#endif // OMP5
    for (unsigned long long it = 2000; it >= 600; it -= 400) {
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
  // CHECK: call {{.*}}i32 [[FS1:@.+]](ptr {{.*}}, i32 {{.*}})
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
// CHECK:          ptr @llvm.stacksave.p0()
// CHECK-32:       store i32 %{{.+}}, ptr %__vla_expr
// OMP51:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 60
// CHECK-64:       store i32 %{{.+}}, ptr [[B_ADDR:%.+]],
// CHECK-64:       [[B_CVAL:%.+]] = load i[[SZ]], ptr [[B_ADDR]],

// CHECK-32:       store i32 %{{.+}}, ptr [[B_ADDR:%.+]],
// CHECK-32:       [[B_CVAL:%.+]] = load i[[SZ]], ptr [[B_ADDR]],

// OMP45:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 60
// OMP51:          [[TOBOOL:%.+]] = trunc i8 %{{.+}} to i1
// OMP51:          [[FROMBOOL:%.+]] = zext i1 [[TOBOOL]] to i8
// OMP51:          store i8 [[FROMBOOL]], ptr [[CAP:%.+]],
// OMP51:          [[SIMD_COND:%.+]] = load i[[SZ]], ptr [[CAP]],
// OMP51:          [[IF:%.+]] = trunc i8 %{{.+}} to i1
// CHECK:       br i1 [[IF]], label %[[TRY:[^,]+]], label %[[FAIL:[^,]+]]
// CHECK:       [[TRY]]
// We capture 2 VLA sizes in this target region
// CHECK:       [[CELEMSIZE2:%.+]] = mul nuw i[[SZ]] 2, [[VLA0:%.+]]
// CHECK-64:    [[CSIZE:%.+]] = mul nuw i64 [[CELEMSIZE2]], 2
// CHECK-32:    [[CSZSIZE:%.+]] = mul nuw i32 [[CELEMSIZE2]], 2
// CHECK-32:    [[CSIZE:%.+]] = sext i32 [[CSZSIZE]] to i64

// OMP45-DAG:   [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 1, i32 1, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// OMP45-DAG:   [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// OMP45-DAG:   store ptr [[BPR:%.+]], ptr [[BPARG]]
// OMP45-DAG:   [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// OMP45-DAG:   store ptr [[PR:%.+]], ptr [[PARG]]
// OMP45-DAG:   [[BPR]] = getelementptr inbounds [5 x ptr], ptr [[BP:%.+]], i32 0, i32 0
// OMP45-DAG:   [[PR]] = getelementptr inbounds [5 x ptr], ptr [[P:%.+]], i32 0, i32 0
// OMP45-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BP]], i32 [[IDX0:[0-9]+]]
// OMP45-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [5 x ptr], ptr [[P]], i32 [[IDX0]]
// OMP45-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BP]], i32 [[IDX1:[0-9]+]]
// OMP45-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [5 x ptr], ptr [[P]], i32 [[IDX1]]
// OMP45-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BP]], i32 [[IDX2:[0-9]+]]
// OMP45-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [5 x ptr], ptr [[P]], i32 [[IDX2]]
// OMP45-DAG:   [[BPADDR3:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BP]], i32 [[IDX3:[0-9]+]]
// OMP45-DAG:   [[PADDR3:%.+]] = getelementptr inbounds [5 x ptr], ptr [[P]], i32 [[IDX3]]
// OMP45-DAG:   [[SADDR4:%.+]] = getelementptr inbounds [5 x i64], ptr [[S]], i32 [[IDX4:[0-9]+]]
// OMP45-DAG:   [[BPADDR4:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BP]], i32 [[IDX4]]
// OMP45-DAG:   [[PADDR4:%.+]] = getelementptr inbounds [5 x ptr], ptr [[P]], i32 [[IDX4]]
// OMP51-DAG:   [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 1, i32 1, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// OMP51-DAG:   [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// OMP51-DAG:   store ptr [[BPR:%.+]], ptr [[BPARG]]
// OMP51-DAG:   [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// OMP51-DAG:   store ptr [[PR:%.+]], ptr [[PARG]]
// OMP51-DAG:   [[BPR]] = getelementptr inbounds [6 x  ptr], ptr [[BP:%.+]], i32 0, i32 0
// OMP51-DAG:   [[PR]] = getelementptr inbounds [6 x  ptr], ptr [[P:%.+]], i32 0, i32 0
// OMP51-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [6 x  ptr], ptr [[BP]], i32 [[IDX0:[0-9]+]]
// OMP51-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [6 x  ptr], ptr [[P]], i32 [[IDX0]]
// OMP51-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [6 x  ptr], ptr [[BP]], i32 [[IDX1:[0-9]+]]
// OMP51-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [6 x  ptr], ptr [[P]], i32 [[IDX1]]
// OMP51-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [6 x  ptr], ptr [[BP]], i32 [[IDX2:[0-9]+]]
// OMP51-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [6 x  ptr], ptr [[P]], i32 [[IDX2]]
// OMP51-DAG:   [[BPADDR3:%.+]] = getelementptr inbounds [6 x  ptr], ptr [[BP]], i32 [[IDX3:[0-9]+]]
// OMP51-DAG:   [[PADDR3:%.+]] = getelementptr inbounds [6 x  ptr], ptr [[P]], i32 [[IDX3]]
// OMP51-DAG:   [[SADDR4:%.+]] = getelementptr inbounds [6 x  i64], ptr [[S]], i32 [[IDX4:[0-9]+]]
// OMP51-DAG:   [[BPADDR4:%.+]] = getelementptr inbounds [6 x  ptr], ptr [[BP]], i32 [[IDX4]]
// OMP51-DAG:   [[PADDR4:%.+]] = getelementptr inbounds [6 x  ptr], ptr [[P]], i32 [[IDX4]]
// OMP51-DAG:   [[BPADDR5:%.+]] = getelementptr inbounds [6 x  ptr], ptr [[BP]], i32 [[IDX5:[0-9]+]]
// OMP51-DAG:   [[PADDR5:%.+]] = getelementptr inbounds [6 x  ptr], ptr [[P]], i32 [[IDX5]]

// The names below are not necessarily consistent with the names used for the
// addresses above as some are repeated.
// CHECK-DAG:   store ptr %{{.+}}, ptr {{%[^,]+}},
// CHECK-DAG:   store ptr %{{.+}}, ptr {{%[^,]+}},

// CHECK-DAG:   store i[[SZ]] [[B_CVAL]], ptr {{%[^,]+}},
// CHECK-DAG:   store i[[SZ]] [[B_CVAL]], ptr {{%[^,]+}},

// CHECK-DAG:   store i[[SZ]] 2, ptr {{%[^,]+}},
// CHECK-DAG:   store i[[SZ]] 2, ptr {{%[^,]+}},

// CHECK-DAG:   store i[[SZ]] %{{.+}}, ptr {{%[^,]+}},
// CHECK-DAG:   store i[[SZ]] %{{.+}}, ptr {{%[^,]+}},

// CHECK-DAG:   store ptr %{{.+}}, ptr {{%[^,]+}},
// CHECK-DAG:   store ptr %{{.+}}, ptr {{%[^,]+}},
// CHECK-DAG:   store i64 [[CSIZE]], ptr {{%[^,]+}}

// OMP51-DAG:   store i[[SZ]] [[SIMD_COND]], ptr {{%[^,]+}}
// OMP51-DAG:   store i[[SZ]] [[SIMD_COND]], ptr {{%[^,]+}}

// CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]

// CHECK:       [[FAIL]]
// OMP45:       call void [[HVT7:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// OMP51:       call void [[HVT7:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]

//
// CHECK: define {{.*}}[[FSTATIC]]
//
// CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 50
// CHECK:       br i1 [[IF]], label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CHECK:       [[IFTHEN]]
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 1, i32 1, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CHECK-DAG:   [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CHECK-DAG:   store ptr [[BPR:%.+]], ptr [[BPARG]]
// CHECK-DAG:   [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CHECK-DAG:   store ptr [[PR:%.+]], ptr [[PARG]]
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [4 x ptr], ptr [[BP:%.+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [4 x ptr], ptr [[P:%.+]], i32 0, i32 0

// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [4 x ptr], ptr [[BP]], i32 0, i32 0
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [4 x ptr], ptr [[P]], i32 0, i32 0
// CHECK-DAG:   store i[[SZ]] [[VAL0:%.+]], ptr [[BPADDR0]],
// CHECK-DAG:   store i[[SZ]] [[VAL0]], ptr [[PADDR0]],

// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [4 x ptr], ptr [[BP]], i32 0, i32 1
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [4 x ptr], ptr [[P]], i32 0, i32 1
// CHECK-DAG:   store i[[SZ]] [[VAL1:%.+]], ptr [[BPADDR1]],
// CHECK-DAG:   store i[[SZ]] [[VAL1]], ptr [[PADDR1]],

// CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [4 x ptr], ptr [[BP]], i32 0, i32 2
// CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [4 x ptr], ptr [[P]], i32 0, i32 2
// CHECK-DAG:   store i[[SZ]] [[VAL2:%.+]], ptr [[BPADDR2]],
// CHECK-DAG:   store i[[SZ]] [[VAL2]], ptr [[PADDR2]],

// CHECK-DAG:   [[BPADDR3:%.+]] = getelementptr inbounds [4 x ptr], ptr [[BP]], i32 0, i32 3
// CHECK-DAG:   [[PADDR3:%.+]] = getelementptr inbounds [4 x ptr], ptr [[P]], i32 0, i32 3
// CHECK-DAG:   store ptr [[VAL3:%.+]], ptr [[BPADDR3]],
// CHECK-DAG:   store ptr [[VAL3]], ptr [[PADDR3]],

// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT6:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
// CHECK-NEXT:  br label %[[IFEND:.+]]
// CHECK:       [[IFELSE]]
// CHECK:       call void [[HVT6]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[IFEND]]
// CHECK:       [[IFEND]]

//
// CHECK: define {{.*}}[[FTEMPLATE]]
//
// CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 40
// CHECK:       br i1 [[IF]], label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CHECK:       [[IFTHEN]]
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 1, i32 1, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CHECK-DAG:   [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CHECK-DAG:   store ptr [[BPR:%.+]], ptr [[BPARG]]
// CHECK-DAG:   [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CHECK-DAG:   store ptr [[PR:%.+]], ptr [[PARG]]
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [3 x ptr], ptr [[BP:%.+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [3 x ptr], ptr [[P:%.+]], i32 0, i32 0

// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [3 x ptr], ptr [[BP]], i32 0, i32 0
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [3 x ptr], ptr [[P]], i32 0, i32 0
// CHECK-DAG:   store i[[SZ]] [[VAL0:%.+]], ptr [[BPADDR0]],
// CHECK-DAG:   store i[[SZ]] [[VAL0]], ptr [[PADDR0]],

// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [3 x ptr], ptr [[BP]], i32 0, i32 1
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [3 x ptr], ptr [[P]], i32 0, i32 1
// CHECK-DAG:   store i[[SZ]] [[VAL1:%.+]], ptr [[BPADDR1]],
// CHECK-DAG:   store i[[SZ]] [[VAL1]], ptr [[PADDR1]],

// CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [3 x ptr], ptr [[BP]], i32 0, i32 2
// CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [3 x ptr], ptr [[P]], i32 0, i32 2
// CHECK-DAG:   store ptr [[VAL2:%.+]], ptr [[BPADDR2]],
// CHECK-DAG:   store ptr [[VAL2]], ptr [[PADDR2]],

// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT5:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
// CHECK-NEXT:  br label %[[IFEND:.+]]
// CHECK:       [[IFELSE]]
// CHECK:       call void [[HVT:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[IFEND]]
// CHECK:       [[IFEND]]

// Check that the offloading functions are emitted and that the arguments are
// correct and loaded correctly for the target regions of the callees of bar().

// CHECK:       define internal void [[HVT7]]
// Create local storage for each capture.
// CHECK:       [[LOCAL_THIS:%.+]] = alloca ptr
// CHECK:       [[LOCAL_B:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_VLA1:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_VLA2:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_C:%.+]] = alloca ptr
// OMP51:       [[LOCAL_SIMD_COND_CASTED:%.+]] = alloca i[[SZ]],
// CHECK-DAG:   store ptr [[ARG_THIS:%.+]], ptr [[LOCAL_THIS]]
// CHECK-DAG:   store i[[SZ]] [[ARG_B:%.+]], ptr [[LOCAL_B]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA1:%.+]], ptr [[LOCAL_VLA1]]
// CHECK-DAG:   store i[[SZ]] [[ARG_VLA2:%.+]], ptr [[LOCAL_VLA2]]
// CHECK-DAG:   store ptr [[ARG_C:%.+]], ptr [[LOCAL_C]]
// Store captures in the context.
// CHECK-DAG:   [[REF_THIS:%.+]] = load ptr, ptr [[LOCAL_THIS]],
// CHECK-DAG:   [[VAL_VLA1:%.+]] = load i[[SZ]], ptr [[LOCAL_VLA1]],
// CHECK-DAG:   [[VAL_VLA2:%.+]] = load i[[SZ]], ptr [[LOCAL_VLA2]],
// CHECK-DAG:   [[REF_C:%.+]] = load ptr, ptr [[LOCAL_C]],
// OMP51-DAG:   [[SIMD_COND:%.+]] = load i8, ptr [[LOCAL_SIMD_COND_CASTED]],
// OMP51-DAG:   trunc i8 [[SIMD_COND]] to i1
// OMP45-NOT:   !nontemporal
// OMP51:       store double {{.*}}!nontemporal
// OMP51:       load double, {{.*}}!nontemporal
// OMP51:       store double {{.*}}!nontemporal

// CHECK:       define internal void [[HVT6]]
// Create local storage for each capture.
// CHECK:       [[LOCAL_A:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_AA:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_AAA:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_B:%.+]] = alloca ptr
// CHECK-DAG:   store i[[SZ]] [[ARG_A:%.+]], ptr [[LOCAL_A]]
// CHECK-DAG:   store i[[SZ]] [[ARG_AA:%.+]], ptr [[LOCAL_AA]]
// CHECK-DAG:   store i[[SZ]] [[ARG_AAA:%.+]], ptr [[LOCAL_AAA]]
// CHECK-DAG:   store ptr [[ARG_B:%.+]], ptr [[LOCAL_B]]
// Store captures in the context.
// CHECK-DAG:   [[REF_B:%.+]] = load ptr, ptr [[LOCAL_B]],

// CHECK:       define internal void [[HVT5]]
// Create local storage for each capture.
// CHECK:       [[LOCAL_A:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_AA:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_B:%.+]] = alloca ptr
// CHECK-DAG:   store i[[SZ]] [[ARG_A:%.+]], ptr [[LOCAL_A]]
// CHECK-DAG:   store i[[SZ]] [[ARG_AA:%.+]], ptr [[LOCAL_AA]]
// CHECK-DAG:   store ptr [[ARG_B:%.+]], ptr [[LOCAL_B]]
// Store captures in the context.
// CHECK-DAG:   [[REF_B:%.+]] = load ptr, ptr [[LOCAL_B]],

// OMP45-NOT: !{!"llvm.loop.vectorize.enable", i1 false}
// TOMP45-NOT: !{!"llvm.loop.vectorize.enable", i1 false}
// OMP51: !{!"llvm.loop.vectorize.enable", i1 false}
// TOMP51: !{!"llvm.loop.vectorize.enable", i1 false}

#endif
