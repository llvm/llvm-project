// Test host codegen.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix OMP45
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix OMP45
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix OMP45
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix OMP45

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix OMP50
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix OMP50
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix OMP50
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix OMP50

// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -D_DOMP51 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix OMP51
// RUN: %clang_cc1 -fopenmp -x c++ -fopenmp-version=51 -D_DOMP51 -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -fopenmp-version=51 -D_DOMP51 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-64 --check-prefix OMP51
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=51 -D_DOMP51 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix OMP51
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -D_DOMP51 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -D_DOMP51 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix CHECK --check-prefix CHECK-32 --check-prefix OMP51

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// Test target codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-64
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCHECK --check-prefix TCHECK-32

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-DAG: [[IDENT_T:%.+]] = type { i32, i32, i32, i32, ptr }
// CHECK-DAG: [[KMP_TASK_T_WITH_PRIVATES:%.+]] = type { [[KMP_TASK_T:%.+]], [[KMP_PRIVATES_T:%.+]] }
// CHECK-DAG: [[KMP_TASK_T]] = type { ptr, ptr, i32, {{%.+}}, {{%.+}} }
// CHECK-DAG: [[TT:%.+]] = type { i64, i8 }
// CHECK-DAG: [[S1:%.+]] = type { double }
// CHECK-DAG: [[S2:%.+]] = type { i32, i32, i32 }
// CHECK-DAG: [[ENTTY:%.+]] = type { ptr, ptr, i[[SZ:32|64]], i32, i32 }
// CHECK-DAG: [[ANON_T:%.+]] = type { ptr, i32, i32 }
// CHECK-32-DAG: [[KMP_PRIVATES_T]] = type { [2 x i64], ptr, i32, [2 x ptr], [2 x ptr] }
// CHECK-64-DAG: [[KMP_PRIVATES_T]] = type { ptr, [2 x ptr], [2 x ptr], [2 x i64], i32 }

// TCHECK: [[ENTTY:%.+]] = type { ptr, ptr, i{{32|64}}, i32, i32 }

// We have 9 target regions, but only 8 that actually will generate offloading
// code and have mapped arguments, and only 6 have all-constant map sizes.

// CHECK-DAG: [[SIZET:@.+]] = private unnamed_addr constant [2 x i64] [i64 0, i64 4]
// CHECK-DAG: [[MAPT:@.+]] = private unnamed_addr constant [2 x i64] [i64 544, i64 800]
// CHECK-DAG: [[SIZET2:@.+]] = private unnamed_addr constant [1 x i{{32|64}}] [i64 2]
// CHECK-DAG: [[MAPT2:@.+]] = private unnamed_addr constant [1 x i64] [i64 800]
// CHECK-DAG: [[SIZET3:@.+]] = private unnamed_addr constant [2 x i64] [i64 4, i64 2]
// CHECK-DAG: [[MAPT3:@.+]] = private unnamed_addr constant [2 x i64] [i64 800, i64 800]
// CHECK-DAG: [[SIZET4:@.+]] = private unnamed_addr constant [9 x i64] [i64 4, i64 40, i64 {{4|8}}, i64 0, i64 400, i64 {{4|8}}, i64 {{4|8}}, i64 0, i64 {{12|16}}]
// CHECK-DAG: [[MAPT4:@.+]] = private unnamed_addr constant [9 x i64] [i64 800, i64 547, i64 800, i64 547, i64 547, i64 800, i64 800, i64 547, i64 547]
// CHECK-DAG: [[SIZET5:@.+]] = private unnamed_addr constant [3 x i64] [i64 4, i64 2, i64 40]
// CHECK-DAG: [[MAPT5:@.+]] = private unnamed_addr constant [3 x i64] [i64 800, i64 800, i64 547]
// CHECK-DAG: [[SIZET6:@.+]] = private unnamed_addr constant [4 x i64] [i64 4, i64 2, i64 1, i64 40]
// CHECK-DAG: [[MAPT6:@.+]] = private unnamed_addr constant [4 x i64] [i64 800, i64 800, i64 800, i64 547]
// CHECK-DAG: [[SIZET7:@.+]] = private unnamed_addr constant [5 x i64] [i64 8, i64 4, i64 {{4|8}}, i64 {{4|8}}, i64 0]
// CHECK-DAG: [[MAPT7:@.+]] = private unnamed_addr constant [5 x i64] [i64 547, i64 800, i64 800, i64 800, i64 547]
// CHECK-DAG: [[SIZET9:@.+]] = private unnamed_addr constant [1 x i64] [i64 12]
// CHECK-DAG: [[MAPT10:@.+]] = private unnamed_addr constant [1 x i64] [i64 35]
// CHECK-DAG: @{{.*}} = weak constant i8 0
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
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK: @{{.+}} = weak constant [[ENTTY]]
// TCHECK-NOT: @{{.+}} = weak constant [[ENTTY]]

// Check target registration is registered as a Ctor.
// CHECK: appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 0, ptr @.omp_offloading.requires_reg, ptr null }]


template<typename tx, typename ty>
struct TT{
  tx X;
  ty Y;
};

int global;
extern int global;

// CHECK: define {{.*}}[[FOO:@.+]](
int foo(int n) {
  // CHECK: [[OFFLOADBPTR:%.+]] = alloca [2 x ptr], align
  // CHECK: [[OFFLOADPTR:%.+]] = alloca [2 x ptr], align
  // CHECK: [[OFFLOADMAPPER:%.+]] = alloca [2 x ptr], align
  int a = 0;
  short aa = 0;
  float b[10];
  float bn[n];
  double c[5][10];
  double cn[5][n];
  TT<long long, char> d;
  static long *plocal;

// CHECK:       [[ADD:%.+]] = add nsw i32
// CHECK:       store i32 [[ADD]], ptr [[DEVICE_CAP:%.+]],
// CHECK:       [[DEV:%.+]] = load i32, ptr [[DEVICE_CAP]],
// CHECK:       [[DEVICE:%.+]] = sext i32 [[DEV]] to i64
// CHECK:       [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE]], i32 {{.+}}, i32 {{.+}}, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CHECK-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT0:@.+]]()
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
#pragma omp target device(global + a)
  {
  }

  // CHECK: [[BPRGEP:%.+]] = getelementptr inbounds [2 x ptr], ptr [[OFFLOADBPTR]], i32 0, i32 0
  // CHECK: [[PRGEP:%.+]] = getelementptr inbounds [2 x ptr], ptr [[OFFLOADPTR]], i32 0, i32 0
  // CHECK: [[BPRGEP:%.+]] = getelementptr inbounds [2 x ptr], ptr [[OFFLOADBPTR]], i32 0, i32 0
  // CHECK: [[PRGEP:%.+]] = getelementptr inbounds [2 x ptr], ptr [[OFFLOADPTR]], i32 0, i32 0
  // CHECK: [[DEVICE:%.+]] = sext i32 {{%.+}} to i64
  // CHECK-32: [[TASK:%.+]] = call ptr @__kmpc_omp_target_task_alloc(ptr {{.+}}, i32 %0, i32 1, i32 60, i32 12, ptr [[OMP_TASK_ENTRY:@.+]], i64 [[DEVICE]])
  // CHECK-64: [[TASK:%.+]] = call ptr @__kmpc_omp_target_task_alloc(ptr {{.+}}, i32 %0, i32 1, i64 104, i64 16, ptr [[OMP_TASK_ENTRY:@.+]], i64 [[DEVICE]])
  // CHECK: [[TASK_WITH_PRIVATES_GEP:%.+]] = getelementptr inbounds [[KMP_TASK_T_WITH_PRIVATES]], ptr [[TASK]], i32 0, i32 1
  // CHECK-32: [[SIZEGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], ptr [[TASK_WITH_PRIVATES_GEP]], i32 0, i32 0
  // CHECK-32: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[SIZEGEP]], ptr align 4 [[SIZET]], i32 16, i1 false)
  // CHECK-32: [[FPBPRGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], ptr [[TASK_WITH_PRIVATES_GEP]], i32 0, i32 3
  // CHECK-32: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[FPBPRGEP]], ptr align 4 [[BPRGEP]], i32 8, i1 false)
  // CHECK-32: [[FPPRGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], ptr [[TASK_WITH_PRIVATES_GEP]], i32 0, i32 4
  // CHECK-32: call void @llvm.memcpy.p0.p0.i32(ptr align 4 [[FPPRGEP]], ptr align 4 [[PRGEP]], i32 8, i1 false)
  // CHECK-64: [[FPBPRGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], ptr [[TASK_WITH_PRIVATES_GEP]], i32 0, i32 1
  // CHECK-64: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[FPBPRGEP]], ptr align 8 [[BPRGEP]], i64 16, i1 false)
  // CHECK-64: [[FPPRGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], ptr [[TASK_WITH_PRIVATES_GEP]], i32 0, i32 2
  // CHECK-64: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[FPPRGEP]], ptr align 8 [[PRGEP]], i64 16, i1 false)
  // CHECK-64: [[SIZEGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], ptr [[TASK_WITH_PRIVATES_GEP]], i32 0, i32 3
  // CHECK-64: call void @llvm.memcpy.p0.p0.i64(ptr align 8 [[SIZEGEP]], ptr align 8 [[SIZET]], i64 16, i1 false)
  // CHECK: call i32 @__kmpc_omp_task(ptr {{.+}}, i32 {{.+}}, ptr [[TASK]])
  #pragma omp target device(global + a) nowait
  {
    static int local1;
    *plocal = global;
    local1 = global;
  }

  // CHECK:       call void [[HVT1:@.+]](i[[SZ]] {{[^,]+}})
  #pragma omp target if(0) firstprivate(global)
  {
    global += 1;
  }

// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 {{.+}}, i32 {{.+}}, i32 {{.+}}, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CHECK-DAG:   [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CHECK-DAG:   store ptr [[BP:%.+]], ptr [[BPARG]]
// CHECK-DAG:   [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CHECK-DAG:   store ptr [[P:%.+]], ptr [[PARG]]
// CHECK-DAG:   [[BP]] = getelementptr inbounds [1 x ptr], ptr [[BPR:%[^,]+]], i32 0, i32 0
// CHECK-DAG:   [[P]] = getelementptr inbounds [1 x ptr], ptr [[PR:%[^,]+]], i32 0, i32 0
// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [1 x ptr], ptr [[BPR]], i32 0, i32 [[IDX0:[0-9]+]]
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [1 x ptr], ptr [[PR]], i32 0, i32 [[IDX0]]
// CHECK-DAG:   store i[[SZ]] [[BP0:%[^,]+]], ptr [[BPADDR0]]
// CHECK-DAG:   store i[[SZ]] [[P0:%[^,]+]], ptr [[PADDR0]]

// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT2:@.+]](i[[SZ]] {{[^,]+}})
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
#pragma omp target if (1)
  {
    aa += 1;
  }

// CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 10
// CHECK:       br i1 [[IF]], label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CHECK:       [[IFTHEN]]
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 {{.+}}, i32 {{.+}}, i32 {{.+}}, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CHECK-DAG:   [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CHECK-DAG:   store ptr [[BPR:%.+]], ptr [[BPARG]]
// CHECK-DAG:   [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CHECK-DAG:   store ptr [[PR:%.+]], ptr [[PARG]]
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [2 x ptr], ptr [[BP:%[^,]+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [2 x ptr], ptr [[P:%[^,]+]], i32 0, i32 0

// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [2 x ptr], ptr [[BP]], i32 0, i32 0
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [2 x ptr], ptr [[P]], i32 0, i32 0
// CHECK-DAG:   store i[[SZ]] [[BP0:%[^,]+]], ptr [[BPADDR0]]
// CHECK-DAG:   store i[[SZ]] [[P0:%[^,]+]], ptr [[PADDR0]]

// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [2 x ptr], ptr [[BP]], i32 0, i32 1
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [2 x ptr], ptr [[P]], i32 0, i32 1
// CHECK-DAG:   store i[[SZ]] [[BP1:%[^,]+]], ptr [[BPADDR1]]
// CHECK-DAG:   store i[[SZ]] [[P1:%[^,]+]], ptr [[PADDR1]]
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
#pragma omp target if (n > 10)
  {
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
  // CHECK:       br i1 [[IF]], label %[[TRY:[^,]+]], label %[[IFELSE:[^,]+]]
  // CHECK:       [[TRY]]
  // CHECK-64:    [[BNSIZE:%.+]] = mul nuw i64 [[VLA0:%.+]], 4
  // CHECK-32:    [[BNSZSIZE:%.+]] = mul nuw i32 [[VLA0:%.+]], 4
  // CHECK-32:    [[BNSIZE:%.+]] = sext i32 [[BNSZSIZE]] to i64
  // CHECK:       [[CNELEMSIZE2:%.+]] = mul nuw i[[SZ]] 5, [[VLA1:%.+]]
  // CHECK-64:    [[CNSIZE:%.+]] = mul nuw i64 [[CNELEMSIZE2]], 8
  // CHECK-32:    [[CNSZSIZE:%.+]] = mul nuw i32 [[CNELEMSIZE2]], 8
  // CHECK-32:    [[CNSIZE:%.+]] = sext i32 [[CNSZSIZE]] to i64

// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 {{.+}}, i32 {{.+}}, i32 {{.+}}, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CHECK-DAG:   [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CHECK-DAG:   store ptr [[BPR:%.+]], ptr [[BPARG]]
// CHECK-DAG:   [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CHECK-DAG:   store ptr [[PR:%.+]], ptr [[PARG]]
// CHECK-DAG:   [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CHECK-DAG:   store ptr [[SZ4:%.+]], ptr [[SARG]]
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [9 x ptr], ptr [[BP:%[^,]+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [9 x ptr], ptr [[P:%[^,]+]], i32 0, i32 0
// CHECK-DAG:   [[SZ4]] = getelementptr inbounds [9 x i64], ptr [[PSZ:%[^,]+]], i32 0, i32 0

// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [9 x ptr], ptr [[BP]], i32 0, i32 [[IDX0:0]]
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [9 x ptr], ptr [[P]], i32 0, i32 [[IDX0]]
// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [9 x ptr], ptr [[BP]], i32 0, i32 [[IDX1:1]]
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [9 x ptr], ptr [[P]], i32 0, i32 [[IDX1]]
// CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [9 x ptr], ptr [[BP]], i32 0, i32 [[IDX2:2]]
// CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [9 x ptr], ptr [[P]], i32 0, i32 [[IDX2]]
// CHECK-DAG:   [[BPADDR3:%.+]] = getelementptr inbounds [9 x ptr], ptr [[BP]], i32 0, i32 [[IDX3:3]]
// CHECK-DAG:   [[PADDR3:%.+]] = getelementptr inbounds [9 x ptr], ptr [[P]], i32 0, i32 [[IDX3]]
// CHECK-DAG:   [[PSZ3:%.+]] = getelementptr inbounds [9 x i64], ptr [[PSZ]], i32 0, i32 [[IDX3]]
// CHECK-DAG:   [[BPADDR4:%.+]] = getelementptr inbounds [9 x ptr], ptr [[BP]], i32 0, i32 [[IDX4:4]]
// CHECK-DAG:   [[PADDR4:%.+]] = getelementptr inbounds [9 x ptr], ptr [[P]], i32 0, i32 [[IDX4]]
// CHECK-DAG:   [[BPADDR5:%.+]] = getelementptr inbounds [9 x ptr], ptr [[BP]], i32 0, i32 [[IDX5:5]]
// CHECK-DAG:   [[PADDR5:%.+]] = getelementptr inbounds [9 x ptr], ptr [[P]], i32 0, i32 [[IDX5]]
// CHECK-DAG:   [[BPADDR6:%.+]] = getelementptr inbounds [9 x ptr], ptr [[BP]], i32 0, i32 [[IDX6:6]]
// CHECK-DAG:   [[PADDR6:%.+]] = getelementptr inbounds [9 x ptr], ptr [[P]], i32 0, i32 [[IDX6]]
// CHECK-DAG:   [[BPADDR7:%.+]] = getelementptr inbounds [9 x ptr], ptr [[BP]], i32 0, i32 [[IDX7:7]]
// CHECK-DAG:   [[PADDR7:%.+]] = getelementptr inbounds [9 x ptr], ptr [[P]], i32 0, i32 [[IDX7]]
// CHECK-DAG:   [[PSZ7:%.+]] = getelementptr inbounds [9 x i64], ptr [[PSZ]], i32 0, i32 [[IDX7]]
// CHECK-DAG:   [[BPADDR8:%.+]] = getelementptr inbounds [9 x ptr], ptr [[BP]], i32 0, i32 [[IDX8:8]]
// CHECK-DAG:   [[PADDR8:%.+]] = getelementptr inbounds [9 x ptr], ptr [[P]], i32 0, i32 [[IDX8]]

// The names below are not necessarily consistent with the names used for the
// addresses above as some are repeated.
// CHECK-DAG:   store i[[SZ]] [[VLA0]], ptr [[BPADDR2]]
// CHECK-DAG:   store i[[SZ]] [[VLA0]], ptr [[PADDR2]]

// CHECK-DAG:   store i[[SZ]] [[VLA1]], ptr [[BPADDR6]]
// CHECK-DAG:   store i[[SZ]] [[VLA1]], ptr [[PADDR6]]

// CHECK-DAG:   store i[[SZ]] 5, ptr [[BPADDR5]]
// CHECK-DAG:   store i[[SZ]] 5, ptr [[PADDR5]]

// CHECK-DAG:   store i[[SZ]] [[A_CVAL]], ptr [[BPADDR0]]
// CHECK-DAG:   store i[[SZ]] [[A_CVAL]], ptr [[PADDR0]]

// CHECK-DAG:   store ptr %{{.+}}, ptr [[BPADDR1]]
// CHECK-DAG:   store ptr %{{.+}}, ptr [[PADDR1]]

// CHECK-DAG:   store ptr %{{.+}}, ptr [[BPADDR3]]
// CHECK-DAG:   store ptr %{{.+}}, ptr [[PADDR3]]
// CHECK-DAG:   store i64 [[BNSIZE]], ptr [[PSZ3]]

// CHECK-DAG:   store ptr %{{.+}}, ptr [[BPADDR4]]
// CHECK-DAG:   store ptr %{{.+}}, ptr [[PADDR4]]

// CHECK-DAG:   store ptr %{{.+}}, ptr [[BPADDR7]]
// CHECK-DAG:   store ptr %{{.+}}, ptr [[PADDR7]]
// CHECK-DAG:   store i64 [[CNSIZE]], ptr [[PSZ7]]

// CHECK-DAG:   store ptr %{{.+}}, ptr [[BPADDR8]]
// CHECK-DAG:   store ptr %{{.+}}, ptr [[PADDR8]]

// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT4:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
// CHECK-NEXT:  br label %[[IFEND:.+]]
// CHECK:       [[IFELSE]]
// CHECK:       call void [[HVT4]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[IFEND]]

// CHECK:       [[IFEND]]
#pragma omp target if (n > 20)
  {
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

// CHECK:       define internal void [[HVT0]]()

// CHECK: define internal void [[HVT0_:@.+]](ptr noundef {{%[^,]+}}, i[[SZ]] noundef {{%[^,]+}})
// CHECK: define internal {{.*}}i32 [[OMP_TASK_ENTRY]](i32 {{.*}}%0, ptr noalias noundef %1)
// CHECK-DAG: [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:%.+]], i32 {{.+}}, i32 {{.+}}, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CHECK-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CHECK-DAG: store ptr [[BPR:%.+]], ptr [[BPARG]]
// CHECK-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CHECK-DAG: store ptr [[PR:%.+]], ptr [[PARG]]
// CHECK-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CHECK-DAG: store ptr [[SIZE:%.+]], ptr [[SARG]]
// CHECK-DAG: [[DEVICE]] = sext i32 [[DEV:%.+]] to i64
// CHECK-DAG: [[DEV]] = load i32, ptr [[DEVADDR:%.+]], align
// CHECK-DAG: [[DEVADDR]] = getelementptr inbounds [[ANON_T]], ptr {{%.+}}, i32 0, i32 2
// CHECK-DAG: [[BPR]] = load ptr, ptr [[FPPTR_BPR:%.+]], align
// CHECK-DAG: [[PR]] = load ptr, ptr [[FPPTR_PR:%.+]], align
// CHECK-DAG: [[SIZE]] = load ptr, ptr [[FPPTR_SIZE:%.+]], align
// CHECK-DAG: call void {{%[0-9]+}}(ptr {{%[^,]+}}, ptr [[FPPTR_PLOCAL:%.+]], ptr [[FPPTR_GLOBAL:%.+]], ptr [[FPPTR_BPR]], ptr [[FPPTR_PR]], ptr [[FPPTR_SIZE]])
// CHECK-DAG: [[PLOCALADDR:%.+]] = load ptr, ptr [[FPPTR_PLOCAL]], align
// CHECK-DAG: {{%.+}} = load ptr, ptr [[FPPTR_GLOBAL:%.+]], align
// CHECK: [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT: br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
// CHECK: [[FAIL]]
// CHECK: [[PLOCAL:%.+]] = load ptr, ptr [[PLOCALADDR]], align
// CHECK: [[GLOBAL:%.+]] = load i32, ptr {{@.+}}, align
// CHECK-32: store i32 [[GLOBAL]], ptr [[GLOBALCAST:%.+]], align
// CHECK-64: store i32 [[GLOBAL]], ptr [[GLOBALCAST:%.+]], align
// CHECK: [[GLOBAL:%.+]] = load i[[SZ]], ptr [[GLOBALCAST]], align
// CHECK: call void [[HVT0_]](ptr [[PLOCAL]], i[[SZ]] [[GLOBAL]])
// CHECK-NEXT: br label %[[END]]
// CHECK: [[END]]

// CHECK:       define internal void [[HVT1]](i[[SZ]] noundef %{{.+}})
// Create stack storage and store argument in there.
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       store i[[SZ]] %{{.+}}, ptr [[AA_ADDR]], align
// CHECK-64:    load i32, ptr [[AA_ADDR]], align
// CHECK-32:    load i32, ptr [[AA_ADDR]], align

// CHECK:       define internal void [[HVT2]](i[[SZ]] noundef %{{.+}})
// Create stack storage and store argument in there.
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       store i[[SZ]] %{{.+}}, ptr [[AA_ADDR]], align
// CHECK:       load i16, ptr [[AA_ADDR]], align

// CHECK:       define internal void [[HVT3]]
// Create stack storage and store argument in there.
// CHECK:       [[A_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK:       [[AA_ADDR:%.+]] = alloca i[[SZ]], align
// CHECK-DAG:   store i[[SZ]] %{{.+}}, ptr [[A_ADDR]], align
// CHECK-DAG:   store i[[SZ]] %{{.+}}, ptr [[AA_ADDR]], align
// CHECK-64-DAG:load i32, ptr [[A_ADDR]], align
// CHECK-32-DAG:load i32, ptr [[A_ADDR]], align
// CHECK-DAG:   load i16, ptr [[AA_ADDR]], align

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

// Use captures.
// CHECK-64-DAG:   load i32, ptr [[LOCAL_A]]
// CHECK-32-DAG:   load i32, ptr [[LOCAL_A]]
// CHECK-DAG:   getelementptr inbounds [10 x float], ptr [[REF_B]], i[[SZ]] 0, i[[SZ]] 2
// CHECK-DAG:   getelementptr inbounds float, ptr [[REF_BN]], i[[SZ]] 3
// CHECK-DAG:   getelementptr inbounds [5 x [10 x double]], ptr [[REF_C]], i[[SZ]] 0, i[[SZ]] 1
// CHECK-DAG:   getelementptr inbounds double, ptr [[REF_CN]], i[[SZ]] %{{.+}}
// CHECK-DAG:   getelementptr inbounds [[TT]], ptr [[REF_D]], i32 0, i32 0

template<typename tx>
tx ftemplate(int n) {
  tx a = 0;
  short aa = 0;
  tx b[10];

  #pragma omp target if(n>40)
  {
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

  #pragma omp target if(n>50)
  {
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

    #pragma omp target if(n>60)
    {
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
// CHECK-64:       store i32 %{{.+}}, ptr [[B_CADDR:%.+]],
// CHECK-64:       [[B_CVAL:%.+]] = load i[[SZ]], ptr [[B_CADDR]],

// CHECK-32:       store i32 %{{.+}}, ptr %__vla_expr
// CHECK-32:       store i32 %{{.+}}, ptr [[B_CADDR:%.+]],
// CHECK-32:       [[B_CVAL:%.+]] = load i[[SZ]], ptr [[B_CADDR]],

// CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 60
// CHECK:       br i1 [[IF]], label %[[TRY:[^,]+]], label %[[IFELSE:[^,]+]]
// CHECK:       [[TRY]]
// We capture 2 VLA sizes in this target region
// CHECK:       [[CELEMSIZE2:%.+]] = mul nuw i[[SZ]] 2, [[VLA0:%.+]]
// CHECK-64:    [[CSIZE:%.+]] = mul nuw i64 [[CELEMSIZE2]], 2
// CHECK-32:    [[CSZSIZE:%.+]] = mul nuw i32 [[CELEMSIZE2]], 2
// CHECK-32:    [[CSIZE:%.+]] = sext i32 [[CSZSIZE]] to i64

// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 {{.+}}, i32 {{.+}}, i32 {{.+}}, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CHECK-DAG:   [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CHECK-DAG:   store ptr [[BPR:%.+]], ptr [[BPARG]]
// CHECK-DAG:   [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CHECK-DAG:   store ptr [[PR:%.+]], ptr [[PARG]]
// CHECK-DAG:   [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CHECK-DAG:   store ptr [[SZ7:%.+]], ptr [[SARG]]
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [5 x ptr], ptr [[BP:%.+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [5 x ptr], ptr [[P:%.+]], i32 0, i32 0
// CHECK-DAG:   [[SZ7]] = getelementptr inbounds [5 x i64], ptr [[PSZ:%.+]], i32 0, i32 0
// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BP]], i32 0, i32 [[IDX0:0]]
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [5 x ptr], ptr [[P]], i32 0, i32 [[IDX0]]
// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BP]], i32 0, i32 [[IDX1:1]]
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [5 x ptr], ptr [[P]], i32 0, i32 [[IDX1]]
// CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BP]], i32 0, i32 [[IDX2:2]]
// CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [5 x ptr], ptr [[P]], i32 0, i32 [[IDX2]]
// CHECK-DAG:   [[BPADDR3:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BP]], i32 0, i32 [[IDX3:3]]
// CHECK-DAG:   [[PADDR3:%.+]] = getelementptr inbounds [5 x ptr], ptr [[P]], i32 0, i32 [[IDX3]]
// CHECK-DAG:   [[BPADDR4:%.+]] = getelementptr inbounds [5 x ptr], ptr [[BP]], i32 0, i32 [[IDX4:4]]
// CHECK-DAG:   [[PADDR4:%.+]] = getelementptr inbounds [5 x ptr], ptr [[P]], i32 0, i32 [[IDX4]]
// CHECK-DAG:   [[PSZ4:%.+]] = getelementptr inbounds [5 x i64], ptr [[PSZ:%.+]], i32 0, i32 [[IDX4]]

// The names below are not necessarily consistent with the names used for the
// addresses above as some are repeated.
// CHECK-DAG:   store ptr %{{.+}}, ptr [[BPADDR4]]
// CHECK-DAG:   store ptr %{{.+}}, ptr [[PADDR4]]
// CHECK-DAG:   store i64 [[CSIZE]], ptr [[PSZ4]]

// CHECK-DAG:   store i[[SZ]] [[VLA0]], ptr [[BPADDR3]]
// CHECK-DAG:   store i[[SZ]] [[VLA0]], ptr [[PADDR3]]

// CHECK-DAG:   store i[[SZ]] 2, ptr [[BPADDR2]]
// CHECK-DAG:   store i[[SZ]] 2, ptr [[PADDR2]]

// CHECK-DAG:   store i[[SZ]] [[B_CVAL]], ptr [[BPADDR1]]
// CHECK-DAG:   store i[[SZ]] [[B_CVAL]], ptr [[PADDR1]]

// CHECK-DAG:   store ptr [[THIS:%.+]], ptr [[BPADDR0]]
// CHECK-DAG:   store ptr [[A:%.+]], ptr [[PADDR0]]

// CHECK:       [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// CHECK-NEXT:  br i1 [[ERROR]], label %[[FAIL:.+]], label %[[END:[^,]+]]
// CHECK:       [[FAIL]]
// CHECK:       call void [[HVT7:@.+]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[END]]
// CHECK:       [[END]]
// CHECK-NEXT:  br label %[[IFEND:.+]]
// CHECK:       [[IFELSE]]
// CHECK:       call void [[HVT7]]({{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}}, {{[^,]+}})
// CHECK-NEXT:  br label %[[IFEND]]

// CHECK:       [[IFEND]]

//
// CHECK: define {{.*}}[[FSTATIC]]
//
// CHECK:       [[IF:%.+]] = icmp sgt i32 {{[^,]+}}, 50
// CHECK:       br i1 [[IF]], label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CHECK:       [[IFTHEN]]
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 {{.+}}, i32 {{.+}}, i32 {{.+}}, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CHECK-DAG:   [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CHECK-DAG:   store ptr [[BPR:%.+]], ptr [[BPARG]]
// CHECK-DAG:   [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CHECK-DAG:   store ptr [[PR:%.+]], ptr [[PARG]]
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [4 x ptr], ptr [[BP:%.+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [4 x ptr], ptr [[P:%.+]], i32 0, i32 0

// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [4 x ptr], ptr [[BP]], i32 0, i32 0
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [4 x ptr], ptr [[P]], i32 0, i32 0
// CHECK-DAG:   store i[[SZ]] [[VAL0:%[^,]+]], ptr [[BPADDR0]]
// CHECK-DAG:   store i[[SZ]] [[VAL0]], ptr [[PADDR0]]

// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [4 x ptr], ptr [[BP]], i32 0, i32 1
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [4 x ptr], ptr [[P]], i32 0, i32 1
// CHECK-DAG:   store i[[SZ]] [[VAL1:%[^,]+]], ptr [[BPADDR1]]
// CHECK-DAG:   store i[[SZ]] [[VAL1]], ptr [[PADDR1]]

// CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [4 x ptr], ptr [[BP]], i32 0, i32 2
// CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [4 x ptr], ptr [[P]], i32 0, i32 2
// CHECK-DAG:   store i[[SZ]] [[VAL2:%[^,]+]], ptr [[BPADDR2]]
// CHECK-DAG:   store i[[SZ]] [[VAL2]], ptr [[PADDR2]]

// CHECK-DAG:   [[BPADDR3:%.+]] = getelementptr inbounds [4 x ptr], ptr [[BP]], i32 0, i32 3
// CHECK-DAG:   [[PADDR3:%.+]] = getelementptr inbounds [4 x ptr], ptr [[P]], i32 0, i32 3
// CHECK-DAG:   store ptr [[VAL3:%[^,]+]], ptr [[BPADDR3]]
// CHECK-DAG:   store ptr [[VAL3]], ptr [[PADDR3]]

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
// CHECK-DAG:   [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 {{.+}}, i32 {{.+}}, i32 {{.+}}, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CHECK-DAG:   [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CHECK-DAG:   store ptr [[BPR:%.+]], ptr [[BPARG]]
// CHECK-DAG:   [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CHECK-DAG:   store ptr [[PR:%.+]], ptr [[PARG]]
// CHECK-DAG:   [[BPR]] = getelementptr inbounds [3 x ptr], ptr [[BP:%.+]], i32 0, i32 0
// CHECK-DAG:   [[PR]] = getelementptr inbounds [3 x ptr], ptr [[P:%.+]], i32 0, i32 0

// CHECK-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [3 x ptr], ptr [[BP]], i32 0, i32 0
// CHECK-DAG:   [[PADDR0:%.+]] = getelementptr inbounds [3 x ptr], ptr [[P]], i32 0, i32 0
// CHECK-DAG:   store i[[SZ]] [[VAL0:%[^,]+]], ptr [[BPADDR0]]
// CHECK-DAG:   store i[[SZ]] [[VAL0]], ptr [[PADDR0]]

// CHECK-DAG:   [[BPADDR1:%.+]] = getelementptr inbounds [3 x ptr], ptr [[BP]], i32 0, i32 1
// CHECK-DAG:   [[PADDR1:%.+]] = getelementptr inbounds [3 x ptr], ptr [[P]], i32 0, i32 1
// CHECK-DAG:   store i[[SZ]] [[VAL1:%[^,]+]], ptr [[BPADDR1]]
// CHECK-DAG:   store i[[SZ]] [[VAL1]], ptr [[PADDR1]]

// CHECK-DAG:   [[BPADDR2:%.+]] = getelementptr inbounds [3 x ptr], ptr [[BP]], i32 0, i32 2
// CHECK-DAG:   [[PADDR2:%.+]] = getelementptr inbounds [3 x ptr], ptr [[P]], i32 0, i32 2
// CHECK-DAG:   store ptr [[VAL2:%[^,]+]], ptr [[BPADDR2]]
// CHECK-DAG:   store ptr [[VAL2]], ptr [[PADDR2]]

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

// OMP45: define internal void @__omp_offloading_{{.+}}_{{.+}}bar{{.+}}_l{{[0-9]+}}(i[[SZ]] noundef %{{.+}})

// OMP45: define {{.*}}@{{.*}}zee{{.*}}

// OMP45:       [[LOCAL_THIS:%.+]] = alloca ptr
// OMP45:       [[BP:%.+]] = alloca [1 x ptr]
// OMP45:       [[P:%.+]] = alloca [1 x ptr]
// OMP45:       [[LOCAL_THIS1:%.+]] = load ptr, ptr [[LOCAL_THIS]]

// OMP45:       call void @__kmpc_critical(
// OMP45:       [[ARR_IDX:%.+]] = getelementptr inbounds [[S2]], ptr [[LOCAL_THIS1]], i[[SZ]] 0
// OMP45:       [[ARR_IDX2:%.+]] = getelementptr inbounds [[S2]], ptr [[LOCAL_THIS1]], i[[SZ]] 0

// OMP45-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [1 x ptr], ptr [[BP]], i32 0, i32 0
// OMP45-DAG:   [[PADDR0:%.+]] =  getelementptr inbounds [1 x ptr], ptr [[P]], i32 0, i32 0
// OMP45-DAG:   store ptr [[ARR_IDX]], ptr [[BPADDR0]]
// OMP45-DAG:   store ptr [[ARR_IDX2]], ptr [[PADDR0]]

// OMP45:       [[BPR:%.+]] = getelementptr inbounds [1 x ptr], ptr [[BP]], i32 0, i32 0
// OMP45:       [[PR:%.+]] = getelementptr inbounds [1 x ptr], ptr [[P]], i32 0, i32 0
// OMP45:       [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 {{.+}}, i32 {{.+}}, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// OMP45-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// OMP45-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
// OMP45:       [[FAIL]]
// OMP45:       call void [[HVT0:@.+]](ptr [[LOCAL_THIS1]])
// OMP45-NEXT:  br label %[[END]]
// OMP45:       [[END]]
// OMP45:       call void @__kmpc_end_critical(

// Check that the offloading functions are emitted and that the arguments are
// correct and loaded correctly for the target regions of the callees of bar().

// CHECK:       define internal void [[HVT7]]
// Create local storage for each capture.
// CHECK:       [[LOCAL_THIS:%.+]] = alloca ptr
// CHECK:       [[LOCAL_B:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_VLA1:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_VLA2:%.+]] = alloca i[[SZ]]
// CHECK:       [[LOCAL_C:%.+]] = alloca ptr
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
// Use captures.
// CHECK-DAG:   getelementptr inbounds [[S1]], ptr [[REF_THIS]], i32 0, i32 0
// CHECK-64-DAG:load i32, ptr [[LOCAL_B]]
// CHECK-32-DAG:load i32, ptr [[LOCAL_B]]
// CHECK-DAG:   getelementptr inbounds i16, ptr [[REF_C]], i[[SZ]] %{{.+}}


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
// CHECK-DAG:      [[REF_B:%.+]] = load ptr, ptr [[LOCAL_B]],
// Use captures.
// CHECK-64-DAG:   load i32, ptr [[LOCAL_A]]
// CHECK-DAG:      load i16, ptr [[LOCAL_AA]]
// CHECK-DAG:      load i8, ptr [[LOCAL_AAA]]
// CHECK-32-DAG:   load i32, ptr [[LOCAL_A]]
// CHECK-DAG:      getelementptr inbounds [10 x i32], ptr [[REF_B]], i[[SZ]] 0, i[[SZ]] 2

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
// Use captures.
// CHECK-64-DAG:   load i32, ptr [[LOCAL_A]]
// CHECK-32-DAG:   load i32, ptr [[LOCAL_A]]
// CHECK-DAG:   load i16, ptr [[LOCAL_AA]]
// CHECK-DAG:   getelementptr inbounds [10 x i32], ptr [[REF_B]], i[[SZ]] 0, i[[SZ]] 2

// OMP50: define internal void @__omp_offloading_{{.+}}_{{.+}}bar{{.+}}_l{{[0-9]+}}(i[[SZ]] noundef %{{.+}})

// OMP50: define {{.*}}@{{.*}}zee{{.*}}

// OMP50:       [[LOCAL_THIS:%.+]] = alloca ptr
// OMP50:       [[BP:%.+]] = alloca [1 x ptr]
// OMP50:       [[P:%.+]] = alloca [1 x ptr]
// OMP50:       [[LOCAL_THIS1:%.+]] = load ptr, ptr [[LOCAL_THIS]]
// OMP50:       [[ARR_IDX:%.+]] = getelementptr inbounds [[S2]], ptr [[LOCAL_THIS1]], i[[SZ]] 0
// OMP50:       [[ARR_IDX2:%.+]] = getelementptr inbounds [[S2]], ptr [[LOCAL_THIS1]], i[[SZ]] 0

// OMP50-DAG:   [[BPADDR0:%.+]] = getelementptr inbounds [1 x ptr], ptr [[BP]], i32 0, i32 0
// OMP50-DAG:   [[PADDR0:%.+]] =  getelementptr inbounds [1 x ptr], ptr [[P]], i32 0, i32 0
// OMP50-DAG:   store ptr [[ARR_IDX]], ptr [[BPADDR0]]
// OMP50-DAG:   store ptr [[ARR_IDX2]], ptr [[PADDR0]]

// OMP50:       [[BPR:%.+]] = getelementptr inbounds [1 x ptr], ptr [[BP]], i32 0, i32 0
// OMP50:       [[PR:%.+]] = getelementptr inbounds [1 x ptr], ptr [[P]], i32 0, i32 0
// OMP50:       [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 {{.+}}, i32 {{.+}}, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// OMP50-NEXT:  [[ERROR:%.+]] = icmp ne i32 [[RET]], 0
// OMP50-NEXT:  br i1 [[ERROR]], label %[[FAIL:[^,]+]], label %[[END:[^,]+]]
// OMP50:       [[FAIL]]
// OMP50:       call void [[HVT0:@.+]](ptr [[LOCAL_THIS1]])
// OMP50-NEXT:  br label %[[END]]
// OMP50:       [[END]]

void bar () {
#define pragma_target _Pragma("omp target")
pragma_target
{
  global = 0;
#pragma omp parallel shared(global)
  global = 1;
}
}

class S2 {
  int a, b, c;

public:
  void zee() {
#pragma omp critical
    #pragma omp target map(this[0])
      a++;
  }
};

#ifdef _DOMP51
void thread_limit_target(int TargetTL, int TeamsTL) {

#pragma omp target
{}
// OMP51: call i32 @__tgt_target_kernel({{.*}}, i64 -1, i32 -1, i32 0,

#pragma omp target
#pragma omp teams
{}
// OMP51: call i32 @__tgt_target_kernel({{.*}}, i64 -1, i32 0, i32 0,

#pragma omp target thread_limit(TargetTL)
{}
// OMP51: [[TL:%.*]] = load {{.*}} %TargetTL.addr
// OMP51: store {{.*}} [[TL]], {{.*}} [[CEA:%.*]]
// OMP51: load {{.*}} [[CEA]]
// OMP51: [[CE:%.*]] = load {{.*}} [[CEA]]
// OMP51: call ptr @__kmpc_omp_task_alloc({{.*@.omp_task_entry.*}})
// OMP51: call i32 [[OMP_TASK_ENTRY]]

#pragma omp target thread_limit(TargetTL)
#pragma omp teams
{}
// OMP51: [[TL:%.*]] = load {{.*}} %TargetTL.addr
// OMP51: store {{.*}} [[TL]], {{.*}} [[CEA:%.*]]
// OMP51: load {{.*}} [[CEA]]
// OMP51: call ptr @__kmpc_omp_task_alloc({{.*@.omp_task_entry.*}})
// OMP51: call i32 [[OMP_TASK_ENTRY]]

#pragma omp target
#pragma omp teams thread_limit(TeamsTL)
{}
// OMP51: load {{.*}} %TeamsTL.addr
// OMP51: [[TeamsL:%.*]] = load {{.*}} %TeamsTL.addr
// OMP51: call i32 @__tgt_target_kernel({{.*}}, i64 -1, i32 0, i32 [[TeamsL]],

#pragma omp target thread_limit(TargetTL)
#pragma omp teams thread_limit(TeamsTL)
{}
// OMP51: load {{.*}} %TeamsTL.addr
// OMP51: [[TeamsL:%.*]] = load {{.*}} %TeamsTL.addr
// OMP51: call ptr @__kmpc_omp_task_alloc({{.*@.omp_task_entry.*}})
// OMP51: call i32 [[OMP_TASK_ENTRY]]

}
#endif
// Check that the offloading functions are called after setting thread_limit in the task entry functions

// OMP51: define internal {{.*}}i32 [[OMP_TASK_ENTRY:@.+]](i32 {{.*}}%0, ptr noalias noundef %1)
// OMP51: call void @__kmpc_set_thread_limit(ptr @{{.+}}, i32 %{{.+}}, i32 %{{.+}})
// OMP51: call i32 @__tgt_target_kernel({{.*}}, i64 -1, i32 -1,

// OMP51: define internal {{.*}}i32 [[OMP_TASK_ENTRY:@.+]](i32 {{.*}}%0, ptr noalias noundef %1)
// OMP51: call void @__kmpc_set_thread_limit(ptr @{{.+}}, i32 %{{.+}}, i32 %{{.+}})
// OMP51: call i32 @__tgt_target_kernel({{.*}}, i64 -1, i32 0,

// OMP51: define internal {{.*}}i32 [[OMP_TASK_ENTRY:@.+]](i32 {{.*}}%0, ptr noalias noundef %1)
// OMP51: call void @__kmpc_set_thread_limit(ptr @{{.+}}, i32 %{{.+}}, i32 %{{.+}})
// OMP51: call i32 @__tgt_target_kernel({{.*}}, i64 -1, i32 0,


// CHECK:     define internal void @.omp_offloading.requires_reg()
// CHECK:     call void @__tgt_register_requires(i64 1)
// CHECK:     ret void

int main () {
  S2 bar;
  bar.zee();
}

#endif
