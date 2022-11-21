// Test host codegen.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// Test target teams distribute parallel for simd codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s -check-prefix=TCHECK
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s -check-prefix=TCHECK
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s -check-prefix=TCHECK
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s -check-prefix=TCHECK

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=45 -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// Check that no target code is emitted if no omptests flag was provided.
// RUN: %clang_cc1 -verify -fopenmp -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK-NTARGET

// RUN: %clang_cc1 -verify -fopenmp-simd -fopenmp-version=45 -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// SIMD-ONLY2-NOT: {{__kmpc|__tgt}}

// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

// Test target teams distribute parallel for simd codegen - host bc file has to be created first.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s -check-prefix=TCHECK
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s -check-prefix=TCHECK
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s -check-prefix=TCHECK
// RUN: %clang_cc1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s -check-prefix=TCHECK

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}

// Check that no target code is emitted if no omptests flag was provided.
// RUN: %clang_cc1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s -check-prefix=CHECK-NTARGET

// RUN: %clang_cc1 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// SIMD-ONLY2-NOT: {{__kmpc|__tgt}}

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK-DAG: [[SA:%.+]] = type { [4 x i32] }
// CHECK-DAG: [[SB:%.+]] = type { [8 x i32] }
// CHECK-DAG: [[SC:%.+]] = type { [16 x i32] }
// CHECK-DAG: [[SD:%.+]] = type { [32 x i32] }
// CHECK-DAG: [[SE:%.+]] = type { [64 x i32] }
// CHECK-DAG: [[ST1:%.+]] = type { [228 x i32] }
// CHECK-DAG: [[ST2:%.+]] = type { [1128 x i32] }
// CHECK-DAG: [[ENTTY:%.+]] = type { ptr, ptr, i[[SZ:32|64]], i32, i32 }

// TCHECK:    [[ENTTY:%.+]] = type { ptr, ptr, i[[SZ:32|64]], i32, i32 }

// CHECK-DAG: [[A1:@.+]] = internal global [[SA]]
// CHECK-DAG: [[A2:@.+]] ={{.*}} global [[SA]]
// CHECK-DAG: [[B1:@.+]] ={{.*}} global [[SB]]
// CHECK-DAG: [[B2:@.+]] ={{.*}} global [[SB]]
// CHECK-DAG: [[C1:@.+]] = internal global [[SC]]
// CHECK-DAG: [[D1:@.+]] ={{.*}} global [[SD]]
// CHECK-DAG: [[E1:@.+]] ={{.*}} global [[SE]]
// CHECK-DAG: [[T1:@.+]] ={{.*}} global [[ST1]]
// CHECK-DAG: [[T2:@.+]] ={{.*}} global [[ST2]]

// CHECK-NTARGET-DAG: [[SA:%.+]] = type { [4 x i32] }
// CHECK-NTARGET-DAG: [[SB:%.+]] = type { [8 x i32] }
// CHECK-NTARGET-DAG: [[SC:%.+]] = type { [16 x i32] }
// CHECK-NTARGET-DAG: [[SD:%.+]] = type { [32 x i32] }
// CHECK-NTARGET-DAG: [[SE:%.+]] = type { [64 x i32] }
// CHECK-NTARGET-DAG: [[ST1:%.+]] = type { [228 x i32] }
// CHECK-NTARGET-DAG: [[ST2:%.+]] = type { [1128 x i32] }
// CHECK-NTARGET-NOT: type { ptr, ptr, %
// CHECK-NTARGET-NOT: type { i32, %

// We have 7 target regions

// CHECK-DAG: {{@.+}} = weak{{.*}} constant i8 0
// TCHECK-NOT: {{@.+}} = weak{{.*}} constant i8 0
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 4]
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 800]
// CHECK-DAG: {{@.+}} = weak{{.*}} constant i8 0
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 4]
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 800]
// CHECK-DAG: {{@.+}} = weak{{.*}} constant i8 0
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 4]
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 800]
// CHECK-DAG: {{@.+}} = weak{{.*}} constant i8 0
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 4]
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 800]
// CHECK-DAG: {{@.+}} = weak{{.*}} constant i8 0
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 4]
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 800]
// CHECK-DAG: {{@.+}} = weak{{.*}} constant i8 0
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 4]
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 800]
// CHECK-DAG: {{@.+}} = weak{{.*}} constant i8 0
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 4]
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 800]
// CHECK-DAG: {{@.+}} = weak{{.*}} constant i8 0
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 4]
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 800]
// CHECK-DAG: {{@.+}} = weak{{.*}} constant i8 0
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 4]
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 800]
// CHECK-DAG: {{@.+}} = weak{{.*}} constant i8 0
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 4]
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 800]
// CHECK-DAG: {{@.+}} = weak{{.*}} constant i8 0
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 4]
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 800]
// CHECK-DAG: {{@.+}} = weak{{.*}} constant i8 0
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 4]
// CHECK-DAG: {{@.+}} = private unnamed_addr constant [1 x i64] [i64 800]

// CHECK-NTARGET-NOT: weak constant i8 0
// CHECK-NTARGET-NOT: private unnamed_addr constant [1 x i

// CHECK-DAG: [[NAMEPTR1:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME1:__omp_offloading_[0-9a-f]+_[0-9a-f]+__Z.+_l[0-9]+]]\00"
// CHECK-DAG: [[ENTRY1:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR1]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// CHECK-DAG: [[NAMEPTR2:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME2:.+]]\00"
// CHECK-DAG: [[ENTRY2:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR2]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// CHECK-DAG: [[NAMEPTR3:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME3:.+]]\00"
// CHECK-DAG: [[ENTRY3:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR3]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// CHECK-DAG: [[NAMEPTR4:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME4:.+]]\00"
// CHECK-DAG: [[ENTRY4:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR4]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// CHECK-DAG: [[NAMEPTR5:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME5:.+]]\00"
// CHECK-DAG: [[ENTRY5:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR5]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// CHECK-DAG: [[NAMEPTR6:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME6:.+]]\00"
// CHECK-DAG: [[ENTRY6:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR6]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// CHECK-DAG: [[NAMEPTR7:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME7:.+]]\00"
// CHECK-DAG: [[ENTRY7:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR7]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// CHECK-DAG: [[NAMEPTR8:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME8:.+]]\00"
// CHECK-DAG: [[ENTRY8:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR8]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// CHECK-DAG: [[NAMEPTR9:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME9:.+]]\00"
// CHECK-DAG: [[ENTRY9:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR9]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// CHECK-DAG: [[NAMEPTR10:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME10:.+]]\00"
// CHECK-DAG: [[ENTRY10:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR10]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// CHECK-DAG: [[NAMEPTR11:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME11:.+]]\00"
// CHECK-DAG: [[ENTRY11:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR11]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// CHECK-DAG: [[NAMEPTR12:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME12:.+]]\00"
// CHECK-DAG: [[ENTRY12:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR12]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1

// TCHECK-DAG: [[NAMEPTR1:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME1:__omp_offloading_[0-9a-f]+_[0-9a-f]+__Z.+_l[0-9]+]]\00"
// TCHECK-DAG: [[ENTRY1:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR1]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// TCHECK-DAG: [[NAMEPTR2:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME2:.+]]\00"
// TCHECK-DAG: [[ENTRY2:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR2]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// TCHECK-DAG: [[NAMEPTR3:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME3:.+]]\00"
// TCHECK-DAG: [[ENTRY3:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR3]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// TCHECK-DAG: [[NAMEPTR4:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME4:.+]]\00"
// TCHECK-DAG: [[ENTRY4:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR4]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// TCHECK-DAG: [[NAMEPTR5:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME5:.+]]\00"
// TCHECK-DAG: [[ENTRY5:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR5]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// TCHECK-DAG: [[NAMEPTR6:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME6:.+]]\00"
// TCHECK-DAG: [[ENTRY6:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR6]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// TCHECK-DAG: [[NAMEPTR7:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME7:.+]]\00"
// TCHECK-DAG: [[ENTRY7:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR7]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// TCHECK-DAG: [[NAMEPTR8:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME8:.+]]\00"
// TCHECK-DAG: [[ENTRY8:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR8]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// TCHECK-DAG: [[NAMEPTR9:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME9:.+]]\00"
// TCHECK-DAG: [[ENTRY9:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR9]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// TCHECK-DAG: [[NAMEPTR10:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME10:.+]]\00"
// TCHECK-DAG: [[ENTRY10:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR10]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// TCHECK-DAG: [[NAMEPTR11:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME11:.+]]\00"
// TCHECK-DAG: [[ENTRY11:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR11]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1
// TCHECK-DAG: [[NAMEPTR12:@.+]] = internal unnamed_addr constant [{{.*}} x i8] c"[[NAME12:.+]]\00"
// TCHECK-DAG: [[ENTRY12:@.+]] = weak{{.*}} constant [[ENTTY]] { ptr @{{.*}}, ptr [[NAMEPTR12]], i[[SZ]] 0, i32 0, i32 0 }, section "omp_offloading_entries", align 1

// We have 4 initializers, one for the 500 priority, another one for 501, or more for the default priority, and the last one for the offloading registration function.
// CHECK: @llvm.global_ctors = appending global [4 x { i32, ptr, ptr }] [
// CHECK-SAME: { i32, ptr, ptr } { i32 500, ptr [[P500:@[^,]+]], ptr null },
// CHECK-SAME: { i32, ptr, ptr } { i32 501, ptr [[P501:@[^,]+]], ptr null },
// CHECK-SAME: { i32, ptr, ptr } { i32 65535, ptr [[PMAX:@[^,]+]], ptr null },

// CHECK-NTARGET: @llvm.global_ctors = appending global [3   x { i32, ptr, ptr }] [

extern int *R;

struct SA {
  int arr[4];
  void foo() {
    int a = *R;
    a += 1;
    *R = a;
  }
  SA() {
    int a = *R;
    a += 2;
    *R = a;
  }
  ~SA() {
    int a = *R;
    a += 3;
    *R = a;
  }
};

struct SB {
  int arr[8];
  void foo() {
    int a = *R;
    #pragma omp target teams distribute parallel for simd
    for (int i = 0; i < 10; ++i)
      a += 4;
    *R = a;
  }
  SB() {
    int a = *R;
    a += 5;
    *R = a;
  }
  ~SB() {
    int a = *R;
    a += 6;
    *R = a;
  }
};

struct SC {
  int arr[16];
  void foo() {
    int a = *R;
    a += 7;
    *R = a;
  }
  SC() {
    int a = *R;
    #pragma omp target teams distribute parallel for simd
    for (int i = 0; i < 10; ++i)
      a += 8;
    *R = a;
  }
  ~SC() {
    int a = *R;
    a += 9;
    *R = a;
  }
};

struct SD {
  int arr[32];
  void foo() {
    int a = *R;
    a += 10;
    *R = a;
  }
  SD() {
    int a = *R;
    a += 11;
    *R = a;
  }
  ~SD() {
    int a = *R;
    #pragma omp target teams distribute parallel for simd
    for (int i = 0; i < 10; ++i)
      a += 12;
    *R = a;
  }
};

struct SE {
  int arr[64];
  void foo() {
    int a = *R;
    #pragma omp target teams distribute parallel for simd if(target: 0)
    for (int i = 0; i < 10; ++i)
      a += 13;
    *R = a;
  }
  SE() {
    int a = *R;
    #pragma omp target teams distribute parallel for simd
    for (int i = 0; i < 10; ++i)
      a += 14;
    *R = a;
  }
  ~SE() {
    int a = *R;
    #pragma omp target teams distribute parallel for simd
    for (int i = 0; i < 10; ++i)
      a += 15;
    *R = a;
  }
};

template <int x>
struct ST {
  int arr[128 + x];
  void foo() {
    int a = *R;
    #pragma omp target teams distribute parallel for simd
    for (int i = 0; i < 10; ++i)
      a += 16 + x;
    *R = a;
  }
  ST() {
    int a = *R;
    #pragma omp target teams distribute parallel for simd
    for (int i = 0; i < 10; ++i)
      a += 17 + x;
    *R = a;
  }
  ~ST() {
    int a = *R;
    #pragma omp target teams distribute parallel for simd
    for (int i = 0; i < 10; ++i)
      a += 18 + x;
    *R = a;
  }
};

// We have to make sure we us all the target regions:
//CHECK-DAG: define internal void @[[NAME1]](
//CHECK-DAG: call void @[[NAME1]](
//CHECK-DAG: define internal void @[[NAME2]](
//CHECK-DAG: call void @[[NAME2]](
//CHECK-DAG: define internal void @[[NAME3]](
//CHECK-DAG: call void @[[NAME3]](
//CHECK-DAG: define internal void @[[NAME4]](
//CHECK-DAG: call void @[[NAME4]](
//CHECK-DAG: define internal void @[[NAME5]](
//CHECK-DAG: call void @[[NAME5]](
//CHECK-DAG: define internal void @[[NAME6]](
//CHECK-DAG: call void @[[NAME6]](
//CHECK-DAG: define internal void @[[NAME7]](
//CHECK-DAG: call void @[[NAME7]](
//CHECK-DAG: define internal void @[[NAME8]](
//CHECK-DAG: call void @[[NAME8]](
//CHECK-DAG: define internal void @[[NAME9]](
//CHECK-DAG: call void @[[NAME9]](
//CHECK-DAG: define internal void @[[NAME10]](
//CHECK-DAG: call void @[[NAME10]](
//CHECK-DAG: define internal void @[[NAME11]](
//CHECK-DAG: call void @[[NAME11]](
//CHECK-DAG: define internal void @[[NAME12]](
//CHECK-DAG: call void @[[NAME12]](

//TCHECK-DAG: define weak{{.*}} void @[[NAME1]](
//TCHECK-DAG: define weak{{.*}} void @[[NAME2]](
//TCHECK-DAG: define weak{{.*}} void @[[NAME3]](
//TCHECK-DAG: define weak{{.*}} void @[[NAME4]](
//TCHECK-DAG: define weak{{.*}} void @[[NAME5]](
//TCHECK-DAG: define weak{{.*}} void @[[NAME6]](
//TCHECK-DAG: define weak{{.*}} void @[[NAME7]](
//TCHECK-DAG: define weak{{.*}} void @[[NAME8]](
//TCHECK-DAG: define weak{{.*}} void @[[NAME9]](
//TCHECK-DAG: define weak{{.*}} void @[[NAME10]](
//TCHECK-DAG: define weak{{.*}} void @[[NAME11]](
//TCHECK-DAG: define weak{{.*}} void @[[NAME12]](

// CHECK-NTARGET-NOT: __tgt_target

// TCHECK-NOT: __tgt_target

// We have 2 initializers with priority 500
//CHECK: define internal void [[P500]](
//CHECK:     call void @{{.+}}()
//CHECK:     call void @{{.+}}()
//CHECK-NOT: call void @{{.+}}()
//CHECK:     ret void

// We have 1 initializers with priority 501
//CHECK: define internal void [[P501]](
//CHECK:     call void @{{.+}}()
//CHECK-NOT: call void @{{.+}}()
//CHECK:     ret void

// We have 6 initializers with default priority
//CHECK: define internal void [[PMAX]](
//CHECK:     call void @{{.+}}()
//CHECK:     call void @{{.+}}()
//CHECK:     call void @{{.+}}()
//CHECK:     call void @{{.+}}()
//CHECK:     call void @{{.+}}()
//CHECK:     call void @{{.+}}()
//CHECK-NOT: call void @{{.+}}()
//CHECK:     ret void

static __attribute__((init_priority(500))) SA a1;
SA a2;
SB __attribute__((init_priority(500))) b1;
SB __attribute__((init_priority(501))) b2;
static SC c1;
SD d1;
SE e1;
ST<100> t1;
ST<1000> t2;


int bar(int a){
  int r = a;

  a1.foo();
  a2.foo();
  b1.foo();
  b2.foo();
  c1.foo();
  d1.foo();
  e1.foo();
  t1.foo();
  t2.foo();

  #pragma omp target teams distribute parallel for simd
  for (int i = 0; i < 10; ++i)
    ++r;

  return r + *R;
}

// Check metadata is properly generated:
// CHECK:     !omp_offload.info = !{!{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}}
// CHECK-DAG: = !{i32 0, i32 [[DEVID:-?[0-9]+]], i32 [[FILEID:-?[0-9]+]], !"_ZN2SB3fooEv", i32 245, i32 0, i32 {{[0-9]+}}}
// CHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2SDD1Ev", i32 297, i32 0, i32 {{[0-9]+}}}
// CHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2SEC1Ev", i32 315, i32 0, i32 {{[0-9]+}}}
// CHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2SED1Ev", i32 322, i32 0, i32 {{[0-9]+}}}
// CHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2STILi1000EE3fooEv", i32 334, i32 0, i32 {{[0-9]+}}}
// CHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2STILi100EEC1Ev", i32 341, i32 0, i32 {{[0-9]+}}}
// CHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_Z3bari", i32 446, i32 0, i32 {{[0-9]+}}}
// CHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2STILi100EED1Ev", i32 348, i32 0, i32 {{[0-9]+}}}
// CHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2STILi1000EEC1Ev", i32 341, i32 0, i32 {{[0-9]+}}}
// CHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2STILi1000EED1Ev", i32 348, i32 0, i32 {{[0-9]+}}}
// CHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2STILi100EE3fooEv", i32 334, i32 0, i32 {{[0-9]+}}}
// CHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2SCC1Ev", i32 271, i32 0, i32 {{[0-9]+}}}

// TCHECK:     !omp_offload.info = !{!{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}, !{{[0-9]+}}}
// TCHECK-DAG: = !{i32 0, i32 [[DEVID:-?[0-9]+]], i32 [[FILEID:-?[0-9]+]], !"_ZN2SB3fooEv", i32 245, i32 0, i32 {{[0-9]+}}}
// TCHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2SDD1Ev", i32 297, i32 0, i32 {{[0-9]+}}}
// TCHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2SEC1Ev", i32 315, i32 0, i32 {{[0-9]+}}}
// TCHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2SED1Ev", i32 322, i32 0, i32 {{[0-9]+}}}
// TCHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2STILi1000EE3fooEv", i32 334, i32 0, i32 {{[0-9]+}}}
// TCHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2STILi100EEC1Ev", i32 341, i32 0, i32 {{[0-9]+}}}
// TCHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_Z3bari", i32 446, i32 0, i32 {{[0-9]+}}}
// TCHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2STILi100EED1Ev", i32 348, i32 0, i32 {{[0-9]+}}}
// TCHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2STILi1000EEC1Ev", i32 341, i32 0, i32 {{[0-9]+}}}
// TCHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2STILi1000EED1Ev", i32 348, i32 0, i32 {{[0-9]+}}}
// TCHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2STILi100EE3fooEv", i32 334, i32 0, i32 {{[0-9]+}}}
// TCHECK-DAG: = !{i32 0, i32 [[DEVID]], i32 [[FILEID]], !"_ZN2SCC1Ev", i32 271, i32 0, i32 {{[0-9]+}}}

#endif
