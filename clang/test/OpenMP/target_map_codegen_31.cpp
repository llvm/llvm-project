// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DUSE -DCK31A -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK31A,CK31A-64,CK31A-USE
// RUN: %clang_cc1 -DUSE -DCK31A -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31A,CK31A-64,CK31A-USE
// RUN: %clang_cc1 -DUSE -DCK31A -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31A,CK31A-32,CK31A-USE
// RUN: %clang_cc1 -DUSE -DCK31A -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31A,CK31A-32,CK31A-USE

// RUN: %clang_cc1 -DUSE -DCK31A -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DUSE -DCK31A -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DUSE -DCK31A -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DUSE -DCK31A -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s

// RUN: %clang_cc1 -DCK31A -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK31A,CK31A-64,CK31A-NOUSE
// RUN: %clang_cc1 -DCK31A -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31A,CK31A-64,CK31A-NOUSE
// RUN: %clang_cc1 -DCK31A -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31A,CK31A-32,CK31A-NOUSE
// RUN: %clang_cc1 -DCK31A -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31A,CK31A-32,CK31A-NOUSE

// RUN: %clang_cc1 -DCK31A -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK31A -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK31A -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK31A -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// SIMD-ONLY18-NOT: {{__kmpc|__tgt}}
#ifdef CK31A

// CK31A: [[ST:%.+]] = type { i32, i32 }
struct ST {
  int i;
  int j;
};

// CK31A-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
//
// CK31A: [[SIZE00:@.+]] = private {{.*}}constant [7 x i64] [i64 0, i64 4, i64 4, i64 4, i64 0, i64 4, i64 4]
// PRESENT=0x1000 | TARGET_PARAM=0x20 = 0x1020
// CK31A-USE: [[MTYPE00:@.+]] = private {{.*}}constant [7 x i64] [i64 [[#0x1020]],
// CK31A-NOUSE: [[MTYPE00:@.+]] = private {{.*}}constant [7 x i64] [i64 [[#0x1000]],
//
// MEMBER_OF_1=0x1000000000000 | PRESENT=0x1000 | FROM=0x2 | TO=0x1 = 0x1000000001003
// MEMBER_OF_1=0x1000000000000 | FROM=0x2 | TO=0x1 = 0x1000000000003
// PRESENT=0x1000 | TARGET_PARAM=0x20 | FROM=0x2 | TO=0x1 = 0x1023
// CK31A-USE-SAME: {{^}} i64 [[#0x1000000001003]], i64 [[#0x1000000000003]], i64 [[#0x1023]],
// CK31A-NOUSE-SAME: {{^}} i64 [[#0x1000000001003]], i64 [[#0x1000000000003]], i64 [[#0x1003]],
//
// PRESENT=0x1000 | TARGET_PARAM=0x20 = 0x1020
// MEMBER_OF_5=0x5000000000000 | PRESENT=0x1000 | FROM=0x2 | TO=0x1 = 0x5000000001003
// MEMBER_OF_5=0x5000000000000 | FROM=0x2 | TO=0x1 = 0x5000000000003
// CK31A-USE-SAME: {{^}} i64 [[#0x1020]], i64 [[#0x5000000001003]], i64 [[#0x5000000000003]]]
// CK31A-NOUSE-SAME: {{^}} i64 [[#0x1000]], i64 [[#0x5000000001003]], i64 [[#0x5000000000003]]]

// CK31A-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK31A: [[SIZE01:@.+]] = private {{.*}}constant [1 x i[[Z:64|32]]] [i[[Z:64|32]] 4]
//
// PRESENT=0x1000 | CLOSE=0x400 | TARGET_PARAM=0x20 | ALWAYS=0x4 | FROM=0x2 | TO=0x1 = 0x1427
// CK31A-USE: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 [[#0x1427]]]
// CK31A-NOUSE: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 [[#0x1407]]]

// CK31A-LABEL: explicit_maps_single{{.*}}(
void explicit_maps_single (int ii){
  // CK31A: alloca i32

  // Map of a scalar.
  // CK31A: [[A:%.+]] = alloca i32
  int a = ii;

  // CK31A: [[ST1:%.+]] = alloca [[ST]]
  // CK31A: [[ST2:%.+]] = alloca [[ST]]
  struct ST st1;
  struct ST st2;

// Make sure the struct picks up present even if another element of the struct
// doesn't have present.
// Region 00
// CK31A: [[ST1_J:%.+]] = getelementptr inbounds nuw [[ST]], ptr [[ST1]], i{{.+}} 0, i{{.+}} 1
// CK31A: [[ST1_I:%.+]] = getelementptr inbounds nuw [[ST]], ptr [[ST1]], i{{.+}} 0, i{{.+}} 0
// CK31A: [[ST2_I:%.+]] = getelementptr inbounds nuw [[ST]], ptr [[ST2]], i{{.+}} 0, i{{.+}} 0
// CK31A: [[ST2_J:%.+]] = getelementptr inbounds nuw [[ST]], ptr [[ST2]], i{{.+}} 0, i{{.+}} 1
// CK31A-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK31A-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK31A-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK31A-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK31A-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK31A-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CK31A-DAG: store ptr [[SIZES:%.+]], ptr [[SARG]]
// CK31A-DAG: [[SIZES]] = getelementptr inbounds [7 x i64], ptr [[S:%.+]], i{{.+}} 0, i{{.+}} 0
// CK31A-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK31A-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// st1
// CK31A-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK31A-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK31A-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK31A-DAG: store ptr [[ST1]], ptr [[BP0]]
// CK31A-DAG: store ptr [[ST1_I]], ptr [[P0]]
// CK31A-DAG: store i64 %{{.+}}, ptr [[S0]]

// st1.j
// CK31A-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK31A-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK31A-DAG: store ptr [[ST1]], ptr [[BP1]]
// CK31A-DAG: store ptr [[ST1_J]], ptr [[P1]]

// st1.i
// CK31A-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK31A-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK31A-DAG: store ptr [[ST1]], ptr [[BP2]]
// CK31A-DAG: store ptr [[ST1_I]], ptr [[P2]]

// a
// CK31A-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
// CK31A-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
// CK31A-DAG: store ptr [[A]], ptr [[BP3]]
// CK31A-DAG: store ptr [[A]], ptr [[P3]]

// st2
// CK31A-DAG: [[BP4:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 4
// CK31A-DAG: [[P4:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 4
// CK31A-DAG: [[S4:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 4
// CK31A-DAG: store ptr [[ST2]], ptr [[BP4]]
// CK31A-DAG: store ptr [[ST2_I]], ptr [[P4]]
// CK31A-DAG: store i64 %{{.+}}, ptr [[S4]]

// st2.i
// CK31A-DAG: [[BP5:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 5
// CK31A-DAG: [[P5:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 5
// CK31A-DAG: store ptr [[ST2]], ptr [[BP5]]
// CK31A-DAG: store ptr [[ST2_I]], ptr [[P5]]

// st2.j
// CK31A-DAG: [[BP6:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 6
// CK31A-DAG: [[P6:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 6
// CK31A-DAG: store ptr [[ST2]], ptr [[BP6]]
// CK31A-DAG: store ptr [[ST2_J]], ptr [[P6]]

// CK31A-USE: call void [[CALL00:@.+]](ptr [[ST1]], ptr [[A]], ptr [[ST2]])
// CK31A-NOUSE: call void [[CALL00:@.+]]()
#pragma omp target map(tofrom                                     \
                       : st1.i) map(present, tofrom               \
                                    : a, st1.j, st2.i) map(tofrom \
                                                           : st2.j)
  {
#ifdef USE
    st1.i++;
    a++;
    st1.j++;
    st2.i++;
    st2.j++;
#endif
  }

// Always Close Present.
// Region 01
// CK31A-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK31A-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK31A-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK31A-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK31A-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK31A-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK31A-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK31A-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK31A-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK31A-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK31A-DAG: store ptr [[VAR0]], ptr [[P0]]

// CK31A-USE: call void [[CALL01:@.+]](ptr {{[^,]+}})
// CK31A-NOUSE: call void [[CALL01:@.+]]()
#pragma omp target map(always close present tofrom \
                       : a)
  {
#ifdef USE
    a++;
#endif
  }
}
// CK31A: define {{.+}}[[CALL00]]
// CK31A: define {{.+}}[[CALL01]]

#endif // CK31A
#endif
