// RUN: %clang_cc1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s

// expected-no-diagnostics
#ifndef HEADER
#define HEADER

// CHECK: @.[[KERNEL00:__omp_offloading_.*foov_l[0-9]+]].region_id = weak constant i8 0
// CHECK: [[SIZE00:@.+]] = private unnamed_addr constant [2 x i64] [i64 {{8|4}}, i64 8]
// CHECK: [[MYTYPE00:@.+]] = private unnamed_addr constant [2 x i64] [i64 35, i64 19]

// CHECK: @.[[KERNEL01:__omp_offloading_.*foov_l[0-9]+]].region_id = weak constant i8 0
// CHECK: [[SIZE01:@.+]] = private unnamed_addr constant [2 x i64] [i64 {{8|4}}, i64 4]
// CHECK: [[MYTYPE01:@.+]] = private unnamed_addr constant [2 x i64] [i64 35, i64 19]

// CHECK: @.[[KERNEL02:__omp_offloading_.*foov_l[0-9]+]].region_id = weak constant i8 0
// CHECK: [[SIZE02:@.+]] = private unnamed_addr constant [2 x i64] [i64 {{8|4}}, i64 4]
// CHECK: [[MYTYPE02:@.+]] = private unnamed_addr constant [2 x i64] [i64 35, i64 19]

// CHECK: [[SIZE03:@.+]] = private unnamed_addr constant [1 x i64] [i64 4]
// CHECK: [[MYTYPE03:@.+]] = private unnamed_addr constant [1 x i64] [i64 51]

extern void *malloc (int __size) throw () __attribute__ ((__malloc__));

// CHECK-LABEL: define{{.*}}@_Z3foov{{.*}}(
void foo() {
  int *ptr = (int *) malloc(3 * sizeof(int));

// Region 00
//   &ptr, &ptr, sizeof(ptr), TO | FROM | PARAM
//   &ptr, &ptr[0], 2 * sizeof(ptr[0]), TO | FROM | PTR_AND_OBJ
//
// CHECK-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.[[KERNEL00]].region_id, ptr [[ARGS:%.+]])
// CHECK-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CHECK-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CHECK-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CHECK-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CHECK-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CHECK-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
//
// CHECK-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CHECK-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CHECK-DAG: store ptr [[VAR0:%ptr]], ptr [[BP0]]
// CHECK-DAG: store ptr [[VAR0]], ptr [[P0]]
//
// CHECK-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CHECK-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CHECK-DAG: store ptr [[VAR0:%ptr]], ptr [[BP1]]
// CHECK-DAG: store ptr [[RVAR00:%.+]], ptr [[P1]]
//
// CHECK-DAG: [[RVAR00]] = getelementptr inbounds {{.*}}[[RVAR0:%.+]], i{{.+}} 0
// CHECK-DAG: [[RVAR0]] = load ptr, ptr [[VAR0]]
//
// CHECK-DAG: call void @[[KERNEL00]](ptr [[VAR0]])
  #pragma omp target map(ptr, ptr[0:2])
  {
    ptr[1] = 6;
  }

// Region 01
//   &ptr, &ptr, sizeof(ptr), TO | FROM | PARAM
//   &ptr, &ptr[2], sizeof(ptr[2]), TO | FROM | PTR_AND_OBJ
//
// CHECK-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.[[KERNEL01]].region_id, ptr [[ARGS:%.+]])
// CHECK-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CHECK-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CHECK-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CHECK-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CHECK-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CHECK-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
//
// CHECK-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CHECK-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CHECK-DAG: store ptr [[VAR0:%ptr]], ptr [[BP0]]
// CHECK-DAG: store ptr [[VAR0]], ptr [[P0]]
//
// CHECK-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CHECK-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CHECK-DAG: store ptr [[VAR0:%ptr]], ptr [[BP1]]
// CHECK-DAG: store ptr [[RVAR02:%.+]], ptr [[P1]]
//
// CHECK-DAG: [[RVAR02]] = getelementptr inbounds {{.*}}[[RVAR0:%.+]], i{{.+}} 2
// CHECK-DAG: [[RVAR0]] = load ptr, ptr [[VAR0]]
//
// CHECK-DAG: call void @[[KERNEL01]](ptr [[VAR0]])
  #pragma omp target map(ptr, ptr[2])
  {
    ptr[2] = 8;
  }

// Region 02
//   &ptr, &ptr, sizeof(ptr), TO | FROM | PARAM
//   &ptr, &ptr[2], sizeof(ptr[2]), TO | FROM | PTR_AND_OBJ
//
// CHECK-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.[[KERNEL02]].region_id, ptr [[ARGS:%.+]])
// CHECK-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CHECK-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CHECK-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CHECK-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CHECK-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CHECK-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
//
// CHECK-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CHECK-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CHECK-DAG: store ptr [[VAR0:%ptr]], ptr [[BP0]]
// CHECK-DAG: store ptr [[VAR0]], ptr [[P0]]
//
// CHECK-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CHECK-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CHECK-DAG: store ptr [[VAR0:%ptr]], ptr [[BP1]]
// CHECK-DAG: store ptr [[RVAR02:%.+]], ptr [[P1]]
//
// CHECK-DAG: [[RVAR02]] = getelementptr inbounds {{.*}}[[RVAR0:%.+]], i{{.+}} 2
// CHECK-DAG: [[RVAR0]] = load ptr, ptr [[VAR0]]
//
// CHECK-DAG: call void @[[KERNEL02]](ptr [[VAR0]])
  #pragma omp target map(ptr[2], ptr)
  {
    ptr[2] = 9;
  }

// Region 03
//   &ptr, &ptr[2], sizeof(ptr[2]), TO | FROM | PARAM | PTR_AND_OBJ
//   FIXME: PARAM seems to be redundant here.
//
// CHECK-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[BPGEP:.+]], ptr [[PGEP:.+]], ptr [[SIZE03]], ptr [[MYTYPE03]], ptr null, ptr null)
// CHECK-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CHECK-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
//
// CHECK-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CHECK-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CHECK-DAG: store ptr [[VAR0:%ptr]], ptr [[BP0]]
// CHECK-DAG: store ptr [[RVAR02:%.+]], ptr [[P0]]
//
// CHECK-DAG: [[RVAR02]] = getelementptr inbounds {{.*}}[[RVAR0:%.+]], i{{.+}} 2
// CHECK-DAG: [[RVAR0]] = load ptr, ptr [[VAR0]]
  #pragma omp target data map(ptr, ptr[2])
  {
    ptr[2] = 10;
  }
}

// CHECK-LABEL: define internal void
// CHECK-SAME: @[[KERNEL00]](ptr {{[^,]*}}[[PTR:%[^,]+]])
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[PTR_ADDR:%.*]] = alloca ptr
// CHECK-NEXT:    store ptr [[PTR]], ptr [[PTR_ADDR]]
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[PTR_ADDR]]
// CHECK-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[TMP0]]
// CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, ptr [[TMP1]], i{{.*}} 1
// CHECK-NEXT:    store i32 6, ptr [[ARRAYIDX]]
// CHECK-NEXT:    ret void

// CHECK-LABEL: define internal void
// CHECK-SAME: @[[KERNEL01]](ptr {{[^,]*}}[[PTR:%[^,]+]])
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[PTR_ADDR:%.*]] = alloca ptr
// CHECK-NEXT:    store ptr [[PTR]], ptr [[PTR_ADDR]]
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[PTR_ADDR]]
// CHECK-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[TMP0]]
// CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, ptr [[TMP1]], i{{.*}} 2
// CHECK-NEXT:    store i32 8, ptr [[ARRAYIDX]]
// CHECK-NEXT:    ret void

// CHECK-LABEL: define internal void
// CHECK-SAME: @[[KERNEL02]](ptr {{[^,]*}}[[PTR:%[^,]+]])
// CHECK-NEXT:  entry:
// CHECK-NEXT:    [[PTR_ADDR:%.*]] = alloca ptr
// CHECK-NEXT:    store ptr [[PTR]], ptr [[PTR_ADDR]]
// CHECK-NEXT:    [[TMP0:%.*]] = load ptr, ptr [[PTR_ADDR]]
// CHECK-NEXT:    [[TMP1:%.*]] = load ptr, ptr [[TMP0]]
// CHECK-NEXT:    [[ARRAYIDX:%.*]] = getelementptr inbounds i32, ptr [[TMP1]], i{{.*}} 2
// CHECK-NEXT:    store i32 9, ptr [[ARRAYIDX]]
// CHECK-NEXT:    ret void
#endif
