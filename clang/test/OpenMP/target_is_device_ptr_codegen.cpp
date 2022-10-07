// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32

// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK1

double *g;

// CK1: @g ={{.*}} global ptr
// CK1: [[SIZES00:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} {{8|4}}]
// CK1: [[TYPES00:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK1: [[SIZES01:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] {{8|4}}]
// CK1: [[TYPES01:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK1: [[SIZES02:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] {{8|4}}]
// CK1: [[TYPES02:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK1: [[SIZES03:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] {{8|4}}]
// CK1: [[TYPES03:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK1: [[SIZES04:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] {{8|4}}]
// CK1: [[TYPES04:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK1: [[SIZES05:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] {{8|4}}]
// CK1: [[TYPES05:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK1: [[SIZES06:@.+]] = {{.+}}constant [2 x i[[sz]]] [i[[sz]] {{8|4}}, i[[sz]] {{8|4}}]
// CK1: [[TYPES06:@.+]] = {{.+}}constant [2 x i64] [i64 288, i64 288]

// CK1-LABEL: @_Z3foo{{.*}}(
template<typename T>
void foo(float *&lr, T *&tr) {
  float *l;
  T *t;

// CK1-DAG: [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK1-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK1-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK1-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK1-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK1-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK1-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK1-DAG: store ptr [[VAL:%.+]], ptr [[BP1]]
// CK1-DAG: store ptr [[VAL]], ptr [[P1]]
// CK1-DAG: [[VAL]] = load ptr, ptr [[ADDR:@g]],

// CK1: call void [[KERNEL:@.+]](ptr [[VAL]])
#pragma omp target is_device_ptr(g)
  {
    ++g;
  }

// CK1-DAG: [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK1-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK1-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK1-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK1-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK1-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK1-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK1-DAG: store ptr [[VAL:%.+]], ptr [[BP1]]
// CK1-DAG: store ptr [[VAL]], ptr [[P1]]
// CK1-DAG: [[VAL]] = load ptr, ptr [[ADDR:%.+]],

// CK1: call void [[KERNEL:@.+]](ptr [[VAL]])
#pragma omp target is_device_ptr(l)
  {
    ++l;
  }

// CK1-DAG: [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK1-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK1-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK1-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK1-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK1-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK1-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK1-DAG: store ptr [[VAL:%.+]], ptr [[BP1]]
// CK1-DAG: store ptr [[VAL]], ptr [[P1]]
// CK1-DAG: [[VAL]] = load ptr, ptr [[ADDR:%.+]],

// CK1: call void [[KERNEL:@.+]](ptr [[VAL]])
#pragma omp target is_device_ptr(t)
  {
    ++t;
  }

// CK1-DAG: [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK1-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK1-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK1-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK1-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK1-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK1-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK1-DAG: store ptr [[VAL:%.+]], ptr [[BP1]]
// CK1-DAG: store ptr [[VAL]], ptr [[P1]]
// CK1-DAG: [[VAL]] = load ptr, ptr [[ADDR:%.+]],
// CK1-DAG: [[ADDR]] = load ptr, ptr [[ADDR2:%.+]],

// CK1: call void [[KERNEL:@.+]](ptr [[VAL]])
#pragma omp target is_device_ptr(lr)
  {
    ++lr;
  }

// CK1-DAG: [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK1-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK1-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK1-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK1-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK1-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK1-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK1-DAG: store ptr [[VAL:%.+]], ptr [[BP1]]
// CK1-DAG: store ptr [[VAL]], ptr [[P1]]
// CK1-DAG: [[VAL]] = load ptr, ptr [[ADDR:%.+]],
// CK1-DAG: [[ADDR]] = load ptr, ptr [[ADDR2:%.+]],

// CK1: call void [[KERNEL:@.+]](ptr [[VAL]])
#pragma omp target is_device_ptr(tr)
  {
    ++tr;
  }

// CK1-DAG: [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK1-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK1-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK1-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK1-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK1-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK1-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK1-DAG: store ptr [[VAL:%.+]], ptr [[BP1]]
// CK1-DAG: store ptr [[VAL]], ptr [[P1]]
// CK1-DAG: [[VAL]] = load ptr, ptr [[ADDR:%.+]],
// CK1-DAG: [[ADDR]] = load ptr, ptr [[ADDR2:%.+]],

// CK1: call void [[KERNEL:@.+]](ptr [[VAL]])
#pragma omp target is_device_ptr(tr, lr)
  {
    ++tr;
  }

// CK1-DAG: [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK1-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK1-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK1-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK1-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK1-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
// CK1-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
// CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
// CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
// CK1-DAG: store ptr [[VAL:%.+]], ptr [[BP1]]
// CK1-DAG: store ptr [[VAL]], ptr [[P1]]
// CK1-DAG: [[VAL]] = load ptr, ptr [[ADDR:%.+]],
// CK1-DAG: [[ADDR]] = load ptr, ptr [[ADDR2:%.+]],

// CK1-DAG: [[_BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 1
// CK1-DAG: [[_P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 1
// CK1-DAG: store ptr [[_VAL:%.+]], ptr [[_BP1]]
// CK1-DAG: store ptr [[_VAL]], ptr [[_P1]]
// CK1-DAG: [[_VAL]] = load ptr, ptr [[_ADDR:%.+]],
// CK1-DAG: [[_ADDR]] = load ptr, ptr [[_ADDR2:%.+]],

// CK1: call void [[KERNEL:@.+]](ptr [[VAL]], ptr [[_VAL]])
#pragma omp target is_device_ptr(tr, lr)
  {
    ++tr,++lr;
  }
}

void bar(float *&a, int *&b) {
  foo<int>(a,b);
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-32
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-32

// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
#ifdef CK2

// CK2: [[ST:%.+]] = type { ptr, ptr }

template <typename T>
struct ST {
  T *a;
  double *&b;
  ST(double *&b) : a(0), b(b) {}

  // CK2-LABEL: @{{.*}}foo{{.*}}
  void foo(double *&arg) {
    int *la = 0;

// CK2-DAG: [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK2-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK2-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK2-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK2-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK2-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK2-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK2-DAG: [[A:%.*]] = getelementptr inbounds [[STRUCT_ST:%.*]], ptr [[THIS1:%.+]], i32 0, i32 0
// CK2-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: store ptr [[THIS1]], ptr [[BP0]]
// CK2-DAG: store ptr [[A]], ptr [[P0]]
#pragma omp target is_device_ptr(a)
    {
      a++;
    }

// CK2-DAG: [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK2-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK2-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK2-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK2-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK2-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CK2-DAG: store ptr [[SIZE:%.+]], ptr [[SARG]]
// CK2-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK2-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK2-DAG: [[S:%[^,]+]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CK2-DAG: [[SIZE:%[^,]+]] = getelementptr inbounds [2 x i64], ptr %.offload_sizes, i32 0, i32 0
// CK2-DAG: store i64 [[S]], ptr [[SIZE]]
// CK2-DAG: [[B:%.*]] = getelementptr inbounds [[STRUCT_ST]], ptr [[THIS1]], i32 0, i32 1
// CK2-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG:  store ptr [[THIS1]], ptr [[BP0]]
// CK2-DAG: store ptr [[B]], ptr [[P0]]
#pragma omp target is_device_ptr(b)
    {
      b++;
    }

// CK2-DAG: [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK2-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK2-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK2-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK2-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK2-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
// CK2-DAG: store ptr [[SIZE:%.+]], ptr [[SARG]]
// CK2-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK2-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK2-DAG: [[A8:%.*]] = getelementptr inbounds [[STRUCT_ST]], ptr [[THIS1]], i32 0, i32 0
// CK2-DAG: [[B9:%.*]] = getelementptr inbounds [[STRUCT_ST]], ptr [[THIS1]], i32 0, i32 1
// CK2-DAG: [[S:%[^,]+]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
// CK2-DAG: store i64 [[S]], ptr [[SIZE:%.+]]

// CK2-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG:  store ptr [[THIS1]], ptr [[BP0]]
// CK2-DAG: store ptr [[A8]], ptr [[TMP64:%.+]]
#pragma omp target is_device_ptr(a, b)
    {
      a++;
      b++;
    }
  }
};

void bar(double *arg){
  ST<double> A(arg);
  A.foo(arg);
  ++arg;
}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-32
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-32

// RUN: %clang_cc1 -DCK3 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK3 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
#ifdef CK3

// CK3-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[SZ:64|32]]] [i{{64|32}} {{8|4}}]
// OMP_MAP_TARGET_PARAM = 0x20 | OMP_MAP_TO = 0x1 = 0x21
// CK3-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x21]]]
void bar() {
  __attribute__((aligned(64))) double *ptr;
  // CK3-DAG: [[RET:%.+]] = call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 [[DEVICE:.+]], i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
  // CK3-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
  // CK3-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
  // CK3-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
  // CK3-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
  // CK3-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK3-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK3-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK3-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK3-DAG: store ptr [[PTR:%.+]], ptr [[BP1]]
  // CK3-DAG: store ptr [[PTR]], ptr [[P1]]

  // CK3: call void [[KERNEL:@.+]](ptr [[PTR]])
#pragma omp target is_device_ptr(ptr)
  *ptr = 0;
}
#endif
#endif
