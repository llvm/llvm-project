// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK25 -std=c++11 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK25 --check-prefix CK25-64
// RUN: %clang_cc1 -DCK25 -std=c++11 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK25 --check-prefix CK25-64
// RUN: %clang_cc1 -DCK25 -std=c++11 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK25 --check-prefix CK25-32
// RUN: %clang_cc1 -DCK25 -std=c++11 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK25 --check-prefix CK25-32

// RUN: %clang_cc1 -DCK25 -std=c++11 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK25 --check-prefix CK25-64
// RUN: %clang_cc1 -DCK25 -std=c++11 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -std=c++11 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK25 --check-prefix CK25-64
// RUN: %clang_cc1 -DCK25 -std=c++11 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK25 --check-prefix CK25-32
// RUN: %clang_cc1 -DCK25 -std=c++11 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -std=c++11 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK25 --check-prefix CK25-32

// RUN: %clang_cc1 -DCK25 -std=c++11 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK25 --check-prefix CK25-64
// RUN: %clang_cc1 -DCK25 -std=c++11 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -std=c++11 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK25 --check-prefix CK25-64
// RUN: %clang_cc1 -DCK25 -std=c++11 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK25 --check-prefix CK25-32
// RUN: %clang_cc1 -DCK25 -std=c++11 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -std=c++11 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK25 --check-prefix CK25-32

// RUN: %clang_cc1 -DCK25 -std=c++11 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY24 %s
// RUN: %clang_cc1 -DCK25 -std=c++11 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY24 %s
// RUN: %clang_cc1 -DCK25 -std=c++11 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY24 %s
// RUN: %clang_cc1 -DCK25 -std=c++11 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY24 %s
// SIMD-ONLY24-NOT: {{__kmpc|__tgt}}
#ifdef CK25
// CK25: [[ST:%.+]] = type { i32, float }
// CK25: [[CA00:%.+]] = type { ptr }
// CK25: [[CA01:%.+]] = type { ptr }

// CK25-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK25: [[SIZE00:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK25: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 33]

// CK25-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK25: [[SIZE01:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK25: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 33]

// CK25-LABEL: explicit_maps_with_inner_lambda{{.*}}(

template <int X, typename T>
struct CC {
  T A;
  float B;

  int foo(T arg) {
// Region 00
// CK25-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK25-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK25-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK25-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK25-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK25-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK25-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK25-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK25-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK25-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK25-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK25-DAG: [[SEC0]] = getelementptr {{.*}}ptr [[VAR0:%.+]], i{{.+}} 0, i{{.+}} 0

// CK25: call void [[CALL00:@.+]](ptr {{[^,]+}})
#pragma omp target map(to \
                       : A)
    {
      [&]() {
        A += 1;
      }();
    }

// Region 01
// CK25-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
// CK25-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
// CK25-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
// CK25-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
// CK25-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
// CK25-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK25-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK25-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK25-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK25-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK25-DAG: store ptr [[VAR0]], ptr [[P0]]

// CK25: call void [[CALL01:@.+]](ptr {{[^,]+}})
#pragma omp target map(to \
                       : arg)
    {
      [&]() {
        arg += 1;
      }();
    }

    return A+arg;
  }
};

int explicit_maps_with_inner_lambda(int a){
  CC<123,int> c;
  return c.foo(a);
}

// CK25: define {{.+}}[[CALL00]](ptr noundef [[VAL:%.+]])
// CK25: store ptr [[VAL]], ptr [[VALADDR:%[^,]+]],
// CK25: [[VAL1:%.+]] = load ptr, ptr [[VALADDR]],
// CK25: [[VALADDR1:%.+]] = getelementptr inbounds nuw [[CA00]], ptr [[CA:%[^,]+]], i32 0, i32 0
// CK25: store ptr [[VAL1]], ptr [[VALADDR1]],
// CK25: call void {{.*}}[[LAMBDA:@.+]]{{.*}}(ptr {{[^,]*}} [[CA]])

// CK25: define {{.+}}[[LAMBDA]]

// CK25: define {{.+}}[[CALL01]](ptr {{.*}}[[VAL:%.+]])
// CK25: store ptr [[VAL]], ptr [[VALADDR:%[^,]+]],
// CK25: [[VAL1:%.+]] = load ptr, ptr [[VALADDR]],
// CK25: [[VALADDR1:%.+]] = getelementptr inbounds nuw [[CA01]], ptr [[CA:%[^,]+]], i32 0, i32 0
// CK25: store ptr [[VAL1]], ptr [[VALADDR1]],
// CK25: call void {{.*}}[[LAMBDA2:@.+]]{{.*}}(ptr {{[^,]*}} [[CA]])

// CK25: define {{.+}}[[LAMBDA2]]
#endif // CK25
#endif
