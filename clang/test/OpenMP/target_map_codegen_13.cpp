// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK14 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK14 --check-prefix CK14-64
// RUN: %clang_cc1 -DCK14 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK14 --check-prefix CK14-64
// RUN: %clang_cc1 -DCK14 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK14 --check-prefix CK14-32
// RUN: %clang_cc1 -DCK14 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK14 --check-prefix CK14-32

// RUN: %clang_cc1 -DCK14 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK14 --check-prefix CK14-64
// RUN: %clang_cc1 -DCK14 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK14 --check-prefix CK14-64
// RUN: %clang_cc1 -DCK14 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK14 --check-prefix CK14-32
// RUN: %clang_cc1 -DCK14 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK14 --check-prefix CK14-32

// RUN: %clang_cc1 -DCK14 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK14 --check-prefix CK14-64
// RUN: %clang_cc1 -DCK14 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK14 --check-prefix CK14-64
// RUN: %clang_cc1 -DCK14 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK14 --check-prefix CK14-32
// RUN: %clang_cc1 -DCK14 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK14 --check-prefix CK14-32

// RUN: %clang_cc1 -DCK14 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY13 %s
// RUN: %clang_cc1 -DCK14 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY13 %s
// RUN: %clang_cc1 -DCK14 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY13 %s
// RUN: %clang_cc1 -DCK14 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY13 %s
// SIMD-ONLY13-NOT: {{__kmpc|__tgt}}
#ifdef CK14

// CK14-DAG: [[ST:%.+]] = type { i32, double }

// CK14-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0


// CK14-DAG: [[SIZES:@.+]] = {{.+}}constant [4 x i64] [i64 0, i64 4, i64 8, i64 4]
// Map types:
// - OMP_MAP_TARGET_PARAM = 32
// - OMP_MAP_TO + OMP_MAP_FROM | OMP_MAP_IMPLICIT | OMP_MAP_MEMBER_OF = 281474976711171
// - OMP_MAP_PRIVATE_VAL + OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK14-DAG: [[TYPES:@.+]] = {{.+}}constant [4 x i64] [i64 32, i64 281474976711171, i64 281474976711171, i64 800]

class SSS {
public:
  int a;
  double b;

  void foo(int c) {
    #pragma omp target
    {
      a += c;
      b += (double)c;
    }
  }

  SSS(int a, double b) : a(a), b(b) {}
};

// CK14-LABEL: implicit_maps_class{{.*}}(
void implicit_maps_class (int a){
  SSS sss(a, (double)a);

  // CK14: define {{.*}}void @{{.+}}foo{{.+}}(ptr {{[^,]+}}, i32 {{[^,]+}})
  // CK14-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
  // CK14-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
  // CK14-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
  // CK14-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
  // CK14-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
  // CK14-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
  // CK14-DAG: store ptr [[SIZES:%.+]], ptr [[SARG]]
  // CK14-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK14-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK14-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]], i32 0, i32 0

  // CK14-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK14-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK14-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i32 0, i32 0
  // CK14-DAG: store ptr [[DECL:%.+]], ptr [[BP0]]
  // CK14-DAG: store ptr [[A:%.+]], ptr [[P0]]
  // CK14-DAG: store i64 %{{.+}}, ptr [[S0]]

  // CK14-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 1
  // CK14-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 1
  // CK14-DAG: store ptr [[DECL]], ptr [[BP1]]
  // CK14-DAG: store ptr [[A]], ptr [[P1]]

  // CK14-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 2
  // CK14-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 2
  // CK14-DAG: store ptr [[DECL]], ptr [[BP2]]
  // CK14-DAG: store ptr %{{.+}}, ptr [[P2]]

  // CK14-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 3
  // CK14-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 3
  // CK14-DAG: store i[[sz:64|32]] [[VAL:%.+]], ptr [[BP3]]
  // CK14-DAG: store i[[sz]] [[VAL]], ptr [[P3]]
  // CK14-DAG: [[VAL]] = load i[[sz]], ptr [[ADDR:%.+]],
  // CK14-64-DAG: store i32 {{.+}}, ptr [[ADDR]],

  // CK14: call void [[KERNEL:@.+]](ptr [[DECL]], i[[sz]] {{.+}})
  sss.foo(123);
}

// CK14: define internal void [[KERNEL]](ptr noundef [[THIS:%.+]], i[[sz]] noundef [[ARG:%.+]])
// CK14: [[ADDR0:%.+]] = alloca ptr,
// CK14: [[ADDR1:%.+]] = alloca i[[sz]],
// CK14: store ptr [[THIS]], ptr [[ADDR0]],
// CK14: store i[[sz]] [[ARG]], ptr [[ADDR1]],
// CK14: [[REF0:%.+]] = load ptr, ptr [[ADDR0]],
// CK14-64: {{.+}} = load i32,  ptr [[ADDR1]],
// CK14-32: {{.+}} = load i32, ptr [[ADDR1]],
// CK14: {{.+}} = getelementptr inbounds nuw [[ST]], ptr [[REF0]], i32 0, i32 0

#endif // CK14
#endif
