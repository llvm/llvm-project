// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK15 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK15 --check-prefix CK15-64
// RUN: %clang_cc1 -DCK15 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK15 --check-prefix CK15-64
// RUN: %clang_cc1 -DCK15 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK15 --check-prefix CK15-32
// RUN: %clang_cc1 -DCK15 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK15 --check-prefix CK15-32

// RUN: %clang_cc1 -DCK15 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK15 --check-prefix CK15-64
// RUN: %clang_cc1 -DCK15 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK15 --check-prefix CK15-64
// RUN: %clang_cc1 -DCK15 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK15 --check-prefix CK15-32
// RUN: %clang_cc1 -DCK15 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK15 --check-prefix CK15-32

// RUN: %clang_cc1 -DCK15 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK15 --check-prefix CK15-64
// RUN: %clang_cc1 -DCK15 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK15 --check-prefix CK15-64
// RUN: %clang_cc1 -DCK15 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK15 --check-prefix CK15-32
// RUN: %clang_cc1 -DCK15 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK15 --check-prefix CK15-32

// RUN: %clang_cc1 -DCK15 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY14 %s
// RUN: %clang_cc1 -DCK15 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY14 %s
// RUN: %clang_cc1 -DCK15 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY14 %s
// RUN: %clang_cc1 -DCK15 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY14 %s
// SIMD-ONLY14-NOT: {{__kmpc|__tgt}}
#ifdef CK15

// CK15: [[ST:%.+]] = type { i32, double }
// CK15: [[SIZES:@.+]] = {{.+}}constant [4 x i64] [i64 0, i64 4, i64 8, i64 4]
// Map types:
// - OMP_MAP_TARGET_PARAM = 32
// - OMP_MAP_TO + OMP_MAP_FROM | OMP_MAP_IMPLICIT | OMP_MAP_MEMBER_OF = 281474976711171
// - OMP_MAP_PRIVATE_VAL + OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK15: [[TYPES:@.+]] = {{.+}}constant [4 x i64] [i64 32, i64 281474976711171, i64 281474976711171, i64 800]

// CK15: [[SIZES2:@.+]] = {{.+}}constant [4 x i64] [i64 0, i64 4, i64 8, i64 4]
// Map types:
// - OMP_MAP_TARGET_PARAM = 32
// - OMP_MAP_TO + OMP_MAP_FROM | OMP_MAP_IMPLICIT | OMP_MAP_MEMBER_OF = 281474976711171
// - OMP_MAP_PRIVATE_VAL + OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK15: [[TYPES2:@.+]] = {{.+}}constant [4 x i64] [i64 32, i64 281474976711171, i64 281474976711171, i64 800]

template<int x>
class SSST {
public:
  int a;
  double b;

  void foo(int c) {
    #pragma omp target
    {
      a += c + x;
      b += (double)(c + x);
    }
  }
  template<int y>
  void bar(int c) {
    #pragma omp target
    {
      a += c + x + y;
      b += (double)(c + x + y);
    }
  }

  SSST(int a, double b) : a(a), b(b) {}
};

// CK15-LABEL: implicit_maps_templated_class{{.*}}(
void implicit_maps_templated_class (int a){
  SSST<123> ssst(a, (double)a);

  // CK15: define {{.*}}void @{{.+}}foo{{.+}}(ptr {{[^,]+}}, i32 {{[^,]+}})
  // CK15-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
  // CK15-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
  // CK15-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
  // CK15-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
  // CK15-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
  // CK15-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
  // CK15-DAG: store ptr [[SIZES:%.+]], ptr [[SARG]]
  // CK15-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK15-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK15-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]], i32 0, i32 0

  // CK15-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK15-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK15-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i32 0, i32 0
  // CK15-DAG: store ptr [[DECL:%.+]], ptr [[BP0]]
  // CK15-DAG: store ptr [[A:%.+]], ptr [[P0]]
  // CK15-DAG: store i64 %{{.+}}, ptr [[S0]]

  // CK15-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 1
  // CK15-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 1
  // CK15-DAG: store ptr [[DECL]], ptr [[BP1]]
  // CK15-DAG: store ptr [[A]], ptr [[P1]]

  // CK15-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 2
  // CK15-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 2
  // CK15-DAG: store ptr [[DECL]], ptr [[BP2]]
  // CK15-DAG: store ptr %{{.+}}, ptr [[P2]]

  // CK15-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 3
  // CK15-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 3
  // CK15-DAG: store i[[sz:64|32]] [[VAL:%.+]], ptr [[BP3]]
  // CK15-DAG: store i[[sz]] [[VAL]], ptr [[P3]]
  // CK15-DAG: [[VAL]] = load i[[sz]], ptr [[ADDR:%.+]],
  // CK15-64-DAG: store i32 {{.+}}, ptr [[ADDR]],

  // CK15: call void [[KERNEL:@.+]](ptr [[DECL]], i[[sz]] {{.+}})
  ssst.foo(456);

  // CK15: define {{.*}}void @{{.+}}bar{{.+}}(ptr {{[^,]+}}, i32 {{[^,]+}})
  // CK15-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
  // CK15-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
  // CK15-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
  // CK15-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
  // CK15-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
  // CK15-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
  // CK15-DAG: store ptr [[SIZES:%.+]], ptr [[SARG]]
  // CK15-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK15-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK15-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]], i32 0, i32 0

  // CK15-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK15-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK15-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i32 0, i32 0
  // CK15-DAG: store ptr [[DECL:%.+]], ptr [[BP0]]
  // CK15-DAG: store ptr [[A:%.+]], ptr [[P0]]
  // CK15-DAG: store i64 %{{.+}}, ptr [[S0]]

  // CK15-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 1
  // CK15-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 1
  // CK15-DAG: store ptr [[DECL]], ptr [[BP1]]
  // CK15-DAG: store ptr [[A]], ptr [[P1]]

  // CK15-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 2
  // CK15-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 2
  // CK15-DAG: store ptr [[DECL]], ptr [[BP2]]
  // CK15-DAG: store ptr %{{.+}}, ptr [[P2]]

  // CK15-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 3
  // CK15-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 3
  // CK15-DAG: store i[[sz]] [[VAL:%.+]], ptr [[BP3]]
  // CK15-DAG: store i[[sz]] [[VAL]], ptr [[P3]]
  // CK15-DAG: [[VAL]] = load i[[sz]], ptr [[ADDR:%.+]],
  // CK15-64-DAG: store i32 {{.+}}, ptr [[ADDR]],

  // CK15: call void [[KERNEL2:@.+]](ptr [[DECL]], i[[sz]] {{.+}})
  ssst.bar<210>(789);
}

// CK15: define internal void [[KERNEL]](ptr noundef [[THIS:%.+]], i[[sz]] noundef [[ARG:%.+]])
// CK15: [[ADDR0:%.+]] = alloca ptr,
// CK15: [[ADDR1:%.+]] = alloca i[[sz]],
// CK15: store ptr [[THIS]], ptr [[ADDR0]],
// CK15: store i[[sz]] [[ARG]], ptr [[ADDR1]],
// CK15: [[REF0:%.+]] = load ptr, ptr [[ADDR0]],
// CK15-64: {{.+}} = load i32,  ptr [[ADDR1]],
// CK15-32: {{.+}} = load i32, ptr [[ADDR1]],
// CK15: {{.+}} = getelementptr inbounds nuw [[ST]], ptr [[REF0]], i32 0, i32 0

// CK15: define internal void [[KERNEL2]](ptr noundef [[THIS:%.+]], i[[sz]] noundef [[ARG:%.+]])
// CK15: [[ADDR0:%.+]] = alloca ptr,
// CK15: [[ADDR1:%.+]] = alloca i[[sz]],
// CK15: store ptr [[THIS]], ptr [[ADDR0]],
// CK15: store i[[sz]] [[ARG]], ptr [[ADDR1]],
// CK15: [[REF0:%.+]] = load ptr, ptr [[ADDR0]],
// CK15-64: {{.+}} = load i32,  ptr [[ADDR1]],
// CK15-32: {{.+}} = load i32, ptr [[ADDR1]],
// CK15: {{.+}} = getelementptr inbounds nuw [[ST]], ptr [[REF0]], i32 0, i32 0

#endif // CK15
#endif
