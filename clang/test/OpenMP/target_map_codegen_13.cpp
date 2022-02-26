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

  // CK14: define {{.*}}void @{{.+}}foo{{.+}}([[ST]]* {{[^,]+}}, i32 {{[^,]+}})
  // CK14-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 4, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], i64* [[SIZES:%[^,]+]], {{.+}}[[TYPES]]{{.+}}, i8** null, i8** null)
  // CK14-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK14-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK14-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]], i32 0, i32 0

  // CK14-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK14-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK14-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i32 0, i32 0
  // CK14-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
  // CK14-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK14-DAG: store [[ST]]* [[DECL:%.+]], [[ST]]** [[CBP0]]
  // CK14-DAG: store i32* [[A:%.+]], i32** [[CP0]]
  // CK14-DAG: store i64 %{{.+}}, i64* [[S0]]

  // CK14-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 1
  // CK14-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 1
  // CK14-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[ST]]**
  // CK14-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK14-DAG: store [[ST]]* [[DECL]], [[ST]]** [[CBP1]]
  // CK14-DAG: store i32* [[A]], i32** [[CP1]]

  // CK14-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 2
  // CK14-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 2
  // CK14-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [[ST]]**
  // CK14-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to double**
  // CK14-DAG: store [[ST]]* [[DECL]], [[ST]]** [[CBP2]]
  // CK14-DAG: store double* %{{.+}}, double** [[CP2]]

  // CK14-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 3
  // CK14-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 3
  // CK14-DAG: [[CBP3:%.+]] = bitcast i8** [[BP3]] to i[[sz:64|32]]*
  // CK14-DAG: [[CP3:%.+]] = bitcast i8** [[P3]] to i[[sz]]*
  // CK14-DAG: store i[[sz]] [[VAL:%.+]], i[[sz]]* [[CBP3]]
  // CK14-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP3]]
  // CK14-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK14-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to i32*
  // CK14-64-DAG: store i32 {{.+}}, i32* [[CADDR]],

  // CK14: call void [[KERNEL:@.+]]([[ST]]* [[DECL]], i[[sz]] {{.+}})
  sss.foo(123);
}

// CK14: define internal void [[KERNEL]]([[ST]]* noundef [[THIS:%.+]], i[[sz]] noundef [[ARG:%.+]])
// CK14: [[ADDR0:%.+]] = alloca [[ST]]*,
// CK14: [[ADDR1:%.+]] = alloca i[[sz]],
// CK14: store [[ST]]* [[THIS]], [[ST]]** [[ADDR0]],
// CK14: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR1]],
// CK14: [[REF0:%.+]] = load [[ST]]*, [[ST]]** [[ADDR0]],
// CK14-64: [[CADDR1:%.+]] = bitcast i[[sz]]* [[ADDR1]] to i32*
// CK14-64: {{.+}} = load i32,  i32* [[CADDR1]],
// CK14-32: {{.+}} = load i32, i32* [[ADDR1]],
// CK14: {{.+}} = getelementptr inbounds [[ST]], [[ST]]* [[REF0]], i32 0, i32 0

#endif // CK14
#endif
