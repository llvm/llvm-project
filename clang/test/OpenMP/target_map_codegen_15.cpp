// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK16 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK16 --check-prefix CK16-64
// RUN: %clang_cc1 -DCK16 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK16 --check-prefix CK16-64
// RUN: %clang_cc1 -DCK16 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK16 --check-prefix CK16-32
// RUN: %clang_cc1 -DCK16 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK16 --check-prefix CK16-32

// RUN: %clang_cc1 -DCK16 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK16 --check-prefix CK16-64
// RUN: %clang_cc1 -DCK16 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK16 --check-prefix CK16-64
// RUN: %clang_cc1 -DCK16 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK16 --check-prefix CK16-32
// RUN: %clang_cc1 -DCK16 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK16 --check-prefix CK16-32

// RUN: %clang_cc1 -DCK16 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK16 --check-prefix CK16-64
// RUN: %clang_cc1 -DCK16 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK16 --check-prefix CK16-64
// RUN: %clang_cc1 -DCK16 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK16 --check-prefix CK16-32
// RUN: %clang_cc1 -DCK16 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK16 --check-prefix CK16-32

// RUN: %clang_cc1 -DCK16 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY15 %s
// RUN: %clang_cc1 -DCK16 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY15 %s
// RUN: %clang_cc1 -DCK16 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY15 %s
// RUN: %clang_cc1 -DCK16 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY15 %s
// SIMD-ONLY15-NOT: {{__kmpc|__tgt}}
#ifdef CK16

// CK16-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// Map types:
// - OMP_MAP_PRIVATE_VAL + OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK16-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 800]

template<int y>
int foo(int d) {
  int res = d;
  #pragma omp target
  {
    res += y;
  }
  return res;
}
// CK16-LABEL: implicit_maps_templated_function{{.*}}(
void implicit_maps_templated_function (int a){
  int i = a;

  // CK16: define {{.*}}i32 @{{.+}}foo{{.+}}(i32 {{[^,]+}})
  // CK16-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null, i8** null)
  // CK16-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK16-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0

  // CK16-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK16-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK16-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz:64|32]]*
  // CK16-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
  // CK16-DAG: store i[[sz]] [[VAL:%.+]], i[[sz]]* [[CBP1]]
  // CK16-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
  // CK16-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK16-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to i32*
  // CK16-64-DAG: store i32 {{.+}}, i32* [[CADDR]],

  // CK16: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
  i = foo<543>(i);
}
// CK16: define internal void [[KERNEL]](i[[sz]] [[ARG:%.+]])
// CK16: [[ADDR:%.+]] = alloca i[[sz]],
// CK16: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR]],
// CK16-64: [[CADDR:%.+]] = bitcast i64* [[ADDR]] to i32*
// CK16-64: {{.+}} = load i32, i32* [[CADDR]],
// CK16-32: {{.+}} = load i32, i32* [[ADDR]],

#endif // CK16
#endif
