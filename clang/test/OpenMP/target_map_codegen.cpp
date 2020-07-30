// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///
/// Implicit maps.
///

///==========================================================================///
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK1 --check-prefix CK1-32
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK1 --check-prefix CK1-32

// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s

// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK1 --check-prefix CK1-32
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK1 --check-prefix CK1-32

// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK1 --check-prefix CK1-32
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK1 --check-prefix CK1-32

// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

#ifdef CK1

class B {
public:
  static double VAR;
  B() {
  }

  static void modify(int &res) {
#pragma omp target map(tofrom \
                       : res)
    {
      res = B::VAR;
    }
  }
};
double B::VAR = 1.0;

// CK1-LABEL: @.__omp_offloading_{{.*}}implicit_maps_integer{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK1-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK1-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 800]

// CK1-LABEL: implicit_maps_integer{{.*}}(
void implicit_maps_integer (int a){
  // CK1: call void{{.*}}modify
  B::modify(a);
  int i = a;

  // CK1-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK1-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK1-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz:64|32]]*
  // CK1-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
  // CK1-DAG: store i[[sz]] [[VAL:%[^,]+]], i[[sz]]* [[CBP1]]
  // CK1-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
  // CK1-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK1-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to i32*
  // CK1-64-DAG: store i32 {{.+}}, i32* [[CADDR]],

  // CK1: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
  #pragma omp target
  {
   ++i;
  }
}

// CK1: define internal void [[KERNEL]](i[[sz]] [[ARG:%.+]])
// CK1: [[ADDR:%.+]] = alloca i[[sz]],
// CK1: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR]],
// CK1-64: [[CADDR:%.+]] = bitcast i64* [[ADDR]] to i32*
// CK1-64: {{.+}} = load i32, i32* [[CADDR]],
// CK1-32: {{.+}} = load i32, i32* [[ADDR]],

#endif


///==========================================================================///
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2 --check-prefix CK2-32
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2 --check-prefix CK2-32

// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2 --check-prefix CK2-32
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2 --check-prefix CK2-32

// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2 --check-prefix CK2-32
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK2 --check-prefix CK2-32

// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
#ifdef CK2

// CK2-LABEL: @.__omp_offloading_{{.*}}implicit_maps_reference{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK2: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK2: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 800]
// CK2-LABEL: @.__omp_offloading_{{.*}}implicit_maps_reference{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK2: [[SIZES2:@.+]] = {{.+}}constant [1 x i64] zeroinitializer
// Map types: OMP_MAP_IS_PTR | OMP_MAP_IMPLICIT = 544
// CK2: [[TYPES2:@.+]] = {{.+}}constant [1 x i64] [i64 544]

// CK2-LABEL: implicit_maps_reference{{.*}}(
void implicit_maps_reference (int a, int *b){
  int &i = a;
  // CK2-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK2-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK2-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK2-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK2-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK2-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz:64|32]]*
  // CK2-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
  // CK2-DAG: store i[[sz]] [[VAL:%[^,]+]], i[[sz]]* [[CBP1]]
  // CK2-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
  // CK2-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK2-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to i32*
  // CK2-64-DAG: store i32 {{.+}}, i32* [[CADDR]],

  // CK2: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
  #pragma omp target
  {
   ++i;
  }

  int *&p = b;
  // CK2-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES2]]{{.+}}, {{.+}}[[TYPES2]]{{.+}}, i8** null)
  // CK2-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK2-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK2-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK2-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK2-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK2-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK2-DAG: store i32* [[VAL:%[^,]+]], i32** [[CBP1]]
  // CK2-DAG: store i32* [[VAL]], i32** [[CP1]]
  // CK2-DAG: [[VAL]] = load i32*, i32** [[ADDR:%.+]],
  // CK2-DAG: [[ADDR]] = load i32**, i32*** [[ADDR2:%.+]],

  // CK2: call void [[KERNEL2:@.+]](i32* [[VAL]])
  #pragma omp target
  {
   ++p;
  }
}

// CK2: define internal void [[KERNEL]](i[[sz]] [[ARG:%.+]])
// CK2: [[ADDR:%.+]] = alloca i[[sz]],
// CK2: [[REF:%.+]] = alloca i32*,
// CK2: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR]],
// CK2-64: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to i32*
// CK2-64: store i32* [[CADDR]], i32** [[REF]],
// CK2-64: [[RVAL:%.+]] = load i32*, i32** [[REF]],
// CK2-64: {{.+}} = load i32, i32* [[RVAL]],
// CK2-32: store i32* [[ADDR]], i32** [[REF]],
// CK2-32: [[RVAL:%.+]] = load i32*, i32** [[REF]],
// CK2-32: {{.+}} = load i32, i32* [[RVAL]],

// CK2: define internal void [[KERNEL2]](i32* [[ARG:%.+]])
// CK2: [[ADDR:%.+]] = alloca i32*,
// CK2: [[REF:%.+]] = alloca i32**,
// CK2: store i32* [[ARG]], i32** [[ADDR]],
// CK2: store i32** [[ADDR]], i32*** [[REF]],
// CK2: [[T:%.+]] = load i32**, i32*** [[REF]],
// CK2: [[TT:%.+]] = load i32*, i32** [[T]],
// CK2: getelementptr inbounds i32, i32* [[TT]], i32 1
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3 --check-prefix CK3-32
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3 --check-prefix CK3-32

// RUN: %clang_cc1 -DCK3 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3 --check-prefix CK3-32
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3 --check-prefix CK3-32

// RUN: %clang_cc1 -DCK3 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3 --check-prefix CK3-32
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK3 --check-prefix CK3-32

// RUN: %clang_cc1 -DCK3 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK3 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY2 %s
// SIMD-ONLY2-NOT: {{__kmpc|__tgt}}
#ifdef CK3

// CK3-LABEL: @.__omp_offloading_{{.*}}implicit_maps_parameter{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK3-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK3-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 800]

// CK3-LABEL: implicit_maps_parameter{{.*}}(
void implicit_maps_parameter (int a){

  // CK3-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK3-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK3-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK3-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK3-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK3-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz:64|32]]*
  // CK3-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
  // CK3-DAG: store i[[sz]] [[VAL:%[^,]+]], i[[sz]]* [[CBP1]]
  // CK3-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
  // CK3-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK3-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to i32*
  // CK3-64-DAG: store i32 {{.+}}, i32* [[CADDR]],

  // CK3: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
  #pragma omp target
  {
   ++a;
  }
}

// CK3: define internal void [[KERNEL]](i[[sz]] [[ARG:%.+]])
// CK3: [[ADDR:%.+]] = alloca i[[sz]],
// CK3: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR]],
// CK3-64: [[CADDR:%.+]] = bitcast i64* [[ADDR]] to i32*
// CK3-64: {{.+}} = load i32, i32* [[CADDR]],
// CK3-32: {{.+}} = load i32, i32* [[ADDR]],

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK4 --check-prefix CK4-32
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK4 --check-prefix CK4-32

// RUN: %clang_cc1 -DCK4 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK4 --check-prefix CK4-32
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK4 --check-prefix CK4-32

// RUN: %clang_cc1 -DCK4 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK4 --check-prefix CK4-32
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK4 --check-prefix CK4-32

// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY3 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY3 %s
// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY3 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY3 %s
// SIMD-ONLY3-NOT: {{__kmpc|__tgt}}
#ifdef CK4

// CK4-LABEL: @.__omp_offloading_{{.*}}implicit_maps_nested_integer{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK4-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK4-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 800]

// CK4-LABEL: implicit_maps_nested_integer{{.*}}(
void implicit_maps_nested_integer (int a){
  int i = a;

  // The captures in parallel are by reference. Only the capture in target is by
  // copy.

  // CK4: call void {{.+}}@__kmpc_fork_call({{.+}} [[KERNELP1:@.+]] to void (i32*, i32*, ...)*), i32* {{.+}})
  // CK4: define internal void [[KERNELP1]](i32* {{[^,]+}}, i32* {{[^,]+}}, i32* {{[^,]+}})
  #pragma omp parallel
  {
    // CK4-DAG: call i32 @__tgt_target_teams_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null, i32 1, i32 0)
    // CK4-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
    // CK4-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
    // CK4-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
    // CK4-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
    // CK4-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz:64|32]]*
    // CK4-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
    // CK4-DAG: store i[[sz]] [[VAL:%[^,]+]], i[[sz]]* [[CBP1]]
    // CK4-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
    // CK4-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
    // CK4-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to i32*
    // CK4-64-DAG: store i32 {{.+}}, i32* [[CADDR]],

    // CK4: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
    #pragma omp target
    {
      #pragma omp parallel
      {
        ++i;
      }
    }
  }
}

// CK4: define internal void [[KERNEL]](i[[sz]] [[ARG:%.+]])
// CK4: [[ADDR:%.+]] = alloca i[[sz]],
// CK4: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR]],
// CK4-64: [[CADDR:%.+]] = bitcast i64* [[ADDR]] to i32*
// CK4-64: call void {{.+}}@__kmpc_fork_call({{.+}} [[KERNELP2:@.+]] to void (i32*, i32*, ...)*), i32* [[CADDR]])
// CK4-32: call void {{.+}}@__kmpc_fork_call({{.+}} [[KERNELP2:@.+]] to void (i32*, i32*, ...)*), i32* [[ADDR]])
// CK4: define internal void [[KERNELP2]](i32* {{[^,]+}}, i32* {{[^,]+}}, i32* {{[^,]+}})
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK5 --check-prefix CK5-32
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK5 --check-prefix CK5-32

// RUN: %clang_cc1 -DCK5 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK5 --check-prefix CK5-32
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK5 --check-prefix CK5-32

// RUN: %clang_cc1 -DCK5 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK5 --check-prefix CK5-32
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK5 --check-prefix CK5-32

// RUN: %clang_cc1 -DCK5 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY4 %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY4 %s
// RUN: %clang_cc1 -DCK5 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY4 %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY4 %s
// SIMD-ONLY4-NOT: {{__kmpc|__tgt}}
#ifdef CK5

// CK5-LABEL: @.__omp_offloading_{{.*}}implicit_maps_nested_integer_and_enum{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK5-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK5-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 800]

// CK5-LABEL: implicit_maps_nested_integer_and_enum{{.*}}(
void implicit_maps_nested_integer_and_enum (int a){
  enum Bla {
    SomeEnum = 0x09
  };

  // Using an enum should not change the mapping information.
  int  i = a;

  // CK5-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK5-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK5-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK5-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK5-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK5-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz:64|32]]*
  // CK5-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
  // CK5-DAG: store i[[sz]] [[VAL:%[^,]+]], i[[sz]]* [[CBP1]]
  // CK5-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
  // CK5-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK5-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to i32*
  // CK5-64-DAG: store i32 {{.+}}, i32* [[CADDR]],

  // CK5: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
  #pragma omp target
  {
    ++i;
    i += SomeEnum;
  }
}

// CK5: define internal void [[KERNEL]](i[[sz]] [[ARG:%.+]])
// CK5: [[ADDR:%.+]] = alloca i[[sz]],
// CK5: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR]],
// CK5-64: [[CADDR:%.+]] = bitcast i64* [[ADDR]] to i32*
// CK5-64: {{.+}} = load i32, i32* [[CADDR]],
// CK5-32: {{.+}} = load i32, i32* [[ADDR]],

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK6 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK6 --check-prefix CK6-64
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK6 --check-prefix CK6-64
// RUN: %clang_cc1 -DCK6 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK6 --check-prefix CK6-32
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK6 --check-prefix CK6-32

// RUN: %clang_cc1 -DCK6 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK6 --check-prefix CK6-64
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK6 --check-prefix CK6-64
// RUN: %clang_cc1 -DCK6 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK6 --check-prefix CK6-32
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK6 --check-prefix CK6-32

// RUN: %clang_cc1 -DCK6 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK6 --check-prefix CK6-64
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK6 --check-prefix CK6-64
// RUN: %clang_cc1 -DCK6 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK6 --check-prefix CK6-32
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK6 --check-prefix CK6-32

// RUN: %clang_cc1 -DCK6 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY5 %s
// RUN: %clang_cc1 -DCK6 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY5 %s
// RUN: %clang_cc1 -DCK6 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY5 %s
// RUN: %clang_cc1 -DCK6 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY5 %s
// SIMD-ONLY5-NOT: {{__kmpc|__tgt}}
#ifdef CK6
// CK6-DAG: @.__omp_offloading_{{.*}}implicit_maps_host_global{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK6-DAG: [[GBL:@Gi]] = global i32 0
// CK6-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK6-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 800]

// CK6-LABEL: implicit_maps_host_global{{.*}}(
int Gi;
void implicit_maps_host_global (int a){
  // CK6-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK6-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK6-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK6-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK6-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK6-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz:64|32]]*
  // CK6-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
  // CK6-DAG: store i[[sz]] [[VAL:%[^,]+]], i[[sz]]* [[CBP1]]
  // CK6-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
  // CK6-64-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK6-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to i32*
  // CK6-64-DAG: store i32 [[GBLVAL:%.+]], i32* [[CADDR]],
  // CK6-64-DAG: [[GBLVAL]] = load i32, i32* [[GBL]],
  // CK6-32-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[GBLVAL:%.+]],

  // CK6: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
  #pragma omp target
  {
    ++Gi;
  }
}

// CK6: define internal void [[KERNEL]](i[[sz]] [[ARG:%.+]])
// CK6: [[ADDR:%.+]] = alloca i[[sz]],
// CK6: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR]],
// CK6-64: [[CADDR:%.+]] = bitcast i64* [[ADDR]] to i32*
// CK6-64: {{.+}} = load i32, i32* [[CADDR]],
// CK6-32: {{.+}} = load i32, i32* [[ADDR]],

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK7 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK7 --check-prefix CK7-64
// RUN: %clang_cc1 -DCK7 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7  --check-prefix CK7-64
// RUN: %clang_cc1 -DCK7 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7  --check-prefix CK7-32
// RUN: %clang_cc1 -DCK7 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7  --check-prefix CK7-32

// RUN: %clang_cc1 -DCK7 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK7 --check-prefix CK7-64
// RUN: %clang_cc1 -DCK7 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7  --check-prefix CK7-64
// RUN: %clang_cc1 -DCK7 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7  --check-prefix CK7-32
// RUN: %clang_cc1 -DCK7 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7  --check-prefix CK7-32

// RUN: %clang_cc1 -DCK7 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK7 --check-prefix CK7-64
// RUN: %clang_cc1 -DCK7 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7  --check-prefix CK7-64
// RUN: %clang_cc1 -DCK7 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7  --check-prefix CK7-32
// RUN: %clang_cc1 -DCK7 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK7  --check-prefix CK7-32

// RUN: %clang_cc1 -DCK7 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// RUN: %clang_cc1 -DCK7 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// RUN: %clang_cc1 -DCK7 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// RUN: %clang_cc1 -DCK7 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY6 %s
// SIMD-ONLY6-NOT: {{__kmpc|__tgt}}
#ifdef CK7

// For a 32-bit targets, the value doesn't fit the size of the pointer,
// therefore it is passed by reference with a map 'to' specification.

// CK7-LABEL: @.__omp_offloading_{{.*}}implicit_maps_double{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK7-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK7-64-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 800]
// Map types: OMP_MAP_TO  | OMP_MAP_PRIVATE | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 673
// CK7-32-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 673]

// CK7-LABEL: implicit_maps_double{{.*}}(
void implicit_maps_double (int a){
  double d = (double)a;

  // CK7-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK7-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK7-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK7-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK7-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0

  // CK7-64-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz:64|32]]*
  // CK7-64-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
  // CK7-64-DAG: store i[[sz]] [[VAL:%[^,]+]], i[[sz]]* [[CBP1]]
  // CK7-64-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
  // CK7-64-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK7-64-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to double*
  // CK7-64-64-DAG: store double {{.+}}, double* [[CADDR]],

  // CK7-32-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double**
  // CK7-32-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double**
  // CK7-32-DAG: store double* [[DECL:%[^,]+]], double** [[CBP1]]
  // CK7-32-DAG: store double* [[DECL]], double** [[CP1]]

  // CK7-64: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
  // CK7-32: call void [[KERNEL:@.+]](double* [[DECL]])
  #pragma omp target
  {
    d += 1.0;
  }
}

// CK7-64: define internal void [[KERNEL]](i[[sz]] [[ARG:%.+]])
// CK7-64: [[ADDR:%.+]] = alloca i[[sz]],
// CK7-64: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR]],
// CK7-64: [[CADDR:%.+]] = bitcast i64* [[ADDR]] to double*
// CK7-64: {{.+}} = load double, double* [[CADDR]],

// CK7-32: define internal void [[KERNEL]](double* {{.+}}[[ARG:%.+]])
// CK7-32: [[ADDR:%.+]] = alloca double*,
// CK7-32: store double* [[ARG]], double** [[ADDR]],
// CK7-32: [[REF:%.+]] = load double*, double** [[ADDR]],
// CK7-32: {{.+}} = load double, double* [[REF]],

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8

// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8

// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK8

// RUN: %clang_cc1 -DCK8 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY7 %s
// RUN: %clang_cc1 -DCK8 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY7 %s
// RUN: %clang_cc1 -DCK8 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY7 %s
// RUN: %clang_cc1 -DCK8 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY7 %s
// SIMD-ONLY7-NOT: {{__kmpc|__tgt}}
#ifdef CK8

// CK8-LABEL: @.__omp_offloading_{{.*}}implicit_maps_float{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK8-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK8-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 800]

// CK8-LABEL: implicit_maps_float{{.*}}(
void implicit_maps_float (int a){
  float f = (float)a;

  // CK8-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK8-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK8-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK8-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK8-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK8-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz:64|32]]*
  // CK8-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
  // CK8-DAG: store i[[sz]] [[VAL:%[^,]+]], i[[sz]]* [[CBP1]]
  // CK8-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
  // CK8-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK8-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to float*
  // CK8-DAG: store float {{.+}}, float* [[CADDR]],

  // CK8: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
  #pragma omp target
  {
    f += 1.0;
  }
}

// CK8: define internal void [[KERNEL]](i[[sz]] [[ARG:%.+]])
// CK8: [[ADDR:%.+]] = alloca i[[sz]],
// CK8: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR]],
// CK8: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to float*
// CK8: {{.+}} = load float, float* [[CADDR]],

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK9 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK9
// RUN: %clang_cc1 -DCK9 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9
// RUN: %clang_cc1 -DCK9 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9
// RUN: %clang_cc1 -DCK9 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9

// RUN: %clang_cc1 -DCK9 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK9
// RUN: %clang_cc1 -DCK9 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9
// RUN: %clang_cc1 -DCK9 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9
// RUN: %clang_cc1 -DCK9 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9

// RUN: %clang_cc1 -DCK9 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK9
// RUN: %clang_cc1 -DCK9 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9
// RUN: %clang_cc1 -DCK9 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9
// RUN: %clang_cc1 -DCK9 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK9

// RUN: %clang_cc1 -DCK9 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK9 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK9 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// RUN: %clang_cc1 -DCK9 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY8 %s
// SIMD-ONLY8-NOT: {{__kmpc|__tgt}}
#ifdef CK9

// CK9-LABEL: @.__omp_offloading_{{.*}}implicit_maps_array{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK9-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 16]
// Map types: OMP_MAP_TO + OMP_MAP_FROM + OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 547
// CK9-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 547]

// CK9-LABEL: implicit_maps_array{{.*}}(
void implicit_maps_array (int a){
  double darr[2] = {(double)a, (double)a};

  // CK9-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK9-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK9-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK9-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK9-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK9-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [2 x double]**
  // CK9-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [2 x double]**
  // CK9-DAG: store [2 x double]* [[DECL:%[^,]+]], [2 x double]** [[CBP1]]
  // CK9-DAG: store [2 x double]* [[DECL]], [2 x double]** [[CP1]]

  // CK9: call void [[KERNEL:@.+]]([2 x double]* [[DECL]])
  #pragma omp target
  {
    darr[0] += 1.0;
    darr[1] += 1.0;
  }
}

// CK9: define internal void [[KERNEL]]([2 x double]* {{.+}}[[ARG:%.+]])
// CK9: [[ADDR:%.+]] = alloca [2 x double]*,
// CK9: store [2 x double]* [[ARG]], [2 x double]** [[ADDR]],
// CK9: [[REF:%.+]] = load [2 x double]*, [2 x double]** [[ADDR]],
// CK9: {{.+}} = getelementptr inbounds [2 x double], [2 x double]* [[REF]], i{{64|32}} 0, i{{64|32}} 0
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK10 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10

// RUN: %clang_cc1 -DCK10 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10

// RUN: %clang_cc1 -DCK10 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK10

// RUN: %clang_cc1 -DCK10 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY9 %s
// RUN: %clang_cc1 -DCK10 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY9 %s
// RUN: %clang_cc1 -DCK10 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY9 %s
// RUN: %clang_cc1 -DCK10 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY9 %s
// SIMD-ONLY9-NOT: {{__kmpc|__tgt}}
#ifdef CK10

// CK10-LABEL: @.__omp_offloading_{{.*}}implicit_maps_pointer{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK10-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] zeroinitializer
// Map types: OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 544
// CK10-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 544]

// CK10-LABEL: implicit_maps_pointer{{.*}}(
void implicit_maps_pointer (){
  double *ddyn;

  // CK10-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK10-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK10-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK10-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK10-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK10-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double**
  // CK10-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double**
  // CK10-DAG: store double* [[PTR:%[^,]+]], double** [[CBP1]]
  // CK10-DAG: store double* [[PTR]], double** [[CP1]]

  // CK10: call void [[KERNEL:@.+]](double* [[PTR]])
  #pragma omp target
  {
    ddyn[0] += 1.0;
    ddyn[1] += 1.0;
  }
}

// CK10: define internal void [[KERNEL]](double* {{.*}}[[ARG:%.+]])
// CK10: [[ADDR:%.+]] = alloca double*,
// CK10: store double* [[ARG]], double** [[ADDR]],
// CK10: [[REF:%.+]] = load double*, double** [[ADDR]],
// CK10: {{.+}} = getelementptr inbounds double, double* [[REF]], i{{64|32}} 0

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK11 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK11
// RUN: %clang_cc1 -DCK11 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK11
// RUN: %clang_cc1 -DCK11 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK11
// RUN: %clang_cc1 -DCK11 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK11

// RUN: %clang_cc1 -DCK11 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -DCK11 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -DCK11 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// RUN: %clang_cc1 -DCK11 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY10 %s
// SIMD-ONLY10-NOT: {{__kmpc|__tgt}}
#ifdef CK11

// CK11-LABEL: @.__omp_offloading_{{.*}}implicit_maps_double_complex{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK11-DAG: [[SIZES:@.+]] = {{.+}}constant [2 x i64] [i64 16, i64 {{8|4}}]
// Map types: OMP_MAP_TO  | OMP_MAP_FROM | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 547
// CK11-DAG: [[TYPES:@.+]] = {{.+}}constant [2 x i64] [i64 547, i64 547]

// CK11-LABEL: implicit_maps_double_complex{{.*}}(
void implicit_maps_double_complex (int a, int *b){
  double _Complex dc = (double)a;

  // CK11-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 2, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK11-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK11-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK11-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK11-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK11-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to { double, double }**
  // CK11-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to { double, double }**
  // CK11-DAG: store { double, double }* [[PTR:%[^,]+]], { double, double }** [[CBP1]]
  // CK11-DAG: store { double, double }* [[PTR]], { double, double }** [[CP1]]

  // CK11: call void [[KERNEL:@.+]]({ double, double }* [[PTR]], i32** %{{.+}})
  #pragma omp target defaultmap(tofrom:scalar)
  {
   dc *= dc; *b = 1;
  }
}

// CK11: define internal void [[KERNEL]]({ double, double }* {{.*}}[[ARG:%.+]], i32** {{.*}})
// CK11: [[ADDR:%.+]] = alloca { double, double }*,
// CK11: store { double, double }* [[ARG]], { double, double }** [[ADDR]],
// CK11: [[REF:%.+]] = load { double, double }*, { double, double }** [[ADDR]],
// CK11: {{.+}} = getelementptr inbounds { double, double }, { double, double }* [[REF]], i32 0, i32 0
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK12 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK12 --check-prefix CK12-64
// RUN: %clang_cc1 -DCK12 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12 --check-prefix CK12-64
// RUN: %clang_cc1 -DCK12 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12 --check-prefix CK12-32
// RUN: %clang_cc1 -DCK12 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12 --check-prefix CK12-32

// RUN: %clang_cc1 -DCK12 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK12 --check-prefix CK12-64
// RUN: %clang_cc1 -DCK12 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12 --check-prefix CK12-64
// RUN: %clang_cc1 -DCK12 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12 --check-prefix CK12-32
// RUN: %clang_cc1 -DCK12 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12 --check-prefix CK12-32

// RUN: %clang_cc1 -DCK12 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK12 --check-prefix CK12-64
// RUN: %clang_cc1 -DCK12 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12 --check-prefix CK12-64
// RUN: %clang_cc1 -DCK12 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12 --check-prefix CK12-32
// RUN: %clang_cc1 -DCK12 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK12 --check-prefix CK12-32

// RUN: %clang_cc1 -DCK12 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY11 %s
// RUN: %clang_cc1 -DCK12 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY11 %s
// RUN: %clang_cc1 -DCK12 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY11 %s
// RUN: %clang_cc1 -DCK12 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY11 %s
// SIMD-ONLY11-NOT: {{__kmpc|__tgt}}
#ifdef CK12

// For a 32-bit targets, the value doesn't fit the size of the pointer,
// therefore it is passed by reference with a map 'to' specification.

// CK12-LABEL: @.__omp_offloading_{{.*}}implicit_maps_float_complex{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// CK12-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 8]
// Map types: OMP_MAP_PRIVATE_VAL + OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK12-64-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 800]
// Map types: OMP_MAP_TO  | OMP_MAP_PRIVATE | OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 673
// CK12-32-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 673]

// CK12-LABEL: implicit_maps_float_complex{{.*}}(
void implicit_maps_float_complex (int a){
  float _Complex fc = (float)a;

  // CK12-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK12-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK12-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK12-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK12-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0

  // CK12-64-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz:64|32]]*
  // CK12-64-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
  // CK12-64-DAG: store i[[sz]] [[VAL:%[^,]+]], i[[sz]]* [[CBP1]]
  // CK12-64-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
  // CK12-64-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK12-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to { float, float }*
  // CK12-64-DAG: store { float, float } {{.+}}, { float, float }* [[CADDR]],

  // CK12-32-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to { float, float }**
  // CK12-32-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to { float, float }**
  // CK12-32-DAG: store { float, float }* [[DECL:%[^,]+]], { float, float }** [[CBP1]]
  // CK12-32-DAG: store { float, float }* [[DECL]], { float, float }** [[CP1]]

  // CK12-64: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
  // CK12-32: call void [[KERNEL:@.+]]({ float, float }* [[DECL]])
  #pragma omp target
  {
    fc *= fc;
  }
}

// CK12-64: define internal void [[KERNEL]](i[[sz]] [[ARG:%.+]])
// CK12-64: [[ADDR:%.+]] = alloca i[[sz]],
// CK12-64: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR]],
// CK12-64: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to { float, float }*
// CK12-64: {{.+}} = getelementptr inbounds { float, float }, { float, float }* [[CADDR]], i32 0, i32 0

// CK12-32: define internal void [[KERNEL]]({ float, float }* {{.+}}[[ARG:%.+]])
// CK12-32: [[ADDR:%.+]] = alloca { float, float }*,
// CK12-32: store { float, float }* [[ARG]], { float, float }** [[ADDR]],
// CK12-32: [[REF:%.+]] = load { float, float }*, { float, float }** [[ADDR]],
// CK12-32: {{.+}} = getelementptr inbounds { float, float }, { float, float }* [[REF]], i32 0, i32 0
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK13 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13

// RUN: %clang_cc1 -DCK13 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13

// RUN: %clang_cc1 -DCK13 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK13

// RUN: %clang_cc1 -DCK13 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY12 %s
// RUN: %clang_cc1 -DCK13 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY12 %s
// RUN: %clang_cc1 -DCK13 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY12 %s
// RUN: %clang_cc1 -DCK13 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY12 %s
// SIMD-ONLY12-NOT: {{__kmpc|__tgt}}
#ifdef CK13

// CK13-LABEL: @.__omp_offloading_{{.*}}implicit_maps_variable_length_array{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0

// We don't have a constant map size for VLAs.
// Map types:
//  - OMP_MAP_PRIVATE_VAL + OMP_MAP_TARGET_PARAM + OMP_MAP_IMPLICIT = 800 (vla size)
//  - OMP_MAP_PRIVATE_VAL + OMP_MAP_TARGET_PARAM + OMP_MAP_IMPLICIT = 800 (vla size)
//  - OMP_MAP_TO + OMP_MAP_FROM + OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 547
// CK13-DAG: [[TYPES:@.+]] = {{.+}}constant [3 x i64] [i64 800, i64 800, i64 547]

// CK13-LABEL: implicit_maps_variable_length_array{{.*}}(
void implicit_maps_variable_length_array (int a){
  double vla[2][a];

  // CK13-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 3, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], i64* [[SGEP:%[^,]+]], {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK13-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK13-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK13-DAG: [[SGEP]] = getelementptr inbounds {{.+}}[[SS:%[^,]+]], i32 0, i32 0

  // CK13-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK13-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK13-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[SS]], i32 0, i32 0
  // CK13-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[sz:64|32]]*
  // CK13-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[sz]]*
  // CK13-DAG: store i[[sz]] 2, i[[sz]]* [[CBP0]]
  // CK13-DAG: store i[[sz]] 2, i[[sz]]* [[CP0]]
  // CK13-DAG: store i64 {{8|4}}, i64* [[S0]],

  // CK13-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 1
  // CK13-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 1
  // CK13-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[SS]], i32 0, i32 1
  // CK13-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz]]*
  // CK13-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
  // CK13-DAG: store i[[sz]] [[VAL:%.+]], i[[sz]]* [[CBP1]]
  // CK13-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
  // CK13-DAG: store i64 {{8|4}}, i64* [[S1]],

  // CK13-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 2
  // CK13-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 2
  // CK13-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[SS]], i32 0, i32 2
  // CK13-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to double**
  // CK13-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to double**
  // CK13-DAG: store double* [[DECL:%.+]], double** [[CBP2]]
  // CK13-DAG: store double* [[DECL]], double** [[CP2]]
  // CK13-DAG: store i64 [[VALS2:%.+]], i64* [[S2]],
  // CK13-DAG: [[VALS2]] = {{mul nuw i64 %.+, 8|sext i32 %.+ to i64}}

  // CK13: call void [[KERNEL:@.+]](i[[sz]] {{.+}}, i[[sz]] {{.+}}, double* [[DECL]])
  #pragma omp target
  {
    vla[1][3] += 1.0;
  }
}

// CK13: define internal void [[KERNEL]](i[[sz]] [[VLA0:%.+]], i[[sz]] [[VLA1:%.+]], double* {{.*}}[[ARG:%.+]])
// CK13: [[ADDR0:%.+]] = alloca i[[sz]],
// CK13: [[ADDR1:%.+]] = alloca i[[sz]],
// CK13: [[ADDR2:%.+]] = alloca double*,
// CK13: store i[[sz]] [[VLA0]], i[[sz]]* [[ADDR0]],
// CK13: store i[[sz]] [[VLA1]], i[[sz]]* [[ADDR1]],
// CK13: store double* [[ARG]], double** [[ADDR2]],
// CK13: {{.+}} = load i[[sz]],  i[[sz]]* [[ADDR0]],
// CK13: {{.+}} = load i[[sz]],  i[[sz]]* [[ADDR1]],
// CK13: [[REF:%.+]] = load double*, double** [[ADDR2]],
// CK13: {{.+}} = getelementptr inbounds double, double* [[REF]], i[[sz]] %{{.+}}
#endif
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
  // CK14-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 4, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], i64* [[SIZES:%[^,]+]], {{.+}}[[TYPES]]{{.+}}, i8** null)
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
  // CK14-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i32 0, i32 1
  // CK14-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[ST]]**
  // CK14-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK14-DAG: store [[ST]]* [[DECL]], [[ST]]** [[CBP1]]
  // CK14-DAG: store i32* [[A]], i32** [[CP1]]
  // CK14-DAG: store i64 4, i64* [[S1]]

  // CK14-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 2
  // CK14-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 2
  // CK14-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i32 0, i32 2
  // CK14-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [[ST]]**
  // CK14-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to double**
  // CK14-DAG: store [[ST]]* [[DECL]], [[ST]]** [[CBP2]]
  // CK14-DAG: store double* %{{.+}}, double** [[CP2]]
  // CK14-DAG: store i64 8, i64* [[S2]]

  // CK14-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 3
  // CK14-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 3
  // CK14-DAG: [[S3:%.+]] = getelementptr inbounds {{.+}}[[S]], i32 0, i32 3
  // CK14-DAG: [[CBP3:%.+]] = bitcast i8** [[BP3]] to i[[sz:64|32]]*
  // CK14-DAG: [[CP3:%.+]] = bitcast i8** [[P3]] to i[[sz]]*
  // CK14-DAG: store i[[sz]] [[VAL:%.+]], i[[sz]]* [[CBP3]]
  // CK14-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP3]]
  // CK14-DAG: store i64 4, i64* [[S3]]
  // CK14-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK14-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to i32*
  // CK14-64-DAG: store i32 {{.+}}, i32* [[CADDR]],

  // CK14: call void [[KERNEL:@.+]]([[ST]]* [[DECL]], i[[sz]] {{.+}})
  sss.foo(123);
}

// CK14: define internal void [[KERNEL]]([[ST]]* [[THIS:%.+]], i[[sz]] [[ARG:%.+]])
// CK14: [[ADDR0:%.+]] = alloca [[ST]]*,
// CK14: [[ADDR1:%.+]] = alloca i[[sz]],
// CK14: store [[ST]]* [[THIS]], [[ST]]** [[ADDR0]],
// CK14: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR1]],
// CK14: [[REF0:%.+]] = load [[ST]]*, [[ST]]** [[ADDR0]],
// CK14-64: [[CADDR1:%.+]] = bitcast i[[sz]]* [[ADDR1]] to i32*
// CK14-64: {{.+}} = load i32,  i32* [[CADDR1]],
// CK14-32: {{.+}} = load i32, i32* [[ADDR1]],
// CK14: {{.+}} = getelementptr inbounds [[ST]], [[ST]]* [[REF0]], i32 0, i32 0

#endif
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
// Map types:
// - OMP_MAP_TARGET_PARAM = 32
// - OMP_MAP_TO + OMP_MAP_FROM | OMP_MAP_IMPLICIT | OMP_MAP_MEMBER_OF = 281474976711171
// - OMP_MAP_PRIVATE_VAL + OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK15: [[TYPES:@.+]] = {{.+}}constant [4 x i64] [i64 32, i64 281474976711171, i64 281474976711171, i64 800]

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

  // CK15: define {{.*}}void @{{.+}}foo{{.+}}([[ST]]* {{[^,]+}}, i32 {{[^,]+}})
  // CK15-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 4, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], i64* [[SIZES:%[^,]+]], {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK15-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK15-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK15-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]], i32 0, i32 0

  // CK15-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK15-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK15-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i32 0, i32 0
  // CK15-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
  // CK15-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK15-DAG: store [[ST]]* [[DECL:%.+]], [[ST]]** [[CBP0]]
  // CK15-DAG: store i32* [[A:%.+]], i32** [[CP0]]
  // CK15-DAG: store i64 %{{.+}}, i64* [[S0]]

  // CK15-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 1
  // CK15-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 1
  // CK15-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i32 0, i32 1
  // CK15-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[ST]]**
  // CK15-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK15-DAG: store [[ST]]* [[DECL]], [[ST]]** [[CBP1]]
  // CK15-DAG: store i32* [[A]], i32** [[CP1]]
  // CK15-DAG: store i64 4, i64* [[S1]]

  // CK15-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 2
  // CK15-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 2
  // CK15-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i32 0, i32 2
  // CK15-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [[ST]]**
  // CK15-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to double**
  // CK15-DAG: store [[ST]]* [[DECL]], [[ST]]** [[CBP2]]
  // CK15-DAG: store double* %{{.+}}, double** [[CP2]]
  // CK15-DAG: store i64 8, i64* [[S2]]

  // CK15-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 3
  // CK15-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 3
  // CK15-DAG: [[S3:%.+]] = getelementptr inbounds {{.+}}[[S]], i32 0, i32 3
  // CK15-DAG: [[CBP3:%.+]] = bitcast i8** [[BP3]] to i[[sz:64|32]]*
  // CK15-DAG: [[CP3:%.+]] = bitcast i8** [[P3]] to i[[sz]]*
  // CK15-DAG: store i[[sz]] [[VAL:%.+]], i[[sz]]* [[CBP3]]
  // CK15-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP3]]
  // CK15-DAG: store i64 4, i64* [[S3]]
  // CK15-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK15-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to i32*
  // CK15-64-DAG: store i32 {{.+}}, i32* [[CADDR]],

  // CK15: call void [[KERNEL:@.+]]([[ST]]* [[DECL]], i[[sz]] {{.+}})
  ssst.foo(456);

  // CK15: define {{.*}}void @{{.+}}bar{{.+}}([[ST]]* {{[^,]+}}, i32 {{[^,]+}})
  // CK15-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 4, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], i64* [[SIZES:[^,]+]], {{.+}}[[TYPES2]]{{.+}}, i8** null)
  // CK15-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK15-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK15-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]], i32 0, i32 0

  // CK15-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK15-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK15-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i32 0, i32 0
  // CK15-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
  // CK15-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK15-DAG: store [[ST]]* [[DECL:%.+]], [[ST]]** [[CBP0]]
  // CK15-DAG: store i32* [[A:%.+]], i32** [[CP0]]
  // CK15-DAG: store i64 %{{.+}}, i64* [[S0]]

  // CK15-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 1
  // CK15-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 1
  // CK15-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i32 0, i32 1
  // CK15-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[ST]]**
  // CK15-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK15-DAG: store [[ST]]* [[DECL]], [[ST]]** [[CBP1]]
  // CK15-DAG: store i32* [[A]], i32** [[CP1]]
  // CK15-DAG: store i64 4, i64* [[S1]]

  // CK15-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 2
  // CK15-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 2
  // CK15-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i32 0, i32 2
  // CK15-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [[ST]]**
  // CK15-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to double**
  // CK15-DAG: store [[ST]]* [[DECL]], [[ST]]** [[CBP2]]
  // CK15-DAG: store double* %{{.+}}, double** [[CP2]]
  // CK15-DAG: store i64 8, i64* [[S2]]

  // CK15-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 3
  // CK15-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 3
  // CK15-DAG: [[S3:%.+]] = getelementptr inbounds {{.+}}[[S]], i32 0, i32 3
  // CK15-DAG: [[CBP3:%.+]] = bitcast i8** [[BP3]] to i[[sz]]*
  // CK15-DAG: [[CP3:%.+]] = bitcast i8** [[P3]] to i[[sz]]*
  // CK15-DAG: store i[[sz]] [[VAL:%.+]], i[[sz]]* [[CBP3]]
  // CK15-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP3]]
  // CK15-DAG: store i64 4, i64* [[S3]]
  // CK15-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK15-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to i32*
  // CK15-64-DAG: store i32 {{.+}}, i32* [[CADDR]],

  // CK15: call void [[KERNEL2:@.+]]([[ST]]* [[DECL]], i[[sz]] {{.+}})
  ssst.bar<210>(789);
}

// CK15: define internal void [[KERNEL]]([[ST]]* [[THIS:%.+]], i[[sz]] [[ARG:%.+]])
// CK15: [[ADDR0:%.+]] = alloca [[ST]]*,
// CK15: [[ADDR1:%.+]] = alloca i[[sz]],
// CK15: store [[ST]]* [[THIS]], [[ST]]** [[ADDR0]],
// CK15: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR1]],
// CK15: [[REF0:%.+]] = load [[ST]]*, [[ST]]** [[ADDR0]],
// CK15-64: [[CADDR1:%.+]] = bitcast i[[sz]]* [[ADDR1]] to i32*
// CK15-64: {{.+}} = load i32,  i32* [[CADDR1]],
// CK15-32: {{.+}} = load i32, i32* [[ADDR1]],
// CK15: {{.+}} = getelementptr inbounds [[ST]], [[ST]]* [[REF0]], i32 0, i32 0

// CK15: define internal void [[KERNEL2]]([[ST]]* [[THIS:%.+]], i[[sz]] [[ARG:%.+]])
// CK15: [[ADDR0:%.+]] = alloca [[ST]]*,
// CK15: [[ADDR1:%.+]] = alloca i[[sz]],
// CK15: store [[ST]]* [[THIS]], [[ST]]** [[ADDR0]],
// CK15: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR1]],
// CK15: [[REF0:%.+]] = load [[ST]]*, [[ST]]** [[ADDR0]],
// CK15-64: [[CADDR1:%.+]] = bitcast i[[sz]]* [[ADDR1]] to i32*
// CK15-64: {{.+}} = load i32,  i32* [[CADDR1]],
// CK15-32: {{.+}} = load i32, i32* [[ADDR1]],
// CK15: {{.+}} = getelementptr inbounds [[ST]], [[ST]]* [[REF0]], i32 0, i32 0

#endif
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
  // CK16-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
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

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK17 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK17
// RUN: %clang_cc1 -DCK17 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17
// RUN: %clang_cc1 -DCK17 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17
// RUN: %clang_cc1 -DCK17 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17

// RUN: %clang_cc1 -DCK17 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK17
// RUN: %clang_cc1 -DCK17 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17
// RUN: %clang_cc1 -DCK17 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17
// RUN: %clang_cc1 -DCK17 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17

// RUN: %clang_cc1 -DCK17 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK17
// RUN: %clang_cc1 -DCK17 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17
// RUN: %clang_cc1 -DCK17 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17
// RUN: %clang_cc1 -DCK17 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK17

// RUN: %clang_cc1 -DCK17 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -DCK17 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -DCK17 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// RUN: %clang_cc1 -DCK17 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY16 %s
// SIMD-ONLY16-NOT: {{__kmpc|__tgt}}
#ifdef CK17

// CK17-DAG: [[ST:%.+]] = type { i32, double }
// CK17-LABEL: @.__omp_offloading_{{.*}}implicit_maps_struct{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK17-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 {{16|12}}]
// Map types: OMP_MAP_TO + OMP_MAP_FROM + OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 547
// CK17-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 547]

class SSS {
public:
  int a;
  double b;
};

// CK17-LABEL: implicit_maps_struct{{.*}}(
void implicit_maps_struct (int a){
  SSS s = {a, (double)a};

  // CK17-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK17-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK17-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK17-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK17-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK17-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[ST]]**
  // CK17-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [[ST]]**
  // CK17-DAG: store [[ST]]* [[DECL:%.+]], [[ST]]** [[CBP1]]
  // CK17-DAG: store [[ST]]* [[DECL]], [[ST]]** [[CP1]]

  // CK17: call void [[KERNEL:@.+]]([[ST]]* [[DECL]])
  #pragma omp target
  {
    s.a += 1;
    s.b += 1.0;
  }
}

// CK17: define internal void [[KERNEL]]([[ST]]* {{.+}}[[ARG:%.+]])
// CK17: [[ADDR:%.+]] = alloca [[ST]]*,
// CK17: store [[ST]]* [[ARG]], [[ST]]** [[ADDR]],
// CK17: [[REF:%.+]] = load [[ST]]*, [[ST]]** [[ADDR]],
// CK17: {{.+}} = getelementptr inbounds [[ST]], [[ST]]* [[REF]], i32 0, i32 0
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK18 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK18 --check-prefix CK18-64
// RUN: %clang_cc1 -DCK18 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK18 --check-prefix CK18-64
// RUN: %clang_cc1 -DCK18 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK18 --check-prefix CK18-32
// RUN: %clang_cc1 -DCK18 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK18 --check-prefix CK18-32

// RUN: %clang_cc1 -DCK18 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK18 --check-prefix CK18-64
// RUN: %clang_cc1 -DCK18 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK18 --check-prefix CK18-64
// RUN: %clang_cc1 -DCK18 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK18 --check-prefix CK18-32
// RUN: %clang_cc1 -DCK18 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK18 --check-prefix CK18-32

// RUN: %clang_cc1 -DCK18 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK18 --check-prefix CK18-64
// RUN: %clang_cc1 -DCK18 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK18 --check-prefix CK18-64
// RUN: %clang_cc1 -DCK18 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK18 --check-prefix CK18-32
// RUN: %clang_cc1 -DCK18 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK18 --check-prefix CK18-32

// RUN: %clang_cc1 -DCK18 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY17 %s
// RUN: %clang_cc1 -DCK18 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY17 %s
// RUN: %clang_cc1 -DCK18 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY17 %s
// RUN: %clang_cc1 -DCK18 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY17 %s
// SIMD-ONLY17-NOT: {{__kmpc|__tgt}}
#ifdef CK18

// CK18-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// Map types:
// - OMP_MAP_PRIVATE_VAL + OMP_MAP_TARGET_PARAM | OMP_MAP_IMPLICIT = 800
// CK18-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 800]

template<typename T>
int foo(T d) {
  #pragma omp target
  {
    d += (T)1;
  }
  return d;
}
// CK18-LABEL: implicit_maps_template_type_capture{{.*}}(
void implicit_maps_template_type_capture (int a){
  int i = a;

  // CK18: define {{.*}}i32 @{{.+}}foo{{.+}}(i32 {{[^,]+}})
  // CK18-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null)
  // CK18-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK18-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0

  // CK18-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK18-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK18-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[sz:64|32]]*
  // CK18-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[sz]]*
  // CK18-DAG: store i[[sz]] [[VAL:%.+]], i[[sz]]* [[CBP1]]
  // CK18-DAG: store i[[sz]] [[VAL]], i[[sz]]* [[CP1]]
  // CK18-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK18-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to i32*
  // CK18-64-DAG: store i32 {{.+}}, i32* [[CADDR]],

  // CK18: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
  i = foo(i);
}
// CK18: define internal void [[KERNEL]](i[[sz]] [[ARG:%.+]])
// CK18: [[ADDR:%.+]] = alloca i[[sz]],
// CK18: store i[[sz]] [[ARG]], i[[sz]]* [[ADDR]],
// CK18-64: [[CADDR:%.+]] = bitcast i64* [[ADDR]] to i32*
// CK18-64: {{.+}} = load i32, i32* [[CADDR]],
// CK18-32: {{.+}} = load i32, i32* [[ADDR]],

#endif
///==========================================================================///
// RUN: %clang_cc1 -DUSE -DCK19 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK19,CK19-64,CK19-USE
// RUN: %clang_cc1 -DUSE -DCK19 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK19,CK19-64,CK19-USE
// RUN: %clang_cc1 -DUSE -DCK19 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK19,CK19-32,CK19-USE
// RUN: %clang_cc1 -DUSE -DCK19 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK19,CK19-32,CK19-USE

// RUN: %clang_cc1 -DUSE -DCK19 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK19,CK19-64,CK19-USE
// RUN: %clang_cc1 -DUSE -DCK19 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK19,CK19-64,CK19-USE
// RUN: %clang_cc1 -DUSE -DCK19 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK19,CK19-32,CK19-USE
// RUN: %clang_cc1 -DUSE -DCK19 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK19,CK19-32,CK19-USE

// RUN: %clang_cc1 -DUSE -DCK19 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK19,CK19-64,CK19-USE
// RUN: %clang_cc1 -DUSE -DCK19 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK19,CK19-64,CK19-USE
// RUN: %clang_cc1 -DUSE -DCK19 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK19,CK19-32,CK19-USE
// RUN: %clang_cc1 -DUSE -DCK19 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK19,CK19-32,CK19-USE

// RUN: %clang_cc1 -DUSE -DCK19 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DUSE -DCK19 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DUSE -DCK19 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DUSE -DCK19 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s

// RUN: %clang_cc1 -DCK19 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK19,CK19-64,CK19-NOUSE
// RUN: %clang_cc1 -DCK19 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK19,CK19-64,CK19-NOUSE
// RUN: %clang_cc1 -DCK19 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK19,CK19-32,CK19-NOUSE
// RUN: %clang_cc1 -DCK19 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK19,CK19-32,CK19-NOUSE

// RUN: %clang_cc1 -DCK19 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK19,CK19-64,CK19-NOUSE
// RUN: %clang_cc1 -DCK19 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK19,CK19-64,CK19-NOUSE
// RUN: %clang_cc1 -DCK19 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK19,CK19-32,CK19-NOUSE
// RUN: %clang_cc1 -DCK19 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK19,CK19-32,CK19-NOUSE

// RUN: %clang_cc1 -DCK19 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK19,CK19-64,CK19-NOUSE
// RUN: %clang_cc1 -DCK19 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK19,CK19-64,CK19-NOUSE
// RUN: %clang_cc1 -DCK19 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK19,CK19-32,CK19-NOUSE
// RUN: %clang_cc1 -DCK19 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK19,CK19-32,CK19-NOUSE

// RUN: %clang_cc1 -DCK19 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK19 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK19 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK19 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s


// SIMD-ONLY18-NOT: {{__kmpc|__tgt}}
#ifdef CK19

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE00:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 32]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE00n:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19: [[MTYPE00n:@.+]] = private {{.*}}constant [1 x i64] [i64 32]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE01:@.+]] = private {{.*}}constant [1 x i64] [i64 400]
// CK19: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 33]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE02:@.+]] = private {{.*}}constant [1 x i64] [i64 240]
// CK19: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i64] [i64 34]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE03:@.+]] = private {{.*}}constant [1 x i64] [i64 240]
// CK19: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE04:@.+]] = private {{.*}}constant [1 x i64] [i64 400]
// CK19: [[MTYPE04:@.+]] = private {{.*}}constant [1 x i64] [i64 32]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE05:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19: [[MTYPE05:@.+]] = private {{.*}}constant [1 x i64] [i64 33]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[MTYPE06:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[MTYPE07:@.+]] = private {{.*}}constant [1 x i64] [i64 32]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE08:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19: [[MTYPE08:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE09:@.+]] = private {{.*}}constant [1 x i64] [i64 {{8|4}}]
// CK19: [[MTYPE09:@.+]] = private {{.*}}constant [1 x i64] [i64 34]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE10:@.+]] = private {{.*}}constant [1 x i64] [i64 240]
// CK19: [[MTYPE10:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE11:@.+]] = private {{.*}}constant [1 x i64] [i64 240]
// CK19: [[MTYPE11:@.+]] = private {{.*}}constant [1 x i64] [i64 32]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE12:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19: [[MTYPE12:@.+]] = private {{.*}}constant [1 x i64] [i64 33]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[MTYPE13:@.+]] = private {{.*}}constant [1 x i64] [i64 32]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[MTYPE14:@.+]] = private {{.*}}constant [1 x i64] [i64 33]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE15:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19: [[MTYPE15:@.+]] = private {{.*}}constant [1 x i64] [i64 34]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE16:@.+]] = private {{.*}}constant [2 x i64] [i64 800, i64 33]
// CK19-NOUSE: [[MTYPE16:@.+]] = private {{.*}}constant [1 x i64] [i64 33]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[SIZE17:@.+]] = private {{.*}}constant [2 x i64] [i64 {{8|4}}, i64 240]
// CK19-USE: [[MTYPE17:@.+]] = private {{.*}}constant [2 x i64] [i64 800, i64 34]
// CK19-NOUSE: [[SIZE17:@.+]] = private {{.*}}constant [1 x i64] [i64 240]
// CK19-NOUSE: [[MTYPE17:@.+]] = private {{.*}}constant [1 x i64] [i64 34]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[SIZE18:@.+]] = private {{.*}}constant [2 x i64] [i64 {{8|4}}, i64 240]
// CK19-USE: [[MTYPE18:@.+]] = private {{.*}}constant [2 x i64] [i64 800, i64 35]
// CK19-NOUSE: [[SIZE18:@.+]] = private {{.*}}constant [1 x i64] [i64 240]
// CK19-NOUSE: [[MTYPE18:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE19:@.+]] = private {{.*}}constant [2 x i64] [i64 800, i64 32]
// CK19-NOUSE: [[MTYPE19:@.+]] = private {{.*}}constant [1 x i64] [i64 32]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[SIZE20:@.+]] = private {{.*}}constant [2 x i64] [i64 {{8|4}}, i64 4]
// CK19-USE: [[MTYPE20:@.+]] = private {{.*}}constant [2 x i64] [i64 800, i64 33]
// CK19-NOUSE: [[SIZE20:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19-NOUSE: [[MTYPE20:@.+]] = private {{.*}}constant [1 x i64] [i64 33]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE21:@.+]] = private {{.*}}constant [2 x i64] [i64 800, i64 35]
// CK19-NOUSE: [[MTYPE21:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[SIZE22:@.+]] = private {{.*}}constant [2 x i64] [i64 {{8|4}}, i64 4]
// CK19-USE: [[MTYPE22:@.+]] = private {{.*}}constant [2 x i64] [i64 800, i64 35]
// CK19-NOUSE: [[SIZE22:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19-NOUSE: [[MTYPE22:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE23:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19: [[MTYPE23:@.+]] = private {{.*}}constant [1 x i64] [i64 39]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE24:@.+]] = private {{.*}}constant [1 x i64] [i64 480]
// CK19: [[MTYPE24:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE25:@.+]] = private {{.*}}constant [1 x i64] [i64 16]
// CK19: [[MTYPE25:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE26:@.+]] = private {{.*}}constant [1 x i64] [i64 24]
// CK19: [[MTYPE26:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE27:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK19: [[MTYPE27:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE28:@.+]] = private {{.*}}constant [3 x i64] [i64 {{8|4}}, i64 {{8|4}}, i64 16]
// CK19: [[MTYPE28:@.+]] = private {{.*}}constant [3 x i64] [i64 35, i64 16, i64 19]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE29:@.+]] = private {{.*}}constant [3 x i64] [i64 {{8|4}}, i64 {{8|4}}, i64 4]
// CK19: [[MTYPE29:@.+]] = private {{.*}}constant [3 x i64] [i64 35, i64 16, i64 19]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE30:@.+]] = private {{.*}}constant [4 x i64] [i64 800, i64 800, i64 800, i64 35]
// CK19-NOUSE: [[MTYPE30:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[SIZE31:@.+]] = private {{.*}}constant [4 x i64] [i64 {{8|4}}, i64 {{8|4}}, i64 {{8|4}}, i64 40]
// CK19-USE: [[MTYPE31:@.+]] = private {{.*}}constant [4 x i64] [i64 800, i64 800, i64 800, i64 35]
// CK19-NOUSE: [[SIZE31:@.+]] = private {{.*}}constant [1 x i64] [i64 40]
// CK19-NOUSE: [[MTYPE31:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE32:@.+]] = private {{.*}}constant [1 x i64] [i64 13728]
// CK19: [[MTYPE32:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE33:@.+]] = private {{.*}}constant [1 x i64] [i64 13728]
// CK19: [[MTYPE33:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE34:@.+]] = private {{.*}}constant [1 x i64] [i64 13728]
// CK19: [[MTYPE34:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[MTYPE35:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE36:@.+]] = private {{.*}}constant [1 x i64] [i64 208]
// CK19: [[MTYPE36:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE37:@.+]] = private {{.*}}constant [3 x i64] [i64 800, i64 800, i64 35]
// CK19-NOUSE: [[MTYPE37:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE38:@.+]] = private {{.*}}constant [3 x i64] [i64 800, i64 800, i64 35]
// CK19-NOUSE: [[MTYPE38:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE39:@.+]] = private {{.*}}constant [3 x i64] [i64 800, i64 800, i64 35]
// CK19-NOUSE: [[MTYPE39:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[MTYPE40:@.+]] = private {{.*}}constant [3 x i64] [i64 800, i64 800, i64 35]
// CK19-NOUSE: [[MTYPE40:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19-USE: [[SIZE41:@.+]] = private {{.*}}constant [3 x i64] [i64 {{8|4}}, i64 {{8|4}}, i64 208]
// CK19-USE: [[MTYPE41:@.+]] = private {{.*}}constant [3 x i64] [i64 800, i64 800, i64 35]
// CK19-NOUSE: [[SIZE41:@.+]] = private {{.*}}constant [1 x i64] [i64 208]
// CK19-NOUSE: [[MTYPE41:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE42:@.+]] = private {{.*}}constant [3 x i64] [i64 {{8|4}}, i64 {{8|4}}, i64 104]
// CK19: [[MTYPE42:@.+]] = private {{.*}}constant [3 x i64] [i64 35, i64 16, i64 19]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[MTYPE43:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK19-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK19: [[SIZE44:@.+]] = private {{.*}}constant [1 x i64] [i64 320]
// CK19: [[MTYPE44:@.+]] = private {{.*}}constant [1 x i64] [i64 34]

// CK19-LABEL: explicit_maps_single{{.*}}(
void explicit_maps_single (int ii){
  // Map of a scalar.
  int a = ii;

  // Region 00
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-DAG: store i32* [[VAR0]], i32** [[CP0]]

  // CK19-USE: call void [[CALL00:@.+]](i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL00:@.+]]()
  #pragma omp target map(alloc:a)
  {
#ifdef USE
    ++a;
#endif
  }

  // Map of a scalar in nested region.
  int b = a;

  // Region 00n
  // CK19-DAG: call i32 @__tgt_target_teams_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00n]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00n]]{{.+}}, i8** null, i32 1, i32 0)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-DAG: store i32* [[VAR0]], i32** [[CP0]]

  // CK19-USE: call void [[CALL00n:@.+]](i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL00n:@.+]]()
  #pragma omp target map(alloc:b)
  #pragma omp parallel
  {
#ifdef USE
    ++b;
#endif
  }

  // Map of an array.
  int arra[100];

  // Region 01
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [100 x i32]**
  // CK19-DAG: store [100 x i32]* [[VAR0:%.+]], [100 x i32]** [[CBP0]]
  // CK19-DAG: store [100 x i32]* [[VAR0]], [100 x i32]** [[CP0]]

  // CK19-USE: call void [[CALL01:@.+]]([100 x i32]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL01:@.+]]()
  #pragma omp target map(to:arra)
  {
#ifdef USE
    arra[50]++;
#endif
  }

  // Region 02
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [100 x i32]* [[VAR0:%.+]], [100 x i32]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%[^,]+]], i32** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} 20

  // CK19-USE: call void [[CALL02:@.+]]([100 x i32]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL02:@.+]]()
  #pragma omp target map(from:arra[20:60])
  {
#ifdef USE
    arra[50]++;
#endif
  }

  // Region 03
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [100 x i32]* [[VAR0:%.+]], [100 x i32]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%[^,]+]], i32** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} 0

  // CK19-USE: call void [[CALL03:@.+]]([100 x i32]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL03:@.+]]()
  #pragma omp target map(tofrom:arra[:60])
  {
#ifdef USE
    arra[50]++;
#endif
  }

  // Region 04
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE04]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE04]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [100 x i32]* [[VAR0:%.+]], [100 x i32]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%[^,]+]], i32** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} 0

  // CK19-USE: call void [[CALL04:@.+]]([100 x i32]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL04:@.+]]()
  #pragma omp target map(alloc:arra[:])
  {
#ifdef USE
    arra[50]++;
#endif
  }

  // Region 05
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE05]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE05]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [100 x i32]* [[VAR0:%.+]], [100 x i32]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%[^,]+]], i32** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} 15

  // CK19-USE: call void [[CALL05:@.+]]([100 x i32]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL05:@.+]]()
  #pragma omp target map(to:arra[15])
  {
#ifdef USE
    arra[15]++;
#endif
  }

  // Region 06
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE06]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [100 x i32]* [[VAR0:%.+]], [100 x i32]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%[^,]+]], i32** [[CP0]]
  // CK19-DAG: store i{{.+}} [[CSVAL0:%[^,]+]], i{{.+}}* [[S0]]
  // CK19-DAG: [[CSVAL0]] = {{mul nuw i.+ %.*, 4|sext i32 .+ to i64}}
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} %{{.*}}

  // CK19-USE: call void [[CALL06:@.+]]([100 x i32]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL06:@.+]]()
  #pragma omp target map(tofrom:arra[ii:ii+23])
  {
#ifdef USE
    arra[50]++;
#endif
  }

  // Region 07
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE07]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [100 x i32]* [[VAR0:%.+]], [100 x i32]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%[^,]+]], i32** [[CP0]]
  // CK19-DAG: store i{{.+}} [[CSVAL0:%[^,]+]], i{{.+}}* [[S0]]
  // CK19-DAG: [[CSVAL0]] = {{mul nuw i.+ %.*, 4|sext i32 .+ to i64}}
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} 0

  // CK19-USE: call void [[CALL07:@.+]]([100 x i32]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL07:@.+]]()
  #pragma omp target map(alloc:arra[:ii])
  {
#ifdef USE
    arra[50]++;
#endif
  }

  // Region 08
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE08]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE08]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [100 x i32]* [[VAR0:%.+]], [100 x i32]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%[^,]+]], i32** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} %{{.*}}

  // CK19-USE: call void [[CALL08:@.+]]([100 x i32]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL08:@.+]]()
  #pragma omp target map(tofrom:arra[ii])
  {
#ifdef USE
    arra[15]++;
#endif
  }

  // Map of a pointer.
  int *pa;

  // Region 09
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE09]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE09]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32***
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32***
  // CK19-DAG: store i32** [[VAR0:%.+]], i32*** [[CBP0]]
  // CK19-DAG: store i32** [[VAR0]], i32*** [[CP0]]

  // CK19-USE: call void [[CALL09:@.+]](i32** {{[^,]+}})
  // CK19-NOUSE: call void [[CALL09:@.+]]()
  #pragma omp target map(from:pa)
  {
#ifdef USE
    pa[50]++;
#endif
  }

  // Region 10
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE10]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE10]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 20
  // CK19-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK19-USE: call void [[CALL10:@.+]](i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL10:@.+]]()
  #pragma omp target map(tofrom:pa[20:60])
  {
#ifdef USE
    pa[50]++;
#endif
  }

  // Region 11
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE11]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE11]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 0
  // CK19-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK19-USE: call void [[CALL11:@.+]](i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL11:@.+]]()
  #pragma omp target map(alloc:pa[:60])
  {
#ifdef USE
    pa[50]++;
#endif
  }

  // Region 12
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE12]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE12]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 15
  // CK19-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK19-USE: call void [[CALL12:@.+]](i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL12:@.+]]()
  #pragma omp target map(to:pa[15])
  {
#ifdef USE
    pa[15]++;
#endif
  }

  // Region 13
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE13]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-DAG: store i{{.+}} [[CSVAL0:%[^,]+]], i{{.+}}* [[S0]]
  // CK19-DAG: [[CSVAL0]] = {{mul nuw i64 %.*, 4|sext i32 .+ to i64}}
  // CK19-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} %{{.*}}
  // CK19-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK19-USE: call void [[CALL13:@.+]](i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL13:@.+]]()
  #pragma omp target map(alloc:pa[ii-23:ii])
  {
#ifdef USE
    pa[50]++;
#endif
  }

  // Region 14
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE14]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-DAG: store i{{.+}} [[CSVAL0:%[^,]+]], i{{.+}}* [[S0]]
  // CK19-DAG: [[CSVAL0]] = {{mul nuw i64 %.*, 4|sext i32 .+ to i64}}
  // CK19-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 0
  // CK19-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK19-USE: call void [[CALL14:@.+]](i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL14:@.+]]()
  #pragma omp target map(to:pa[:ii])
  {
#ifdef USE
    pa[50]++;
#endif
  }

  // Region 15
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE15]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE15]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} %{{.*}}
  // CK19-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK19-USE: call void [[CALL15:@.+]](i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL15:@.+]]()
  #pragma omp target map(from:pa[ii+12])
  {
#ifdef USE
    pa[15]++;
#endif
  }

  // Map of a variable-size array.
  int va[ii];

  // Region 16
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|2}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[MTYPE16]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z:64|32]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CP0]]
  // CK19-USE-DAG: store i{{.+}} {{8|4}}, i{{.+}}* [[S0]]

  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK19-USE-DAG: store i32* [[VAR1:%.+]], i32** [[CBP1]]
  // CK19-USE-DAG: store i32* [[VAR1]], i32** [[CP1]]
  // CK19-USE-DAG: store i{{.+}} [[CSVAL1:%[^,]+]], i{{.+}}* [[S1]]
  // CK19-USE-DAG: [[CSVAL1]] = {{mul nuw i64 %.*, 4|sext i32 .+ to i64}}

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-NOUSE-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-NOUSE-DAG: store i32* [[VAR0]], i32** [[CP0]]
  // CK19-NOUSE-DAG: store i{{.+}} [[CSVAL0:%[^,]+]], i{{.+}}* [[S0]]
  // CK19-NOUSE-DAG: [[CSVAL0]] = {{mul nuw i64 %.*, 4|sext i32 .+ to i64}}

  // CK19-USE: call void [[CALL16:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL16:@.+]]()
  #pragma omp target map(to:va)
  {
#ifdef USE
   va[50]++;
#endif
  }

  // Region 17
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|2}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[SIZE17]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[MTYPE17]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CP0]]

  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK19-USE-DAG: store i32* [[VAR1:%.+]], i32** [[CBP1]]
  // CK19-USE-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
  // CK19-USE-DAG: [[SEC1]] = getelementptr {{.*}}i32* [[VAR1]], i{{.+}} 20

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-NOUSE-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-NOUSE-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[VAR0]], i{{.+}} 20

  // CK19-USE: call void [[CALL17:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL17:@.+]]()
  #pragma omp target map(from:va[20:60])
  {
#ifdef USE
   va[50]++;
#endif
  }

  // Region 18
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|2}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[SIZE18]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[MTYPE18]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CP0]]

  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK19-USE-DAG: store i32* [[VAR1:%.+]], i32** [[CBP1]]
  // CK19-USE-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
  // CK19-USE-DAG: [[SEC1]] = getelementptr {{.*}}i32* [[VAR1]], i{{.+}} 0

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-NOUSE-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-NOUSE-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[VAR0]], i{{.+}} 0

  // CK19-USE: call void [[CALL18:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL18:@.+]]()
  #pragma omp target map(tofrom:va[:60])
  {
#ifdef USE
   va[50]++;
#endif
  }

  // Region 19
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|2}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[MTYPE19]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CP0]]
  // CK19-USE-DAG: store i{{.+}} {{8|4}}, i{{.+}}* [[S0]]

  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK19-USE-DAG: store i32* [[VAR1:%.+]], i32** [[CBP1]]
  // CK19-USE-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
  // CK19-USE-DAG: store i{{.+}} [[CSVAL1:%[^,]+]], i{{.+}}* [[S1]]
  // CK19-USE-DAG: [[CSVAL1]] = {{mul nuw i64 %.*, 4|sext i32 .+ to i64}}
  // CK19-USE-DAG: [[SEC1]] = getelementptr {{.*}}i32* [[VAR1]], i{{.+}} 0

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-NOUSE-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-NOUSE-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-NOUSE-DAG: store i{{.+}} [[CSVAL0:%[^,]+]], i{{.+}}* [[S0]]
  // CK19-NOUSE-DAG: [[CSVAL0]] = {{mul nuw i64 %.*, 4|sext i32 .+ to i64}}
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[VAR0]], i{{.+}} 0

  // CK19-USE: call void [[CALL19:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL19:@.+]]()
  #pragma omp target map(alloc:va[:])
  {
#ifdef USE
   va[50]++;
#endif
  }

  // Region 20
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|2}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[SIZE20]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[MTYPE20]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CP0]]

  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK19-USE-DAG: store i32* [[VAR1:%.+]], i32** [[CBP1]]
  // CK19-USE-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
  // CK19-USE-DAG: [[SEC1]] = getelementptr {{.*}}i32* [[VAR1]], i{{.+}} 15

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-NOUSE-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-NOUSE-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[VAR0]], i{{.+}} 15

  // CK19-USE: call void [[CALL20:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL20:@.+]]()
  #pragma omp target map(to:va[15])
  {
#ifdef USE
   va[15]++;
#endif
  }

  // Region 21
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|2}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[MTYPE21]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CP0]]
  // CK19-USE-DAG: store i{{.+}} {{8|4}}, i{{.+}}* [[S0]]

  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK19-USE-DAG: store i32* [[VAR1:%.+]], i32** [[CBP1]]
  // CK19-USE-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
  // CK19-USE-DAG: store i{{.+}} [[CSVAL1:%[^,]+]], i{{.+}}* [[S1]]
  // CK19-USE-DAG: [[CSVAL1]] = {{mul nuw i64 %.*, 4|sext i32 .+ to i64}}
  // CK19-USE-DAG: [[SEC1]] = getelementptr {{.*}}i32* [[VAR1]], i{{.+}} %{{.+}}

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-NOUSE-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-NOUSE-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-NOUSE-DAG: store i{{.+}} [[CSVAL0:%[^,]+]], i{{.+}}* [[S0]]
  // CK19-NOUSE-DAG: [[CSVAL0]] = {{mul nuw i64 %.*, 4|sext i32 .+ to i64}}
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[VAR0]], i{{.+}} %{{.+}}

  // CK19-USE: call void [[CALL21:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL21:@.+]]()
  #pragma omp target map(tofrom:va[ii:ii+23])
  {
#ifdef USE
   va[50]++;
#endif
  }

  // Region 22
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|2}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[SIZE22]], {{.+}}getelementptr {{.+}}[{{1|2}} x i{{.+}}]* [[MTYPE22]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] {{%.+}}, i[[Z]]* [[CP0]]

  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK19-USE-DAG: store i32* [[VAR1:%.+]], i32** [[CBP1]]
  // CK19-USE-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
  // CK19-USE-DAG: [[SEC1]] = getelementptr {{.*}}i32* [[VAR1]], i{{.+}} %{{.+}}

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-NOUSE-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-NOUSE-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[VAR0]], i{{.+}} %{{.+}}

  // CK19-USE: call void [[CALL22:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL22:@.+]]()
  #pragma omp target map(tofrom:va[ii])
  {
#ifdef USE
   va[15]++;
#endif
  }

  // Always.
  // Region 23
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE23]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE23]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK19-DAG: store i32* [[VAR0]], i32** [[CP0]]

  // CK19-USE: call void [[CALL23:@.+]](i32* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL23:@.+]]()
  #pragma omp target map(always, tofrom: a)
  {
#ifdef USE
   a++;
#endif
  }

  // Multidimensional arrays.
  int marr[4][5][6];
  int ***mptr;

  // Region 24
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE24]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE24]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [4 x [5 x [6 x i32]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [4 x [5 x [6 x i32]]]**
  // CK19-DAG: store [4 x [5 x [6 x i32]]]* [[VAR0:%.+]], [4 x [5 x [6 x i32]]]** [[CBP0]]
  // CK19-DAG: store [4 x [5 x [6 x i32]]]* [[VAR0]], [4 x [5 x [6 x i32]]]** [[CP0]]

  // CK19-USE: call void [[CALL24:@.+]]([4 x [5 x [6 x i32]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL24:@.+]]()
  #pragma omp target map(tofrom: marr)
  {
#ifdef USE
   marr[1][2][3]++;
#endif
  }

  // Region 25
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE25]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE25]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [4 x [5 x [6 x i32]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [4 x [5 x [6 x i32]]]* [[VAR0:%.+]], [4 x [5 x [6 x i32]]]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[6 x i32]* [[SEC00:[^,]+]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[SEC00]] = getelementptr {{.*}}[5 x [6 x i32]]* [[SEC000:[^,]+]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[SEC000]] = getelementptr {{.*}}[4 x [5 x [6 x i32]]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

  // CK19-USE: call void [[CALL25:@.+]]([4 x [5 x [6 x i32]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL25:@.+]]()
  #pragma omp target map(tofrom: marr[1][2][2:4])
  {
#ifdef USE
   marr[1][2][3]++;
#endif
  }

  // Region 26
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE26]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE26]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [4 x [5 x [6 x i32]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [4 x [5 x [6 x i32]]]* [[VAR0:%.+]], [4 x [5 x [6 x i32]]]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[6 x i32]* [[SEC00:[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[SEC00]] = getelementptr {{.*}}[5 x [6 x i32]]* [[SEC000:[^,]+]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[SEC000]] = getelementptr {{.*}}[4 x [5 x [6 x i32]]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

  // CK19-USE: call void [[CALL26:@.+]]([4 x [5 x [6 x i32]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL26:@.+]]()
  #pragma omp target map(tofrom: marr[1][2][:])
  {
#ifdef USE
   marr[1][2][3]++;
#endif
  }

  // Region 27
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE27]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE27]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [4 x [5 x [6 x i32]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [4 x [5 x [6 x i32]]]* [[VAR0:%.+]], [4 x [5 x [6 x i32]]]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[6 x i32]* [[SEC00:[^,]+]], i{{.+}} 0, i{{.+}} 3
  // CK19-DAG: [[SEC00]] = getelementptr {{.*}}[5 x [6 x i32]]* [[SEC000:[^,]+]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[SEC000]] = getelementptr {{.*}}[4 x [5 x [6 x i32]]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

  // CK19-USE: call void [[CALL27:@.+]]([4 x [5 x [6 x i32]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL27:@.+]]()
  #pragma omp target map(tofrom: marr[1][2][3])
  {
#ifdef USE
   marr[1][2][3]++;
#endif
  }

  // Region 28
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[SIZE28]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE28]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32****
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32****
  // CK19-DAG: store i32*** [[VAR0:%.+]], i32**** [[CBP0]]
  // CK19-DAG: store i32*** [[SEC0:%.+]], i32**** [[CP0]]
  // CK19-DAG: [[VAR0]] = load i32***, i32**** [[PTR:%[^,]+]],
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32*** [[SEC00:[^,]+]], i{{.+}} 1
  // CK19-DAG: [[SEC00]] = load i32***, i32**** [[PTR]],

  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32****
  // CK19-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32***
  // CK19-DAG: store i32*** [[SEC0]], i32**** [[CBP1]]
  // CK19-DAG: store i32** [[SEC1:%.+]], i32*** [[CP1]]
  // CK19-DAG: [[SEC1]] = getelementptr {{.*}}i32** [[SEC11:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC11]] = load i32**, i32*** [[SEC111:%[^,]+]],
  // CK19-DAG: [[SEC111]] = getelementptr {{.*}}i32*** [[SEC1111:[^,]+]], i{{.+}} 1
  // CK19-DAG: [[SEC1111]] = load i32***, i32**** [[PTR]],

  // CK19-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to i32***
  // CK19-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to i32**
  // CK19-DAG: store i32** [[SEC1]], i32*** [[CBP2]]
  // CK19-DAG: store i32* [[SEC2:%.+]], i32** [[CP2]]
  // CK19-DAG: [[SEC2]] = getelementptr {{.*}}i32* [[SEC22:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC22]] = load i32*, i32** [[SEC222:%[^,]+]],
  // CK19-DAG: [[SEC222]] = getelementptr {{.*}}i32** [[SEC2222:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC2222]] = load i32**, i32*** [[SEC22222:%[^,]+]],
  // CK19-DAG: [[SEC22222]] = getelementptr {{.*}}i32*** [[SEC222222:[^,]+]], i{{.+}} 1
  // CK19-DAG: [[SEC222222]] = load i32***, i32**** [[PTR]],

  // CK19-USE: call void [[CALL28:@.+]](i32*** {{[^,]+}})
  // CK19-NOUSE: call void [[CALL28:@.+]]()
  #pragma omp target map(tofrom: mptr[1][2][2:4])
  {
#ifdef USE
    mptr[1][2][3]++;
#endif
  }

  // Region 29
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[SIZE29]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE29]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32****
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32****
  // CK19-DAG: store i32*** [[VAR0:%.+]], i32**** [[CBP0]]
  // CK19-DAG: store i32*** [[SEC0:%.+]], i32**** [[CP0]]
  // CK19-DAG: [[VAR0]] = load i32***, i32**** [[PTR:%[^,]+]],
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32*** [[SEC00:[^,]+]], i{{.+}} 1
  // CK19-DAG: [[SEC00]] = load i32***, i32**** [[PTR]],

  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32****
  // CK19-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32***
  // CK19-DAG: store i32*** [[SEC0]], i32**** [[CBP1]]
  // CK19-DAG: store i32** [[SEC1:%.+]], i32*** [[CP1]]
  // CK19-DAG: [[SEC1]] = getelementptr {{.*}}i32** [[SEC11:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC11]] = load i32**, i32*** [[SEC111:%[^,]+]],
  // CK19-DAG: [[SEC111]] = getelementptr {{.*}}i32*** [[SEC1111:[^,]+]], i{{.+}} 1
  // CK19-DAG: [[SEC1111]] = load i32***, i32**** [[PTR]],

  // CK19-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to i32***
  // CK19-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to i32**
  // CK19-DAG: store i32** [[SEC1]], i32*** [[CBP2]]
  // CK19-DAG: store i32* [[SEC2:%.+]], i32** [[CP2]]
  // CK19-DAG: [[SEC2]] = getelementptr {{.*}}i32* [[SEC22:[^,]+]], i{{.+}} 3
  // CK19-DAG: [[SEC22]] = load i32*, i32** [[SEC222:%[^,]+]],
  // CK19-DAG: [[SEC222]] = getelementptr {{.*}}i32** [[SEC2222:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC2222]] = load i32**, i32*** [[SEC22222:%[^,]+]],
  // CK19-DAG: [[SEC22222]] = getelementptr {{.*}}i32*** [[SEC222222:[^,]+]], i{{.+}} 1
  // CK19-DAG: [[SEC222222]] = load i32***, i32**** [[PTR]],

  // CK19-USE: call void [[CALL29:@.+]](i32*** {{[^,]+}})
  // CK19-NOUSE: call void [[CALL29:@.+]]()
  #pragma omp target map(tofrom: mptr[1][2][3])
  {
#ifdef USE
    mptr[1][2][3]++;
#endif
  }

  // Multidimensional VLA.
  double mva[23][ii][ii+5];

  // Region 30
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|4}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[{{1|4}} x i{{.+}}]* [[MTYPE30]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] 23, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] 23, i[[Z]]* [[CP0]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S0]]
  //
  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[Z]]*
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] [[VAR1:%.+]], i[[Z]]* [[CBP1]]
  // CK19-USE-DAG: store i[[Z]] [[VAR11:%.+]], i[[Z]]* [[CP1]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S1]]
  // CK19-64-USE-DAG: [[VAR1]] = zext i32 %{{[^,]+}} to i64
  // CK19-64-USE-DAG: [[VAR11]] = zext i32 %{{[^,]+}} to i64
  //
  // CK19-USE-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to i[[Z]]*
  // CK19-USE-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] [[VAR2:%.+]], i[[Z]]* [[CBP2]]
  // CK19-USE-DAG: store i[[Z]] [[VAR22:%.+]], i[[Z]]* [[CP2]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S2]]
  // CK19-64-USE-DAG: [[VAR2]] = zext i32 %{{[^,]+}} to i64
  // CK19-64-USE-DAG: [[VAR22]] = zext i32 %{{[^,]+}} to i64
  //
  // CK19-USE-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
  // CK19-USE-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
  // CK19-USE-DAG: [[S3:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 3
  // CK19-USE-DAG: [[CBP3:%.+]] = bitcast i8** [[BP3]] to double**
  // CK19-USE-DAG: [[CP3:%.+]] = bitcast i8** [[P3]] to double**
  // CK19-USE-DAG: store double* [[VAR3:%.+]], double** [[CBP3]]
  // CK19-USE-DAG: store double* [[VAR3]], double** [[CP3]]
  // CK19-USE-DAG: store i64 [[CSVAL3:%[^,]+]], i64* [[S3]]
  // CK19-USE-DAG: [[CSVAL3]] = {{mul nuw i64 %[^,]+, 8|sext i32 .+ to i64}}

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to double**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to double**
  // CK19-NOUSE-DAG: store double* [[VAR0:%.+]], double** [[CBP0]]
  // CK19-NOUSE-DAG: store double* [[VAR0]], double** [[CP0]]
  // CK19-NOUSE-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  // CK19-NOUSE-DAG: [[CSVAL0]] = {{mul nuw i64 %[^,]+, 8|sext i32 .+ to i64}}

  // CK19-USE: call void [[CALL30:@.+]](i[[Z]] 23, i[[Z]] %{{[^,]+}}, i[[Z]] %{{[^,]+}}, double* %{{[^,]+}})
  // CK19-NOUSE: call void [[CALL30:@.+]]()
  #pragma omp target map(tofrom: mva)
  {
#ifdef USE
    mva[1][2][3]++;
#endif
  }

  // Region 31
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|4}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[{{1|4}} x i{{.+}}]* [[SIZE31]], {{.+}}getelementptr {{.+}}[{{1|4}} x i{{.+}}]* [[MTYPE31]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  //
  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] 23, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] 23, i[[Z]]* [[CP0]]
  //
  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[Z]]*
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] [[VAR1:%.+]], i[[Z]]* [[CBP1]]
  // CK19-USE-DAG: store i[[Z]] [[VAR11:%.+]], i[[Z]]* [[CP1]]
  //
  // CK19-USE-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to i[[Z]]*
  // CK19-USE-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] [[VAR2:%.+]], i[[Z]]* [[CBP2]]
  // CK19-USE-DAG: store i[[Z]] [[VAR22:%.+]], i[[Z]]* [[CP2]]
  //
  // CK19-USE-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
  // CK19-USE-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
  // CK19-USE-DAG: [[CBP3:%.+]] = bitcast i8** [[BP3]] to double**
  // CK19-USE-DAG: [[CP3:%.+]] = bitcast i8** [[P3]] to double**
  // CK19-USE-DAG: store double* [[VAR3:%.+]], double** [[CBP3]]
  // CK19-USE-DAG: store double* [[SEC3:%.+]], double** [[CP3]]
  // CK19-USE-DAG: [[SEC3]] = getelementptr {{.*}}double* [[SEC33:%.+]], i[[Z]] 0
  // CK19-USE-DAG: [[SEC33]] = getelementptr {{.*}}double* [[SEC333:%.+]], i[[Z]] [[IDX3:%.+]]
  // CK19-USE-DAG: [[IDX3]] = mul nsw i[[Z]] %{{[^,]+}}, %{{[^,]+}}
  // CK19-USE-DAG: [[SEC333]] = getelementptr {{.*}}double* [[VAR3]], i[[Z]] [[IDX33:%.+]]
  // CK19-USE-DAG: [[IDX33]] = mul nsw i[[Z]] 1, %{{[^,]+}}

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to double**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to double**
  // CK19-NOUSE-DAG: store double* [[VAR0:%.+]], double** [[CBP0]]
  // CK19-NOUSE-DAG: store double* [[SEC0:%.+]], double** [[CP0]]
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.*}}double* [[SEC00:%.+]], i[[Z:64|32]] 0
  // CK19-NOUSE-DAG: [[SEC00]] = getelementptr {{.*}}double* [[SEC000:%.+]], i[[Z]] [[IDX0:%.+]]
  // CK19-NOUSE-DAG: [[IDX0]] = mul nsw i[[Z]] %{{[^,]+}}, %{{[^,]+}}
  // CK19-NOUSE-DAG: [[SEC000]] = getelementptr {{.*}}double* [[VAR0]], i[[Z]] [[IDX00:%.+]]
  // CK19-NOUSE-DAG: [[IDX00]] = mul nsw i[[Z]] 1, %{{[^,]+}}

  // CK19-USE: call void [[CALL31:@.+]](i[[Z]] 23, i[[Z]] %{{[^,]+}}, i[[Z]] %{{[^,]+}}, double* %{{[^,]+}})
  // CK19-NOUSE: call void [[CALL31:@.+]]()
  #pragma omp target map(tofrom: mva[1][ii-2][:5])
  {
#ifdef USE
    mva[1][2][3]++;
#endif
  }

  // Multidimensional array sections.
  double marras[11][12][13];
  double mvlaas[11][ii][13];
  double ***mptras;

  // Region 32
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE32]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE32]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to  [11 x [12 x [13 x double]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [11 x [12 x [13 x double]]]**
  // CK19-DAG: store [11 x [12 x [13 x double]]]* [[VAR0:%.+]], [11 x [12 x [13 x double]]]** [[CBP0]]
  // CK19-DAG: store [11 x [12 x [13 x double]]]* [[VAR0]], [11 x [12 x [13 x double]]]** [[CP0]]

  // CK19-USE: call void [[CALL32:@.+]]([11 x [12 x [13 x double]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL32:@.+]]()
  #pragma omp target map(marras)
  {
#ifdef USE
    marras[1][2][3]++;
#endif
  }

  // Region 33
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE33]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE33]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to  [11 x [12 x [13 x double]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [12 x [13 x double]]**
  // CK19-DAG: store [11 x [12 x [13 x double]]]* [[VAR0:%.+]], [11 x [12 x [13 x double]]]** [[CBP0]]
  // CK19-DAG: store [12 x [13 x double]]* [[SEC0:%.+]], [12 x [13 x double]]** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.+}}[11 x [12 x [13 x double]]]* [[VAR0]], i[[Z]] 0, i[[Z]] 0

  // CK19-USE: call void [[CALL33:@.+]]([11 x [12 x [13 x double]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL33:@.+]]()
  #pragma omp target map(marras[:])
  {
#ifdef USE
    marras[1][2][3]++;
#endif
  }

  // Region 34
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE34]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE34]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to  [11 x [12 x [13 x double]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [12 x [13 x double]]**
  // CK19-DAG: store [11 x [12 x [13 x double]]]* [[VAR0:%.+]], [11 x [12 x [13 x double]]]** [[CBP0]]
  // CK19-DAG: store [12 x [13 x double]]* [[SEC0:%.+]], [12 x [13 x double]]** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.+}}[11 x [12 x [13 x double]]]* [[VAR0]], i[[Z]] 0, i[[Z]] 0

  // CK19-USE: call void [[CALL34:@.+]]([11 x [12 x [13 x double]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL34:@.+]]()
  #pragma omp target map(marras[:][:][:])
  {
#ifdef USE
    marras[1][2][3]++;
#endif
  }

  // Region 35
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE35]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0

  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to  [11 x [12 x [13 x double]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [13 x double]**
  // CK19-DAG: store [11 x [12 x [13 x double]]]* [[VAR0:%.+]], [11 x [12 x [13 x double]]]** [[CBP0]]
  // CK19-DAG: store [13 x double]* [[SEC0:%.+]], [13 x double]** [[CP0]]
  // CK19-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.+}}[12 x [13 x double]]* [[SEC00:%[^,]+]], i[[Z]] 0, i[[Z]] 0
  // CK19-DAG: [[SEC00]] = getelementptr {{.+}}[11 x [12 x [13 x double]]]* [[VAR0]], i[[Z]] 0, i[[Z]] 1
  // CK19-DAG: [[CSVAL0]] = {{mul nuw i64 %[^,]+, 104|sext i32 .+ to i64}}

  // CK19-USE: call void [[CALL35:@.+]]([11 x [12 x [13 x double]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL35:@.+]]()
  #pragma omp target map(marras[1][:ii][:])
  {
#ifdef USE
    marras[1][2][3]++;
#endif
  }

  // Region 36
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE36]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE36]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to  [11 x [12 x [13 x double]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [13 x double]**
  // CK19-DAG: store [11 x [12 x [13 x double]]]* [[VAR0:%.+]], [11 x [12 x [13 x double]]]** [[CBP0]]
  // CK19-DAG: store [13 x double]* [[SEC0:%.+]], [13 x double]** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.+}}[13 x double]* [[SEC00:%[^,]+]], i{{.+}} 0
  // CK19-DAG: [[SEC00]] = getelementptr {{.+}}[12 x [13 x double]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[SEC000]] = getelementptr {{.+}}[11 x [12 x [13 x double]]]* [[VAR0]], i{{.+}} 0, i{{.+}} 0

  // CK19-USE: call void [[CALL36:@.+]]([11 x [12 x [13 x double]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL36:@.+]]()
  #pragma omp target map(marras[:1][:2][:13])
  {
#ifdef USE
    marras[1][2][3]++;
#endif
  }

  // Region 37
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|3}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[{{1|3}} x i{{.+}}]* [[MTYPE37]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CP0]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S0]]
  //
  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[Z]]*
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] [[VAR1:%.+]], i[[Z]]* [[CBP1]]
  // CK19-USE-DAG: store i[[Z]] [[VAR11:%.+]], i[[Z]]* [[CP1]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S1]]
  //
  // CK19-USE-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [13 x double]**
  // CK19-USE-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to [13 x double]**
  // CK19-USE-DAG: store [13 x double]* [[VAR2:%.+]], [13 x double]** [[CBP2]]
  // CK19-USE-DAG: store [13 x double]* [[VAR2]], [13 x double]** [[CP2]]
  // CK19-USE-DAG: store i64 [[CSVAL2:%[^,]+]], i64* [[S2]]
  // CK19-USE-DAG: [[CSVAL2]] = {{mul nuw i64 %[^,]+, 104|sext i32 .+ to i64}}

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [13 x double]**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [13 x double]**
  // CK19-NOUSE-DAG: store [13 x double]* [[VAR0:%.+]], [13 x double]** [[CBP0]]
  // CK19-NOUSE-DAG: store [13 x double]* [[VAR0]], [13 x double]** [[CP0]]
  // CK19-NOUSE-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  // CK19-NOUSE-DAG: [[CSVAL0]] = {{mul nuw i64 %[^,]+, 104|sext i32 .+ to i64}}

  // CK19-USE: call void [[CALL37:@.+]](i[[Z]] 11, i[[Z]] %{{[^,]+}}, [13 x double]* %{{[^,]+}})
  // CK19-NOUSE: call void [[CALL37:@.+]]()
  #pragma omp target map(mvlaas)
  {
#ifdef USE
    mvlaas[1][2][3]++;
#endif
  }

  // Region 38
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|3}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[{{1|3}} x i{{.+}}]* [[MTYPE38]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CP0]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S0]]
  //
  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[Z]]*
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] [[VAR1:%.+]], i[[Z]]* [[CBP1]]
  // CK19-USE-DAG: store i[[Z]] [[VAR11:%.+]], i[[Z]]* [[CP1]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S1]]
  //
  // CK19-USE-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [13 x double]**
  // CK19-USE-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to [13 x double]**
  // CK19-USE-DAG: store [13 x double]* [[VAR2:%.+]], [13 x double]** [[CBP2]]
  // CK19-USE-DAG: store [13 x double]* [[SEC2:%.+]], [13 x double]** [[CP2]]
  // CK19-USE-DAG: store i64 [[CSVAL2:%[^,]+]], i64* [[S2]]
  // CK19-USE-DAG: [[SEC2]] = getelementptr {{.+}}[13 x double]* [[VAR2]], i[[Z]] [[SEC22:%[^,]+]]
  // CK19-USE-DAG: [[SEC22]] = mul nsw i[[Z]] 0, %{{[^,]+}}
  // CK19-USE-DAG: [[CSVAL2]] = {{mul nuw i64 %[^,]+, 104|sext i32 .+ to i64}}

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [13 x double]**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [13 x double]**
  // CK19-NOUSE-DAG: store [13 x double]* [[VAR0:%.+]], [13 x double]** [[CBP0]]
  // CK19-NOUSE-DAG: store [13 x double]* [[SEC0:%.+]], [13 x double]** [[CP0]]
  // CK19-NOUSE-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.+}}[13 x double]* [[VAR0]], i[[Z]] [[SEC00:%[^,]+]]
  // CK19-NOUSE-DAG: [[SEC00]] = mul nsw i[[Z]] 0, %{{[^,]+}}
  // CK19-NOUSE-DAG: [[CSVAL0]] = {{mul nuw i64 %[^,]+, 104|sext i32 .+ to i64}}

  // CK19-USE: call void [[CALL38:@.+]](i[[Z]] 11, i[[Z]] %{{[^,]+}}, [13 x double]* %{{[^,]+}})
  // CK19-NOUSE: call void [[CALL38:@.+]]()
  #pragma omp target map(mvlaas[:])
  {
#ifdef USE
    mvlaas[1][2][3]++;
#endif
  }

  // Region 39
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|3}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[{{1|3}} x i{{.+}}]* [[MTYPE39]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CP0]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S0]]
  //
  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[Z]]*
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] [[VAR1:%.+]], i[[Z]]* [[CBP1]]
  // CK19-USE-DAG: store i[[Z]] [[VAR11:%.+]], i[[Z]]* [[CP1]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S1]]
  //
  // CK19-USE-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [13 x double]**
  // CK19-USE-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to [13 x double]**
  // CK19-USE-DAG: store [13 x double]* [[VAR2:%.+]], [13 x double]** [[CBP2]]
  // CK19-USE-DAG: store [13 x double]* [[SEC2:%.+]], [13 x double]** [[CP2]]
  // CK19-USE-DAG: store i64 [[CSVAL2:%[^,]+]], i64* [[S2]]
  // CK19-USE-DAG: [[SEC2]] = getelementptr {{.+}}[13 x double]* [[VAR2]], i[[Z]] [[SEC22:%[^,]+]]
  // CK19-USE-DAG: [[SEC22]] = mul nsw i[[Z]] 0, %{{[^,]+}}
  // CK19-USE-DAG: [[CSVAL2]] = {{mul nuw i64 %[^,]+, 104|sext i32 .+ to i64}}

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [13 x double]**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [13 x double]**
  // CK19-NOUSE-DAG: store [13 x double]* [[VAR0:%.+]], [13 x double]** [[CBP0]]
  // CK19-NOUSE-DAG: store [13 x double]* [[SEC0:%.+]], [13 x double]** [[CP0]]
  // CK19-NOUSE-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.+}}[13 x double]* [[VAR0]], i[[Z]] [[SEC00:%[^,]+]]
  // CK19-NOUSE-DAG: [[SEC00]] = mul nsw i[[Z]] 0, %{{[^,]+}}
  // CK19-NOUSE-DAG: [[CSVAL0]] = {{mul nuw i64 %[^,]+, 104|sext i32 .+ to i64}}

  // CK19-USE: call void [[CALL39:@.+]](i[[Z]] 11, i[[Z]] %{{[^,]+}}, [13 x double]* %{{[^,]+}})
  // CK19-NOUSE: call void [[CALL39:@.+]]()
  #pragma omp target map(mvlaas[:][:][:])
  {
#ifdef USE
    mvlaas[1][2][3]++;
#endif
  }

  // Region 40
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|3}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[{{1|3}} x i{{.+}}]* [[MTYPE40]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CP0]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S0]]
  //
  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[Z]]*
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] [[VAR1:%.+]], i[[Z]]* [[CBP1]]
  // CK19-USE-DAG: store i[[Z]] [[VAR11:%.+]], i[[Z]]* [[CP1]]
  // CK19-USE-DAG: store i64 {{8|4}}, i64* [[S1]]
  //
  // CK19-USE-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [13 x double]**
  // CK19-USE-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to [13 x double]**
  // CK19-USE-DAG: store [13 x double]* [[VAR2:%.+]], [13 x double]** [[CBP2]]
  // CK19-USE-DAG: store [13 x double]* [[SEC2:%.+]], [13 x double]** [[CP2]]
  // CK19-USE-DAG: store i64 [[CSVAL2:%[^,]+]], i64* [[S2]]
  // CK19-USE-DAG: [[SEC2]] = getelementptr {{.+}}[13 x double]* [[SEC22:%[^,]+]], i[[Z]] 0
  // CK19-USE-DAG: [[SEC22]] = getelementptr {{.+}}[13 x double]* [[VAR2]], i[[Z]] [[SEC222:%[^,]+]]
  // CK19-USE-DAG: [[SEC222]] = mul nsw i[[Z]] 1, %{{[^,]+}}

  // CK19-NOUSE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-NOUSE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [13 x double]**
  // CK19-NOUSE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [13 x double]**
  // CK19-NOUSE-DAG: store [13 x double]* [[VAR0:%.+]], [13 x double]** [[CBP0]]
  // CK19-NOUSE-DAG: store [13 x double]* [[SEC0:%.+]], [13 x double]** [[CP0]]
  // CK19-NOUSE-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  // CK19-NOUSE-DAG: [[SEC0]] = getelementptr {{.+}}[13 x double]* [[SEC00:%[^,]+]], i[[Z]] 0
  // CK19-NOUSE-DAG: [[SEC00]] = getelementptr {{.+}}[13 x double]* [[VAR0]], i[[Z]] [[SEC000:%[^,]+]]
  // CK19-NOUSE-DAG: [[SEC000]] = mul nsw i[[Z]] 1, %{{[^,]+}}

  // CK19-USE: call void [[CALL40:@.+]](i[[Z]] 11, i[[Z]] %{{[^,]+}}, [13 x double]* %{{[^,]+}})
  // CK19-NOUSE: call void [[CALL40:@.+]]()
  #pragma omp target map(mvlaas[1][:ii][:])
  {
#ifdef USE
    mvlaas[1][2][3]++;
#endif
  }

  // Region 41
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 {{1|3}}, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[{{1|3}} x i{{.+}}]* [[SIZE41]], {{.+}}getelementptr {{.+}}[{{1|3}} x i{{.+}}]* [[MTYPE41]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  //
  // CK19-USE-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i[[Z]]*
  // CK19-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CBP0]]
  // CK19-USE-DAG: store i[[Z]] 11, i[[Z]]* [[CP0]]
  //
  // CK19-USE-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-USE-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[Z]]*
  // CK19-USE-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[Z]]*
  // CK19-USE-DAG: store i[[Z]] [[VAR1:%.+]], i[[Z]]* [[CBP1]]
  // CK19-USE-DAG: store i[[Z]] [[VAR11:%.+]], i[[Z]]* [[CP1]]
  //
  // CK19-USE-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-USE-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [13 x double]**
  // CK19-USE-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to [13 x double]**
  // CK19-USE-DAG: store [13 x double]* [[VAR2:%.+]], [13 x double]** [[CBP2]]
  // CK19-USE-DAG: store [13 x double]* [[SEC2:%.+]], [13 x double]** [[CP2]]
  // CK19-USE-DAG: [[SEC2]] = getelementptr {{.+}}[13 x double]* [[SEC22:%[^,]+]], i[[Z]] 0
  // CK19-USE-DAG: [[SEC22]] = getelementptr {{.+}}[13 x double]* [[VAR2]], i[[Z]] [[SEC222:%[^,]+]]
  // CK19-USE-DAG: [[SEC222]] = mul nsw i[[Z]] 0, %{{[^,]+}}
  // CK19-USE-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2

  // CK19-NO-USE-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-NO-USE-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [13 x double]**
  // CK19-NO-USE-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [13 x double]**
  // CK19-NO-USE-DAG: store [13 x double]* [[VAR0:%.+]], [13 x double]** [[CBP0]]
  // CK19-NO-USE-DAG: store [13 x double]* [[SEC0:%.+]], [13 x double]** [[CP0]]
  // CK19-NO-USE-DAG: [[SEC0]] = getelementptr {{.+}}[13 x double]* [[SEC00:%[^,]+]], i[[Z]] 0
  // CK19-NO-USE-DAG: [[SEC00]] = getelementptr {{.+}}[13 x double]* [[VAR0]], i[[Z]] [[SEC000:%[^,]+]]
  // CK19-NO-USE-DAG: [[SEC000]] = mul nsw i[[Z]] 0, %{{[^,]+}}

  // CK19-USE: call void [[CALL41:@.+]](i[[Z]] 11, i[[Z]] %{{[^,]+}}, [13 x double]* %{{[^,]+}})
  // CK19-NOUSE: call void [[CALL41:@.+]]()
  #pragma omp target map(mvlaas[:1][:2][:13])
  {
#ifdef USE
    mvlaas[1][2][3]++;
#endif
  }

  // Region 42
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[SIZE42]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE42]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to double****
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to double****
  // CK19-DAG: store double*** [[VAR0:%.+]], double**** [[CBP0]]
  // CK19-DAG: store double*** [[SEC0:%.+]], double**** [[CP0]]
  // CK19-DAG: [[VAR0]] = load double***, double**** [[PTR:%[^,]+]],
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}double*** [[SEC00:[^,]+]], i{{.+}} 0
  // CK19-DAG: [[SEC00]] = load double***, double**** [[PTR]],

  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double****
  // CK19-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double***
  // CK19-DAG: store double*** [[SEC0]], double**** [[CBP1]]
  // CK19-DAG: store double** [[SEC1:%.+]], double*** [[CP1]]
  // CK19-DAG: [[SEC1]] = getelementptr {{.*}}double** [[SEC11:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC11]] = load double**, double*** [[SEC111:%[^,]+]],
  // CK19-DAG: [[SEC111]] = getelementptr {{.*}}double*** [[SEC1111:[^,]+]], i{{.+}} 0
  // CK19-DAG: [[SEC1111]] = load double***, double**** [[PTR]],

  // CK19-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to double***
  // CK19-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to double**
  // CK19-DAG: store double** [[SEC1]], double*** [[CBP2]]
  // CK19-DAG: store double* [[SEC2:%.+]], double** [[CP2]]
  // CK19-DAG: [[SEC2]] = getelementptr {{.*}}double* [[SEC22:[^,]+]], i{{.+}} 0
  // CK19-DAG: [[SEC22]] = load double*, double** [[SEC222:%[^,]+]],
  // CK19-DAG: [[SEC222]] = getelementptr {{.*}}double** [[SEC2222:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC2222]] = load double**, double*** [[SEC22222:%[^,]+]],
  // CK19-DAG: [[SEC22222]] = getelementptr {{.*}}double*** [[SEC222222:[^,]+]], i{{.+}} 0
  // CK19-DAG: [[SEC222222]] = load double***, double**** [[PTR]],

  // CK19-USE: call void [[CALL42:@.+]](double*** {{[^,]+}})
  // CK19-NOUSE: call void [[CALL42:@.+]]()
  #pragma omp target map(mptras[:1][2][:13])
  {
#ifdef USE
    mptras[1][2][3]++;
#endif
  }

  // Region 43 - the memory is not contiguous for this map - will map the whole last dimension.
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE43]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0

  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [11 x [12 x [13 x double]]]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [13 x double]**
  // CK19-DAG: store [11 x [12 x [13 x double]]]* [[VAR0:%.+]], [11 x [12 x [13 x double]]]** [[CBP0]]
  // CK19-DAG: store [13 x double]* [[SEC0:%.+]], [13 x double]** [[CP0]]
  // CK19-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.+}}[12 x [13 x double]]* [[SEC00:%[^,]+]], i[[Z]] 0, i[[Z]] 0
  // CK19-DAG: [[SEC00]] = getelementptr {{.+}}[11 x [12 x [13 x double]]]* [[VAR0]], i[[Z]] 0, i[[Z]] 1
  // CK19-DAG: [[CSVAL0]] = {{mul nuw i64 %[^,]+, 104|sext i32 .+ to i64}}

  // CK19-USE: call void [[CALL43:@.+]]([11 x [12 x [13 x double]]]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL43:@.+]]()
  #pragma omp target map(marras[1][:ii][1:])
  {
#ifdef USE
    marras[1][2][3]++;
#endif
  }

  // Region 44
  // CK19-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE44]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE44]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK19-DAG: store [100 x i32]* [[VAR0:%.+]], [100 x i32]** [[CBP0]]
  // CK19-DAG: store i32* [[SEC0:%[^,]+]], i32** [[CP0]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} 20

  // CK19-USE: call void [[CALL44:@.+]]([100 x i32]* {{[^,]+}})
  // CK19-NOUSE: call void [[CALL44:@.+]]()
  #pragma omp target map(from:arra[20:])
  {
#ifdef USE
    arra[50]++;
#endif
  }

}

// CK19: define {{.+}}[[CALL00]]
// CK19: define {{.+}}[[CALL01]]
// CK19: define {{.+}}[[CALL02]]
// CK19: define {{.+}}[[CALL03]]
// CK19: define {{.+}}[[CALL04]]
// CK19: define {{.+}}[[CALL05]]
// CK19: define {{.+}}[[CALL06]]
// CK19: define {{.+}}[[CALL07]]
// CK19: define {{.+}}[[CALL08]]
// CK19: define {{.+}}[[CALL09]]
// CK19: define {{.+}}[[CALL10]]
// CK19: define {{.+}}[[CALL11]]
// CK19: define {{.+}}[[CALL12]]
// CK19: define {{.+}}[[CALL13]]
// CK19: define {{.+}}[[CALL14]]
// CK19: define {{.+}}[[CALL15]]
// CK19: define {{.+}}[[CALL16]]
// CK19: define {{.+}}[[CALL17]]
// CK19: define {{.+}}[[CALL18]]
// CK19: define {{.+}}[[CALL19]]
// CK19: define {{.+}}[[CALL20]]
// CK19: define {{.+}}[[CALL21]]
// CK19: define {{.+}}[[CALL22]]
// CK19: define {{.+}}[[CALL23]]
// CK19: define {{.+}}[[CALL24]]
// CK19: define {{.+}}[[CALL25]]
// CK19: define {{.+}}[[CALL26]]
// CK19: define {{.+}}[[CALL27]]
// CK19: define {{.+}}[[CALL28]]
// CK19: define {{.+}}[[CALL29]]
// CK19: define {{.+}}[[CALL30]]
// CK19: define {{.+}}[[CALL31]]
// CK19: define {{.+}}[[CALL32]]
// CK19: define {{.+}}[[CALL33]]
// CK19: define {{.+}}[[CALL34]]
// CK19: define {{.+}}[[CALL35]]
// CK19: define {{.+}}[[CALL36]]
// CK19: define {{.+}}[[CALL37]]
// CK19: define {{.+}}[[CALL38]]
// CK19: define {{.+}}[[CALL39]]
// CK19: define {{.+}}[[CALL40]]
// CK19: define {{.+}}[[CALL41]]
// CK19: define {{.+}}[[CALL42]]
// CK19: define {{.+}}[[CALL43]]
// CK19: define {{.+}}[[CALL44]]

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-32
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-32

// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-32
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-32

// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-32
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK20 --check-prefix CK20-32

// RUN: %clang_cc1 -DCK20 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK20 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK20 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK20 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY19 %s
// SIMD-ONLY19-NOT: {{__kmpc|__tgt}}
#ifdef CK20

// CK20-LABEL: @.__omp_offloading_{{.*}}explicit_maps_references_and_function_args{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK20: [[SIZE00:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK20: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 33]

// CK20-LABEL: @.__omp_offloading_{{.*}}explicit_maps_references_and_function_args{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK20: [[SIZE01:@.+]] = private {{.*}}constant [1 x i64] [i64 20]
// CK20: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 33]

// CK20-LABEL: @.__omp_offloading_{{.*}}explicit_maps_references_and_function_args{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK20: [[SIZE02:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK20: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i64] [i64 34]

// CK20-LABEL: @.__omp_offloading_{{.*}}explicit_maps_references_and_function_args{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK20: [[SIZE03:@.+]] = private {{.*}}constant [1 x i64] [i64 12]
// CK20: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i64] [i64 34]

// CK20-LABEL: explicit_maps_references_and_function_args{{.*}}(
void explicit_maps_references_and_function_args (int a, float b, int (&c)[10], float *d){

  int &aa = a;
  float &bb = b;
  int (&cc)[10] = c;
  float *&dd = d;

  // Region 00
  // CK20-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK20-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK20-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK20-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK20-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK20-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
  // CK20-DAG: store i32* [[RVAR00:%.+]], i32** [[CP0]]
  // CK20-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK20-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK20: call void [[CALL00:@.+]](i32* {{[^,]+}})
  #pragma omp target map(to:aa)
  {
    aa += 1;
  }

  // Region 01
  // CK20-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null)
  // CK20-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK20-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK20-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [10 x i32]**
  // CK20-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK20-DAG: store [10 x i32]* [[RVAR0:%.+]], [10 x i32]** [[CBP0]]
  // CK20-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK20-DAG: [[SEC0]] = getelementptr {{.*}}[10 x i32]* [[RVAR00:%.+]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[RVAR0]] = load [10 x i32]*, [10 x i32]** [[VAR0:%[^,]+]]
  // CK20-DAG: [[RVAR00]] = load [10 x i32]*, [10 x i32]** [[VAR0]]

  // CK20: call void [[CALL01:@.+]]([10 x i32]* {{[^,]+}})
  #pragma omp target map(to:cc[:5])
  {
    cc[3] += 1;
  }

  // Region 02
  // CK20-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}}, i8** null)
  // CK20-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK20-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK20-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK20-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
  // CK20-DAG: store float* [[VAR0:%.+]], float** [[CBP0]]
  // CK20-DAG: store float* [[VAR0]], float** [[CP0]]

  // CK20: call void [[CALL02:@.+]](float* {{[^,]+}})
  #pragma omp target map(from:b)
  {
    b += 1.0f;
  }

  // Region 03
  // CK20-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}}, i8** null)
  // CK20-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK20-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK20-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK20-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
  // CK20-DAG: store float* [[RVAR0:%.+]], float** [[CBP0]]
  // CK20-DAG: store float* [[SEC0:%.+]], float** [[CP0]]
  // CK20-DAG: [[RVAR0]] = load float*, float** [[VAR0:%[^,]+]]
  // CK20-DAG: [[SEC0]] = getelementptr {{.*}}float* [[RVAR00:%.+]], i{{.+}} 2
  // CK20-DAG: [[RVAR00]] = load float*, float** [[VAR0]]

  // CK20: call void [[CALL03:@.+]](float* {{[^,]+}})
  #pragma omp target map(from:d[2:3])
  {
    d[2] += 1.0f;
  }
}

// CK20: define {{.+}}[[CALL00]]
// CK20: define {{.+}}[[CALL01]]
// CK20: define {{.+}}[[CALL02]]
// CK20: define {{.+}}[[CALL03]]

#endif
///==========================================================================///
// RUN: %clang_cc1 -DUSE -DCK21 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK21,CK21-64,CK21-USE
// RUN: %clang_cc1 -DUSE -DCK21 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-64,CK21-USE
// RUN: %clang_cc1 -DUSE -DCK21 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-USE
// RUN: %clang_cc1 -DUSE -DCK21 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-USE

// RUN: %clang_cc1 -DUSE -DCK21 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK21,CK21-64,CK21-USE
// RUN: %clang_cc1 -DUSE -DCK21 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-64,CK21-USE
// RUN: %clang_cc1 -DUSE -DCK21 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-USE
// RUN: %clang_cc1 -DUSE -DCK21 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-USE

// RUN: %clang_cc1 -DUSE -DCK21 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK21,CK21-64,CK21-USE
// RUN: %clang_cc1 -DUSE -DCK21 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-64,CK21-USE
// RUN: %clang_cc1 -DUSE -DCK21 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-USE
// RUN: %clang_cc1 -DUSE -DCK21 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-USE

// RUN: %clang_cc1 -DUSE -DCK21 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY20 %s
// RUN: %clang_cc1 -DUSE -DCK21 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY20 %s
// RUN: %clang_cc1 -DUSE -DCK21 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY20 %s
// RUN: %clang_cc1 -DUSE -DCK21 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY20 %s

// RUN: %clang_cc1 -DCK21 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK21,CK21-64,CK21-NOUSE
// RUN: %clang_cc1 -DCK21 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-64,CK21-NOUSE
// RUN: %clang_cc1 -DCK21 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-NOUSE
// RUN: %clang_cc1 -DCK21 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-NOUSE

// RUN: %clang_cc1 -DCK21 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK21,CK21-64,CK21-NOUSE
// RUN: %clang_cc1 -DCK21 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-64,CK21-NOUSE
// RUN: %clang_cc1 -DCK21 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-NOUSE
// RUN: %clang_cc1 -DCK21 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-NOUSE

// RUN: %clang_cc1 -DCK21 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK21,CK21-64,CK21-NOUSE
// RUN: %clang_cc1 -DCK21 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-64,CK21-NOUSE
// RUN: %clang_cc1 -DCK21 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-NOUSE
// RUN: %clang_cc1 -DCK21 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK21,CK21-32,CK21-NOUSE

// RUN: %clang_cc1 -DCK21 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY20 %s
// RUN: %clang_cc1 -DCK21 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY20 %s
// RUN: %clang_cc1 -DCK21 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY20 %s
// RUN: %clang_cc1 -DCK21 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY20 %s

// SIMD-ONLY20-NOT: {{__kmpc|__tgt}}
#ifdef CK21
// CK21: [[ST:%.+]] = type { i32, i32, float* }

// CK21-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK21: [[MTYPE00:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710659]

// CK21-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK21: [[SIZE01:@.+]] = private {{.*}}constant [1 x i64] [i64 492]
// CK21: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK21-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK21: [[MTYPE02:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710674]

// CK21-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK21: [[SIZE03:@.+]] = private {{.*}}constant [1 x i64] [i64 492]
// CK21: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i64] [i64 34]

// CK21-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK21: [[SIZE04:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK21: [[MTYPE04:@.+]] = private {{.*}}constant [1 x i64] [i64 34]

// CK21-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK21: [[MTYPE05:@.+]] = private {{.*}}constant [3 x i64] [i64 32, i64 281474976710659, i64 281474976710659]

// CK21-LABEL: explicit_maps_template_args_and_members{{.*}}(

template <int X, typename T>
struct CC {
  T A;
  int A2;
  float *B;

  int foo(T arg) {
    float la[X];
    T *lb;

    // Region 00
    // CK21-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
    // CK21-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK21-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
    // CK21-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

    // CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
    // CK21-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
    // CK21-DAG: store [[ST]]* [[VAR0:%.+]], [[ST]]** [[CBP0]]
    // CK21-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
    // CK21-DAG: store i64 {{%.+}}, i64* [[S0]]
    // CK21-DAG: [[SEC0]] = getelementptr {{.*}}[[ST]]* [[VAR0:%.+]], i{{.+}} 0, i{{.+}} 0

    // CK21-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
    // CK21-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
    // CK21-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
    // CK21-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[ST]]**
    // CK21-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
    // CK21-DAG: store [[ST]]* [[VAR1:%.+]], [[ST]]** [[CBP1]]
    // CK21-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
    // CK21-DAG: store i64 {{.+}}, i64* [[S1]]
    // CK21-DAG: [[SEC1]] = getelementptr {{.*}}[[ST]]* [[VAR1:%.+]], i{{.+}} 0, i{{.+}} 0

    // CK21-USE: call void [[CALL00:@.+]]([[ST]]* {{[^,]+}})
    // CK21-NOUSE: call void [[CALL00:@.+]]()
    #pragma omp target map(A)
    {
#ifdef USE
      A += 1;
#endif
    }

    // Region 01
    // CK21-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null)
    // CK21-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK21-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
    // CK21-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
    // CK21-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
    // CK21-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
    // CK21-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
    // CK21-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 0
    // CK21-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

    // CK21-USE: call void [[CALL01:@.+]](i32* {{[^,]+}})
    // CK21-NOUSE: call void [[CALL01:@.+]]()
    #pragma omp target map(lb[:X])
    {
#ifdef USE
      lb[4] += 1;
#endif
    }

    // Region 02
    // CK21-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE02]]{{.+}}, i8** null)
    // CK21-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK21-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
    // CK21-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

    // CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
    // CK21-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float***
    // CK21-DAG: store [[ST]]* [[VAR0:%.+]], [[ST]]** [[CBP0]]
    // CK21-DAG: store float** [[SEC0:%.+]], float*** [[CP0]]
    // CK21-DAG: store i64 {{%.+}}, i64* [[S0]]
    // CK21-DAG: [[SEC0]] = getelementptr {{.*}}[[ST]]* [[VAR0]], i{{.+}} 0, i{{.+}} 2

    // CK21-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
    // CK21-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
    // CK21-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
    // CK21-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to float***
    // CK21-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to float**
    // CK21-DAG: store float** [[SEC0]], float*** [[CBP1]]
    // CK21-DAG: store float* [[SEC1:%.+]], float** [[CP1]]
    // CK21-DAG: store i64 {{.+}}, i64* [[S1]]
    // CK21-DAG: [[SEC1]] = getelementptr {{.*}}float* [[RVAR1:%[^,]+]], i{{.+}} 123
    // CK21-DAG: [[RVAR1]] = load float*, float** [[SEC1_:%[^,]+]]
    // CK21-DAG: [[SEC1_]] = getelementptr {{.*}}[[ST]]* [[VAR0]], i{{.+}} 0, i{{.+}} 2

    // CK21-USE: call void [[CALL02:@.+]]([[ST]]* {{[^,]+}})
    // CK21-NOUSE: call void [[CALL02:@.+]]()
    #pragma omp target map(from:B[X:X+2])
    {
#ifdef USE
      B[2] += 1.0f;
#endif
    }

    // Region 03
    // CK21-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}}, i8** null)
    // CK21-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK21-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [123 x float]**
    // CK21-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [123 x float]**
    // CK21-DAG: store [123 x float]* [[VAR0:%.+]], [123 x float]** [[CBP0]]
    // CK21-DAG: store [123 x float]* [[VAR0]], [123 x float]** [[CP0]]

    // CK21-USE: call void [[CALL03:@.+]]([123 x float]* {{[^,]+}})
    // CK21-NOUSE: call void [[CALL03:@.+]]()
    #pragma omp target map(from:la)
    {
#ifdef USE
      la[3] += 1.0f;
#endif
    }

    // Region 04
    // CK21-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE04]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE04]]{{.+}}, i8** null)
    // CK21-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK21-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
    // CK21-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
    // CK21-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
    // CK21-DAG: store i32* [[VAR0]], i32** [[CP0]]

    // CK21-USE: call void [[CALL04:@.+]](i32* {{[^,]+}})
    // CK21-NOUSE: call void [[CALL04:@.+]]()
    #pragma omp target map(from:arg)
    {
#ifdef USE
      arg +=1;
#endif
    }

    // Make sure the extra flag is passed to the second map.
    // Region 05
    // CK21-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE05]]{{.+}}, i8** null)
    // CK21-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK21-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
    // CK21-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

    // CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
    // CK21-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
    // CK21-DAG: store [[ST]]* [[VAR0:%.+]], [[ST]]** [[CBP0]]
    // CK21-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
    // CK21-DAG: store i64 {{%.+}}, i64* [[S0]]
    // CK21-DAG: [[SEC0]] = getelementptr {{.*}}[[ST]]* [[VAR0]], i{{.+}} 0, i{{.+}} 0

    // CK21-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
    // CK21-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
    // CK21-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
    // CK21-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[ST]]**
    // CK21-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
    // CK21-DAG: store [[ST]]* [[VAR0]], [[ST]]** [[CBP1]]
    // CK21-DAG: store i32* [[SEC0]], i32** [[CP1]]
    // CK21-DAG: store i64 {{.+}}, i64* [[S1]]

    // CK21-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
    // CK21-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
    // CK21-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
    // CK21-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [[ST]]**
    // CK21-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to i32**
    // CK21-DAG: store [[ST]]* [[VAR2:%.+]], [[ST]]** [[CBP2]]
    // CK21-DAG: store i32* [[SEC2:%.+]], i32** [[CP2]]
    // CK21-DAG: store i64 {{.+}}, i64* [[S2]]
    // CK21-DAG: [[SEC2]] = getelementptr {{.*}}[[ST]]* [[VAR2]], i{{.+}} 0, i{{.+}} 1

    // CK21-USE: call void [[CALL05:@.+]]([[ST]]* {{[^,]+}})
    // CK21-NOUSE: call void [[CALL05:@.+]]()
    #pragma omp target map(A, A2)
    {
#ifdef USE
      A += 1;
      A2 += 1;
#endif
    }
    return A;
  }
};

int explicit_maps_template_args_and_members(int a){
  CC<123,int> c;
  return c.foo(a);
}

// CK21: define {{.+}}[[CALL00]]
// CK21: define {{.+}}[[CALL01]]
// CK21: define {{.+}}[[CALL02]]
// CK21: define {{.+}}[[CALL03]]
// CK21: define {{.+}}[[CALL04]]
// CK21: define {{.+}}[[CALL05]]
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK22 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK22 --check-prefix CK22-64
// RUN: %clang_cc1 -DCK22 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22 --check-prefix CK22-64
// RUN: %clang_cc1 -DCK22 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22 --check-prefix CK22-32
// RUN: %clang_cc1 -DCK22 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22 --check-prefix CK22-32

// RUN: %clang_cc1 -DCK22 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK22 --check-prefix CK22-64
// RUN: %clang_cc1 -DCK22 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22 --check-prefix CK22-64
// RUN: %clang_cc1 -DCK22 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22 --check-prefix CK22-32
// RUN: %clang_cc1 -DCK22 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22 --check-prefix CK22-32

// RUN: %clang_cc1 -DCK22 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK22 --check-prefix CK22-64
// RUN: %clang_cc1 -DCK22 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22 --check-prefix CK22-64
// RUN: %clang_cc1 -DCK22 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22 --check-prefix CK22-32
// RUN: %clang_cc1 -DCK22 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK22 --check-prefix CK22-32

// RUN: %clang_cc1 -DCK22 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY21 %s
// RUN: %clang_cc1 -DCK22 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY21 %s
// RUN: %clang_cc1 -DCK22 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY21 %s
// RUN: %clang_cc1 -DCK22 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY21 %s
// SIMD-ONLY21-NOT: {{__kmpc|__tgt}}
#ifdef CK22

// CK22-DAG: [[ST:%.+]] = type { float }
// CK22-DAG: [[STT:%.+]] = type { i32 }

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE00:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK22: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE01:@.+]] = private {{.*}}constant [1 x i64] [i64 400]
// CK22: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE02:@.+]] = private {{.*}}constant [1 x i64] [i64 {{8|4}}]
// CK22: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE03:@.+]] = private {{.*}}constant [1 x i64] [i64 16]
// CK22: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE04:@.+]] = private {{.*}}constant [1 x i64] [i64 20]
// CK22: [[MTYPE04:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE05:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK22: [[MTYPE05:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE06:@.+]] = private {{.*}}constant [1 x i64] [i64 400]
// CK22: [[MTYPE06:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE07:@.+]] = private {{.*}}constant [1 x i64] [i64 {{8|4}}]
// CK22: [[MTYPE07:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE08:@.+]] = private {{.*}}constant [1 x i64] [i64 16]
// CK22: [[MTYPE08:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE09:@.+]] = private {{.*}}constant [1 x i64] [i64 20]
// CK22: [[MTYPE09:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE10:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK22: [[MTYPE10:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE11:@.+]] = private {{.*}}constant [1 x i64] [i64 400]
// CK22: [[MTYPE11:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE12:@.+]] = private {{.*}}constant [1 x i64] [i64 {{8|4}}]
// CK22: [[MTYPE12:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE13:@.+]] = private {{.*}}constant [1 x i64] [i64 16]
// CK22: [[MTYPE13:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK22-LABEL: @.__omp_offloading_{{.*}}explicit_maps_globals{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK22: [[SIZE14:@.+]] = private {{.*}}constant [1 x i64] [i64 20]
// CK22: [[MTYPE14:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

int a;
int c[100];
int *d;

struct ST {
  float fa;
};

ST sa ;
ST sc[100];
ST *sd;

template<typename T>
struct STT {
  T fa;
};

STT<int> sta ;
STT<int> stc[100];
STT<int> *std;

// CK22-LABEL: explicit_maps_globals{{.*}}(
int explicit_maps_globals(void){
  // Region 00
  // CK22-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK22-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK22-DAG: store i32* @a, i32** [[CBP0]]
  // CK22-DAG: store i32* @a, i32** [[CP0]]

  // CK22: call void [[CALL00:@.+]](i32* {{[^,]+}})
  #pragma omp target map(a)
  { a+=1; }

  // Region 01
  // CK22-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null)
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK22-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [100 x i32]**
  // CK22-DAG: store [100 x i32]* @c, [100 x i32]** [[CBP0]]
  // CK22-DAG: store [100 x i32]* @c, [100 x i32]** [[CP0]]

  // CK22: call void [[CALL01:@.+]]([100 x i32]* {{[^,]+}})
  #pragma omp target map(c)
  { c[3]+=1; }

  // Region 02
  // CK22-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}}, i8** null)
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32***
  // CK22-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32***
  // CK22-DAG: store i32** @d, i32*** [[CBP0]]
  // CK22-DAG: store i32** @d, i32*** [[CP0]]

  // CK22: call void [[CALL02:@.+]](i32** {{[^,]+}})
  #pragma omp target map(d)
  { d[3]+=1; }

  // Region 03
  // CK22-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}}, i8** null)
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x i32]**
  // CK22-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK22-DAG: store [100 x i32]* @c, [100 x i32]** [[CBP0]]
  // CK22-DAG: store i32* getelementptr inbounds ([100 x i32], [100 x i32]* @c, i{{.+}} 0, i{{.+}} 1), i32** [[CP0]]

  // CK22: call void [[CALL03:@.+]]([100 x i32]* {{[^,]+}})
  #pragma omp target map(c[1:4])
  { c[3]+=1; }

  // Region 04
  // CK22-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE04]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE04]]{{.+}}, i8** null)
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK22-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK22-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
  // CK22-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK22-DAG: [[RVAR0]] = load i32*, i32** @d
  // CK22-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 2
  // CK22-DAG: [[RVAR00]] = load i32*, i32** @d

  // CK22: call void [[CALL04:@.+]](i32* {{[^,]+}})
  #pragma omp target map(d[2:5])
  { d[3]+=1; }

  // Region 05
  // CK22-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE05]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE05]]{{.+}}, i8** null)
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
  // CK22-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[ST]]**
  // CK22-DAG: store [[ST]]* @sa, [[ST]]** [[CBP0]]
  // CK22-DAG: store [[ST]]* @sa, [[ST]]** [[CP0]]

  // CK22: call void [[CALL05:@.+]]([[ST]]* {{[^,]+}})
  #pragma omp target map(sa)
  { sa.fa+=1; }

  // Region 06
  // CK22-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE06]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE06]]{{.+}}, i8** null)
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x [[ST]]]**
  // CK22-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [100 x [[ST]]]**
  // CK22-DAG: store [100 x [[ST]]]* @sc, [100 x [[ST]]]** [[CBP0]]
  // CK22-DAG: store [100 x [[ST]]]* @sc, [100 x [[ST]]]** [[CP0]]

  // CK22: call void [[CALL06:@.+]]([100 x [[ST]]]* {{[^,]+}})
  #pragma omp target map(sc)
  { sc[3].fa+=1; }

  // Region 07
  // CK22-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE07]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE07]]{{.+}}, i8** null)
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]***
  // CK22-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[ST]]***
  // CK22-DAG: store [[ST]]** @sd, [[ST]]*** [[CBP0]]
  // CK22-DAG: store [[ST]]** @sd, [[ST]]*** [[CP0]]

  // CK22: call void [[CALL07:@.+]]([[ST]]** {{[^,]+}})
  #pragma omp target map(sd)
  { sd[3].fa+=1; }

  // Region 08
  // CK22-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE08]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE08]]{{.+}}, i8** null)
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x [[ST]]]**
  // CK22-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[ST]]**
  // CK22-DAG: store [100 x [[ST]]]* @sc, [100 x [[ST]]]** [[CBP0]]
  // CK22-DAG: store [[ST]]* getelementptr inbounds ([100 x [[ST]]], [100 x [[ST]]]* @sc, i{{.+}} 0, i{{.+}} 1), [[ST]]** [[CP0]]

  // CK22: call void [[CALL08:@.+]]([100 x [[ST]]]* {{[^,]+}})
  #pragma omp target map(sc[1:4])
  { sc[3].fa+=1; }

  // Region 09
  // CK22-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE09]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE09]]{{.+}}, i8** null)
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
  // CK22-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[ST]]**
  // CK22-DAG: store [[ST]]* [[RVAR0:%.+]], [[ST]]** [[CBP0]]
  // CK22-DAG: store [[ST]]* [[SEC0:%.+]], [[ST]]** [[CP0]]
  // CK22-DAG: [[RVAR0]] = load [[ST]]*, [[ST]]** @sd
  // CK22-DAG: [[SEC0]] = getelementptr {{.*}}[[ST]]* [[RVAR00:%.+]], i{{.+}} 2
  // CK22-DAG: [[RVAR00]] = load [[ST]]*, [[ST]]** @sd

  // CK22: call void [[CALL09:@.+]]([[ST]]* {{[^,]+}})
  #pragma omp target map(sd[2:5])
  { sd[3].fa+=1; }

  // Region 10
  // CK22-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE10]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE10]]{{.+}}, i8** null)
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[STT]]**
  // CK22-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[STT]]**
  // CK22-DAG: store [[STT]]* @sta, [[STT]]** [[CBP0]]
  // CK22-DAG: store [[STT]]* @sta, [[STT]]** [[CP0]]

  // CK22: call void [[CALL10:@.+]]([[STT]]* {{[^,]+}})
  #pragma omp target map(sta)
  { sta.fa+=1; }

  // Region 11
  // CK22-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE11]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE11]]{{.+}}, i8** null)
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x [[STT]]]**
  // CK22-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [100 x [[STT]]]**
  // CK22-DAG: store [100 x [[STT]]]* @stc, [100 x [[STT]]]** [[CBP0]]
  // CK22-DAG: store [100 x [[STT]]]* @stc, [100 x [[STT]]]** [[CP0]]

  // CK22: call void [[CALL11:@.+]]([100 x [[STT]]]* {{[^,]+}})
  #pragma omp target map(stc)
  { stc[3].fa+=1; }

  // Region 12
  // CK22-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE12]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE12]]{{.+}}, i8** null)
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[STT]]***
  // CK22-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[STT]]***
  // CK22-DAG: store [[STT]]** @std, [[STT]]*** [[CBP0]]
  // CK22-DAG: store [[STT]]** @std, [[STT]]*** [[CP0]]

  // CK22: call void [[CALL12:@.+]]([[STT]]** {{[^,]+}})
  #pragma omp target map(std)
  { std[3].fa+=1; }

  // Region 13
  // CK22-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE13]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE13]]{{.+}}, i8** null)
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x [[STT]]]**
  // CK22-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[STT]]**
  // CK22-DAG: store [100 x [[STT]]]* @stc, [100 x [[STT]]]** [[CBP0]]
  // CK22-DAG: store [[STT]]* getelementptr inbounds ([100 x [[STT]]], [100 x [[STT]]]* @stc, i{{.+}} 0, i{{.+}} 1),  [[STT]]** [[CP0]]

  // CK22: call void [[CALL13:@.+]]([100 x [[STT]]]* {{[^,]+}})
  #pragma omp target map(stc[1:4])
  { stc[3].fa+=1; }

  // Region 14
  // CK22-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE14]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE14]]{{.+}}, i8** null)
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[STT]]**
  // CK22-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[STT]]**
  // CK22-DAG: store [[STT]]* [[RVAR0:%.+]], [[STT]]** [[CBP0]]
  // CK22-DAG: store [[STT]]* [[SEC0:%.+]], [[STT]]** [[CP0]]
  // CK22-DAG: [[RVAR0]] = load [[STT]]*, [[STT]]** @std
  // CK22-DAG: [[SEC0]] = getelementptr {{.*}}[[STT]]* [[RVAR00:%.+]], i{{.+}} 2
  // CK22-DAG: [[RVAR00]] = load [[STT]]*, [[STT]]** @std

  // CK22: call void [[CALL14:@.+]]([[STT]]* {{[^,]+}})
  #pragma omp target map(std[2:5])
  { std[3].fa+=1; }

  return 0;
}
// CK22: define {{.+}}[[CALL00]]
// CK22: define {{.+}}[[CALL01]]
// CK22: define {{.+}}[[CALL02]]
// CK22: define {{.+}}[[CALL03]]
// CK22: define {{.+}}[[CALL04]]
// CK22: define {{.+}}[[CALL05]]
// CK22: define {{.+}}[[CALL06]]
// CK22: define {{.+}}[[CALL07]]
// CK22: define {{.+}}[[CALL08]]
// CK22: define {{.+}}[[CALL09]]
// CK22: define {{.+}}[[CALL10]]
// CK22: define {{.+}}[[CALL11]]
// CK22: define {{.+}}[[CALL12]]
// CK22: define {{.+}}[[CALL13]]
// CK22: define {{.+}}[[CALL14]]
#endif
///==========================================================================///
// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-32
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-32

// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-32
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-32

// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-32
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK23 --check-prefix CK23-32

// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY22 %s
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY22 %s
// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY22 %s
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY22 %s
// SIMD-ONLY22-NOT: {{__kmpc|__tgt}}
#ifdef CK23

// CK23-LABEL: @.__omp_offloading_{{.*}}explicit_maps_inside_captured{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK23: [[SIZE00:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK23: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK23-LABEL: @.__omp_offloading_{{.*}}explicit_maps_inside_captured{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK23: [[SIZE01:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK23: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK23-LABEL: @.__omp_offloading_{{.*}}explicit_maps_inside_captured{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK23: [[SIZE02:@.+]] = private {{.*}}constant [1 x i64] [i64 400]
// CK23: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK23-LABEL: @.__omp_offloading_{{.*}}explicit_maps_inside_captured{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK23: [[SIZE03:@.+]] = private {{.*}}constant [1 x i64] [i64 {{8|4}}]
// CK23: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK23-LABEL: @.__omp_offloading_{{.*}}explicit_maps_inside_captured{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK23: [[SIZE04:@.+]] = private {{.*}}constant [1 x i64] [i64 16]
// CK23: [[MTYPE04:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK23-LABEL: @.__omp_offloading_{{.*}}explicit_maps_inside_captured{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK23: [[SIZE05:@.+]] = private {{.*}}constant [1 x i64] [i64 16]
// CK23: [[MTYPE05:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK23-LABEL: explicit_maps_inside_captured{{.*}}(
int explicit_maps_inside_captured(int a){
  float b;
  float c[100];
  float *d;

  // CK23: call void @{{.*}}explicit_maps_inside_captured{{.*}}([[SA:%.+]]* {{.*}})
  // CK23: define {{.*}}explicit_maps_inside_captured{{.*}}
  [&](void){
    // Region 00
    // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
    // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
    // CK23-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
    // CK23-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
    // CK23-DAG: store i32* [[VAR00:%.+]], i32** [[CP0]]
    // CK23-DAG: [[VAR0]] = load i32*, i32** [[CAP0:%[^,]+]]
    // CK23-DAG: [[CAP0]] = getelementptr inbounds [[SA]], [[SA]]
    // CK23-DAG: [[VAR00]] = load i32*, i32** [[CAP00:%[^,]+]]
    // CK23-DAG: [[CAP00]] = getelementptr inbounds [[SA]], [[SA]]

    // CK23: call void [[CALL00:@.+]](i32* {{[^,]+}})
    #pragma omp target map(a)
      { a+=1; }
    // Region 01
    // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null)
    // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
    // CK23-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
    // CK23-DAG: store float* [[VAR0:%.+]], float** [[CBP0]]
    // CK23-DAG: store float* [[VAR00:%.+]], float** [[CP0]]
    // CK23-DAG: [[VAR0]] = load float*, float** [[CAP0:%[^,]+]]
    // CK23-DAG: [[CAP0]] = getelementptr inbounds [[SA]], [[SA]]
    // CK23-DAG: [[VAR00]] = load float*, float** [[CAP00:%[^,]+]]
    // CK23-DAG: [[CAP00]] = getelementptr inbounds [[SA]], [[SA]]

    // CK23: call void [[CALL01:@.+]](float* {{[^,]+}})
    #pragma omp target map(b)
      { b+=1; }
    // Region 02
    // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}}, i8** null)
    // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x float]**
    // CK23-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [100 x float]**
    // CK23-DAG: store [100 x float]* [[VAR0:%.+]], [100 x float]** [[CBP0]]
    // CK23-DAG: store [100 x float]* [[VAR00:%.+]], [100 x float]** [[CP0]]
    // CK23-DAG: [[VAR0]] = load [100 x float]*, [100 x float]** [[CAP0:%[^,]+]]
    // CK23-DAG: [[CAP0]] = getelementptr inbounds [[SA]], [[SA]]
    // CK23-DAG: [[VAR00]] = load [100 x float]*, [100 x float]** [[CAP00:%[^,]+]]
    // CK23-DAG: [[CAP00]] = getelementptr inbounds [[SA]], [[SA]]

    // CK23: call void [[CALL02:@.+]]([100 x float]* {{[^,]+}})
    #pragma omp target map(c)
      { c[3]+=1; }

    // Region 03
    // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}}, i8** null)
    // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float***
    // CK23-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float***
    // CK23-DAG: store float** [[VAR0:%.+]], float*** [[CBP0]]
    // CK23-DAG: store float** [[VAR00:%.+]], float*** [[CP0]]
    // CK23-DAG: [[VAR0]] = load float**, float*** [[CAP0:%[^,]+]]
    // CK23-DAG: [[CAP0]] = getelementptr inbounds [[SA]], [[SA]]
    // CK23-DAG: [[VAR00]] = load float**, float*** [[CAP00:%[^,]+]]
    // CK23-DAG: [[CAP00]] = getelementptr inbounds [[SA]], [[SA]]

    // CK23: call void [[CALL03:@.+]](float** {{[^,]+}})
    #pragma omp target map(d)
      { d[3]+=1; }
    // Region 04
    // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE04]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE04]]{{.+}}, i8** null)
    // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x float]**
    // CK23-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
    // CK23-DAG: store [100 x float]* [[VAR0:%.+]], [100 x float]** [[CBP0]]
    // CK23-DAG: store float* [[SEC0:%.+]], float** [[CP0]]
    // CK23-DAG: [[SEC0]] = getelementptr {{.*}}[100 x float]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 2
    // CK23-DAG: [[VAR0]] = load [100 x float]*, [100 x float]** [[CAP0:%[^,]+]]
    // CK23-DAG: [[CAP0]] = getelementptr inbounds [[SA]], [[SA]]
    // CK23-DAG: [[VAR00]] = load [100 x float]*, [100 x float]** [[CAP00:%[^,]+]]
    // CK23-DAG: [[CAP00]] = getelementptr inbounds [[SA]], [[SA]]

    // CK23: call void [[CALL04:@.+]]([100 x float]* {{[^,]+}})
    #pragma omp target map(c[2:4])
      { c[3]+=1; }

    // Region 05
    // CK23-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE05]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE05]]{{.+}}, i8** null)
    // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
    // CK23-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
    // CK23-DAG: store float* [[RVAR0:%.+]], float** [[CBP0]]
    // CK23-DAG: store float* [[SEC0:%.+]], float** [[CP0]]
    // CK23-DAG: [[RVAR0]] = load float*, float** [[VAR0:%[^,]+]]
    // CK23-DAG: [[SEC0]] = getelementptr {{.*}}float* [[RVAR00:%.+]], i{{.+}} 2
    // CK23-DAG: [[RVAR00]] = load float*, float** [[VAR00:%[^,]+]]
    // CK23-DAG: [[VAR0]] = load float**, float*** [[CAP0:%[^,]+]]
    // CK23-DAG: [[CAP0]] = getelementptr inbounds [[SA]], [[SA]]
    // CK23-DAG: [[VAR00]] = load float**, float*** [[CAP00:%[^,]+]]
    // CK23-DAG: [[CAP00]] = getelementptr inbounds [[SA]], [[SA]]

    // CK23: call void [[CALL05:@.+]](float* {{[^,]+}})
    #pragma omp target map(d[2:4])
      { d[3]+=1; }
  }();
  return b;
}

// CK23: define {{.+}}[[CALL00]]
// CK23: define {{.+}}[[CALL01]]
// CK23: define {{.+}}[[CALL02]]
// CK23: define {{.+}}[[CALL03]]
// CK23: define {{.+}}[[CALL04]]
// CK23: define {{.+}}[[CALL05]]
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32

// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32

// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK24 --check-prefix CK24-32

// RUN: %clang_cc1 -DCK24 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY23 %s
// RUN: %clang_cc1 -DCK24 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY23 %s
// RUN: %clang_cc1 -DCK24 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY23 %s
// RUN: %clang_cc1 -DCK24 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY23 %s
// SIMD-ONLY23-NOT: {{__kmpc|__tgt}}
#ifdef CK24

// CK24-DAG: [[SC:%.+]] = type { i32, [[SB:%.+]], [[SB:%.+]]*, [10 x i32] }
// CK24-DAG: [[SB]] = type { i32, [[SA:%.+]], [10 x [[SA:%.+]]], [10 x [[SA:%.+]]*], [[SA:%.+]]* }
// CK24-DAG: [[SA]] = type { i32, [[SA]]*, [10 x i32] }

struct SA{
  int a;
  struct SA *p;
  int b[10];
};
struct SB{
  int a;
  struct SA s;
  struct SA sa[10];
  struct SA *sp[10];
  struct SA *p;
};
struct SC{
  int a;
  struct SB s;
  struct SB *p;
  int b[10];
};

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE01:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710659]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE13:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710659]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE14:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710659]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE15:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710659]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE16:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710659]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE17:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710675]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE18:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710659]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE19:@.+]] = private {{.*}}constant [3 x i64] [i64 32, i64 281474976710659, i64 281474976710675]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE20:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710675]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE21:@.+]] = private {{.*}}constant [3 x i64] [i64 32, i64 281474976710659, i64 281474976710675]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE22:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710659]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE23:@.+]] = private {{.*}}constant [3 x i64] [i64 32, i64 281474976710659, i64 281474976710675]

// CK24-LABEL: @.__omp_offloading_{{.*}}explicit_maps_struct_fields{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK24: [[MTYPE24:@.+]] = private {{.*}}constant [4 x i64] [i64 32, i64 281474976710672, i64 16, i64 19]

// CK24-LABEL: explicit_maps_struct_fields
int explicit_maps_struct_fields(int a){
  SC s;
  SC *p;

// Region 01
// CK24-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 0

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SC]]**
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0]], [[SC]]** [[CBP1]]
// CK24-DAG: store i32* [[SEC0]], i32** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]

// CK24: call void [[CALL01:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(s.a)
  { s.a++; }

//
// Same thing but starting from a pointer.
//
// Region 13
// CK24-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE13]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 0

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SC]]**
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0]], [[SC]]** [[CBP1]]
// CK24-DAG: store i32* [[SEC0]], i32** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL13:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->a)
  { p->a++; }

// Region 14
// CK24-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE14]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SA]]**
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store [[SA]]* [[SEC0:%.+]], [[SA]]** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SB]]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SC]]**
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [[SA]]**
// CK24-DAG: store [[SC]]* [[VAR0]], [[SC]]** [[CBP1]]
// CK24-DAG: store [[SA]]* [[SEC0]], [[SA]]** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL14:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->s.s)
  { p->a++; }

// Region 15
// CK24-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz:64|32]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE15]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SA]]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SB]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SC]]**
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0]], [[SC]]** [[CBP1]]
// CK24-DAG: store i32* [[SEC0]], i32** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL15:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->s.s.a)
  { p->a++; }

// Region 16
// CK24-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE16]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[10 x i32]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 3

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SC]]**
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0]], [[SC]]** [[CBP1]]
// CK24-DAG: store i32* [[SEC0]], i32** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL16:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->b[:5])
  { p->a++; }

// Region 17
// CK24-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE17]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SB]]***
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store [[SB]]** [[SEC0:%.+]], [[SB]]*** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SB]]***
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [[SB]]**
// CK24-DAG: store [[SB]]** [[SEC0]], [[SB]]*** [[CBP1]]
// CK24-DAG: store [[SB]]* [[SEC1:%.+]], [[SB]]** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[[SB]]* [[SEC11:%[^,]+]], i{{.+}} 0
// CK24-DAG: [[SEC11]] = load [[SB]]*, [[SB]]** [[SEC111:%[^,]+]],
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}[[SC]]* [[VAR000:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR000]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL17:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->p[:5])
  { p->a++; }

// Region 18
// CK24-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE18]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SA]]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[10 x [[SA]]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}[[SB]]* [[SEC0000:%[^,]+]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[SEC0000]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SC]]**
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0]], [[SC]]** [[CBP1]]
// CK24-DAG: store i32* [[SEC0]], i32** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL18:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->s.sa[3].a)
  { p->a++; }

// Region 19
// CK24-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE19]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SA]]***
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store [[SA]]** [[SEC0:%.+]], [[SA]]*** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[10 x [[SA]]*]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SB]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SC]]**
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [[SA]]***
// CK24-DAG: store [[SC]]* [[VAR0]], [[SC]]** [[CBP1]]
// CK24-DAG: store [[SA]]** [[SEC0]], [[SA]]*** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]

// CK24-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [[SA]]***
// CK24-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to i32**
// CK24-DAG: store [[SA]]** [[SEC0]], [[SA]]*** [[CBP2]]
// CK24-DAG: store i32* [[SEC1:%.+]], i32** [[CP2]]
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[[SA]]* [[SEC11:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC11]] = load [[SA]]*, [[SA]]** [[SEC111:%[^,]+]],
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}[10 x [[SA]]*]* [[SEC1111:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC1111]] = getelementptr {{.*}}[[SB]]* [[SEC11111:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC11111]] = getelementptr {{.*}}[[SC]]* [[VAR000:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR000]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL19:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->s.sp[3]->a)
  { p->a++; }

// Region 20
// CK24-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE20]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SB]]***
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store [[SB]]** [[SEC0:%.+]], [[SB]]*** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SB]]***
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
// CK24-DAG: store [[SB]]** [[SEC0]], [[SB]]*** [[CBP1]]
// CK24-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[[SB]]* [[SEC11:%[^,]+]], i{{.+}} 0
// CK24-DAG: [[SEC11]] = load [[SB]]*, [[SB]]** [[SEC111:%[^,]+]],
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}[[SC]]* [[VAR000:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR000]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL20:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->p->a)
  { p->a++; }

// Region 21
// CK24-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE21]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SA]]***
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store [[SA]]** [[SEC0:%.+]], [[SA]]*** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SB]]* [[SEC00:[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SC]]**
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [[SA]]***
// CK24-DAG: store [[SC]]* [[VAR0]], [[SC]]** [[CBP1]]
// CK24-DAG: store [[SA]]** [[SEC0]], [[SA]]*** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]

// CK24-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [[SA]]***
// CK24-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to i32**
// CK24-DAG: store [[SA]]** [[SEC0]], [[SA]]*** [[CBP2]]
// CK24-DAG: store i32* [[SEC1:%.+]], i32** [[CP2]]
// CK24-DAG: store i64 {{.+}}, i64* [[S2]]
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[[SA]]* [[SEC11:%[^,]+]], i{{.+}} 0
// CK24-DAG: [[SEC11]] = load [[SA]]*, [[SA]]** [[SEC111:%[^,]+]],
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}[[SB]]* [[SEC1111:[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC1111]] = getelementptr {{.*}}[[SC]]* [[VAR000:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR000]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL21:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->s.p->a)
  { p->a++; }

// Region 22
// CK24-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE22]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[10 x i32]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SA]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}[[SB]]* [[SEC0000:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC0000]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SC]]**
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
// CK24-DAG: store [[SC]]* [[VAR0]], [[SC]]** [[CBP1]]
// CK24-DAG: store i32* [[SEC0]], i32** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL22:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->s.s.b[:2])
  { p->a++; }

// Region 23
// CK24-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE23]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SA]]***
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store [[SA]]** [[SEC0:%.+]], [[SA]]*** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SB]]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SC]]**
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [[SA]]***
// CK24-DAG: store [[SC]]* [[VAR0]], [[SC]]** [[CBP1]]
// CK24-DAG: store [[SA]]** [[SEC0]], [[SA]]*** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]

// CK24-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [[SA]]***
// CK24-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to i32**
// CK24-DAG: store [[SA]]** [[SEC0]], [[SA]]*** [[CBP2]]
// CK24-DAG: store i32* [[SEC1:%.+]], i32** [[CP2]]
// CK24-DAG: store i64 {{.+}}, i64* [[S2]]
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[10 x i32]* [[SEC11:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC11]] = getelementptr {{.*}}[[SA]]* [[SEC111:%[^,]+]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[SEC111]] = load [[SA]]*, [[SA]]** [[SEC1111:%[^,]+]],
// CK24-DAG: [[SEC1111]] = getelementptr {{.*}}[[SB]]* [[SEC11111:%[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC11111]] = getelementptr {{.*}}[[SC]]* [[VAR000:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR000]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL23:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->s.p->b[:2])
  { p->a++; }

// Region 24
// CK24-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 4, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[4 x i{{.+}}]* [[MTYPE24]]{{.+}}, i8** null)
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK24-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SC]]**
// CK24-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SB]]***
// CK24-DAG: store [[SC]]* [[VAR0:%.+]], [[SC]]** [[CBP0]]
// CK24-DAG: store [[SB]]** [[SEC0:%.+]], [[SB]]*** [[CP0]]
// CK24-DAG: store i64 {{%.+}}, i64* [[S0]]
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SB]]***
// CK24-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to [[SA]]***
// CK24-DAG: store [[SB]]** [[SEC0]], [[SB]]*** [[CBP1]]
// CK24-DAG: store [[SA]]** [[SEC1:%.+]], [[SA]]*** [[CP1]]
// CK24-DAG: store i64 {{.+}}, i64* [[S1]]
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[[SB]]* [[SEC11:%[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC11]] = load [[SB]]*, [[SB]]** [[SEC111:%[^,]+]],
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}[[SC]]* [[VAR000:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [[SA]]***
// CK24-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to [[SA]]***
// CK24-DAG: store [[SA]]** [[SEC1]], [[SA]]*** [[CBP2]]
// CK24-DAG: store [[SA]]** [[SEC2:%.+]], [[SA]]*** [[CP2]]
// CK24-DAG: store i64 {{.+}}, i64* [[S2]]
// CK24-DAG: [[SEC2]] = getelementptr {{.*}}[[SA]]* [[SEC22:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC22]] = load [[SA]]*, [[SA]]** [[SEC222:%[^,]+]],
// CK24-DAG: [[SEC222]] = getelementptr {{.*}}[[SB]]* [[SEC2222:%[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC2222]] = load [[SB]]*, [[SB]]** [[SEC22222:%[^,]+]],
// CK24-DAG: [[SEC22222]] = getelementptr {{.*}}[[SC]]* [[VAR0000:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[S3:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[CBP3:%.+]] = bitcast i8** [[BP3]] to [[SA]]***
// CK24-DAG: [[CP3:%.+]] = bitcast i8** [[P3]] to i32**
// CK24-DAG: store [[SA]]** [[SEC2]], [[SA]]*** [[CBP3]]
// CK24-DAG: store i32* [[SEC3:%.+]], i32** [[CP3]]
// CK24-DAG: store i64 {{.+}}, i64* [[S3]]
// CK24-DAG: [[SEC3]] = getelementptr {{.*}}[[SA]]* [[SEC33:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC33]] = load [[SA]]*, [[SA]]** [[SEC333:%[^,]+]],
// CK24-DAG: [[SEC333]] = getelementptr {{.*}}[[SA]]* [[SEC3333:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC3333]] = load [[SA]]*, [[SA]]** [[SEC33333:%[^,]+]],
// CK24-DAG: [[SEC33333]] = getelementptr {{.*}}[[SB]]* [[SEC333333:%[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC333333]] = load [[SB]]*, [[SB]]** [[SEC3333333:%[^,]+]],
// CK24-DAG: [[SEC3333333]] = getelementptr {{.*}}[[SC]]* [[VAR00000:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR000]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR0000]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00000]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL24:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->p->p->p->a)
  { p->a++; }

  return s.a;
}

// CK24: define {{.+}}[[CALL01]]
// CK24: define {{.+}}[[CALL13]]
// CK24: define {{.+}}[[CALL14]]
// CK24: define {{.+}}[[CALL15]]
// CK24: define {{.+}}[[CALL16]]
// CK24: define {{.+}}[[CALL17]]
// CK24: define {{.+}}[[CALL18]]
// CK24: define {{.+}}[[CALL19]]
// CK24: define {{.+}}[[CALL20]]
// CK24: define {{.+}}[[CALL21]]
// CK24: define {{.+}}[[CALL22]]
// CK24: define {{.+}}[[CALL23]]
// CK24: define {{.+}}[[CALL24]]
#endif
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
// CK25: [[CA00:%.+]] = type { [[ST]]* }
// CK25: [[CA01:%.+]] = type { i32* }

// CK25-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK25: [[MTYPE00:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710657]

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
    // CK25-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
    // CK25-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK25-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
    // CK25-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

    // CK25-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK25-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK25-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
    // CK25-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
    // CK25-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
    // CK25-DAG: store [[ST]]* [[VAR0:%.+]], [[ST]]** [[CBP0]]
    // CK25-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
    // CK25-DAG: store i64 {{%.+}}, i64* [[S0]]
    // CK25-DAG: [[SEC0]] = getelementptr {{.*}}[[ST]]* [[VAR0:%.+]], i{{.+}} 0, i{{.+}} 0

    // CK25-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
    // CK25-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
    // CK25-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
    // CK25-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[ST]]**
    // CK25-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
    // CK25-DAG: store [[ST]]* [[VAR0]], [[ST]]** [[CBP1]]
    // CK25-DAG: store i32* [[SEC0]], i32** [[CP1]]
    // CK25-DAG: store i64 {{.+}}, i64* [[S1]]

    // CK25: call void [[CALL00:@.+]]([[ST]]* {{[^,]+}})
    #pragma omp target map(to:A)
    {
      [&]() {
        A += 1;
      }();
    }

    // Region 01
    // CK25-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null)
    // CK25-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK25-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK25-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK25-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK25-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
    // CK25-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
    // CK25-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
    // CK25-DAG: store i32* [[VAR0]], i32** [[CP0]]

    // CK25: call void [[CALL01:@.+]](i32* {{[^,]+}})
    #pragma omp target map(to:arg)
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

// CK25: define {{.+}}[[CALL00]]([[ST]]* [[VAL:%.+]])
// CK25: store [[ST]]* [[VAL]], [[ST]]** [[VALADDR:%[^,]+]],
// CK25: [[VAL1:%.+]] = load [[ST]]*, [[ST]]** [[VALADDR]],
// CK25: [[VALADDR1:%.+]] = getelementptr inbounds [[CA00]], [[CA00]]* [[CA:%[^,]+]], i32 0, i32 0
// CK25: store [[ST]]* [[VAL1]], [[ST]]** [[VALADDR1]],
// CK25: call void {{.*}}[[LAMBDA:@.+]]{{.*}}([[CA00]]* [[CA]])

// CK25: define {{.+}}[[LAMBDA]]

// CK25: define {{.+}}[[CALL01]](i32* {{.*}}[[VAL:%.+]])
// CK25: store i32* [[VAL]], i32** [[VALADDR:%[^,]+]],
// CK25: [[VAL1:%.+]] = load i32*, i32** [[VALADDR]],
// CK25: [[VALADDR1:%.+]] = getelementptr inbounds [[CA01]], [[CA01]]* [[CA:%[^,]+]], i32 0, i32 0
// CK25: store i32* [[VAL1]], i32** [[VALADDR1]],
// CK25: call void {{.*}}[[LAMBDA2:@.+]]{{.*}}([[CA01]]* [[CA]])

// CK25: define {{.+}}[[LAMBDA2]]
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK26 -std=c++11 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK26 --check-prefix CK26-64
// RUN: %clang_cc1 -DCK26 -std=c++11 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-64
// RUN: %clang_cc1 -DCK26 -std=c++11 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-32
// RUN: %clang_cc1 -DCK26 -std=c++11 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-32

// RUN: %clang_cc1 -DCK26 -std=c++11 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK26 --check-prefix CK26-64
// RUN: %clang_cc1 -DCK26 -std=c++11 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -std=c++11 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-64
// RUN: %clang_cc1 -DCK26 -std=c++11 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-32
// RUN: %clang_cc1 -DCK26 -std=c++11 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -std=c++11 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-32

// RUN: %clang_cc1 -DCK26 -std=c++11 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK26 --check-prefix CK26-64
// RUN: %clang_cc1 -DCK26 -std=c++11 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -std=c++11 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-64
// RUN: %clang_cc1 -DCK26 -std=c++11 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-32
// RUN: %clang_cc1 -DCK26 -std=c++11 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -std=c++11 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK26 --check-prefix CK26-32

// RUN: %clang_cc1 -DCK26 -std=c++11 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY25 %s
// RUN: %clang_cc1 -DCK26 -std=c++11 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY25 %s
// RUN: %clang_cc1 -DCK26 -std=c++11 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY25 %s
// RUN: %clang_cc1 -DCK26 -std=c++11 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -std=c++11 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY25 %s
// SIMD-ONLY25-NOT: {{__kmpc|__tgt}}
#ifdef CK26
// CK26: [[ST:%.+]] = type { i32, float*, i32, float* }

// CK26-LABEL: @.__omp_offloading_{{.*}}CC{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK26: [[SIZE00:@.+]] = private {{.*}}constant [2 x i64] [i64 {{32|16}}, i64 4]
// CK26: [[MTYPE00:@.+]] = private {{.*}}constant [2 x i64] [i64 547, i64 35]

// CK26-LABEL: @.__omp_offloading_{{.*}}CC{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK26: [[SIZE01:@.+]] = private {{.*}}constant [2 x i64] [i64 {{32|16}}, i64 4]
// CK26: [[MTYPE01:@.+]] = private {{.*}}constant [2 x i64] [i64 547, i64 35]

// CK26-LABEL: @.__omp_offloading_{{.*}}CC{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK26: [[SIZE02:@.+]] = private {{.*}}constant [2 x i64] [i64 {{32|16}}, i64 4]
// CK26: [[MTYPE02:@.+]] = private {{.*}}constant [2 x i64] [i64 547, i64 35]

// CK26-LABEL: @.__omp_offloading_{{.*}}CC{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK26: [[SIZE03:@.+]] = private {{.*}}constant [2 x i64] [i64 {{32|16}}, i64 4]
// CK26: [[MTYPE03:@.+]] = private {{.*}}constant [2 x i64] [i64 547, i64 35]

// CK26-LABEL: explicit_maps_with_private_class_members{{.*}}(

struct CC {
  int fA;
  float &fB;
  int pA;
  float &pB;

  CC(float &B) : fB(B), pB(B) {

    // CK26: call {{.*}}@__kmpc_fork_call{{.*}} [[OUTCALL:@.+]] to void (i32*, i32*, ...)*
    // define {{.*}}void [[OUTCALL]]
    #pragma omp parallel firstprivate(fA,fB) private(pA,pB)
    {
      // Region 00
      // CK26-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
      // CK26-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
      // CK26-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

      // CK26-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
      // CK26-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[ST]]**
      // CK26-DAG: store [[ST]]* [[VAR0:%.+]], [[ST]]** [[CBP0]]
      // CK26-DAG: store [[ST]]* [[VAR0]], [[ST]]** [[CP0]]

      // CK26-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
      // CK26-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
      // CK26-DAG: store i32* [[VAR1:%.+]], i32** [[CBP1]]
      // CK26-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
      // CK26-DAG: [[VAR1]] = load i32*, i32** [[PVT:%.+]],
      // CK26-DAG: [[SEC1]] = load i32*, i32** [[PVT]],

      // CK26: call void [[CALL00:@.+]]([[ST]]* {{[^,]+}}, i32* {{[^,]+}})
      #pragma omp target map(fA)
      {
        ++fA;
      }

      // Region 01
      // CK26-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null)
      // CK26-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
      // CK26-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

      // CK26-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
      // CK26-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[ST]]**
      // CK26-DAG: store [[ST]]* [[VAR0:%.+]], [[ST]]** [[CBP0]]
      // CK26-DAG: store [[ST]]* [[VAR0]], [[ST]]** [[CP0]]

      // CK26-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to float**
      // CK26-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to float**
      // CK26-DAG: store float* [[VAR1:%.+]], float** [[CBP1]]
      // CK26-DAG: store float* [[SEC1:%.+]], float** [[CP1]]
      // CK26-DAG: [[VAR1]] = load float*, float** [[PVT:%.+]],
      // CK26-DAG: [[SEC1]] = load float*, float** [[PVT]],

      // CK26: call void [[CALL01:@.+]]([[ST]]* {{[^,]+}}, float* {{[^,]+}})
      #pragma omp target map(fB)
      {
        fB += 1.0;
      }

      // Region 02
      // CK26-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE02]]{{.+}}, i8** null)
      // CK26-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
      // CK26-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

      // CK26-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
      // CK26-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[ST]]**
      // CK26-DAG: store [[ST]]* [[VAR0:%.+]], [[ST]]** [[CBP0]]
      // CK26-DAG: store [[ST]]* [[VAR0]], [[ST]]** [[CP0]]

      // CK26-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
      // CK26-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
      // CK26-DAG: store i32* [[VAR1:%.+]], i32** [[CBP1]]
      // CK26-DAG: store i32* [[SEC1:%.+]], i32** [[CP1]]
      // CK26-DAG: [[VAR1]] = load i32*, i32** [[PVT:%.+]],
      // CK26-DAG: [[SEC1]] = load i32*, i32** [[PVT]],

      // CK26: call void [[CALL02:@.+]]([[ST]]* {{[^,]+}}, i32* {{[^,]+}})
      #pragma omp target map(pA)
      {
        ++pA;
      }

      // Region 01
      // CK26-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE03]]{{.+}}, i8** null)
      // CK26-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
      // CK26-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

      // CK26-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
      // CK26-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[ST]]**
      // CK26-DAG: store [[ST]]* [[VAR0:%.+]], [[ST]]** [[CBP0]]
      // CK26-DAG: store [[ST]]* [[VAR0]], [[ST]]** [[CP0]]

      // CK26-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to float**
      // CK26-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to float**
      // CK26-DAG: store float* [[VAR1:%.+]], float** [[CBP1]]
      // CK26-DAG: store float* [[SEC1:%.+]], float** [[CP1]]
      // CK26-DAG: [[VAR1]] = load float*, float** [[PVT:%.+]],
      // CK26-DAG: [[SEC1]] = load float*, float** [[PVT]],

      // CK26: call void [[CALL03:@.+]]([[ST]]* {{[^,]+}}, float* {{[^,]+}})
      #pragma omp target map(pB)
      {
        pB += 1.0;
      }
    }
  }

  int foo() {
    return fA + pA;
  }
};

// Make sure the private instance is used in all target regions.
// CK26: define {{.+}}[[CALL00]]({{.*}}i32*{{.*}}[[PVTARG:%.+]])
// CK26: store i32* [[PVTARG]], i32** [[PVTADDR:%.+]],
// CK26: [[ADDR:%.+]] = load i32*, i32** [[PVTADDR]],
// CK26: store i32* [[ADDR]], i32** [[PVTADDR:%.+]],
// CK26: [[ADDR:%.+]] = load i32*, i32** [[PVTADDR]],
// CK26: [[VAL:%.+]] = load i32, i32* [[ADDR]],
// CK26: add nsw i32 [[VAL]], 1

// CK26: define {{.+}}[[CALL01]]({{.*}}float*{{.*}}[[PVTARG:%.+]])
// CK26: store float* [[PVTARG]], float** [[PVTADDR:%.+]],
// CK26: [[ADDR:%.+]] = load float*, float** [[PVTADDR]],
// CK26: store float* [[ADDR]], float** [[PVTADDR:%.+]],
// CK26: [[ADDR:%.+]] = load float*, float** [[PVTADDR]],
// CK26: [[VAL:%.+]] = load float, float* [[ADDR]],
// CK26: [[EXT:%.+]] = fpext float [[VAL]] to double
// CK26: fadd double [[EXT]], 1.000000e+00

// CK26: define {{.+}}[[CALL02]]({{.*}}i32*{{.*}}[[PVTARG:%.+]])
// CK26: store i32* [[PVTARG]], i32** [[PVTADDR:%.+]],
// CK26: [[ADDR:%.+]] = load i32*, i32** [[PVTADDR]],
// CK26: store i32* [[ADDR]], i32** [[PVTADDR:%.+]],
// CK26: [[ADDR:%.+]] = load i32*, i32** [[PVTADDR]],
// CK26: [[VAL:%.+]] = load i32, i32* [[ADDR]],
// CK26: add nsw i32 [[VAL]], 1

// CK26: define {{.+}}[[CALL03]]({{.*}}float*{{.*}}[[PVTARG:%.+]])
// CK26: store float* [[PVTARG]], float** [[PVTADDR:%.+]],
// CK26: [[ADDR:%.+]] = load float*, float** [[PVTADDR]],
// CK26: store float* [[ADDR]], float** [[PVTADDR:%.+]],
// CK26: [[ADDR:%.+]] = load float*, float** [[PVTADDR]],
// CK26: [[VAL:%.+]] = load float, float* [[ADDR]],
// CK26: [[EXT:%.+]] = fpext float [[VAL]] to double
// CK26: fadd double [[EXT]], 1.000000e+00

int explicit_maps_with_private_class_members(){
  float B;
  CC c(B);
  return c.foo();
}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK27 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK27 --check-prefix CK27-64
// RUN: %clang_cc1 -DCK27 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK27 --check-prefix CK27-64
// RUN: %clang_cc1 -DCK27 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK27 --check-prefix CK27-32
// RUN: %clang_cc1 -DCK27 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK27 --check-prefix CK27-32

// RUN: %clang_cc1 -DCK27 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK27 --check-prefix CK27-64
// RUN: %clang_cc1 -DCK27 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK27 --check-prefix CK27-64
// RUN: %clang_cc1 -DCK27 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK27 --check-prefix CK27-32
// RUN: %clang_cc1 -DCK27 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK27 --check-prefix CK27-32

// RUN: %clang_cc1 -DCK27 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK27 --check-prefix CK27-64
// RUN: %clang_cc1 -DCK27 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK27 --check-prefix CK27-64
// RUN: %clang_cc1 -DCK27 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK27 --check-prefix CK27-32
// RUN: %clang_cc1 -DCK27 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK27 --check-prefix CK27-32

// RUN: %clang_cc1 -DCK27 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY26 %s
// RUN: %clang_cc1 -DCK27 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY26 %s
// RUN: %clang_cc1 -DCK27 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY26 %s
// RUN: %clang_cc1 -DCK27 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY26 %s
// SIMD-ONLY26-NOT: {{__kmpc|__tgt}}
#ifdef CK27

// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27: [[SIZE00:@.+]] = private {{.*}}constant [1 x i64] zeroinitializer
// CK27: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 544]

// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27: [[SIZE01:@.+]] = private {{.*}}constant [1 x i64] zeroinitializer
// CK27: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27: [[SIZE02:@.+]] = private {{.*}}constant [1 x i64] zeroinitializer
// CK27: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27: [[SIZE03:@.+]] = private {{.*}}constant [1 x i64] zeroinitializer
// CK27: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27: [[SIZE05:@.+]] = private {{.*}}constant [1 x i64] zeroinitializer
// CK27: [[MTYPE05:@.+]] = private {{.*}}constant [1 x i64] [i64 32]

// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27: [[SIZE07:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK27: [[MTYPE07:@.+]] = private {{.*}}constant [1 x i64] [i64 288]

// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27-LABEL: @.__omp_offloading_{{.*}}zero_size_section_and_private_maps{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK27: [[SIZE09:@.+]] = private {{.*}}constant [1 x i64] [i64 40]
// CK27: [[MTYPE09:@.+]] = private {{.*}}constant [1 x i64] [i64 161]

// CK27-LABEL: zero_size_section_and_private_maps{{.*}}(
void zero_size_section_and_private_maps (int ii){

  // Map of a pointer.
  int *pa;

  // Region 00
  // CK27-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK27-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK27-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK27-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK27-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK27-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK27-DAG: store i32* [[VAR0]], i32** [[CP0]]

  // CK27: call void [[CALL00:@.+]](i32* {{[^,]+}})
  #pragma omp target
  {
    pa[50]++;
  }

  // Region 01
  // CK27-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null)
  // CK27-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK27-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK27-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK27-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK27-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
  // CK27-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK27-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK27-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 0
  // CK27-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK27: call void [[CALL01:@.+]](i32* {{[^,]+}})
  #pragma omp target map(pa[:0])
  {
    pa[50]++;
  }

  // Region 02
  // CK27-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}}, i8** null)
  // CK27-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK27-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK27-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK27-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK27-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
  // CK27-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK27-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK27-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 0
  // CK27-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK27: call void [[CALL02:@.+]](i32* {{[^,]+}})
  #pragma omp target map(pa[0:0])
  {
    pa[50]++;
  }

  // Region 03
  // CK27-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}}, i8** null)
  // CK27-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK27-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK27-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK27-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK27-DAG: store i32* [[RVAR0:%.+]], i32** [[CBP0]]
  // CK27-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
  // CK27-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK27-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} %{{.+}}
  // CK27-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK27: call void [[CALL03:@.+]](i32* {{[^,]+}})
  #pragma omp target map(pa[ii:0])
  {
    pa[50]++;
  }

  int *pvtPtr;
  int pvtScl;
  int pvtArr[10];

  // Region 04
  // CK27: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 0, i8** null, i8** null, i{{64|32}}* null, i64* null, i8** null)
  // CK27: call void [[CALL04:@.+]]()
  #pragma omp target private(pvtPtr)
  {
    pvtPtr[5]++;
  }

  // Region 05
  // CK27-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE05]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE05]]{{.+}}, i8** null)
  // CK27-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK27-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK27-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK27-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK27-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK27-DAG: store i32* [[VAR0]], i32** [[CP0]]

  // CK27: call void [[CALL05:@.+]](i32* {{[^,]+}})
  #pragma omp target firstprivate(pvtPtr)
  {
    pvtPtr[5]++;
  }

  // Region 06
  // CK27: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 0, i8** null, i8** null, i{{64|32}}* null, i64* null, i8** null)
  // CK27: call void [[CALL06:@.+]]()
  #pragma omp target private(pvtScl)
  {
    pvtScl++;
  }

  // Region 07
  // CK27-DAG: call i32 @__tgt_target_mapper(i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZE07]]{{.+}}, {{.+}}[[MTYPE07]]{{.+}}, i8** null)
  // CK27-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK27-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK27-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK27-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK27-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i[[Z:64|32]]*
  // CK27-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i[[Z]]*
  // CK27-DAG: store i[[Z]] [[VAL:%.+]], i[[Z]]* [[CBP1]]
  // CK27-DAG: store i[[Z]] [[VAL]], i[[Z]]* [[CP1]]
  // CK27-DAG: [[VAL]] = load i[[Z]], i[[Z]]* [[ADDR:%.+]],
  // CK27-64-DAG: [[CADDR:%.+]] = bitcast i[[Z]]* [[ADDR]] to i32*
  // CK27-64-DAG: store i32 {{.+}}, i32* [[CADDR]],

  // CK27: call void [[CALL07:@.+]](i[[Z]] [[VAL]])
  #pragma omp target firstprivate(pvtScl)
  {
    pvtScl++;
  }

  // Region 08
  // CK27: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 0, i8** null, i8** null, i{{64|32}}* null, i64* null, i8** null)
  // CK27: call void [[CALL08:@.+]]()
  #pragma omp target private(pvtArr)
  {
    pvtArr[5]++;
  }

  // Region 09
  // CK27-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE09]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE09]]{{.+}}, i8** null)
  // CK27-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK27-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK27-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [10 x i32]**
  // CK27-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [10 x i32]**
  // CK27-DAG: store [10 x i32]* [[VAR0:%.+]], [10 x i32]** [[CBP0]]
  // CK27-DAG: store [10 x i32]* [[VAR0]], [10 x i32]** [[CP0]]

  // CK27: call void [[CALL09:@.+]]([10 x i32]* {{[^,]+}})
  #pragma omp target firstprivate(pvtArr)
  {
    pvtArr[5]++;
  }
}

// CK27: define {{.+}}[[CALL00]]
// CK27: define {{.+}}[[CALL01]]
// CK27: define {{.+}}[[CALL02]]
// CK27: define {{.+}}[[CALL03]]
// CK27: define {{.+}}[[CALL04]]
// CK27: define {{.+}}[[CALL05]]
// CK27: define {{.+}}[[CALL06]]
// CK27: define {{.+}}[[CALL07]]
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK28 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK28 --check-prefix CK28-64
// RUN: %clang_cc1 -DCK28 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK28 --check-prefix CK28-64
// RUN: %clang_cc1 -DCK28 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK28 --check-prefix CK28-32
// RUN: %clang_cc1 -DCK28 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK28 --check-prefix CK28-32

// RUN: %clang_cc1 -DCK28 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK28 --check-prefix CK28-64
// RUN: %clang_cc1 -DCK28 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK28 --check-prefix CK28-64
// RUN: %clang_cc1 -DCK28 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK28 --check-prefix CK28-32
// RUN: %clang_cc1 -DCK28 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK28 --check-prefix CK28-32

// RUN: %clang_cc1 -DCK28 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK28 --check-prefix CK28-64
// RUN: %clang_cc1 -DCK28 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK28 --check-prefix CK28-64
// RUN: %clang_cc1 -DCK28 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK28 --check-prefix CK28-32
// RUN: %clang_cc1 -DCK28 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK28 --check-prefix CK28-32

// RUN: %clang_cc1 -DCK28 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY27 %s
// RUN: %clang_cc1 -DCK28 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY27 %s
// RUN: %clang_cc1 -DCK28 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY27 %s
// RUN: %clang_cc1 -DCK28 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY27 %s
// SIMD-ONLY27-NOT: {{__kmpc|__tgt}}
#ifdef CK28

// CK28-LABEL: @.__omp_offloading_{{.*}}explicit_maps_pointer_references{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK28: [[SIZE00:@.+]] = private {{.*}}constant [1 x i64] [i64 {{8|4}}]
// CK28: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK28-LABEL: @.__omp_offloading_{{.*}}explicit_maps_pointer_references{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK28: [[SIZE01:@.+]] = private {{.*}}constant [1 x i64] [i64 400]
// CK28: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 35]

// CK28-LABEL: explicit_maps_pointer_references{{.*}}(
void explicit_maps_pointer_references (int *p){
  int *&a = p;

  // Region 00
  // CK28-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK28-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK28-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK28-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK28-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK28-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32***
  // CK28-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32***
  // CK28-DAG: store i32** [[VAR0:%.+]], i32*** [[CBP0]]
  // CK28-DAG: store i32** [[VAR1:%.+]], i32*** [[CP0]]
  // CK28-DAG: [[VAR0]] = load i32**, i32*** [[VAR00:%.+]],
  // CK28-DAG: [[VAR1]] = load i32**, i32*** [[VAR11:%.+]],

  // CK28: call void [[CALL00:@.+]](i32** {{[^,]+}})
  #pragma omp target map(a)
  {
    ++a;
  }

  // Region 01
  // CK28-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null)
  // CK28-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK28-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK28-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK28-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK28-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK28-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK28-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK28-DAG: store i32* [[VAR1:%.+]], i32** [[CP0]]
  // CK28-DAG: [[VAR0]] = load i32*, i32** [[VAR00:%.+]],
  // CK28-DAG: [[VAR00]] = load i32**, i32*** [[VAR000:%.+]],
  // CK28-DAG: [[VAR1]] = getelementptr inbounds i32, i32* [[VAR11:%.+]], i{{64|32}} 2
  // CK28-DAG: [[VAR11]] = load i32*, i32** [[VAR111:%.+]],
  // CK28-DAG: [[VAR111]] = load i32**, i32*** [[VAR1111:%.+]],

  // CK28: call void [[CALL01:@.+]](i32* {{[^,]+}})
  #pragma omp target map(a[2:100])
  {
    ++a;
  }
}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK29 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK29 --check-prefix CK29-64
// RUN: %clang_cc1 -DCK29 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK29 --check-prefix CK29-64
// RUN: %clang_cc1 -DCK29 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK29 --check-prefix CK29-32
// RUN: %clang_cc1 -DCK29 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK29 --check-prefix CK29-32

// RUN: %clang_cc1 -DCK29 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK29 --check-prefix CK29-64
// RUN: %clang_cc1 -DCK29 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK29 --check-prefix CK29-64
// RUN: %clang_cc1 -DCK29 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK29 --check-prefix CK29-32
// RUN: %clang_cc1 -DCK29 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK29 --check-prefix CK29-32

// RUN: %clang_cc1 -DCK29 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK29 --check-prefix CK29-64
// RUN: %clang_cc1 -DCK29 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK29 --check-prefix CK29-64
// RUN: %clang_cc1 -DCK29 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK29 --check-prefix CK29-32
// RUN: %clang_cc1 -DCK29 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK29 --check-prefix CK29-32

// RUN: %clang_cc1 -DCK29 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY28 %s
// RUN: %clang_cc1 -DCK29 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY28 %s
// RUN: %clang_cc1 -DCK29 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY28 %s
// RUN: %clang_cc1 -DCK29 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY28 %s
// SIMD-ONLY28-NOT: {{__kmpc|__tgt}}
#ifdef CK29

// CK29: [[SSA:%.+]] = type { double*, double** }
// CK29: [[SSB:%.+]]  = type { [[SSA]]*, [[SSA]]** }

// CK29-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK29: [[MTYPE00:@.+]] = private {{.*}}constant [3 x i64] [i64 32, i64 281474976710672, i64 19]

// CK29-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK29: [[MTYPE01:@.+]] = private {{.*}}constant [3 x i64] [i64 32, i64 281474976710672, i64 19]

// CK29-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK29: [[MTYPE02:@.+]] = private {{.*}}constant [3 x i64] [i64 32, i64 281474976710672, i64 19]

struct SSA{
  double *p;
  double *&pr;
  SSA(double *&pr) : pr(pr) {}
};

struct SSB{
  SSA *p;
  SSA *&pr;
  SSB(SSA *&pr) : pr(pr) {}

  // CK29-LABEL: define {{.+}}foo
  void foo() {

    // Region 00
    // CK29-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[Z:64|32]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)

    // CK29-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK29-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
    // CK29-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

    // CK29-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SSB]]**
    // CK29-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SSA]]**
    // CK29-DAG: store [[SSB]]* [[VAR0:%.+]], [[SSB]]** [[CBP0]]
    // CK29-DAG: store [[SSA]]** [[VAR00:%.+]], [[SSA]]*** [[CP0]]
    // CK29-DAG: store i64 %{{.+}}, i64* [[S0]]
    // CK29-DAG: [[VAR0]] = load [[SSB]]*, [[SSB]]** %
    // CK29-DAG: [[VAR00]] = getelementptr inbounds [[SSB]], [[SSB]]* [[VAR0]], i32 0, i32 0

    // CK29-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SSA]]***
    // CK29-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double***
    // CK29-DAG: store [[SSA]]** [[VAR00]], [[SSA]]*** [[CBP1]]
    // CK29-DAG: store double** [[VAR1:%.+]], double*** [[CP1]]
    // CK29-DAG: store i64 {{8|4}}, i64* [[S1]]
    // CK29-DAG: [[VAR1]] = load double**, double*** [[VAR1_REF:%.+]],
    // CK29-DAG: [[VAR1_REF]] = getelementptr inbounds [[SSA]], [[SSA]]* %{{.+}}, i32 0, i32 1

    // CK29-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
    // CK29-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
    // CK29-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
    // CK29-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to double***
    // CK29-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to double**
    // CK29-DAG: store double** [[VAR1]], double*** [[CBP2]]
    // CK29-DAG: store double* [[VAR2:%.+]], double** [[CP2]]
    // CK29-DAG: store i64 80, i64* [[S2]]
    // CK29-DAG: [[VAR2]] = getelementptr inbounds double, double* [[VAR22:%.+]], i{{.+}} 0
    // CK29-DAG: [[VAR22]] = load double*, double** %{{.+}},

    // CK29: call void [[CALL00:@.+]]([[SSB]]* {{[^,]+}})
    #pragma omp target map(p->pr[:10])
    {
      p->pr++;
    }

    // Region 01
    // CK29-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null)

    // CK29-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK29-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
    // CK29-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

    // CK29-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SSB]]**
    // CK29-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SSA]]**
    // CK29-DAG: store [[SSB]]* [[VAR0]], [[SSB]]** [[CBP0]]
    // CK29-DAG: store [[SSA]]** [[VAR00:%.+]], [[SSA]]*** [[CP0]]
    // CK29-DAG: store i64 %{{.+}}, i64* [[S0]]
    // CK29-DAG: [[VAR00]] = load [[SSA]]**, [[SSA]]*** [[VAR000:%.+]],
    // CK29-DAG: [[VAR000]] = getelementptr inbounds [[SSB]], [[SSB]]* [[VAR0]], i32 0, i32 1

    // CK29-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SSA]]***
    // CK29-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double***
    // CK29-DAG: store [[SSA]]** [[VAR00]], [[SSA]]*** [[CBP1]]
    // CK29-DAG: store double** [[VAR1:%.+]], double*** [[CP1]]
    // CK29-DAG: store i64 {{8|4}}, i64* [[S1]]
    // CK29-DAG: [[VAR1]] = getelementptr inbounds [[SSA]], [[SSA]]* %{{.+}}, i32 0, i32 0

    // CK29-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
    // CK29-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
    // CK29-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
    // CK29-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to double***
    // CK29-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to double**
    // CK29-DAG: store double** [[VAR1]], double*** [[CBP2]]
    // CK29-DAG: store double* [[VAR2:%.+]], double** [[CP2]]
    // CK29-DAG: store i64 80, i64* [[S2]]
    // CK29-DAG: [[VAR2]] = getelementptr inbounds double, double* [[VAR22:%.+]], i{{.+}} 0
    // CK29-DAG: [[VAR22]] = load double*, double** %{{.+}},

    // CK29: call void [[CALL00:@.+]]([[SSB]]* {{[^,]+}})
    #pragma omp target map(pr->p[:10])
    {
      pr->p++;
    }

    // Region 02
    // CK29-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE02]]{{.+}}, i8** null)

    // CK29-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK29-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
    // CK29-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

    // CK29-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[SSB]]**
    // CK29-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[SSA]]**
    // CK29-DAG: store [[SSB]]* [[VAR0]], [[SSB]]** [[CBP0]]
    // CK29-DAG: store [[SSA]]** [[VAR00:%.+]], [[SSA]]*** [[CP0]]
    // CK29-DAG: store i64 %{{.+}}, i64* [[S0]]
    // CK29-DAG: [[VAR00]] = load [[SSA]]**, [[SSA]]*** [[VAR000:%.+]],
    // CK29-DAG: [[VAR000]] = getelementptr inbounds [[SSB]], [[SSB]]* [[VAR0]], i32 0, i32 1

    // CK29-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[SSA]]***
    // CK29-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double***
    // CK29-DAG: store [[SSA]]** [[VAR00]], [[SSA]]*** [[CBP1]]
    // CK29-DAG: store double** [[VAR1:%.+]], double*** [[CP1]]
    // CK29-DAG: store i64 {{8|4}}, i64* [[S1]]
    // CK29-DAG: [[VAR1]] = load double**, double*** [[VAR1_REF:%.+]],
    // CK29-DAG: [[VAR1_REF]] = getelementptr inbounds [[SSA]], [[SSA]]* %{{.+}}, i32 0, i32 1

    // CK29-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
    // CK29-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
    // CK29-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
    // CK29-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to double***
    // CK29-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to double**
    // CK29-DAG: store double** [[VAR1]], double*** [[CBP2]]
    // CK29-DAG: store double* [[VAR2:%.+]], double** [[CP2]]
    // CK29-DAG: store i64 80, i64* [[S2]]
    // CK29-DAG: [[VAR2]] = getelementptr inbounds double, double* [[VAR22:%.+]], i{{.+}} 0
    // CK29-DAG: [[VAR22]] = load double*, double** %{{.+}},

    // CK29: call void [[CALL00:@.+]]([[SSB]]* {{[^,]+}})
    #pragma omp target map(pr->pr[:10])
    {
      pr->pr++;
    }
  }
};

void explicit_maps_member_pointer_references(SSA *sap) {
  double *d;
  SSA sa(d);
  SSB sb(sap);
  sb.foo();
}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK30 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK30 --check-prefix CK30-64
// RUN: %clang_cc1 -DCK30 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-64
// RUN: %clang_cc1 -DCK30 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-32
// RUN: %clang_cc1 -DCK30 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-32

// RUN: %clang_cc1 -DCK30 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK30 --check-prefix CK30-64
// RUN: %clang_cc1 -DCK30 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-64
// RUN: %clang_cc1 -DCK30 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-32
// RUN: %clang_cc1 -DCK30 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-32

// RUN: %clang_cc1 -DCK30 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK30 --check-prefix CK30-64
// RUN: %clang_cc1 -DCK30 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-64
// RUN: %clang_cc1 -DCK30 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-32
// RUN: %clang_cc1 -DCK30 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK30 --check-prefix CK30-32

// RUN: %clang_cc1 -DCK30 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY30 %s
// RUN: %clang_cc1 -DCK30 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY30 %s
// RUN: %clang_cc1 -DCK30 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY30 %s
// RUN: %clang_cc1 -DCK30 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY30 %s
// SIMD-ONLY30-NOT: {{__kmpc|__tgt}}
#ifdef CK30

// CK30-DAG: [[BASE:%.+]] = type { i32*, i32, i32* }
// CK30-DAG: [[STRUCT:%.+]] = type { [[BASE]], i32*, i32*, i32, i32* }

// CK30-LABEL: @.__omp_offloading_{{.*}}map_with_deep_copy{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// The first element: 0x20 - OMP_MAP_TARGET_PARAM
// 2-4: 0x1000000000003 - OMP_MAP_MEMBER_OF(0) | OMP_MAP_TO | OMP_MAP_FROM - copies all the data in structs excluding deep-copied elements (from &s to &s.ptrBase1, from &s.ptr to &s.ptr1, from &s.ptr1 to end of s).
// 5-6: 0x1000000000013 - OMP_MAP_MEMBER_OF(0) | OMP_MAP_PTR_AND_OBJ | OMP_MAP_TO | OMP_MAP_FROM - deep copy of the pointers + pointee.
// CK30: [[MTYPE00:@.+]] = private {{.*}}constant [6 x i64] [i64 32, i64 281474976710659, i64 281474976710659, i64 281474976710659, i64 281474976710675, i64 281474976710675]

typedef struct {
  int *ptrBase;
  int valBase;
  int *ptrBase1;
} Base;

typedef struct StructWithPtrTag : public Base {
  int *ptr;
  int *ptr2;
  int val;
  int *ptr1;
} StructWithPtr;

// CK30-DAG: call i32 @__tgt_target_mapper(i64 -1, i8* @.__omp_offloading_{{.*}}map_with_deep_copy{{.*}}_l{{[0-9]+}}.region_id, i32 6, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], i64* getelementptr inbounds ([6 x i64], [6 x i64]* [[MTYPE00]], i32 0, i32 0), i8** null)
// CK30-DAG: [[GEPS]] = getelementptr inbounds [6 x i{{64|32}}], [6 x i64]* [[SIZES:%.+]], i32 0, i32 0
// CK30-DAG: [[GEPP]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[PTRS:%.+]], i32 0, i32 0
// CK30-DAG: [[GEPBP]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BASES:%.+]], i32 0, i32 0

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BASES]], i32 0, i32 0
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[BASE_PTR]] to [[STRUCT]]**
// CK30-DAG: store [[STRUCT]]* [[S:%.+]], [[STRUCT]]** [[BC]],
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[PTRS]], i32 0, i32 0
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[PTR]] to [[STRUCT]]**
// CK30-DAG: store [[STRUCT]]* [[S]], [[STRUCT]]** [[BC]],
// CK30-DAG: [[SIZE:%.+]] = getelementptr inbounds [6 x i{{64|32}}], [6 x i64]* [[SIZES]], i32 0, i32 0
// CK30-DAG: store i64 [[S_ALLOC_SIZE:%.+]], i64* [[SIZE]],
// CK30-DAG: [[S_ALLOC_SIZE]] = sdiv exact i64 [[DIFF:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
// CK30-DAG: [[DIFF]] = sub i64 [[S_END_BC:%.+]], [[S_BEGIN_BC:%.+]]
// CK30-DAG: [[S_BEGIN_BC]] = ptrtoint i8* [[S_BEGIN:%.+]] to i64
// CK30-DAG: [[S_END_BC]] = ptrtoint i8* [[S_END:%.+]] to i64
// CK30-DAG: [[S_BEGIN]] = bitcast [[STRUCT]]* [[S]] to i8*
// CK30-DAG: [[S_END]] = getelementptr i8, i8* [[S_LAST:%.+]], i32 1
// CK30-DAG: [[S_LAST]] = getelementptr i8, i8* [[S_BC:%.+]], i{{64|32}} {{55|27}}
// CK30-DAG: [[S_BC]] = bitcast [[STRUCT]]* [[S]] to i8*

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BASES]], i32 0, i32 1
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[BASE_PTR]] to [[STRUCT]]**
// CK30-DAG: store [[STRUCT]]* [[S]], [[STRUCT]]** [[BC]],
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[PTRS]], i32 0, i32 1
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[PTR]] to [[STRUCT]]**
// CK30-DAG: store [[STRUCT]]* [[S]], [[STRUCT]]** [[BC]],
// CK30-DAG: [[SIZE:%.+]] = getelementptr inbounds [6 x i64], [6 x i64]* [[SIZES]], i32 0, i32 1
// CK30-DAG: store i64 [[SIZE1:%.+]], i64* [[SIZE]],
// CK30-DAG: [[SIZE1]] = sdiv exact i64 [[DIFF:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
// CK30-DAG: [[DIFF]] = sub i64 [[S_PTRBASE1_BC:%.+]], [[S_BEGIN_BC:%.+]]
// CK30-DAG: [[S_BEGIN_BC]] = ptrtoint i8* [[S_BEGIN:%.+]] to i64
// CK30-DAG: [[S_PTRBASE1_BC]] = ptrtoint i8* [[S_PTRBASE1:%.+]] to i64
// CK30-DAG: [[S_PTRBASE1]] = bitcast i32** [[S_PTRBASE1_REF:%.+]] to i8*
// CK30-DAG: [[S_BEGIN]] = bitcast [[STRUCT]]* [[S]] to i8*
// CK30-DAG: [[S_PTRBASE1_REF]] = getelementptr inbounds [[BASE]], [[BASE]]* [[BASE_ADDR:%.+]], i32 0, i32 2
// CK30-DAG: [[BASE_ADDR]] = bitcast [[STRUCT]]* [[S]] to [[BASE]]*

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BASES]], i32 0, i32 2
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[BASE_PTR]] to [[STRUCT]]**
// CK30-DAG: store [[STRUCT]]* [[S]], [[STRUCT]]** [[BC]],
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[PTRS]], i32 0, i32 2
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[PTR]] to i32***
// CK30-DAG: store i32** [[PTR1:%.+]], i32*** [[BC]],
// CK30-DAG: [[SIZE:%.+]] = getelementptr inbounds [6 x i64], [6 x i64]* [[SIZES]], i32 0, i32 2
// CK30-DAG: store i64 [[SIZE2:%.+]], i64* [[SIZE]],
// CK30-DAG: [[PTR1]] = getelementptr i32*, i32** [[S_PTRBASE1_REF]], i{{64|32}} 1
// CK30-DAG: [[SIZE2]] = sdiv exact i64 [[DIFF:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
// CK30-DAG: [[DIFF]] = sub i64 [[S_PTR1_BC:%.+]], [[S_PTRBASE1_BC:%.+]]
// CK30-DAG: [[S_PTR1_BC]] = ptrtoint i8* [[S_PTR1:%.+]] to i64
// CK30-DAG: [[S_PTRBASE1_BC]] = ptrtoint i8* [[S_PTRBASE1:%.+]] to i64
// CK30-DAG: [[S_PTR1]] = bitcast i32** [[S_PTR1_REF:%.+]] to i8*
// CK30-DAG: [[S_PTRBASE1]] = bitcast i32** [[PTR1]] to i8*
// CK30-DAG: [[S_PTR1_REF]] = getelementptr inbounds [[STRUCT]], [[STRUCT]]* [[S]], i32 0, i32 4

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BASES]], i32 0, i32 3
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[BASE_PTR]] to [[STRUCT]]**
// CK30-DAG: store [[STRUCT]]* [[S]], [[STRUCT]]** [[BC]],
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[PTRS]], i32 0, i32 3
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[PTR]] to i32***
// CK30-DAG: store i32** [[PTR2:%.+]], i32*** [[BC]],
// CK30-DAG: [[SIZE:%.+]] = getelementptr inbounds [6 x i64], [6 x i64]* [[SIZES]], i32 0, i32 3
// CK30-DAG: store i64 [[SIZE3:%.+]], i64* [[SIZE]],
// CK30-DAG: [[PTR2]] = getelementptr i32*, i32** [[S_PTR1_REF]], i{{64|32}} 1
// CK30-DAG: [[SIZE3]] = sdiv exact i64 [[DIFF:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
// CK30-DAG: [[DIFF]] = sub i64 [[S_END_BC:%.+]], [[S_PTR1_BC:%.+]]
// CK30-DAG: [[S_PTR1_BC]] = ptrtoint i8* [[S_PTR1:%.+]] to i64
// CK30-DAG: [[S_END_BC]] = ptrtoint i8* [[S_END:%.+]] to i64
// CK30-DAG: [[S_PTR1]] = bitcast i32** [[PTR2]] to i8*
// CK30-DAG: [[S_END]] = getelementptr i8, i8* [[S_LAST]], i{{64|32}} 1

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BASES]], i32 0, i32 4
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[BASE_PTR]] to i32***
// CK30-DAG: store i32** [[S_PTR1:%.+]], i32*** [[BC]],
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[PTRS]], i32 0, i32 4
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[PTR]] to i32**
// CK30-DAG: store i32* [[S_PTR1_BEGIN:%.+]], i32** [[BC]],
// CK30-DAG: [[SIZE:%.+]] = getelementptr inbounds [6 x i64], [6 x i64]* [[SIZES]], i32 0, i32 4
// CK30-DAG: store i64 4, i64* [[SIZE]],
// CK30-DAG: [[S_PTR1]] = getelementptr inbounds [[STRUCT]], [[STRUCT]]* [[S]], i32 0, i32 4
// CK30-DAG: [[S_PTR1_BEGIN]] = getelementptr inbounds i32, i32* [[S_PTR1_BEGIN_REF:%.+]], i{{64|32}} 0
// CK30-DAG: [[S_PTR1_BEGIN_REF]] = load i32*, i32** [[S_PTR1:%.+]],
// CK30-DAG: [[S_PTR1]] = getelementptr inbounds [[STRUCT]], [[STRUCT]]* [[S]], i32 0, i32 4

// CK30-DAG: [[BASE_PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[BASES]], i32 0, i32 5
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[BASE_PTR]] to i32***
// CK30-DAG: store i32** [[S_PTRBASE1:%.+]], i32*** [[BC]],
// CK30-DAG: [[PTR:%.+]] = getelementptr inbounds [6 x i8*], [6 x i8*]* [[PTRS]], i32 0, i32 5
// CK30-DAG: [[BC:%.+]] = bitcast i8** [[PTR]] to i32**
// CK30-DAG: store i32* [[S_PTRBASE1_BEGIN:%.+]], i32** [[BC]],
// CK30-DAG: [[SIZE:%.+]] = getelementptr inbounds [6 x i{{64|32}}], [6 x i{{64|32}}]* [[SIZES]], i32 0, i32 5
// CK30-DAG: store i{{64|32}} 4, i{{64|32}}* [[SIZE]],
// CK30-DAG: [[S_PTRBASE1]] = getelementptr inbounds [[BASE]], [[BASE]]* [[S_BASE:%.+]], i32 0, i32 2
// CK30-DAG: [[S_BASE]] = bitcast [[STRUCT]]* [[S]] to [[BASE]]*
// CK30-DAG: [[S_PTRBASE1_BEGIN]] = getelementptr inbounds i32, i32* [[S_PTRBASE1_BEGIN_REF:%.+]], i{{64|32}} 0
// CK30-DAG: [[S_PTRBASE1_BEGIN_REF]] = load i32*, i32** [[S_PTRBASE1:%.+]],
// CK30-DAG: [[S_PTRBASE1]] = getelementptr inbounds [[BASE]], [[BASE]]* [[S_BASE:%.+]], i32 0, i32 2
// CK30-DAG: [[S_BASE]] = bitcast [[STRUCT]]* [[S]] to [[BASE]]*
void map_with_deep_copy() {
  StructWithPtr s;
#pragma omp target map(s, s.ptr1 [0:1], s.ptrBase1 [0:1])
  {
    s.val++;
    s.ptr1[0]++;
    s.ptrBase1[0] = 10001;
  }
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK31 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK31 --check-prefix CK31-64
// RUN: %clang_cc1 -DCK31 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK31 --check-prefix CK31-64
// RUN: %clang_cc1 -DCK31 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK31 --check-prefix CK31-32
// RUN: %clang_cc1 -DCK31 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK31 --check-prefix CK31-32

// RUN: %clang_cc1 -DCK31 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK31 --check-prefix CK31-64
// RUN: %clang_cc1 -DCK31 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK31 --check-prefix CK31-64
// RUN: %clang_cc1 -DCK31 -verify -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK31 --check-prefix CK31-32
// RUN: %clang_cc1 -DCK31 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=45 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK31 --check-prefix CK31-32

// RUN: %clang_cc1 -DCK31 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefix CK31 --check-prefix CK31-64
// RUN: %clang_cc1 -DCK31 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK31 --check-prefix CK31-64
// RUN: %clang_cc1 -DCK31 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK31 --check-prefix CK31-32
// RUN: %clang_cc1 -DCK31 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefix CK31 --check-prefix CK31-32

// RUN: %clang_cc1 -DCK31 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK31 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK31 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK31 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// SIMD-ONLY18-NOT: {{__kmpc|__tgt}}
#ifdef CK31

// CK31-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK31: [[SIZE00:@.+]] = private {{.*}}constant [1 x i[[Z:64|32]]] [i[[Z:64|32]] 4]
// CK31: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 1059]

// CK31-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK31: [[SIZE01:@.+]] = private {{.*}}constant [1 x i[[Z:64|32]]] [i[[Z:64|32]] 4]
// CK31: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 1063]

// CK31-LABEL: explicit_maps_single{{.*}}(
void explicit_maps_single (int ii){
  // Map of a scalar.
  int a = ii;

  // Close.
  // Region 00
  // CK31-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK31-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK31-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK31-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK31-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK31-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK31-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK31-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK31-DAG: store i32* [[VAR0]], i32** [[CP0]]

  // CK31: call void [[CALL00:@.+]](i32* {{[^,]+}})
  #pragma omp target map(close, tofrom: a)
  {
   a++;
  }

  // Always Close.
  // Region 01
  // CK31-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null)
  // CK31-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK31-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK31-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK31-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK31-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK31-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK31-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK31-DAG: store i32* [[VAR0]], i32** [[CP0]]

  // CK31: call void [[CALL01:@.+]](i32* {{[^,]+}})
  #pragma omp target map(always close tofrom: a)
  {
   a++;
  }
}
// CK31: define {{.+}}[[CALL00]]
// CK31: define {{.+}}[[CALL01]]

#endif
///==========================================================================///
// RUN: %clang_cc1 -DUSE -DCK31A -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK31A,CK31A-64,CK31A-USE
// RUN: %clang_cc1 -DUSE -DCK31A -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31A,CK31A-64,CK31A-USE
// RUN: %clang_cc1 -DUSE -DCK31A -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31A,CK31A-32,CK31A-USE
// RUN: %clang_cc1 -DUSE -DCK31A -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31A,CK31A-32,CK31A-USE

// RUN: %clang_cc1 -DUSE -DCK31A -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DUSE -DCK31A -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DUSE -DCK31A -verify -fopenmp-version=51 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DUSE -DCK31A -fopenmp-version=51 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s

// RUN: %clang_cc1 -DCK31A -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK31A,CK31A-64,CK31A-NOUSE
// RUN: %clang_cc1 -DCK31A -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31A,CK31A-64,CK31A-NOUSE
// RUN: %clang_cc1 -DCK31A -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31A,CK31A-32,CK31A-NOUSE
// RUN: %clang_cc1 -DCK31A -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31A,CK31A-32,CK31A-NOUSE

// RUN: %clang_cc1 -DCK31A -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK31A -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK31A -verify -fopenmp-version=51 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK31A -fopenmp-version=51 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// SIMD-ONLY18-NOT: {{__kmpc|__tgt}}
#ifdef CK31A

// CK31A: [[ST:%.+]] = type { i32, i32 }
struct ST {
  int i;
  int j;
};

// CK31A-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
//
// PRESENT=0x1000 | TARGET_PARAM=0x20 = 0x1020
// CK31A: [[MTYPE00:@.+]] = private {{.*}}constant [7 x i64] [i64 [[#0x1020]],
//
// MEMBER_OF_1=0x1000000000000 | FROM=0x2 | TO=0x1 = 0x1000000000003
// MEMBER_OF_1=0x1000000000000 | PRESENT=0x1000 | FROM=0x2 | TO=0x1 = 0x1000000001003
// PRESENT=0x1000 | TARGET_PARAM=0x20 | FROM=0x2 | TO=0x1 = 0x1023
// CK31A-SAME: {{^}} i64 [[#0x1000000000003]], i64 [[#0x1000000001003]], i64 [[#0x1023]],
//
// PRESENT=0x1000 | TARGET_PARAM=0x20 = 0x1020
// MEMBER_OF_5=0x5000000000000 | PRESENT=0x1000 | FROM=0x2 | TO=0x1 = 0x5000000001003
// MEMBER_OF_5=0x5000000000000 | FROM=0x2 | TO=0x1 = 0x5000000000003
// CK31A-SAME: {{^}} i64 [[#0x1020]], i64 [[#0x5000000001003]], i64 [[#0x5000000000003]]]

// CK31A-LABEL: @.__omp_offloading_{{.*}}explicit_maps_single{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK31A: [[SIZE01:@.+]] = private {{.*}}constant [1 x i[[Z:64|32]]] [i[[Z:64|32]] 4]
//
// PRESENT=0x1000 | CLOSE=0x400 | TARGET_PARAM=0x20 | ALWAYS=0x4 | FROM=0x2 | TO=0x1 = 0x1427
// CK31A: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 [[#0x1427]]]

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
  // CK31A: [[ST1_I:%.+]] = getelementptr inbounds [[ST]], [[ST]]* [[ST1]], i{{.+}} 0, i{{.+}} 0
  // CK31A: [[ST1_J:%.+]] = getelementptr inbounds [[ST]], [[ST]]* [[ST1]], i{{.+}} 0, i{{.+}} 1
  // CK31A: [[ST2_I:%.+]] = getelementptr inbounds [[ST]], [[ST]]* [[ST2]], i{{.+}} 0, i{{.+}} 0
  // CK31A: [[ST2_J:%.+]] = getelementptr inbounds [[ST]], [[ST]]* [[ST2]], i{{.+}} 0, i{{.+}} 1
  // CK31A-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 7, i8** [[GEPBP:%[0-9]+]], i8** [[GEPP:%[0-9]+]], i64* [[GEPS:%.+]], i64* getelementptr {{.+}}[7 x i{{.+}}]* [[MTYPE00]]{{.+}})
  // CK31A-DAG: [[GEPS]] = getelementptr inbounds [7 x i64], [7 x i64]* [[S:%.+]], i{{.+}} 0, i{{.+}} 0
  // CK31A-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK31A-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // st1
  // CK31A-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK31A-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK31A-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK31A-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
  // CK31A-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK31A-DAG: store [[ST]]* [[ST1]], [[ST]]** [[CBP0]]
  // CK31A-DAG: store i32* [[ST1_I]], i32** [[CP0]]
  // CK31A-DAG: store i64 %{{.+}}, i64* [[S0]]

  // st1.i
  // CK31A-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK31A-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK31A-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK31A-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[ST]]**
  // CK31A-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK31A-DAG: store [[ST]]* [[ST1]], [[ST]]** [[CBP1]]
  // CK31A-DAG: store i32* [[ST1_I]], i32** [[CP1]]
  // CK31A-DAG: store i64 4, i64* [[S1]]

  // st1.j
  // CK31A-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK31A-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK31A-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK31A-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [[ST]]**
  // CK31A-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to i32**
  // CK31A-DAG: store [[ST]]* [[ST1]], [[ST]]** [[CBP2]]
  // CK31A-DAG: store i32* [[ST1_J]], i32** [[CP2]]
  // CK31A-DAG: store i64 4, i64* [[S2]]

  // a
  // CK31A-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
  // CK31A-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
  // CK31A-DAG: [[S3:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 3
  // CK31A-DAG: [[CBP3:%.+]] = bitcast i8** [[BP3]] to i32**
  // CK31A-DAG: [[CP3:%.+]] = bitcast i8** [[P3]] to i32**
  // CK31A-DAG: store i32* [[A]], i32** [[CBP3]]
  // CK31A-DAG: store i32* [[A]], i32** [[CP3]]
  // CK31A-DAG: store i64 4, i64* [[S3]]

  // st2
  // CK31A-DAG: [[BP4:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 4
  // CK31A-DAG: [[P4:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 4
  // CK31A-DAG: [[S4:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 4
  // CK31A-DAG: [[CBP4:%.+]] = bitcast i8** [[BP4]] to [[ST]]**
  // CK31A-DAG: [[CP4:%.+]] = bitcast i8** [[P4]] to i32**
  // CK31A-DAG: store [[ST]]* [[ST2]], [[ST]]** [[CBP4]]
  // CK31A-DAG: store i32* [[ST2_I]], i32** [[CP4]]
  // CK31A-DAG: store i64 %{{.+}}, i64* [[S4]]

  // st2.i
  // CK31A-DAG: [[BP5:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 5
  // CK31A-DAG: [[P5:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 5
  // CK31A-DAG: [[S5:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 5
  // CK31A-DAG: [[CBP5:%.+]] = bitcast i8** [[BP5]] to [[ST]]**
  // CK31A-DAG: [[CP5:%.+]] = bitcast i8** [[P5]] to i32**
  // CK31A-DAG: store [[ST]]* [[ST2]], [[ST]]** [[CBP5]]
  // CK31A-DAG: store i32* [[ST2_I]], i32** [[CP5]]
  // CK31A-DAG: store i64 4, i64* [[S5]]

  // st2.j
  // CK31A-DAG: [[BP6:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 6
  // CK31A-DAG: [[P6:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 6
  // CK31A-DAG: [[S6:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 6
  // CK31A-DAG: [[CBP6:%.+]] = bitcast i8** [[BP6]] to [[ST]]**
  // CK31A-DAG: [[CP6:%.+]] = bitcast i8** [[P6]] to i32**
  // CK31A-DAG: store [[ST]]* [[ST2]], [[ST]]** [[CBP6]]
  // CK31A-DAG: store i32* [[ST2_J]], i32** [[CP6]]
  // CK31A-DAG: store i64 4, i64* [[S6]]

  // CK31A-USE: call void [[CALL00:@.+]]([[ST]]* [[ST1]], i32* [[A]], [[ST]]* [[ST2]])
  // CK31A-NOUSE: call void [[CALL00:@.+]]()
  #pragma omp target map(tofrom: st1.i) map(present, tofrom: a, st1.j, st2.i) map(tofrom: st2.j)
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
  // CK31A-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}})
  // CK31A-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK31A-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK31A-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK31A-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK31A-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK31A-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK31A-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK31A-DAG: store i32* [[VAR0]], i32** [[CP0]]

  // CK31A-USE: call void [[CALL01:@.+]](i32* {{[^,]+}})
  // CK31A-NOUSE: call void [[CALL01:@.+]]()
  #pragma omp target map(always close present tofrom: a)
  {
#ifdef USE
    a++;
#endif
  }
}
// CK31A: define {{.+}}[[CALL00]]
// CK31A: define {{.+}}[[CALL01]]

#endif
///==========================================================================///
// RUN: %clang_cc1 -DUSE -DCK31B -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK31B,CK31B-64,CK31B-USE
// RUN: %clang_cc1 -DUSE -DCK31B -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31B,CK31B-64,CK31B-USE
// RUN: %clang_cc1 -DUSE -DCK31B -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31B,CK31B-32,CK31B-USE
// RUN: %clang_cc1 -DUSE -DCK31B -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31B,CK31B-32,CK31B-USE

// RUN: %clang_cc1 -DUSE -DCK31B -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DUSE -DCK31B -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DUSE -DCK31B -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DUSE -DCK31B -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -DUSE -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s

// RUN: %clang_cc1 -DCK31B -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s --check-prefixes=CK31B,CK31B-64,CK31B-NOUSE
// RUN: %clang_cc1 -DCK31B -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31B,CK31B-64,CK31B-NOUSE
// RUN: %clang_cc1 -DCK31B -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31B,CK31B-32,CK31B-NOUSE
// RUN: %clang_cc1 -DCK31B -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  %s  --check-prefixes=CK31B,CK31B-32,CK31B-NOUSE

// RUN: %clang_cc1 -DCK31B -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK31B -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK31B -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK31B -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck -allow-deprecated-dag-overlap  --check-prefix SIMD-ONLY18 %s
// SIMD-ONLY18-NOT: {{__kmpc|__tgt}}
#ifdef CK31B

// CK31B: [[ST:%.+]] = type { i32, i32 }

// PRESENT=0x1000 | TARGET_PARAM=0x20 = 0x1020
// MEMBER_OF_1=0x1000000000000 | FROM=0x2 | TO=0x1 = 0x1000000000003
// MEMBER_OF_1=0x1000000000000 | PRESENT=0x1000 | FROM=0x2 | TO=0x1 = 0x1000000001003

// CK31B-LABEL: @.__omp_offloading_{{.*}}test_present_members{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK31B: [[MTYPE00:@.+]] = private {{.*}}constant [3 x i64] [i64 [[#0x1020]],
// CK31B-SAME: {{^}} i64 [[#0x1000000000003]], i64 [[#0x1000000001003]]]

struct ST {
  int i;
  int j;
  // CK31B-LABEL: define {{.*}}test_present_members{{.*}}(
  void test_present_members() {
    // Make sure the struct picks up present even if another element of the
    // struct doesn't have present.
    // Region 00
    // CK31B: [[I:%.+]] = getelementptr inbounds [[ST]], [[ST]]* [[THIS:%.+]], i{{.+}} 0, i{{.+}} 0
    // CK31B: [[J:%.+]] = getelementptr inbounds [[ST]], [[ST]]* [[THIS]], i{{.+}} 0, i{{.+}} 1
    // CK31B-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%[0-9]+]], i8** [[GEPP:%[0-9]+]], i64* [[GEPS:%.+]], i64* getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE00]]{{.+}})
    // CK31B-DAG: [[GEPS]] = getelementptr inbounds [3 x i64], [3 x i64]* [[S:%.+]], i{{.+}} 0, i{{.+}} 0
    // CK31B-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK31B-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // this
    // CK31B-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK31B-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK31B-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
    // CK31B-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
    // CK31B-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
    // CK31B-DAG: store [[ST]]* [[THIS]], [[ST]]** [[CBP0]]
    // CK31B-DAG: store i32* [[I]], i32** [[CP0]]
    // CK31B-DAG: store i64 %{{.+}}, i64* [[S0]]

    // i
    // CK31B-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
    // CK31B-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
    // CK31B-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
    // CK31B-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to [[ST]]**
    // CK31B-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
    // CK31B-DAG: store [[ST]]* [[THIS]], [[ST]]** [[CBP1]]
    // CK31B-DAG: store i32* [[I]], i32** [[CP1]]
    // CK31B-DAG: store i64 4, i64* [[S1]]

    // j
    // CK31B-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
    // CK31B-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
    // CK31B-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
    // CK31B-DAG: [[CBP2:%.+]] = bitcast i8** [[BP2]] to [[ST]]**
    // CK31B-DAG: [[CP2:%.+]] = bitcast i8** [[P2]] to i32**
    // CK31B-DAG: store [[ST]]* [[THIS]], [[ST]]** [[CBP2]]
    // CK31B-DAG: store i32* [[J]], i32** [[CP2]]
    // CK31B-DAG: store i64 4, i64* [[S2]]

    // CK31B-USE: call void [[CALL00:@.+]]([[ST]]* [[THIS]])
    // CK31B-NOUSE: call void [[CALL00:@.+]]()
    #pragma omp target map(tofrom: i) map(present, tofrom: j)
    {
#ifdef USE
      i++;
      j++;
#endif
    }
  }
};

void test() {
  ST s;
  s.test_present_members();
}

// CK31B: define {{.+}}[[CALL00]]

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK32 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK32 --check-prefix CK32-64
// RUN: %clang_cc1 -DCK32 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK32 --check-prefix CK32-64
// RUN: %clang_cc1 -DCK32 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK32 --check-prefix CK32-32
// RUN: %clang_cc1 -DCK32 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK32 --check-prefix CK32-32

// RUN: %clang_cc1 -DCK32 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// RUN: %clang_cc1 -DCK32 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// RUN: %clang_cc1 -DCK32 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// RUN: %clang_cc1 -DCK32 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// SIMD-ONLY32-NOT: {{__kmpc|__tgt}}
#ifdef CK32

// CK32-DAG: [[MTYPE_TO:@.+]] = {{.+}}constant [1 x i64] [i64 33]
// CK32-DAG: [[MTYPE_FROM:@.+]] = {{.+}}constant [1 x i64] [i64 34]

void array_shaping(float *f, int sa) {

  // CK32-DAG: call i32 @__tgt_target_mapper(i64 -1, i8* @{{.+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE_TO]]{{.+}}, i8** null)
  // CK32-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK32-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK32-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK32-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK32-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK32-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0

  // CK32-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK32-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to float**

  // CK32-DAG: store float* [[F1:%.+]], float** [[BPC0]],
  // CK32-DAG: store float* [[F2:%.+]], float** [[PC0]],
  // CK32-DAG: store i64 [[SIZE:%.+]], i64* [[S0]],

  // CK32-DAG: [[F1]] = load float*, float** [[F_ADDR:%.+]],
  // CK32-DAG: [[F2]] = load float*, float** [[F_ADDR]],
  // CK32-64-DAG: [[SIZE]] = mul nuw i64 [[SZ1:%.+]], 4
  // CK32-64-DAG: [[SZ1]] = mul nuw i64 12, %{{.+}}
  // CK32-32-DAG: [[SIZE]] = sext i32 [[SZ1:%.+]] to i64
  // CK32-32-DAG: [[SZ1]] = mul nuw i32 [[SZ2:%.+]], 4
  // CK32-32-DAG: [[SZ2]] = mul nuw i32 12, %{{.+}}
  #pragma omp target map(to:([3][sa][4])f)
  f[0] = 1;
  sa = 1;
  // CK32-DAG: call i32 @__tgt_target_mapper(i64 -1, i8* @{{.+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE_FROM]]{{.+}}, i8** null)
  // CK32-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK32-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK32-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK32-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK32-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK32-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0

  // CK32-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK32-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to float**

  // CK32-DAG: store float* [[F1:%.+]], float** [[BPC0]],
  // CK32-DAG: store float* [[F2:%.+]], float** [[PC0]],
  // CK32-DAG: store i64 [[SIZE:%.+]], i64* [[S0]],

  // CK32-DAG: [[F1]] = load float*, float** [[F_ADDR:%.+]],
  // CK32-DAG: [[F2]] = load float*, float** [[F_ADDR]],
  // CK32-64-DAG: [[SIZE]] = mul nuw i64 [[SZ1:%.+]], 5
  // CK32-64-DAG: [[SZ1]] = mul nuw i64 4, %{{.+}}
  // CK32-32-DAG: [[SIZE]] = sext i32 [[SZ1:%.+]] to i64
  // CK32-32-DAG: [[SZ1]] = mul nuw i32 [[SZ2:%.+]], 5
  // CK32-32-DAG: [[SZ2]] = mul nuw i32 4, %{{.+}}
  #pragma omp target map(from: ([sa][5])f)
  f[0] = 1;
}

#endif
#endif
