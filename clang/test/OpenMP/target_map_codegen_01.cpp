// expected-no-diagnostics
#ifndef HEADER
#define HEADER



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
  // CK2-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null, i8** null)
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
  // CK2-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES2]]{{.+}}, {{.+}}[[TYPES2]]{{.+}}, i8** null, i8** null)
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
#endif // CK2
#endif
