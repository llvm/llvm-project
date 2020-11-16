// expected-no-diagnostics
#ifndef HEADER
#define HEADER

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
// CK21: [[SIZE00:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK21-USE: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 35]
// CK21-NOUSE: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK21-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK21: [[SIZE01:@.+]] = private {{.*}}constant [1 x i64] [i64 492]
// CK21-USE: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 35]
// CK21-NOUSE: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i64] [i64 3]

// CK21-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK21-USE: [[MTYPE02:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710674]
// CK21-NOUSE: [[MTYPE02:@.+]] = private {{.*}}constant [2 x i64] [i64 0, i64 281474976710674]

// CK21-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK21: [[SIZE03:@.+]] = private {{.*}}constant [1 x i64] [i64 492]
// CK21-USE: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i64] [i64 34]
// CK21-NOUSE: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i64] [i64 2]

// CK21-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK21: [[SIZE04:@.+]] = private {{.*}}constant [1 x i64] [i64 4]
// CK21-USE: [[MTYPE04:@.+]] = private {{.*}}constant [1 x i64] [i64 34]
// CK21-NOUSE: [[MTYPE04:@.+]] = private {{.*}}constant [1 x i64] [i64 2]

// CK21-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l{{[0-9]+}}.region_id = weak constant i8 0
// CK21-USE: [[MTYPE05:@.+]] = private {{.*}}constant [3 x i64] [i64 32, i64 281474976710659, i64 281474976710659]
// CK21-NOUSE: [[MTYPE05:@.+]] = private {{.*}}constant [3 x i64] [i64 0, i64 281474976710659, i64 281474976710659]

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
    // CK21-DAG: call i32 @__tgt_target_mapper(i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]]{{.+}}, {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
    // CK21-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK21-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
    // CK21-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
    // CK21-DAG: store [[ST]]* [[VAR0:%.+]], [[ST]]** [[CBP0]]
    // CK21-DAG: store i32* [[SEC0:%.+]], i32** [[CP0]]
    // CK21-DAG: [[SEC0]] = getelementptr {{.*}}[[ST]]* [[VAR0:%.+]], i{{.+}} 0, i{{.+}} 0

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
#endif // CK21
#endif
