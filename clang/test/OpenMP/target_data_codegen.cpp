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

// CK1: [[ST:%.+]] = type { i32, double* }
template <typename T>
struct ST {
  T a;
  double *b;
};

ST<int> gb;
double gc[100];

// CK1: [[SIZE00:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 800]
// CK1: [[MTYPE00:@.+]] = {{.+}}constant [1 x i64] [i64 34]

// CK1: [[SIZE02:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] 4]
// CK1: [[MTYPE02:@.+]] = {{.+}}constant [1 x i64] [i64 33]

// CK1: [[MTYPE03:@.+]] = {{.+}}constant [1 x i64] [i64 37]

// CK1: [[SIZE04:@.+]] = {{.+}}constant [2 x i64] [i64 sdiv exact (i64 sub (i64 ptrtoint (double** getelementptr (double*, double** getelementptr inbounds (%struct.ST, %struct.ST* @gb, i32 0, i32 1), i32 1) to i64), i64 ptrtoint (double** getelementptr inbounds (%struct.ST, %struct.ST* @gb, i32 0, i32 1) to i64)), i64 ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)), i64 24]
// CK1: [[MTYPE04:@.+]] = {{.+}}constant [2 x i64] [i64 32, i64 281474976710673]

// CK1: [[MTYPE05:@.+]] = {{.+}}constant [1 x i64] [i64 1057]

// CK1: [[MTYPE06:@.+]] = {{.+}}constant [1 x i64] [i64 1061]

// CK1-LABEL: _Z3fooi
void foo(int arg) {
  int la;
  float lb[arg];

  // Region 00
  // CK1-DAG: call void @__tgt_target_data_begin_mapper(i64 [[DEV:%[^,]+]], i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK1-DAG: [[DEV]] = sext i32 [[DEVi32:%[^,]+]] to i64
  // CK1-DAG: [[DEVi32]] = load i32, i32* %{{[^,]+}},
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [100 x double]**
  // CK1-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [100 x double]**
  // CK1-DAG: store [100 x double]* @gc, [100 x double]** [[CBP0]]
  // CK1-DAG: store [100 x double]* @gc, [100 x double]** [[CP0]]

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1

  // CK1-DAG: call void @__tgt_target_data_end_mapper(i64 [[DEV:%[^,]+]], i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK1-DAG: [[DEV]] = sext i32 [[DEVi32:%[^,]+]] to i64
  // CK1-DAG: [[DEVi32]] = load i32, i32* %{{[^,]+}},
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
// CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
  #pragma omp target data if(1+3-5) device(arg) map(from: gc)
  {++arg;}

  // Region 01
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target data map(la) if(1+3-4)
  {++arg;}

  // Region 02
  // CK1: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
  // CK1: [[IFTHEN]]
  // CK1-DAG: call void @__tgt_target_data_begin_mapper(i64 4, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}}, i8** null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK1-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK1-DAG: store i32* [[VAR0:%.+]], i32** [[CBP0]]
  // CK1-DAG: store i32* [[VAR0]], i32** [[CP0]]
  // CK1: br label %[[IFEND:[^,]+]]

  // CK1: [[IFELSE]]
  // CK1: br label %[[IFEND]]
  // CK1: [[IFEND]]
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  // CK1: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]

  // CK1: [[IFTHEN]]
  // CK1-DAG: call void @__tgt_target_data_end_mapper(i64 4, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}}, i8** null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
  // CK1: br label %[[IFEND:[^,]+]]
  // CK1: [[IFELSE]]
  // CK1: br label %[[IFEND]]
  // CK1: [[IFEND]]
  #pragma omp target data map(to: arg) if(arg) device(4)
  {++arg;}

  // Region 03
  // CK1-DAG: call void @__tgt_target_data_begin_mapper(i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}}, i8** null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK1-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
  // CK1-DAG: store float* [[VAR0:%.+]], float** [[CBP0]]
  // CK1-DAG: store float* [[VAR0]], float** [[CP0]]
  // CK1-DAG: store i[[sz]] [[CSVAL0:%[^,]+]], i[[sz]]* [[S0]]
  // CK1-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1

  // CK1-DAG: call void @__tgt_target_data_end_mapper(i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}}, i8** null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S]]
  #pragma omp target data map(always, to: lb)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 04
  // CK1-DAG: call void @__tgt_target_data_begin_mapper(i64 -1, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE04]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE04]]{{.+}}, i8** null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
  // CK1-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to double***
  // CK1-DAG: store [[ST]]* @gb, [[ST]]** [[CBP0]]
  // CK1-DAG: store double** getelementptr inbounds ([[ST]], [[ST]]* @gb, i32 0, i32 1), double*** [[CP0]]

  // CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK1-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double***
  // CK1-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double**
  // CK1-DAG: store double** getelementptr inbounds ([[ST]], [[ST]]* @gb, i32 0, i32 1), double*** [[CBP1]]
  // CK1-DAG: store double* [[SEC1:%.+]], double** [[CP1]]
  // CK1-DAG: [[SEC1]] = getelementptr inbounds {{.+}}double* [[SEC11:%[^,]+]], i{{.+}} 0
  // CK1-DAG: [[SEC11]] = load double*, double** getelementptr inbounds ([[ST]], [[ST]]* @gb, i32 0, i32 1),

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1

  // CK1-DAG: call void @__tgt_target_data_end_mapper(i64 -1, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE04]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE04]]{{.+}}, i8** null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
  #pragma omp target data map(to: gb.b[:3])
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 05
  // CK1-DAG: call void @__tgt_target_data_begin_mapper(i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE05]]{{.+}}, i8** null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK1-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
  // CK1-DAG: store float* [[VAR0:%.+]], float** [[CBP0]]
  // CK1-DAG: store float* [[VAR0]], float** [[CP0]]
  // CK1-DAG: store i[[sz]] [[CSVAL0:%[^,]+]], i[[sz]]* [[S0]]
  // CK1-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1

  // CK1-DAG: call void @__tgt_target_data_end_mapper(i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE05]]{{.+}}, i8** null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S]]
  #pragma omp target data map(close, to: lb)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 06
  // CK1-DAG: call void @__tgt_target_data_begin_mapper(i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE06]]{{.+}}, i8** null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK1-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
  // CK1-DAG: store float* [[VAR0:%.+]], float** [[CBP0]]
  // CK1-DAG: store float* [[VAR0]], float** [[CP0]]
  // CK1-DAG: store i[[sz]] [[CSVAL0:%[^,]+]], i[[sz]]* [[S0]]
  // CK1-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1

  // CK1-DAG: call void @__tgt_target_data_end_mapper(i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE06]]{{.+}}, i8** null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S]]
  #pragma omp target data map(always close, to: lb)
  {++arg;}

}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK1A -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK1A --check-prefix CK1A-64
// RUN: %clang_cc1 -DCK1A -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1A --check-prefix CK1A-64
// RUN: %clang_cc1 -DCK1A -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK1A --check-prefix CK1A-32
// RUN: %clang_cc1 -DCK1A -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1A --check-prefix CK1A-32

// RUN: %clang_cc1 -DCK1A -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1A -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1A -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1A -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK1A

// CK1A: [[ST:%.+]] = type { i32, double* }
template <typename T>
struct ST {
  T a;
  double *b;
};

ST<int> gb;
double gc[100];

// PRESENT=0x1000 | TARGET_PARAM=0x20 | TO=0x1 = 0x1021
// CK1A: [[MTYPE00:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x1021]]]

// PRESENT=0x1000 | CLOSE=0x400 | TARGET_PARAM=0x20 | ALWAYS=0x4 | TO=0x1 = 0x1425
// CK1A: [[MTYPE01:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x1425]]]

// CK1A-LABEL: _Z3fooi
void foo(int arg) {
  int la;
  float lb[arg];

  // Region 00
  // CK1A-DAG: call void @__tgt_target_data_begin_mapper(i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz:32|64]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}})
  // CK1A-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1A-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1A-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1A-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1A-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1A-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1A-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK1A-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
  // CK1A-DAG: store float* [[VAR0:%.+]], float** [[CBP0]]
  // CK1A-DAG: store float* [[VAR0]], float** [[CP0]]
  // CK1A-DAG: store i[[sz]] [[CSVAL0:%[^,]+]], i[[sz]]* [[S0]]
  // CK1A-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1A-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1A-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1A: %{{.+}} = add nsw i32 %{{[^,]+}}, 1

  // CK1A-DAG: call void @__tgt_target_data_end_mapper(i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}})
  // CK1A-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
  // CK1A-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
  // CK1A-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S]]
  #pragma omp target data map(present, to: lb)
  {++arg;}

  // Region 01
  // CK1A-DAG: call void @__tgt_target_data_begin_mapper(i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}})
  // CK1A-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1A-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1A-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1A-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1A-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1A-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1A-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK1A-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
  // CK1A-DAG: store float* [[VAR0:%.+]], float** [[CBP0]]
  // CK1A-DAG: store float* [[VAR0]], float** [[CP0]]
  // CK1A-DAG: store i[[sz]] [[CSVAL0:%[^,]+]], i[[sz]]* [[S0]]
  // CK1A-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1A-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1A-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1A: %{{.+}} = add nsw i32 %{{[^,]+}}, 1

  // CK1A-DAG: call void @__tgt_target_data_end_mapper(i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}})
  // CK1A-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
  // CK1A-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
  // CK1A-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S]]
  #pragma omp target data map(always close present, to: lb)
  {++arg;}

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

// CK2: [[ST:%.+]] = type { i32, double* }
template <typename T>
struct ST {
  T a;
  double *b;

  T foo(T arg) {
    // Region 00
    #pragma omp target data map(always, to: b[1:3]) if(a>123) device(arg)
    {arg++;}
    return arg;
  }
};

// CK2: [[MTYPE00:@.+]] = {{.+}}constant [2 x i64] [i64 32, i64 281474976710677]

// CK2-LABEL: _Z3bari
int bar(int arg){
  ST<int> A;
  return A.foo(arg);
}

// Region 00
// CK2: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CK2: [[IFTHEN]]
// CK2-DAG: call void @__tgt_target_data_begin_mapper(i64 [[DEV:%[^,]+]], i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz:64|32]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
// CK2-DAG: [[DEV]] = sext i32 [[DEVi32:%[^,]+]] to i64
// CK2-DAG: [[DEVi32]] = load i32, i32* %{{[^,]+}},
// CK2-DAG: [[GEPBP]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BP:%[^,]+]]
// CK2-DAG: [[GEPP]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[P:%[^,]+]]
// CK2-DAG: [[GEPS]] = getelementptr inbounds [2 x i[[sz]]], [2 x i[[sz]]]* [[S:%[^,]+]]

// CK2-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
// CK2-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to double***
// CK2-DAG: store [[ST]]* [[VAR0:%.+]], [[ST]]** [[CBP0]]
// CK2-DAG: store double** [[SEC0:%.+]], double*** [[CP0]]
// CK2-DAG: store i[[sz]] {{%.+}}, i[[sz]]* [[S0]]
// CK2-DAG: [[SEC0]] = getelementptr inbounds {{.*}}[[ST]]* [[VAR0]], i32 0, i32 1

// CK2-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK2-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK2-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK2-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double***
// CK2-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double**
// CK2-DAG: store double** [[SEC0]], double*** [[CBP1]]
// CK2-DAG: store double* [[SEC1:%.+]], double** [[CP1]]
// CK2-DAG: store i[[sz]] 24, i[[sz]]* [[S1]]
// CK2-DAG: [[SEC1]] = getelementptr inbounds {{.*}}double* [[SEC11:%[^,]+]], i{{.+}} 1
// CK2-DAG: [[SEC11]] = load double*, double** [[SEC111:%[^,]+]],
// CK2-DAG: [[SEC111]] = getelementptr inbounds {{.*}}[[ST]]* [[VAR0]], i32 0, i32 1

// CK2: br label %[[IFEND:[^,]+]]

// CK2: [[IFELSE]]
// CK2: br label %[[IFEND]]
// CK2: [[IFEND]]
// CK2: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
// CK2: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]

// CK2: [[IFTHEN]]
// CK2-DAG: call void @__tgt_target_data_end_mapper(i64 [[DEV:%[^,]+]], i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
// CK2-DAG: [[DEV]] = sext i32 [[DEVi32:%[^,]+]] to i64
// CK2-DAG: [[DEVi32]] = load i32, i32* %{{[^,]+}},
// CK2-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
// CK2-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
// CK2-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S]]
// CK2: br label %[[IFEND:[^,]+]]
// CK2: [[IFELSE]]
// CK2: br label %[[IFEND]]
// CK2: [[IFEND]]
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -verify -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-32
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-32

// RUN: %clang_cc1 -DCK3 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK3 -verify -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// SIMD-ONLY2-NOT: {{__kmpc|__tgt}}
#ifdef CK3

// CK3-LABEL: no_target_devices
void no_target_devices(int arg) {
  // CK3-NOT: tgt_target_data_begin
  // CK3: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  // CK3-NOT: tgt_target_data_end
  // CK3: ret
  #pragma omp target data map(to: arg) if(arg) device(4)
  {++arg;}
}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK4 --check-prefix CK4-32
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK4 --check-prefix CK4-32

// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
#ifdef CK4

// CK4: [[STT:%.+]] = type { i32, double* }
template <typename T>
struct STT {
  T a;
  double *b;

  T foo(T arg) {
    // Region 00
    #pragma omp target data map(always, close to: b[1:3]) if(a>123) device(arg)
    {arg++;}
    return arg;
  }
};

// CK4: [[MTYPE00:@.+]] = {{.+}}constant [2 x i64] [i64 32, i64 281474976711701]

// CK4-LABEL: _Z3bari
int bar(int arg){
  STT<int> A;
  return A.foo(arg);
}

// Region 00
// CK4: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CK4: [[IFTHEN]]
// CK4-DAG: call void @__tgt_target_data_begin_mapper(i64 [[DEV:%[^,]+]], i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz:64|32]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
// CK4-DAG: [[DEV]] = sext i32 [[DEVi32:%[^,]+]] to i64
// CK4-DAG: [[DEVi32]] = load i32, i32* %{{[^,]+}},
// CK4-DAG: [[GEPBP]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BP:%[^,]+]]
// CK4-DAG: [[GEPP]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[P:%[^,]+]]
// CK4-DAG: [[GEPS]] = getelementptr inbounds [2 x i[[sz]]], [2 x i[[sz]]]* [[S:%[^,]+]]

// CK4-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK4-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK4-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK4-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[STT]]**
// CK4-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to double***
// CK4-DAG: store [[STT]]* [[VAR0:%.+]], [[STT]]** [[CBP0]]
// CK4-DAG: store double** [[SEC0:%.+]], double*** [[CP0]]
// CK4-DAG: store i[[sz]] {{%.+}}, i[[sz]]* [[S0]]
// CK4-DAG: [[SEC0]] = getelementptr inbounds {{.*}}[[STT]]* [[VAR0]], i32 0, i32 1

// CK4-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK4-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK4-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
// CK4-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double***
// CK4-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double**
// CK4-DAG: store double** [[SEC0]], double*** [[CBP1]]
// CK4-DAG: store double* [[SEC1:%.+]], double** [[CP1]]
// CK4-DAG: store i[[sz]] 24, i[[sz]]* [[S1]]
// CK4-DAG: [[SEC1]] = getelementptr inbounds {{.*}}double* [[SEC11:%[^,]+]], i{{.+}} 1
// CK4-DAG: [[SEC11]] = load double*, double** [[SEC111:%[^,]+]],
// CK4-DAG: [[SEC111]] = getelementptr inbounds {{.*}}[[STT]]* [[VAR0]], i32 0, i32 1

// CK4: br label %[[IFEND:[^,]+]]

// CK4: [[IFELSE]]
// CK4: br label %[[IFEND]]
// CK4: [[IFEND]]
// CK4: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
// CK4: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]

// CK4: [[IFTHEN]]
// CK4-DAG: call void @__tgt_target_data_end_mapper(i64 [[DEV:%[^,]+]], i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
// CK4-DAG: [[DEV]] = sext i32 [[DEVi32:%[^,]+]] to i64
// CK4-DAG: [[DEVi32]] = load i32, i32* %{{[^,]+}},
// CK4-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
// CK4-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
// CK4-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S]]
// CK4: br label %[[IFEND:[^,]+]]
// CK4: [[IFELSE]]
// CK4: br label %[[IFEND]]
// CK4: [[IFEND]]
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK5 --check-prefix CK5-32
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK5 --check-prefix CK5-32

// RUN: %clang_cc1 -DCK5 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK5 -verify -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// SIMD-ONLY2-NOT: {{__kmpc|__tgt}}#ifdef CK5
#ifdef CK5
struct S1 {
  int i;
};
struct S2 {
  S1 s;
  struct S2 *ps;
};

void test_close_modifier(int arg) {
  S2 *ps;
  // CK5: private unnamed_addr constant [6 x i64] [i64 1059, i64 32, i64 562949953422339, i64 562949953421328, i64 16, i64 1043]
  #pragma omp target data map(close,tofrom: arg, ps->ps->ps->ps->s)
  {
    ++(arg);
  }
}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK6 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK6 --check-prefix CK6-64
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK6 --check-prefix CK6-64
// RUN: %clang_cc1 -DCK6 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK6 --check-prefix CK6-32
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK6 --check-prefix CK6-32

// RUN: %clang_cc1 -DCK6 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK6 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK6 -verify -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK6 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// SIMD-ONLY2-NOT: {{__kmpc|__tgt}}
#ifdef CK6
void test_close_modifier(int arg) {
  // CK6: private unnamed_addr constant [1 x i64] [i64 1059]
  #pragma omp target data map(close,tofrom: arg)
  {++arg;}
}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK7 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK7 --check-prefix CK7-64
// RUN: %clang_cc1 -DCK7 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK7 --check-prefix CK7-64

// RUN: %clang_cc1 -DCK7 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY7 %s
// RUN: %clang_cc1 -DCK7 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY7 %s
// SIMD-ONLY7-NOT: {{__kmpc|__tgt}}
#ifdef CK7
// CK7: test_device_ptr_addr
void test_device_ptr_addr(int arg) {
  int *p;
  // CK7: add nsw i32
  // CK7: add nsw i32
  #pragma omp target data use_device_ptr(p) use_device_addr(arg)
  { ++arg, ++(*p); }
}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK8 --check-prefix CK8-64
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK8 --check-prefix CK8-64
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK8 --check-prefix CK8-32
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK8 --check-prefix CK8-32

// RUN: %clang_cc1 -DCK8 -verify -fopenmp-simd -fopenmp-version=51 -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK8 -fopenmp-simd -fopenmp-version=51 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK8 -verify -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK8 -fopenmp-simd -fopenmp-version=51 -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// SIMD-ONLY2-NOT: {{__kmpc|__tgt}}#ifdef CK8
#ifdef CK8
struct S1 {
  int i;
};
struct S2 {
  S1 s;
  struct S2 *ps;
};

void test_present_modifier(int arg) {
  S2 *ps1;
  S2 *ps2;

  // Make sure the struct picks up present even if another element of the struct
  // doesn't have present.

  // CK8: private unnamed_addr constant [15 x i64]

  // ps1
  //
  // PRESENT=0x1000 | TARGET_PARAM=0x20 = 0x1020
  // MEMBER_OF_1=0x1000000000000 | FROM=0x2 | TO=0x1 = 0x1000000000003
  // MEMBER_OF_1=0x1000000000000 | PTR_AND_OBJ=0x10 | FROM=0x2 | TO=0x1 = 0x1000000000013
  // MEMBER_OF_1=0x1000000000000 | PRESENT=0x1000 | FROM=0x2 | TO=0x1 = 0x1000000001003
  // MEMBER_OF_1=0x1000000000000 | PRESENT=0x1000 | PTR_AND_OBJ=0x10 = 0x1000000001010
  // PRESENT=0x1000 | PTR_AND_OBJ=0x10 = 0x1010
  // PRESENT=0x1000 | PTR_AND_OBJ=0x10 | FROM=0x2 | TO=0x1 = 0x1013
  //
  // CK8-SAME: {{^}} [i64 [[#0x1020]], i64 [[#0x1000000000003]],
  // CK8-SAME: {{^}} i64 [[#0x1000000000013]], i64 [[#0x1000000001003]],
  // CK8-SAME: {{^}} i64 [[#0x1000000001010]], i64 [[#0x1010]], i64 [[#0x1013]],

  // arg
  //
  // PRESENT=0x1000 | TARGET_PARAM=0x20 | FROM=0x2 | TO=0x1 = 0x1023
  //
  // CK8-SAME: {{^}} i64 [[#0x1023]],

  // ps2
  //
  // PRESENT=0x1000 | TARGET_PARAM=0x20 = 0x1020
  // MEMBER_OF_9=0x9000000000000 | PRESENT=0x1000 | FROM=0x2 | TO=0x1 = 0x9000000001003
  // MEMBER_OF_9=0x9000000000000 | PRESENT=0x1000 | PTR_AND_OBJ=0x10 | FROM=0x2 | TO=0x1 = 0x9000000001013
  // MEMBER_OF_9=0x9000000000000 | FROM=0x2 | TO=0x1 = 0x9000000000003
  // MEMBER_OF_9=0x9000000000000 | PTR_AND_OBJ=0x10 = 0x9000000000010
  // PTR_AND_OBJ=0x10 = 0x10
  // PTR_AND_OBJ=0x10 | FROM=0x2 | TO=0x1 = 0x13
  //
  // CK8-SAME: {{^}} i64 [[#0x1020]], i64 [[#0x9000000001003]],
  // CK8-SAME: {{^}} i64 [[#0x9000000001013]], i64 [[#0x9000000000003]],
  // CK8-SAME: {{^}} i64 [[#0x9000000000010]], i64 [[#0x10]], i64 [[#0x13]]]
  #pragma omp target data map(tofrom: ps1->s) \
                          map(present,tofrom: arg, ps1->ps->ps->ps->s, ps2->s) \
                          map(tofrom: ps2->ps->ps->ps->s)
  {
    ++(arg);
  }
}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK9 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK9 --check-prefix CK9-64
// RUN: %clang_cc1 -DCK9 -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK9 --check-prefix CK9-64
// RUN: %clang_cc1 -DCK9 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK9 --check-prefix CK9-32
// RUN: %clang_cc1 -DCK9 -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK9 --check-prefix CK9-32

// RUN: %clang_cc1 -DCK9 -verify -fopenmp-simd -fopenmp-version=51 -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK9 -fopenmp-simd -fopenmp-version=51 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK9 -verify -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK9 -fopenmp-simd -fopenmp-version=51 -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// SIMD-ONLY2-NOT: {{__kmpc|__tgt}}
#ifdef CK9
void test_present_modifier(int arg) {
  // PRESENT=0x1000 | TARGET_PARAM=0x20 | FROM=0x2 | TO=0x1 = 0x1023
  // CK9: private unnamed_addr constant [1 x i64] [i64 [[#0x1023]]]
  #pragma omp target data map(present,tofrom: arg)
  {++arg;}
}
#endif
#endif
