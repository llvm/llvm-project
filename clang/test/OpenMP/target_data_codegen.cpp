// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK1 -verify -Wno-vla -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -verify -Wno-vla -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32

// RUN: %clang_cc1 -DCK1 -verify -Wno-vla -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -verify -Wno-vla -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK1

// CK1: [[ST:%.+]] = type { i32, ptr }
template <typename T>
struct ST {
  T a;
  double *b;
};

ST<int> gb;
double gc[100];

// CK1: [[SIZE00:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 800]
// CK1: [[MTYPE00:@.+]] = {{.+}}constant [1 x i64] [i64 2]

// CK1: [[SIZE02:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] 4]
// CK1: [[MTYPE02:@.+]] = {{.+}}constant [1 x i64] [i64 1]

// CK1: [[MTYPE03:@.+]] = {{.+}}constant [1 x i64] [i64 5]

// CK1: [[SIZE04:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 24]
// CK1: [[MTYPE04:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 281474976710673]

// CK1: [[MTYPE05:@.+]] = {{.+}}constant [1 x i64] [i64 1025]

// CK1: [[MTYPE06:@.+]] = {{.+}}constant [1 x i64] [i64 1029]

// CK1-LABEL: _Z3fooi
void foo(int arg) {
  int la;
  float lb[arg];

  // Region 00
  // CK1-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 [[DEV:%[^,]+]], i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[SIZE00]], ptr [[MTYPE00]], ptr null, ptr null)
  // CK1-DAG: [[DEV]] = sext i32 [[DEVi32:%[^,]+]] to i64
  // CK1-DAG: [[DEVi32]] = load i32, ptr %{{[^,]+}},
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: store ptr @gc, ptr [[BP0]]
  // CK1-DAG: store ptr @gc, ptr [[P0]]

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1

  // CK1-DAG: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 [[DEV]], i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[SIZE00]], ptr [[MTYPE00]], ptr null, ptr null)
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
  // CK1-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 4, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[SIZE02]], ptr [[MTYPE02]], ptr null, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
  // CK1-DAG: store ptr [[VAR0]], ptr [[P0]]
  // CK1: br label %[[IFEND:[^,]+]]

  // CK1: [[IFELSE]]
  // CK1: br label %[[IFEND]]
  // CK1: [[IFEND]]
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  // CK1: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]

  // CK1: [[IFTHEN]]
  // CK1-DAG: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 4, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[SIZE02]], ptr [[MTYPE02]], ptr null, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
  // CK1: br label %[[IFEND:[^,]+]]
  // CK1: [[IFELSE]]
  // CK1: br label %[[IFEND]]
  // CK1: [[IFEND]]
  #pragma omp target data map(to: arg) if(arg) device(4)
  {++arg;}

  // Region 03
  // CK1-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE03]], ptr null, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
  // CK1-DAG: store ptr [[VAR0]], ptr [[P0]]
  // CK1-DAG: store i[[sz]] [[CSVAL0:%[^,]+]], ptr [[S0]]
  // CK1-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1

  // CK1-DAG: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE03]], ptr null, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S]]
  #pragma omp target data map(always, to: lb)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 04
  // CK1-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 2, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%[^,]+]], ptr [[MTYPE04]], ptr null, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[PSZ:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[PS0:%.+]] = getelementptr inbounds {{.+}}[[PSZ]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: store ptr @gb, ptr [[BP0]]
  // CK1-DAG: store ptr getelementptr inbounds nuw ([[ST]], ptr @gb, i32 0, i32 1), ptr [[P0]]
  // CK1-DAG: [[DIV:%.+]] = sdiv exact i64 sub (i64 ptrtoint (ptr getelementptr (ptr, ptr getelementptr inbounds nuw (%struct.ST, ptr @gb, i32 0, i32 1), i32 1) to i64), i64 ptrtoint (ptr getelementptr inbounds nuw (%struct.ST, ptr @gb, i32 0, i32 1) to i64)), ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
  // CK1-DAG: store i64 [[DIV]], ptr [[PS0]],

  // CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK1-DAG: store ptr getelementptr inbounds nuw ([[ST]], ptr @gb, i32 0, i32 1), ptr [[BP1]]
  // CK1-DAG: store ptr [[SEC1:%.+]], ptr [[P1]]
  // CK1-DAG: [[SEC1]] = getelementptr inbounds {{.+}}ptr [[SEC11:%[^,]+]], i{{.+}} 0
  // CK1-DAG: [[SEC11]] = load ptr, ptr getelementptr inbounds nuw ([[ST]], ptr @gb, i32 0, i32 1),

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1

  // CK1-DAG: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 -1, i32 2, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%[^,]+]], ptr [[MTYPE04]], ptr null, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[PSZ]]
  #pragma omp target data map(to: gb.b[:3])
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 05
  // CK1-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE05]], ptr null, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
  // CK1-DAG: store ptr [[VAR0]], ptr [[P0]]
  // CK1-DAG: store i[[sz]] [[CSVAL0:%[^,]+]], ptr [[S0]]
  // CK1-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1

  // CK1-DAG: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE05]], ptr null, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S]]
  #pragma omp target data map(close, to: lb)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 06
  // CK1-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE06]], ptr null, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
  // CK1-DAG: store ptr [[VAR0]], ptr [[P0]]
  // CK1-DAG: store i[[sz]] [[CSVAL0:%[^,]+]], ptr [[S0]]
  // CK1-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1

  // CK1-DAG: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE06]], ptr null, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S]]
  #pragma omp target data map(always close, to: lb)
  {++arg;}

}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK1A -verify -Wno-vla  -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK1A --check-prefix CK1A-64
// RUN: %clang_cc1 -DCK1A -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1A --check-prefix CK1A-64
// RUN: %clang_cc1 -DCK1A -verify -Wno-vla  -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK1A --check-prefix CK1A-32
// RUN: %clang_cc1 -DCK1A -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1A --check-prefix CK1A-32

// RUN: %clang_cc1 -DCK1A -verify -Wno-vla  -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1A -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1A -verify -Wno-vla  -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1A -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK1A

// CK1A: [[ST:%.+]] = type { i32, ptr }
template <typename T>
struct ST {
  T a;
  double *b;
};

ST<int> gb;
double gc[100];

// PRESENT=0x1000 | TO=0x1 = 0x1001
// CK1A: [[MTYPE00Begin:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x1001]]]

// TO=0x1 = 0x1
// CK1A: [[MTYPE00End:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x1]]]

// PRESENT=0x1000 | CLOSE=0x400 | ALWAYS=0x4 | TO=0x1 = 0x1405
// CK1A: [[MTYPE01Begin:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x1405]]]

// CLOSE=0x400 | ALWAYS=0x4 | TO=0x1 = 0x405
// CK1A: [[MTYPE01End:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x405]]]

// CK1A-LABEL: _Z3fooi
void foo(int arg) {
  int la;
  float lb[arg];

  // Region 00
  // CK1A-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE00Begin]]{{.+}})
  // CK1A-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1A-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1A-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1A-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1A-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1A-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1A-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
  // CK1A-DAG: store ptr [[VAR0]], ptr [[P0]]
  // CK1A-DAG: store i[[sz:32|64]] [[CSVAL0:%[^,]+]], ptr [[S0]]
  // CK1A-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1A-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1A-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1A: %{{.+}} = add nsw i32 %{{[^,]+}}, 1

  // CK1A-DAG: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE00End]]{{.+}})
  // CK1A-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
  // CK1A-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
  // CK1A-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S]]
  #pragma omp target data map(present, to: lb)
  {++arg;}

  // Region 01
  // CK1A-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE01Begin]]{{.+}})
  // CK1A-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1A-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1A-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1A-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1A-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1A-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1A-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
  // CK1A-DAG: store ptr [[VAR0]], ptr [[P0]]
  // CK1A-DAG: store i[[sz]] [[CSVAL0:%[^,]+]], ptr [[S0]]
  // CK1A-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1A-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1A-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1A: %{{.+}} = add nsw i32 %{{[^,]+}}, 1

  // CK1A-DAG: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE01End]]{{.+}})
  // CK1A-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
  // CK1A-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
  // CK1A-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S]]
  #pragma omp target data map(always close present, to: lb)
  {++arg;}

}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK2 -verify -Wno-vla  -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -verify -Wno-vla  -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-32
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-32

// RUN: %clang_cc1 -DCK2 -verify -Wno-vla  -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -verify -Wno-vla  -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK2 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
#ifdef CK2

// CK2: [[ST:%.+]] = type { i32, ptr }
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

// CK2: [[SIZE00:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 24]
// CK2: [[MTYPE00:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 281474976710677]

// CK2-LABEL: _Z3bari
int bar(int arg){
  ST<int> A;
  return A.foo(arg);
}

// Region 00
// CK2-DAG: [[DEV:%[^,]+]] = sext i32 [[DEVi32:%[^,]+]] to i64
// CK2-DAG: [[DEVi32]] = load i32, ptr %{{[^,]+}},
// CK2: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CK2: [[IFTHEN]]
// CK2-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 [[DEV]], i32 2, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%[^,]+]], ptr [[MTYPE00]], ptr null, ptr null)
// CK2-DAG: [[GEPBP]] = getelementptr inbounds [2 x ptr], ptr [[BP:%[^,]+]]
// CK2-DAG: [[GEPP]] = getelementptr inbounds [2 x ptr], ptr [[P:%[^,]+]]
// CK2-DAG: [[GEPS]] = getelementptr inbounds [2 x i64], ptr [[PS:%[^,]+]]

// CK2-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: [[PS0:%.+]] = getelementptr inbounds [2 x i64], ptr [[PS]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK2-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK2-DAG: store i64 {{%.+}}, ptr [[PS0]],
// CK2-DAG: [[SEC0]] = getelementptr inbounds {{.*}}ptr [[VAR0]], i32 0, i32 1

// CK2-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK2-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK2-DAG: store ptr [[SEC0]], ptr [[BP1]]
// CK2-DAG: store ptr [[SEC1:%.+]], ptr [[P1]]
// CK2-DAG: [[SEC1]] = getelementptr inbounds {{.*}}ptr [[SEC11:%[^,]+]], i{{.+}} 1
// CK2-DAG: [[SEC11]] = load ptr, ptr [[SEC111:%[^,]+]],
// CK2-DAG: [[SEC111]] = getelementptr inbounds {{.*}}ptr [[VAR0]], i32 0, i32 1

// CK2: br label %[[IFEND:[^,]+]]

// CK2: [[IFELSE]]
// CK2: br label %[[IFEND]]
// CK2: [[IFEND]]
// CK2: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
// CK2: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]

// CK2: [[IFTHEN]]
// CK2-DAG: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 [[DEV]], i32 2, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%[^,]+]], ptr [[MTYPE00]], ptr null, ptr null)
// CK2-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
// CK2-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
// CK2-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[PS]]
// CK2: br label %[[IFEND:[^,]+]]
// CK2: [[IFELSE]]
// CK2: br label %[[IFEND]]
// CK2: [[IFEND]]
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK3 -verify -Wno-vla  -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -verify -Wno-vla  -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-32
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-32

// RUN: %clang_cc1 -DCK3 -verify -Wno-vla  -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK3 -verify -Wno-vla  -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK3 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
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
// RUN: %clang_cc1 -DCK4 -verify -Wno-vla  -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck %s  --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -verify -Wno-vla  -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK4 --check-prefix CK4-32
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck %s  --check-prefix CK4 --check-prefix CK4-32

// RUN: %clang_cc1 -DCK4 -verify -Wno-vla  -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK4 -verify -Wno-vla  -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
#ifdef CK4

// CK4: [[STT:%.+]] = type { i32, ptr }
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

// CK4: [[SIZE00:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 24]
// CK4: [[MTYPE00:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 281474976711701]

// CK4-LABEL: _Z3bari
int bar(int arg){
  STT<int> A;
  return A.foo(arg);
}

// Region 00
// CK4-DAG: [[DEV:%[^,]+]] = sext i32 [[DEVi32:%[^,]+]] to i64
// CK4-DAG: [[DEVi32]] = load i32, ptr %{{[^,]+}},
// CK4: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CK4: [[IFTHEN]]
// CK4-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 [[DEV]], i32 2, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE00]], ptr null, ptr null)
// CK4-DAG: [[GEPBP]] = getelementptr inbounds [2 x ptr], ptr [[BP:%[^,]+]]
// CK4-DAG: [[GEPP]] = getelementptr inbounds [2 x ptr], ptr [[P:%[^,]+]]
// CK4-DAG: [[GEPS]] = getelementptr inbounds [2 x i64], ptr [[PS:%[^,]+]]

// CK4-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK4-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK4-DAG: [[PS0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i{{.+}} 0, i{{.+}} 0
// CK4-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK4-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK4-DAG: store i64 {{%.+}}, ptr [[PS0]],
// CK4-DAG: [[SEC0]] = getelementptr inbounds {{.*}}ptr [[VAR0]], i32 0, i32 1

// CK4-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK4-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK4-DAG: store ptr [[SEC0]], ptr [[BP1]]
// CK4-DAG: store ptr [[SEC1:%.+]], ptr [[P1]]
// CK4-DAG: [[SEC1]] = getelementptr inbounds {{.*}}ptr [[SEC11:%[^,]+]], i{{.+}} 1
// CK4-DAG: [[SEC11]] = load ptr, ptr [[SEC111:%[^,]+]],
// CK4-DAG: [[SEC111]] = getelementptr inbounds {{.*}}ptr [[VAR0]], i32 0, i32 1

// CK4: br label %[[IFEND:[^,]+]]

// CK4: [[IFELSE]]
// CK4: br label %[[IFEND]]
// CK4: [[IFEND]]
// CK4: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
// CK4: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]

// CK4: [[IFTHEN]]
// CK4-DAG: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 [[DEV]], i32 2, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE00]], ptr null, ptr null)
// CK4-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
// CK4-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P]]
// CK4-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[PS]]
// CK4: br label %[[IFEND:[^,]+]]
// CK4: [[IFELSE]]
// CK4: br label %[[IFEND]]
// CK4: [[IFEND]]
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK5 -verify -Wno-vla  -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck %s  --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -verify -Wno-vla  -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK5 --check-prefix CK5-32
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck %s  --check-prefix CK5 --check-prefix CK5-32

// RUN: %clang_cc1 -DCK5 -verify -Wno-vla  -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK5 -verify -Wno-vla  -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
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
// CK5: private unnamed_addr constant [5 x i64] [i64 1027, i64 0, i64 562949953421328, i64 16, i64 1043]
#pragma omp target data map(close, tofrom \
                            : arg, ps->ps->ps->ps->s)
  {
    ++(arg);
  }
}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK6 -verify -Wno-vla  -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK6 --check-prefix CK6-64
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck %s  --check-prefix CK6 --check-prefix CK6-64
// RUN: %clang_cc1 -DCK6 -verify -Wno-vla  -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK6 --check-prefix CK6-32
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck %s  --check-prefix CK6 --check-prefix CK6-32

// RUN: %clang_cc1 -DCK6 -verify -Wno-vla  -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK6 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK6 -verify -Wno-vla  -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK6 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// SIMD-ONLY2-NOT: {{__kmpc|__tgt}}
#ifdef CK6
void test_close_modifier(int arg) {
// CK6: private unnamed_addr constant [1 x i64] [i64 1027]
#pragma omp target data map(close, tofrom \
                            : arg)
  {++arg;}
}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK7 -verify -Wno-vla  -fopenmp -fopenmp-targets=x86_64 -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK7 --check-prefix CK7-64
// RUN: %clang_cc1 -DCK7 -fopenmp -fopenmp-targets=x86_64 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=x86_64 -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck %s  --check-prefix CK7 --check-prefix CK7-64

// RUN: %clang_cc1 -DCK7 -verify -Wno-vla  -fopenmp-simd -fopenmp-targets=x86_64 -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY7 %s
// RUN: %clang_cc1 -DCK7 -fopenmp-simd -fopenmp-targets=x86_64 -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=x86_64 -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY7 %s
// SIMD-ONLY7-NOT: {{__kmpc|__tgt}}
#ifdef CK7
// CK7: private unnamed_addr constant [2 x i64] [i64 64, i64 64]
// CK7: private unnamed_addr constant [2 x i64] [i64 3, i64 64]
// CK7-NOT: private unnamed_addr constant [2 x i64] [i64 64, i64 3]
// CK7: test_device_ptr_addr
void test_device_ptr_addr(int arg) {
  int *p;
  // CK7: add nsw i32
  // CK7: add nsw i32
  #pragma omp target data use_device_ptr(p) use_device_addr(arg)
  { ++arg, ++(*p); }

  short x[10];
  short *xp = &x[0];

  x[1] = 111;

  #pragma omp target data map(tofrom: x) use_device_addr(xp[1:3])
  {
    xp[1] = 222;
  }
}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK8 -verify -Wno-vla  -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK8 --check-prefix CK8-64
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck %s  --check-prefix CK8 --check-prefix CK8-64
// RUN: %clang_cc1 -DCK8 -verify -Wno-vla  -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK8 --check-prefix CK8-32
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck %s  --check-prefix CK8 --check-prefix CK8-32

// RUN: %clang_cc1 -DCK8 -verify -Wno-vla  -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK8 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK8 -verify -Wno-vla  -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK8 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
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
  // CK8: private unnamed_addr constant [11 x i64] [i64 0, i64 {{4|8}}, i64 {{4|8}}, i64 4, i64 4, i64 4, i64 0, i64 4, i64 {{4|8}}, i64 {{4|8}}, i64 4]
  // CK8: private unnamed_addr constant [11 x i64]

// ps1
//
// PRESENT=0x1000 = 0x1000
// MEMBER_OF_1=0x1000000000000 | PRESENT=0x1000 | PTR_AND_OBJ=0x10 = 0x1000000001010
// PRESENT=0x1000 | PTR_AND_OBJ=0x10 = 0x1010
// PRESENT=0x1000 | PTR_AND_OBJ=0x10 | FROM=0x2 | TO=0x1 = 0x1013
// MEMBER_OF_1=0x1000000000000 | FROM=0x2 | TO=0x1 = 0x1000000000003
//
// CK8-SAME: {{^}} [i64 [[#0x1000]], i64 [[#0x1000000001010]],
// CK8-SAME: {{^}} i64 [[#0x1010]], i64 [[#0x1013]], i64 [[#0x1000000000003]],

// arg
//
// PRESENT=0x1000 | FROM=0x2 | TO=0x1 = 0x1003
//
// CK8-SAME: {{^}} i64 [[#0x1003]],

// ps2
//
// PRESENT=0x1000 = 0x1000
// MEMBER_OF_7=0x7000000000000 | PRESENT=0x1000 | FROM=0x2 | TO=0x1 = 0x7000000001003
// MEMBER_OF_7=0x7000000000000 | PTR_AND_OBJ=0x10 = 0x7000000000010
// PTR_AND_OBJ=0x10 = 0x10
// PTR_AND_OBJ=0x10 | FROM=0x2 | TO=0x1 = 0x13
//
// CK8-SAME: {{^}} i64 [[#0x1000]], i64 [[#0x7000000001003]],
// CK8-SAME: {{^}} i64 [[#0x7000000000010]], i64 [[#0x10]], i64 [[#0x13]]]
#pragma omp target data map(tofrom         \
                            : ps1->s)      \
    map(present, tofrom                    \
        : arg, ps1->ps->ps->ps->s, ps2->s) \
        map(tofrom                         \
            : ps2->ps->ps->ps->s)
  {
    ++(arg);
  }
}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK9 -verify -Wno-vla  -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK9 --check-prefix CK9-64
// RUN: %clang_cc1 -DCK9 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck %s  --check-prefix CK9 --check-prefix CK9-64
// RUN: %clang_cc1 -DCK9 -verify -Wno-vla  -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK9 --check-prefix CK9-32
// RUN: %clang_cc1 -DCK9 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck %s  --check-prefix CK9 --check-prefix CK9-32

// RUN: %clang_cc1 -DCK9 -verify -Wno-vla  -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK9 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK9 -verify -Wno-vla  -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// RUN: %clang_cc1 -DCK9 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify -Wno-vla  %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY2 %s
// SIMD-ONLY2-NOT: {{__kmpc|__tgt}}
#ifdef CK9
void test_present_modifier(int arg) {
// PRESENT=0x1000 | FROM=0x2 | TO=0x1 = 0x1003
// CK9: private unnamed_addr constant [1 x i64] [i64 [[#0x1003]]]
#pragma omp target data map(present, tofrom \
                            : arg)
  {++arg;}
}
#endif
#endif
