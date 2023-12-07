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

// CK1: [[ST:%.+]] = type { i32, ptr }
template <typename T>
struct ST {
  T a;
  double *b;
};

ST<int> gb;
double gc[100];

// CK1: [[IDENT_T:%.+]] = type { i32, i32, i32, i32, ptr }

// CK1: [[KMP_TASK_T_WITH_PRIVATES:%.+]] = type { [[TASK_T:%[^,]+]], [[KMP_PRIVATES_T:%.+]] }
// CK1: [[TASK_T]] = type { ptr, ptr, i32, %{{[^,]+}}, %{{[^,]+}} }
// CK1-32: [[KMP_PRIVATES_T]] = type { [1 x i64], [1 x ptr], [1 x ptr] }
// CK1-64: [[KMP_PRIVATES_T]] = type { [1 x ptr], [1 x ptr], [1 x i64] }

// CK1: [[SIZE00:@.+]] = {{.+}}constant [1 x i64] [i64 800]
// CK1: [[MTYPE00:@.+]] = {{.+}}constant [1 x i64] zeroinitializer

// CK1: [[SIZE02:@.+]] = {{.+}}constant [1 x i64] [i64 4]
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
  // CK1-DAG:     call i32 @__kmpc_omp_task(ptr @{{[^,]+}}, i32 %{{[^,]+}}, ptr [[TASK:%.+]])
  // CK1-DAG:     [[TASK]] = call ptr @__kmpc_omp_target_task_alloc(ptr @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i{{32|64}} {{36|64}}, i{{32|64}} 4, ptr [[OMP_TASK_ENTRY:@[^,]+]], i64 [[DEV:%.+]])
  // CK1-DAG:     [[DEV]] = sext i32 [[DEVi32:%[^,]+]] to i64
  // CK1-DAG:     [[DEVi32]] = load i32, ptr %{{[^,]+}},
  // CK1-DAG:     [[TASK_WITH_PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_T_WITH_PRIVATES]], ptr [[TASK]], i32 0, i32 1
  // CK1-32-DAG:  [[FPBPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], ptr [[TASK_WITH_PRIVATES]], i32 0, i32 1
  // CK1-64-DAG:  [[FPBPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], ptr [[TASK_WITH_PRIVATES]], i32 0, i32 0
  // CK1-DAG:     call void @llvm.memcpy.p0.p0.i{{32|64}}(ptr align {{4|8}} [[FPBPGEP]], ptr align {{4|8}} [[BPGEP:%.+]], i{{32|64}} {{4|8}}, i1 false)
  // CK1-DAG:     [[BPGEP]] = getelementptr inbounds [1 x ptr], ptr [[BP:%.+]], i32 0, i32 0
  // CK1-DAG:     [[BPGEP:%.+]] = getelementptr inbounds [1 x ptr], ptr [[BP]], i32 0, i32 0
  // CK1-DAG:     store ptr @gc, ptr [[BPGEP]], align
  // CK1-32-DAG:  [[FPPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], ptr [[TASK_WITH_PRIVATES]], i32 0, i32 2
  // CK1-64-DAG:  [[FPPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], ptr [[TASK_WITH_PRIVATES]], i32 0, i32 1
  // CK1-DAG:     call void @llvm.memcpy.p0.p0.i{{32|64}}(ptr align {{4|8}} [[FPPGEP]], ptr align {{4|8}} [[PGEP:%.+]], i{{32|64}} {{4|8}}, i1 false)
  // CK1-DAG:     [[PGEP]] = getelementptr inbounds [1 x ptr], ptr [[P:%.+]], i32 0, i32 0
  // CK1-DAG:     [[PGEP:%.+]] = getelementptr inbounds [1 x ptr], ptr [[P]], i32 0, i32 0
  // CK1-DAG:     store ptr @gc, ptr [[PGEP]], align
  // CK1-32-DAG:  [[FPSZGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], ptr [[TASK_WITH_PRIVATES]], i32 0, i32 0
  // CK1-64-DAG:  [[FPSZGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], ptr [[TASK_WITH_PRIVATES]], i32 0, i32 2
  // CK1-DAG:     call void @llvm.memcpy.p0.p0.i{{32|64}}(ptr align {{4|8}} [[FPSZGEP]], ptr align {{4|8}} [[SIZE00]], i{{32|64}} 8, i1 false)
  // CK1-NOT:     __tgt_target_data_end
  #pragma omp target enter data if(1+3-5) device(arg) map(alloc: gc) nowait
  {++arg;}

  // Region 01
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target enter data map(to: la) if(1+3-4)
  {++arg;}

  // Region 02
  // CK1: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
  // CK1: [[IFTHEN]]
  // CK1-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 4, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[SIZE02]], ptr [[MTYPE02]]{{.+}}, ptr null)
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
  // CK1-NOT: __tgt_target_data_end
  #pragma omp target enter data map(to: arg) if(arg) device(4)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 03
  // CK1-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE03]]{{.+}}, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
  // CK1-DAG: store ptr [[VAR0]], ptr [[P0]]
  // CK1-DAG: store i64 [[CSVAL0:%[^,]+]], ptr [[S0]]
  // CK1-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  // CK1-NOT: __tgt_target_data_end
  #pragma omp target enter data map(always, to: lb)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 04
  // CK1-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 2, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE04]]{{.+}}, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: store ptr @gb, ptr [[BP0]]
  // CK1-DAG: store ptr getelementptr inbounds ([[ST]], ptr @gb, i32 0, i32 1), ptr [[P0]]
  // CK1-DAG: [[DIV:%.+]] = sdiv exact i64 sub (i64 ptrtoint (ptr getelementptr (ptr, ptr getelementptr inbounds (%struct.ST, ptr @gb, i32 0, i32 1), i32 1) to i64), i64 ptrtoint (ptr getelementptr inbounds (%struct.ST, ptr @gb, i32 0, i32 1) to i64)), ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
  // CK1-DAG: store i64 [[DIV]], ptr [[S0]],


  // CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK1-DAG: store ptr getelementptr inbounds ([[ST]], ptr @gb, i32 0, i32 1), ptr [[BP1]]
  // CK1-DAG: store ptr [[SEC1:%.+]], ptr [[P1]]
  // CK1-DAG: [[SEC1]] = getelementptr inbounds {{.+}}ptr [[SEC11:%[^,]+]], i{{.+}} 0
  // CK1-DAG: [[SEC11]] = load ptr, ptr getelementptr inbounds ([[ST]], ptr @gb, i32 0, i32 1),

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  // CK1-NOT: __tgt_target_data_end
  #pragma omp target enter data map(to: gb.b[:3])
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 05
  // CK1-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE05]]{{.+}}, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
  // CK1-DAG: store ptr [[VAR0]], ptr [[P0]]
  // CK1-DAG: store i64 [[CSVAL0:%[^,]+]], ptr [[S0]]
  // CK1-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  // CK1-NOT: __tgt_target_data_end
  #pragma omp target enter data map(close, to: lb)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 06
  // CK1-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE06]]{{.+}}, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
  // CK1-DAG: store ptr [[VAR0]], ptr [[P0]]
  // CK1-DAG: store i64 [[CSVAL0:%[^,]+]], ptr [[S0]]
  // CK1-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  // CK1-NOT: __tgt_target_data_end
  #pragma omp target enter data map(always close, to: lb)
  {++arg;}
}


// CK1:     define internal {{.*}}i32 [[OMP_TASK_ENTRY]](i32 {{.*}}%0, ptr noalias noundef %1)
// CK1-DAG: call void @__tgt_target_data_begin_nowait_mapper(ptr @{{.+}}, i64 %{{[^,]+}}, i32 1, ptr [[BPADDR:%[^,]+]], ptr [[PADDR:%[^,]+]], ptr [[SZADDR:%[^,]+]], ptr [[MTYPE00]], ptr null, ptr null)
// CK1-DAG: [[BPADDR]] = load ptr, ptr [[FPBP:%[^,]+]], align
// CK1-DAG: [[PADDR]] = load ptr, ptr [[FPP:%[^,]+]], align
// CK1-DAG: [[SZADDR]] = load ptr, ptr [[FPSZ:%[^,]+]], align
// CK1-DAG: call void {{%.*}}(ptr %{{[^,]+}}, ptr [[FPBP]], ptr [[FPP]], ptr [[FPSZ]])
// CK1:   ret i32 0
// CK1: }

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK1A -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK1A --check-prefix CK1A-64
// RUN: %clang_cc1 -DCK1A -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1A --check-prefix CK1A-64
// RUN: %clang_cc1 -DCK1A -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK1A --check-prefix CK1A-32
// RUN: %clang_cc1 -DCK1A -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1A --check-prefix CK1A-32

// RUN: %clang_cc1 -DCK1A -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1A -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1A -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK1A -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
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
// CK1A: [[MTYPE00:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x1001]]]

// PRESENT=0x1000 | CLOSE=0x400 | ALWAYS=0x4 | TO=0x1 = 0x1405
// CK1A: [[MTYPE01:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x1405]]]

// CK1A-LABEL: _Z3fooi
void foo(int arg) {
  int la;
  float lb[arg];

  // Region 00
  // CK1A-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE00]]{{.+}})
  // CK1A-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1A-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1A-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1A-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1A-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1A-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1A-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
  // CK1A-DAG: store ptr [[VAR0]], ptr [[P0]]
  // CK1A-DAG: store i64 [[CSVAL0:%[^,]+]], ptr [[S0]]
  // CK1A-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1A-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1A-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1A: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  // CK1A-NOT: __tgt_target_data_end
  #pragma omp target enter data map(present, to: lb)
  {++arg;}

  // CK1A: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 01
  // CK1A-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE01]]{{.+}})
  // CK1A-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1A-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1A-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1A-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1A-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1A-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1A-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
  // CK1A-DAG: store ptr [[VAR0]], ptr [[P0]]
  // CK1A-DAG: store i64 [[CSVAL0:%[^,]+]], ptr [[S0]]
  // CK1A-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1A-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1A-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1A: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  // CK1A-NOT: __tgt_target_data_end
  #pragma omp target enter data map(always close present, to: lb)
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

// CK2: [[ST:%.+]] = type { i32, ptr }
template <typename T>
struct ST {
  T a;
  double *b;

  T foo(T arg) {
    // Region 00
    #pragma omp target enter data map(always, to: b[1:3]) if(a>123) device(arg)
    {arg++;}
    return arg;
  }
};

// CK2: [[SIZES:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 24]
// CK2: [[MTYPE00:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 281474976710677]

// CK2-LABEL: _Z3bari
int bar(int arg){
  ST<int> A;
  return A.foo(arg);
}

// Region 00
// CK2: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CK2: [[IFTHEN]]
// CK2-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 [[DEV:%[^,]+]], i32 2, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE00]]{{.+}}, ptr null)
// CK2-DAG: [[DEV]] = sext i32 [[DEVi32:%[^,]+]] to i64
// CK2-DAG: [[DEVi32]] = load i32, ptr %{{[^,]+}},
// CK2-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK2-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK2-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]]

// CK2-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: [[PS0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i{{.+}} 0, i{{.+}} 0
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
// CK2-NOT: __tgt_target_data_end
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
  #pragma omp target enter data map(to: arg) if(arg) device(4)
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

// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY3 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY3 %s
// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY3 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY3 %s
// SIMD-ONLY3-NOT: {{__kmpc|__tgt}}

// RUN: %clang_cc1 -DCK4 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix TCK4 --check-prefix TCK4-64
// RUN: %clang_cc1 -DCK4 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCK4 --check-prefix TCK4-64
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix TCK4 --check-prefix TCK4-32
// RUN: %clang_cc1 -DCK4 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCK4 --check-prefix TCK4-32

// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck --check-prefix SIMD-ONLY4 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY4 %s
// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck --check-prefix SIMD-ONLY4 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-target-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY4 %s
// SIMD-ONLY4-NOT: {{__kmpc|__tgt}}
#ifdef CK4

// CK4-LABEL: device_side_scan
void device_side_scan(int arg) {
  // CK4: tgt_target_data_begin
  // CK4: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  // CK4: ret
  // TCK4-NOT: tgt_target_data_begin
  #pragma omp target enter data map(to: arg) if(arg) device(4)
  {++arg;}
}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK5 --check-prefix CK5-32
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK5 --check-prefix CK5-32

// RUN: %clang_cc1 -DCK5 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK5 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
#ifdef CK5

// CK5: [[STT:%.+]] = type { i32, ptr }
template <typename T>
struct STT {
  T a;
  double *b;

  T foo(T arg) {
    // Region 00
    #pragma omp target enter data map(always close to: b[1:3]) if(a>123) device(arg)
    {arg++;}
    return arg;
  }
};

// CK5: [[SIZES:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 24]
// CK5: [[MTYPE00:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 281474976711701]

// CK5-LABEL: _Z3bari
int bar(int arg){
  STT<int> A;
  return A.foo(arg);
}

// Region 00
// CK5: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CK5: [[IFTHEN]]
// CK5-DAG: call void @__tgt_target_data_begin_mapper(ptr @{{.+}}, i64 [[DEV:%[^,]+]], i32 2, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE00]]{{.+}}, ptr null)
// CK5-DAG: [[DEV]] = sext i32 [[DEVi32:%[^,]+]] to i64
// CK5-DAG: [[DEVi32]] = load i32, ptr %{{[^,]+}},
// CK5-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK5-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK5-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]]

// CK5-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK5-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK5-DAG: [[PS0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i{{.+}} 0, i{{.+}} 0
// CK5-DAG: store ptr [[VAR0:%.+]], ptr [[BP0]]
// CK5-DAG: store ptr [[SEC0:%.+]], ptr [[P0]]
// CK5-DAG: store i64 {{%.+}}, ptr [[PS0]],
// CK5-DAG: [[SEC0]] = getelementptr inbounds {{.*}}ptr [[VAR0]], i32 0, i32 1

// CK5-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK5-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK5-DAG: store ptr [[SEC0]], ptr [[BP1]]
// CK5-DAG: store ptr [[SEC1:%.+]], ptr [[P1]]
// CK5-DAG: [[SEC1]] = getelementptr inbounds {{.*}}ptr [[SEC11:%[^,]+]], i{{.+}} 1
// CK5-DAG: [[SEC11]] = load ptr, ptr [[SEC111:%[^,]+]],
// CK5-DAG: [[SEC111]] = getelementptr inbounds {{.*}}ptr [[VAR0]], i32 0, i32 1

// CK5: br label %[[IFEND:[^,]+]]

// CK5: [[IFELSE]]
// CK5: br label %[[IFEND]]
// CK5: [[IFEND]]
// CK5: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
// CK5-NOT: __tgt_target_data_end
#endif
#endif
