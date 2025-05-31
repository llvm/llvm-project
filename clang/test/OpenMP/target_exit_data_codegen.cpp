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

// CK1: [[IDENT_T:%.+]] = type { i32, i32, i32, i32, ptr }
// CK1: [[KMP_TASK_T_WITH_PRIVATES:%.+]] = type { [[KMP_TASK_T:%[^,]+]], [[KMP_PRIVATES_T:%.+]] }
// CK1: [[KMP_TASK_T]] = type { ptr, ptr, i32, %{{[^,]+}}, %{{[^,]+}} }
// CK1-32: [[KMP_PRIVATES_T]] = type { [1 x i64], [1 x ptr], [1 x ptr] }
// CK1-64: [[KMP_PRIVATES_T]] = type { [1 x ptr], [1 x ptr], [1 x i64] }

// CK1: [[SIZE00:@.+]] = {{.+}}constant [1 x i64] [i64 800]
// CK1: [[MTYPE00:@.+]] = {{.+}}constant [1 x i64] [i64 2]

// CK1: [[SIZE02:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// CK1: [[MTYPE02:@.+]] = {{.+}}constant [1 x i64] zeroinitializer

// CK1: [[MTYPE03:@.+]] = {{.+}}constant [1 x i64] [i64 6]

// CK1: [[SIZE04:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 24]
// CK1: [[MTYPE04:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 281474976710672]

// CK1: [[MTYPE05:@.+]] = {{.+}}constant [1 x i64] [i64 1026]

// CK1: [[MTYPE06:@.+]] = {{.+}}constant [1 x i64] [i64 1030]

// CK1-LABEL: _Z3fooi
void foo(int arg) {
  int la;
  float lb[arg];

  // Region 00
  // CK1-NOT: __tgt_target_data_begin
  // CK1-DAG: call i32 @__kmpc_omp_task(ptr @{{[^,]+}}, i32 %{{[^,]+}}, ptr [[TASK:%.+]])
  // CK1-DAG: [[TASK]] = call ptr @__kmpc_omp_target_task_alloc(ptr @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i[[sz:32|64]] {{36|64}}, i{{32|64}} 4, ptr [[OMP_TASK_ENTRY:@[^,]+]], i64 [[DEV:%.+]])
  // CK1-DAG: [[DEV]] = sext i32 [[DEV32:%.+]] to i64
  // CK1-DAG: [[PRIVATES:%.+]] = getelementptr inbounds nuw [[KMP_TASK_T_WITH_PRIVATES]], ptr [[TASK]], i32 0, i32 1
  // CK1-32-DAG: [[FPBPGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T]], ptr [[PRIVATES]], i32 0, i32 1
  // CK1-64-DAG: [[FPBPGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T]], ptr [[PRIVATES]], i32 0, i32 0
  // CK1-DAG: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{4|8}} [[FPBPGEP]], ptr align {{4|8}} [[BPGEP:%.+]], i[[sz]] {{4|8}}, i1 false)
  // CK1-DAG: [[BPGEP]] = getelementptr inbounds [1 x ptr], ptr [[BP:%.+]], i32 0, i32 0
  // CK1-DAG: [[BPGEP:%.+]] = getelementptr inbounds [1 x ptr], ptr [[BP]], i32 0, i32 0
  // CK1-DAG: store ptr [[GC:@[^,]+]], ptr [[BPGEP]], align
  // CK1-32-DAG: [[FPPGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T]], ptr [[PRIVATES]], i32 0, i32 2
  // CK1-64-DAG: [[FPPGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T]], ptr [[PRIVATES]], i32 0, i32 1
  // CK1-DAG: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{4|8}} [[FPPGEP]], ptr align {{4|8}} [[PGEP:%.+]], i[[sz]] {{4|8}}, i1 false)
  // CK1-DAG: [[PGEP]] = getelementptr inbounds [1 x ptr], ptr [[P:%.+]], i32 0, i32 0
  // CK1-DAG: [[PGEP:%.+]] = getelementptr inbounds [1 x ptr], ptr [[P]], i32 0, i32 0
  // CK1-DAG: store ptr [[GC]], ptr [[PGEP]], align
  // CK1-32-DAG: [[FPSZGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T]], ptr [[PRIVATES]], i32 0, i32 0
  // CK1-64-DAG: [[FPSZGEP:%.+]] = getelementptr inbounds nuw [[KMP_PRIVATES_T]], ptr [[PRIVATES]], i32 0, i32 2
  // CK1-DAG: call void @llvm.memcpy.p0.p0.i[[sz]](ptr align {{4|8}} [[FPSZGEP]], ptr align {{4|8}} [[SIZE00]], i[[sz]] {{4|8}}, i1 false)
  #pragma omp target exit data if(1+3-5) device(arg) map(from: gc) nowait
  {++arg;}

  // Region 01
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target exit data map(release: la) if(1+3-4)
  {++arg;}

  // Region 02
  // CK1-NOT: __tgt_target_data_begin
  // CK1: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
  // CK1: [[IFTHEN]]
  // CK1-DAG: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 4, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[SIZE02]], ptr [[MTYPE02]]{{.+}}, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: store ptr [[VAL0:%[^,]+]], ptr [[BP0]]
  // CK1-DAG: store ptr [[VAL0]], ptr [[P0]]
  // CK1: br label %[[IFEND:[^,]+]]

  // CK1: [[IFELSE]]
  // CK1: br label %[[IFEND]]
  // CK1: [[IFEND]]
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target exit data map(release: arg) if(arg) device(4)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 03
  // CK1-NOT: __tgt_target_data_begin
  // CK1-DAG: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE03]]{{.+}}, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: store ptr [[VAL0:%[^,]+]], ptr [[BP0]]
  // CK1-DAG: store ptr [[VAL0]], ptr [[P0]]
  // CK1-DAG: store i64 [[CSVAL0:%[^,]+]], ptr [[S0]]
  // CK1-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target exit data map(always, from: lb)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 04
  // CK1-NOT: __tgt_target_data_begin
  // CK1-DAG: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 -1, i32 2, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE04]]{{.+}}, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[PS0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: store ptr @gb, ptr [[BP0]]
  // CK1-DAG: store ptr getelementptr inbounds nuw ([[ST]], ptr @gb, i32 0, i32 1), ptr [[P0]]
  // CK1-DAG: [[DIV:%.+]] = sdiv exact i64 sub (i64 ptrtoint (ptr getelementptr (ptr, ptr getelementptr inbounds nuw (%struct.ST, ptr @gb, i32 0, i32 1), i32 1) to i64), i64 ptrtoint (ptr getelementptr inbounds nuw (%struct.ST, ptr @gb, i32 0, i32 1) to i64)), ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
  // CK1-DAG: store i64 [[DIV]], ptr [[PS0]],


  // CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK1-DAG: store ptr getelementptr inbounds nuw ([[ST]], ptr @gb, i32 0, i32 1), ptr [[BP1]]
  // CK1-DAG: store ptr [[SEC1:%[^,]+]], ptr [[P1]]
  // CK1-DAG: [[SEC1]] = getelementptr inbounds {{.+}}ptr [[SEC11:%[^,]+]], i{{.+}} 0
  // CK1-DAG: [[SEC11]] = load ptr, ptr getelementptr inbounds nuw ([[ST]], ptr @gb, i32 0, i32 1),

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target exit data map(release: gb.b[:3])
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 05
  // CK1-NOT: __tgt_target_data_begin
  // CK1-DAG: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE05]]{{.+}}, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: store ptr [[VAL0:%[^,]+]], ptr [[BP0]]
  // CK1-DAG: store ptr [[VAL0]], ptr [[P0]]
  // CK1-DAG: store i64 [[CSVAL0:%[^,]+]], ptr [[S0]]
  // CK1-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target exit data map(close, from: lb)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 06
  // CK1-NOT: __tgt_target_data_begin
  // CK1-DAG: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 -1, i32 1, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE06]]{{.+}}, ptr null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: store ptr [[VAL0:%[^,]+]], ptr [[BP0]]
  // CK1-DAG: store ptr [[VAL0]], ptr [[P0]]
  // CK1-DAG: store i64 [[CSVAL0:%[^,]+]], ptr [[S0]]
  // CK1-64-DAG: [[CSVAL0]] = mul nuw i64 %{{[^,]+}}, 4
  // CK1-32-DAG: [[CSVAL0]] = sext i32 [[CSVAL032:%.+]] to i64
  // CK1-32-DAG: [[CSVAL032]] = mul nuw i32 %{{[^,]+}}, 4
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target exit data map(always close, from: lb)
  {++arg;}
}

// CK1:     define internal {{.*}}i32 [[OMP_TASK_ENTRY]](i32 {{.*}}%{{[^,]+}}, ptr noalias noundef %{{[^,]+}})
// CK1-DAG: call void @__tgt_target_data_end_nowait_mapper(ptr @{{.+}}, i64 %{{[^,]+}}, i32 1, ptr [[BP:%[^,]+]], ptr [[P:%[^,]+]], ptr [[SZ:%[^,]+]], ptr [[MTYPE00]], ptr null, ptr null, i32 0, ptr null, i32 0, ptr null)
// CK1-DAG: [[BP]] = load ptr, ptr [[FPBPADDR:%[^,]+]], align
// CK1-DAG: [[P]] = load ptr, ptr [[FPPADDR:%[^,]+]], align
// CK1-DAG: [[SZ]] = load ptr, ptr [[FPSZADDR:%[^,]+]], align
// CK1-DAG: call void {{%.*}}(ptr %{{[^,]+}}, ptr [[FPBPADDR]], ptr [[FPPADDR]], ptr [[FPSZADDR]])
// CK1:     ret i32 0
// CK1:     }

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
    #pragma omp target exit data map(always, release: b[1:3]) if(a>123) device(arg)
    {arg++;}
    return arg;
  }
};

// CK2: [[SIZES:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 24]
// CK2: [[MTYPE00:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 281474976710676]

// CK2-LABEL: _Z3bari
int bar(int arg){
  ST<int> A;
  return A.foo(arg);
}

// Region 00
// CK2-NOT: __tgt_target_data_begin
// CK2: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CK2: [[IFTHEN]]
// CK2-DAG: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 [[DEV:%[^,]+]], i32 2, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE00]]{{.+}}, ptr null)
// CK2-DAG: [[DEV]] = sext i32 [[DEVi32:%[^,]+]] to i64
// CK2-DAG: [[DEVi32]] = load i32, ptr %{{[^,]+}},
// CK2-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK2-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK2-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]]

// CK2-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: [[PS0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: store ptr [[VAR0:%[^,]+]], ptr [[BP0]]
// CK2-DAG: store ptr [[SEC0:%[^,]+]], ptr [[P0]]
// CK2-DAG: store i64 [[CSVAL0:%[^,]+]], ptr [[PS0]],
// CK2-DAG: [[SEC0]] = getelementptr inbounds {{.*}}ptr [[VAR0]], i32 0, i32 1

// CK2-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK2-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK2-DAG: store ptr [[SEC0]], ptr [[BP1]]
// CK2-DAG: store ptr [[SEC1:%[^,]+]], ptr [[P1]]
// CK2-DAG: [[SEC1]] = getelementptr inbounds {{.*}}ptr [[SEC11:%[^,]+]], i{{.+}} 1
// CK2-DAG: [[SEC11]] = load ptr, ptr [[SEC111:%[^,]+]],
// CK2-DAG: [[SEC111]] = getelementptr inbounds {{.*}}ptr [[VAR0]], i32 0, i32 1

// CK2: br label %[[IFEND:[^,]+]]

// CK2: [[IFELSE]]
// CK2: br label %[[IFEND]]
// CK2: [[IFEND]]
// CK2: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
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
  #pragma omp target exit data map(from: arg) if(arg) device(4)
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

// CK4: [[STT:%.+]] = type { i32, ptr }
template <typename T>
struct STT {
  T a;
  double *b;

  T foo(T arg) {
    // Region 00
    #pragma omp target exit data map(always close, release: b[1:3]) if(a>123) device(arg)
    {arg++;}
    return arg;
  }
};

// CK4: [[SIZES:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 24]
// CK4: [[MTYPE00:@.+]] = {{.+}}constant [2 x i64] [i64 0, i64 281474976711700]

// CK4-LABEL: _Z3bari
int bar(int arg){
  STT<int> A;
  return A.foo(arg);
}

// Region 00
// CK4-NOT: __tgt_target_data_begin
// CK4: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CK4: [[IFTHEN]]
// CK4-DAG: call void @__tgt_target_data_end_mapper(ptr @{{.+}}, i64 [[DEV:%[^,]+]], i32 2, ptr [[GEPBP:%.+]], ptr [[GEPP:%.+]], ptr [[GEPS:%.+]], ptr [[MTYPE00]]{{.+}}, ptr null)
// CK4-DAG: [[DEV]] = sext i32 [[DEVi32:%[^,]+]] to i64
// CK4-DAG: [[DEVi32]] = load i32, ptr %{{[^,]+}},
// CK4-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK4-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK4-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]]

// CK4-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK4-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK4-DAG: [[PS0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i{{.+}} 0, i{{.+}} 0
// CK4-DAG: store ptr [[VAR0:%[^,]+]], ptr [[BP0]]
// CK4-DAG: store ptr [[SEC0:%[^,]+]], ptr [[P0]]
// CK4-DAG: store i64 [[CSVAL0:%[^,]+]], ptr [[PS0]],
// CK4-DAG: [[SEC0]] = getelementptr inbounds {{.*}}ptr [[VAR0]], i32 0, i32 1

// CK4-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK4-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK4-DAG: store ptr [[SEC0]], ptr [[BP1]]
// CK4-DAG: store ptr [[SEC1:%[^,]+]], ptr [[P1]]
// CK4-DAG: [[SEC1]] = getelementptr inbounds {{.*}}ptr [[SEC11:%[^,]+]], i{{.+}} 1
// CK4-DAG: [[SEC11]] = load ptr, ptr [[SEC111:%[^,]+]],
// CK4-DAG: [[SEC111]] = getelementptr inbounds {{.*}}ptr [[VAR0]], i32 0, i32 1

// CK4: br label %[[IFEND:[^,]+]]

// CK4: [[IFELSE]]
// CK4: br label %[[IFEND]]
// CK4: [[IFEND]]
// CK4: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
#endif
#endif
