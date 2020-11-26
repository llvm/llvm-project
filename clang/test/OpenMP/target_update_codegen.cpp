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

// CK1: [[IDENT_T:%.+]] = type { i32, i32, i32, i32, i8* }
// CK1: [[KMP_TASK_T_WITH_PRIVATES:%.+]] = type { [[KMP_TASK_T:%[^,]+]], [[KMP_PRIVATES_T:%.+]] }
// CK1: [[KMP_TASK_T]] = type { i8*, i32 (i32, i8*)*, i32, %{{[^,]+}}, %{{[^,]+}} }
// CK1-32: [[KMP_PRIVATES_T]] = type { [1 x i64], [1 x i8*], [1 x i8*] }
// CK1-64: [[KMP_PRIVATES_T]] = type { [1 x i8*], [1 x i8*], [1 x i64] }

// CK1: [[SIZE00:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 800]
// CK1: [[MTYPE00:@.+]] = {{.+}}constant [1 x i64] [i64 34]

// CK1: [[SIZE02:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] 4]
// CK1: [[MTYPE02:@.+]] = {{.+}}constant [1 x i64] [i64 33]

// CK1: [[MTYPE03:@.+]] = {{.+}}constant [1 x i64] [i64 34]

// CK1: [[SIZE04:@.+]] = {{.+}}constant [2 x i64] [i64 sdiv exact (i64 sub (i64 ptrtoint (double** getelementptr (double*, double** getelementptr inbounds (%struct.ST, %struct.ST* @gb, i32 0, i32 1), i32 1) to i64), i64 ptrtoint (double** getelementptr inbounds (%struct.ST, %struct.ST* @gb, i32 0, i32 1) to i64)), i64 ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)), i64 24]
// CK1: [[MTYPE04:@.+]] = {{.+}}constant [2 x i64] [i64 32, i64 281474976710673]

// CK1-LABEL: _Z3fooi
void foo(int arg) {
  int la;
  float lb[arg];

  // Region 00
  // CK1-DAG: call i32 @__kmpc_omp_task([[IDENT_T]]* @{{[^,]+}}, i32 %{{[^,]+}}, i8* [[TASK:%.+]])
  // CK1-DAG: [[TASK]] = call i8* @__kmpc_omp_target_task_alloc([[IDENT_T]]* @{{[^,]+}}, i32 %{{[^,]+}}, i32 1, i[[sz:32|64]] {{36|64}}, i{{32|64}} 4, i32 (i32, i8*)* bitcast (i32 (i32, [[KMP_TASK_T_WITH_PRIVATES]]*)* [[OMP_TASK_ENTRY:@[^,]+]] to i32 (i32, i8*)*), i64 [[DEV:%.+]])
  // CK1-DAG: [[DEV]] = sext i32 [[DEV32:%.+]] to i64
  // CK1-DAG: [[TASK_WITH_PRIVATES:%.+]] = bitcast i8* [[TASK]] to [[KMP_TASK_T_WITH_PRIVATES]]*
  // CK1-DAG: [[PRIVATES:%.+]] = getelementptr inbounds [[KMP_TASK_T_WITH_PRIVATES]], [[KMP_TASK_T_WITH_PRIVATES]]* [[TASK_WITH_PRIVATES]], i32 0, i32 1
  // CK1-32-DAG: [[FPBPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], [[KMP_PRIVATES_T]]* [[PRIVATES]], i32 0, i32 1
  // CK1-64-DAG: [[FPBPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], [[KMP_PRIVATES_T]]* [[PRIVATES]], i32 0, i32 0
  // CK1-DAG: [[FPBPADDR:%.+]] = bitcast [1 x i8*]* [[FPBPGEP]] to i8*
  // CK1-DAG: [[BPADDR:%.+]] = bitcast i8** [[BPGEP:%.+]] to i8*
  // CK1-DAG: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* align {{4|8}} [[FPBPADDR]], i8* align {{4|8}} [[BPADDR]], i[[sz]] {{4|8}}, i1 false)
  // CK1-DAG: [[BPGEP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP:%.+]], i32 0, i32 0
  // CK1-DAG: [[BPGEP:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP]], i32 0, i32 0
  // CK1-DAG: [[BPADDR:%.+]] = bitcast i8** [[BPGEP]] to [100 x double]**
  // CK1-DAG: store [100 x double]* [[GC:@[^,]+]], [100 x double]** [[BPADDR]], align
  // CK1-32-DAG: [[FPPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], [[KMP_PRIVATES_T]]* [[PRIVATES]], i32 0, i32 2
  // CK1-64-DAG: [[FPPGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], [[KMP_PRIVATES_T]]* [[PRIVATES]], i32 0, i32 1
  // CK1-DAG: [[FPPADDR:%.+]] = bitcast [1 x i8*]* [[FPPGEP]] to i8*
  // CK1-DAG: [[PADDR:%.+]] = bitcast i8** [[PGEP:%.+]] to i8*
  // CK1-DAG: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* align {{4|8}} [[FPPADDR]], i8* align {{4|8}} [[PADDR]], i[[sz]] {{4|8}}, i1 false)
  // CK1-DAG: [[PGEP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P:%.+]], i32 0, i32 0
  // CK1-DAG: [[PGEP:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P]], i32 0, i32 0
  // CK1-DAG: [[PADDR:%.+]] = bitcast i8** [[PGEP]] to [100 x double]**
  // CK1-DAG: store [100 x double]* [[GC]], [100 x double]** [[PADDR]], align
  // CK1-32-DAG: [[FPSZGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], [[KMP_PRIVATES_T]]* [[PRIVATES]], i32 0, i32 0
  // CK1-64-DAG: [[FPSZGEP:%.+]] = getelementptr inbounds [[KMP_PRIVATES_T]], [[KMP_PRIVATES_T]]* [[PRIVATES]], i32 0, i32 2
  // CK1-DAG: [[FPSZADDR:%.+]] = bitcast [1 x i64]* [[FPSZGEP]] to i8*
  // CK1-DAG: call void @llvm.memcpy.p0i8.p0i8.i[[sz]](i8* align {{4|8}} [[FPSZADDR]], i8* align {{4|8}} bitcast ([1 x i64]* [[SIZE00]] to i8*), i[[sz]] {{4|8}}, i1 false)
  #pragma omp target update if(1+3-5) device(arg) from(gc) nowait
  {++arg;}

  // Region 01
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target update to(la) if(1+3-4)
  {++arg;}

  // Region 02
  // CK1: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
  // CK1: [[IFTHEN]]
  // CK1-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 4, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}}, i8** null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK1-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK1-DAG: store i32* [[VAL0:%[^,]+]], i32** [[CBP0]]
  // CK1-DAG: store i32* [[VAL0]], i32** [[CP0]]
  // CK1: br label %[[IFEND:[^,]+]]

  // CK1: [[IFELSE]]
  // CK1: br label %[[IFEND]]
  // CK1: [[IFEND]]
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target update to(arg) if(arg) device(4)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 03
  // CK1-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}}, i8** null)
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK1-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK1-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
  // CK1-DAG: store float* [[VAL0:%[^,]+]], float** [[CBP0]]
  // CK1-DAG: store float* [[VAL0]], float** [[CP0]]
  // CK1-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  // CK1-NOT: __tgt_target_data_end
  #pragma omp target update from(lb)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 04
  // CK1-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE04]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE04]]{{.+}}, i8** null)
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
  // CK1-DAG: store double** getelementptr inbounds ([[ST]], [[ST]]* @gb, i32 0, i32 1), double*** [[CBP1]]
  // CK1-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double**
  // CK1-DAG: store double* [[VAL1:%[^,]+]], double** [[CP1]]
  // CK1-DAG: [[VAL1]] = getelementptr inbounds {{.+}}double* [[SEC11:%.+]], i{{.+}} 0
  // CK1-DAG: [[SEC11]] = load double*, double** getelementptr inbounds ([[ST]], [[ST]]* @gb, i32 0, i32 1),

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  // CK1-NOT: __tgt_target_data_end
  #pragma omp target update to(gb.b[:3])
  {++arg;}
}

// CK1:     define internal {{.*}}i32 [[OMP_TASK_ENTRY]](i32 {{.*}}%{{[^,]+}}, [[KMP_TASK_T_WITH_PRIVATES]]* noalias %{{[^,]+}})
// CK1-DAG: call void @__tgt_target_data_update_nowait_mapper(%struct.ident_t* @{{.+}}, i64 %{{[^,]+}}, i32 1, i8** [[BP:%[^,]+]], i8** [[P:%[^,]+]], i64* [[SZ:%[^,]+]], i64* getelementptr inbounds ([1 x i64], [1 x i64]* [[MTYPE00]], i32 0, i32 0), i8** null, i8** null)
// CK1-DAG: [[BP]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BPADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK1-DAG: [[P]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[PADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK1-DAG: [[SZ]] = getelementptr inbounds [1 x i64], [1 x i64]* [[SZADDR:%[^,]+]], i[[sz]] 0, i[[sz]] 0
// CK1-DAG: [[BPADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPBPADDR:%[^,]+]], align
// CK1-DAG: [[PADDR]] = load [1 x i8*]*, [1 x i8*]** [[FPPADDR:%[^,]+]], align
// CK1-DAG: [[SZADDR]] = load [1 x i64]*, [1 x i64]** [[FPSZADDR:%[^,]+]], align
// CK1-DAG: call void (i8*, ...) %{{.+}}(i8* %{{[^,]+}}, [1 x i8*]** [[FPBPADDR]], [1 x i8*]** [[FPPADDR]], [1 x i64]** [[FPSZADDR]])
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

// CK2: [[ST:%.+]] = type { i32, double* }
template <typename T>
struct ST {
  T a;
  double *b;

  T foo(T arg) {
    // Region 00
    #pragma omp target update from(b[1:3]) if(a>123) device(arg)
    {arg++;}
    return arg;
  }
};

// CK2: [[MTYPE00:@.+]] = {{.+}}constant [2 x i64] [i64 32, i64 281474976710674]

// CK2-LABEL: _Z3bari
int bar(int arg){
  ST<int> A;
  return A.foo(arg);
}

// Region 00
// CK2: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
// CK2: [[IFTHEN]]
// CK2-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 [[DEV:%[^,]+]], i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz:64|32]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
// CK2-DAG: [[DEV]] = sext i32 [[DEVi32:%[^,]+]] to i64
// CK2-DAG: [[DEVi32]] = load i32, i32* %{{[^,]+}},
// CK2-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK2-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
// CK2-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

// CK2-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
// CK2-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
// CK2-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to double***
// CK2-DAG: store [[ST]]* [[VAR0:%[^,]+]], [[ST]]** [[CBP0]]
// CK2-DAG: store double** [[SEC0:%[^,]+]], double*** [[CP0]]
// CK2-DAG: store i[[sz]] {{%.+}}, i[[sz]]* [[S0]]
// CK2-DAG: [[SEC0]] = getelementptr inbounds {{.*}}[[ST]]* [[VAR0]], i32 0, i32 1


// CK2-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK2-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK2-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double***
// CK2-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double**
// CK2-DAG: store double** [[CBPVAL1:%[^,]+]], double*** [[CBP1]]
// CK2-DAG: store double* [[SEC1:%[^,]+]], double** [[CP1]]
// CK2-DAG: [[SEC1]] = getelementptr inbounds {{.*}}double* [[SEC11:%[^,]+]], i{{.+}} 1
// CK2-DAG: [[SEC11]] = load double*, double** [[SEC111:%[^,]+]],
// CK2-DAG: [[SEC111]] = getelementptr inbounds {{.*}}[[ST]]* [[VAR0]], i32 0, i32 1

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
  // CK3-NOT: tgt_target_data_update
  // CK3: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  // CK3: ret
  #pragma omp target update to(arg) if(arg) device(4)
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
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck %s --check-prefix TCK4 --check-prefix TCK4-64
// RUN: %clang_cc1 -DCK4 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCK4 --check-prefix TCK4-64
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck %s --check-prefix TCK4 --check-prefix TCK4-32
// RUN: %clang_cc1 -DCK4 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s --check-prefix TCK4 --check-prefix TCK4-32

// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm-bc %s -o %t-ppc-host.bc
// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o - | FileCheck --check-prefix SIMD-ONLY4 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple powerpc64le-unknown-unknown -fopenmp-targets=powerpc64le-ibm-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-ppc-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY4 %s
// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm-bc %s -o %t-x86-host.bc
// RUN: %clang_cc1 -DCK4 -verify -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-llvm %s -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o - | FileCheck --check-prefix SIMD-ONLY4 %s
// RUN: %clang_cc1 -DCK4 -fopenmp-simd -x c++ -std=c++11 -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -emit-pch -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -x c++ -triple i386-unknown-unknown -fopenmp-targets=i386-pc-linux-gnu -std=c++11 -fopenmp-is-device -fopenmp-host-ir-file-path %t-x86-host.bc -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY4 %s
// SIMD-ONLY4-NOT: {{__kmpc|__tgt}}
#ifdef CK4

// CK4-LABEL: device_side_scan
void device_side_scan(int arg) {
  // CK4: tgt_target_data_update
  // CK4: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  // CK4: ret
  // TCK4-NOT: tgt_target_data_update
  #pragma omp target update from(arg) if(arg) device(4)
  {++arg;}
}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK5 --check-prefix CK5-32
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK5 --check-prefix CK5-32

// RUN: %clang_cc1 -DCK5 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK5 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK5 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

#ifdef CK5

// CK5: [[SIZE00:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// CK5: [[MTYPE00:@.+]] = {{.+}}constant [1 x i64] [i64 33]

// CK5-LABEL: lvalue
void lvalue(int *B, int l, int e) {

  // CK5-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK5-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK5-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK5-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK5-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK5-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK5-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK5-DAG: store i32* [[B_VAL:%.+]], i32** [[BPC0]]
  // CK5-DAG: store i32* [[B_VAL_2:%.+]], i32** [[PC0]]
  // CK5-DAG: [[B_VAL]] = load i32*, i32** [[B_ADDR:%.+]]
  // CK5-DAG: [[B_VAL_2]] = load i32*, i32** [[B_ADDR]]
  #pragma omp target update to(*B)
  *B += e;
  #pragma omp target update from(*B)
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK6 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK6 --check-prefix CK6-64
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK6 --check-prefix CK6-64
// RUN: %clang_cc1 -DCK6 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK6 --check-prefix CK6-32
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK6 --check-prefix CK6-32

// RUN: %clang_cc1 -DCK6 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK6 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK6 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK6 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

#ifdef CK6

// CK6: [[SIZE00:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// CK6: [[MTYPE00:@.+]] = {{.+}}constant [1 x i64] [i64 33]

// CK6-LABEL: lvalue
void lvalue(int *B, int l, int e) {

  // CK6-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK6-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK6-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK6-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK6-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK6-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK6-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK6-DAG: store i32* [[TWO:%.+]], i32** [[BPC0]]
  // CK6-DAG: store i32* [[ADD_PTR:%.+]], i32** [[PC0]]
  // CK6-64-DAG: [[ADD_PTR]] = getelementptr inbounds i32, i32* [[ONE:%.+]], i{{32|64}} [[IDX_EXT:%.+]]
  // CK6-32-DAG: [[ADD_PTR]] = getelementptr inbounds i32, i32* [[ONE:%.+]], i{{32|64}} [[L_VAL:%.+]]
  // CK6-64-DAG: [[IDX_EXT]] = sext i32 [[L_VAL:%.+]] to i64
  // CK6-DAG: [[L_VAL]] = load i32, i32* [[L_ADDR:%.+]]
  // CK6-DAG: store i32 {{.+}}, i32* [[L_ADDR]]
  // CK6-DAG: [[ONE]] = load i32*, i32** [[B_ADDR:%.+]]
  // CK6-DAG: [[TWO]] = load i32*, i32** [[B_ADDR]]
  #pragma omp target update to(*(B+l))
  *(B+l) += e;
  #pragma omp target update from(*(B+l))
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK7 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK7 --check-prefix CK7-64
// RUN: %clang_cc1 -DCK7 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK7 --check-prefix CK7-64
// RUN: %clang_cc1 -DCK7 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK7 --check-prefix CK7-32
// RUN: %clang_cc1 -DCK7 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK7 --check-prefix CK7-32

// RUN: %clang_cc1 -DCK7 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK7 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK7 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK7 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

#ifdef CK7

// CK7: [[SIZE00:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// CK7: [[MTYPE00:@.+]] = {{.+}}constant [1 x i64] [i64 33]

// CK7-LABEL: lvalue
void lvalue(int *B, int l, int e) {

  // CK7-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK7-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK7-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK7-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK7-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK7-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to i32**
  // CK7-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK7-DAG: store i32* [[B_VAL:%.+]], i32** [[BPC0]]
  // CK7-DAG: store i32* [[ARRAY_IDX:%.+]], i32** [[PC0]]
  // CK7-DAG: [[ARRAY_IDX]] = getelementptr inbounds i32, i32* [[ADD_PTR:%.+]], i{{32|64}} [[IDX_PROM:%.+]]
  // CK7-64-DAG: [[ADD_PTR]] = getelementptr inbounds i32, i32* [[ONE:%.+]], i64 [[IDX_EXT:%.+]]
  // CK7-32-DAG: [[ADD_PTR]] = getelementptr inbounds i32, i32* [[B_VAL_2:%.+]], i32 [[L_VAL:%.+]]
  // CK7-32-DAG: [[B_VAL]] = load i32*, i32** [[B_ADDR:%.+]]
  // CK7-32-DAG: [[B_VAL_2]] = load i32*, i32** [[B_ADDR]]
  // CK7-32-DAG: [[L_VAL]] = load i32, i32* [[L_ADDR:%.+]]
  // CK7-32-DAG: [[IDX_PROM]] = load i32, i32* [[L_ADDR]]
  // CK7-64-DAG: [[IDX_EXT:%.+]] = sext i32 [[L_VAL:%.+]] to i64
  // CK7-64-DAG: [[L_VAL:%.+]] = load i32, i32* [[L_ADDR:%.+]]
  // CK7-64-DAG: [[IDX_PROM]] = sext i32 [[L_VAL_2:%.+]] to i64
  // CK7-64-DAG: [[L_VAL_2]] = load i32, i32* [[L_ADDR]]
  #pragma omp target update to((B+l)[l])
  (B+l)[l] += e;
  #pragma omp target update from((B+l)[l])
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK8 --check-prefix CK8-64
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK8 --check-prefix CK8-64
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK8 --check-prefix CK8-32
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK8 --check-prefix CK8-32

// RUN: %clang_cc1 -DCK8 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK8 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK8 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK8 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

#ifdef CK8
// CK8: [[SIZE00:@.+]] = {{.+}}constant [2 x i[[sz:64|32]]] [i{{64|32}} {{8|4}}, i{{64|32}} 4]
// CK8: [[MTYPE00:@.+]] = {{.+}}constant [2 x i64] [i64 33, i64 17]

// CK8-LABEL: lvalue
void lvalue(int **B, int l, int e) {

  // CK8-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}], [2 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}, [2 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK8-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK8-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK8-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK8-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK8-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to i32***
  // CK8-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK8-DAG: store i32** [[ARRAY_IDX_1:%.+]], i32*** [[BPC0]]
  // CK8-DAG: store i32* [[ARRAY_IDX_4:%.+]], i32** [[PC0]]
  // CK8-DAG: store i32** [[ARRAY_IDX_1]], i32*** [[NINE:%.+]]
  // CK8-DAG: [[ARRAY_IDX_1]] = getelementptr inbounds i32*, i32** [[ADD_PTR:%.+]], i{{.+}} 1
  // CK8-64-DAG: [[ADD_PTR]] = getelementptr inbounds i32*, i32** [[B_VAL:%.+]], i{{.+}} [[IDX_EXT:%.+]]
  // CK8-64-DAG: [[IDX_EXT]] = sext i32 [[L_VAL:%.+]] to i64
  // CK8-DAG: [[ARRAY_IDX_4]] = getelementptr inbounds i32, i32* [[FIVE:%.+]], i{{.+}} 2
  // CK8-64-DAG: [[ARRAY_IDX_3:%.+]] = getelementptr inbounds i32*, i32** [[ADD_PTR_2:%.+]], i{{.+}} 1
  // CK8-64-DAG: [[ADD_PTR_2]] = getelementptr inbounds i32*, i32** %3, i{{.+}} [[IDX_EXT_1:%.+]]
  // CK8-64-DAG: [[IDX_EXT_1]] = sext i32 [[L_VAL:%.+]] to i{{.+}}
  // CK8-32-DAG: [[ADD_PTR]] = getelementptr inbounds i32*, i32** [[B_VAL:%.+]], i{{.+}} [[L_VAL:%.+]]
  // CK8-32-DAG: [[ARRAY_IDX_4:%.+]] = getelementptr inbounds i32*, i32** [[ADD_PTR_2:%.+]], i{{.+}} 1
  // CK8-32-DAG: [[ADD_PTR_2]] = getelementptr inbounds i32*, i32** %3, i{{.+}} [[L_VAL:%.+]]
  #pragma omp target update to((B+l)[1][2])
  (B+l)[1][2] += e;
  #pragma omp target update from((B+l)[1][2])
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK9 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK9 --check-prefix CK9-64
// RUN: %clang_cc1 -DCK9 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK9 --check-prefix CK9-64
// RUN: %clang_cc1 -DCK9 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK9 --check-prefix CK9-32
// RUN: %clang_cc1 -DCK9 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK9 --check-prefix CK9-32

// RUN: %clang_cc1 -DCK9 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK9 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK9 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK9 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}

#ifdef CK9

struct S {
  double *p;
};

// CK9: [[MTYPE00:@.+]] = {{.+}}constant [2 x i64] [i64 32, i64 281474976710673]

// CK9-LABEL: lvalue
void lvalue(struct S *s, int l, int e) {

  // CK9-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i{{.+}} [[GSIZE:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}, [2 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK9-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK9-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK9-DAG: [[GSIZE]] = getelementptr inbounds {{.+}}[[SIZE:%[^,]+]]
  //
  // CK9-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK9-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK9-DAG: [[SIZE0:%.+]] = getelementptr inbounds {{.+}}[[SIZE]], i{{.+}} 0, i{{.+}} 1
  // CK9-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to double***
  // CK9-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to double**
  // CK9-DAG: store double** [[P:%.+]], double*** [[BPC0]]
  // CK9-DAG: store double* [[P_VAL:%.+]], double** [[PC0]]
  // CK9-DAG: store i{{.+}} 8, i{{.+}}* [[SIZE0]]
  // CK9-DAG: [[P]] = getelementptr inbounds [[STRUCT_S:%.+]], [[STRUCT_S]]* [[S_VAL:%.+]], i32 0, i32 0
  // CK9-DAG: [[S_VAL]] = load [[STRUCT_S]]*, [[STRUCT_S]]** [[S_ADDR:%.+]]
  // CK9-DAG: [[P_VAL]] = load double*, double** [[P_1:%.+]],
  // CK9-DAG: [[P_1]] = getelementptr inbounds [[STRUCT_S]], [[STRUCT_S]]* [[S_VAL_2:%.+]], i32 0, i32 0
  // CK9-DAG: [[S_VAL_2]] = load [[STRUCT_S]]*, [[STRUCT_S]]** [[S_ADDR:%.+]]
  #pragma omp target update to(*(s->p))
    *(s->p) += e;
  #pragma omp target update from(*(s->p))
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK10 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK10 --check-prefix CK10-64
// RUN: %clang_cc1 -DCK10 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK10 --check-prefix CK10-64
// RUN: %clang_cc1 -DCK10 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK10 --check-prefix CK10-32
// RUN: %clang_cc1 -DCK10 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK10 --check-prefix CK10-32

// RUN: %clang_cc1 -DCK10 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK10 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK10 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK10 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK10

struct S {
  double *p;
};

// CK10: [[MTYPE00:@.+]] = {{.+}}constant [2 x i64] [i64 32, i64 281474976710673]

// CK10-LABEL: lvalue
void lvalue(struct S *s, int l, int e) {

  // CK10-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i{{.+}} [[GSIZE:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}, [2 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK10-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK10-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK10-DAG: [[GSIZE]] = getelementptr inbounds {{.+}}[[SIZE:%[^,]+]]
  //
  // CK10-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK10-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK10-DAG: [[SIZE0:%.+]] = getelementptr inbounds {{.+}}[[SIZE]], i{{.+}} 0, i{{.+}} 1
  // CK10-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to double***
  // CK10-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to double**
  // CK10-DAG: store double** [[P_VAL:%.+]], double*** [[BPC0]]
  // CK10-DAG: store double* [[ADD_PTR:%.+]], double** [[PC0]]
  // CK10-DAG: store i{{.+}} 8, i{{.+}}* [[SIZE0]]
  // CK10-64-DAG: [[ADD_PTR]] = getelementptr inbounds double, double* [[S_P:%.+]], i{{.+}} [[IDX_EXT:%.+]]
  // CK10-32-DAG: [[ADD_PTR]] = getelementptr inbounds double, double* [[S_P:%.+]], i{{.+}} [[L_VAL:%.+]]
  // CK10-64-DAG: [[IDX_EXT]] = sext i32 [[L_VAL:%.+]] to i64
  // CK10-DAG: [[S_P]] = load double*, double** [[P_VAL_1:%.+]]
  // CK10-DAG: getelementptr inbounds {{.+}}, {{.+}}* [[SS_1:%.+]], i32 0, i32 0
  // CK10-DAG: [[SS_1]] = load {{.+}}*, {{.+}}** [[S_ADDR:%.+]]
  #pragma omp target update to(*(s->p+l))
    *(s->p+l) += e;
  #pragma omp target update from(*(s->p+l))
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK11 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK11 --check-prefix CK11-64
// RUN: %clang_cc1 -DCK11 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK11 --check-prefix CK11-64
// RUN: %clang_cc1 -DCK11 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK11 --check-prefix CK11-32
// RUN: %clang_cc1 -DCK11 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK11 --check-prefix CK11-32

// RUN: %clang_cc1 -DCK11 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK11 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK11 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK11 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK11

struct S {
  double *p;
};
// CK11: [[MTYPE00:@.+]] = {{.+}}constant [2 x i64] [i64 32, i64 281474976710673]

// CK11-LABEL: lvalue
void lvalue(struct S *s, int l, int e) {

  // CK11-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i{{.+}} [[GSIZE:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}, [2 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK11-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK11-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK11-DAG: [[GSIZE]] = getelementptr inbounds {{.+}}[[SIZE:%[^,]+]]
  //
  // CK11-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK11-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK11-DAG: [[SIZE0:%.+]] = getelementptr inbounds {{.+}}[[SIZE]], i{{.+}} 0, i{{.+}} 1
  // CK11-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to double***
  // CK11-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to double**
  // CK11-DAG: store double** [[P:%.+]], double*** [[BPC0]]
  // CK11-DAG: store double* [[ARRAY_IDX:%.+]], double** [[PC0]]
  // CK11-DAG: store i{{.+}} 8, i{{.+}} [[SIZE0]]
  // CK11-DAG: [[P]] = getelementptr inbounds [[STRUCT_S:%.+]], [[STRUCT_S]]* [[SS_1:%.+]], i32 0, i32 0
  // CK11-DAG: [[ARRAY_IDX]] = getelementptr inbounds double, double* [[ADD_PTR:%.+]], i{{.+}} 3
  // CK11-64-DAG: [[ADD_PTR]] = getelementptr inbounds double, double* [[S_P:%.+]], i{{.+}} [[IDX_EXT:%.+]]
  // CK11-32-DAG: [[ADD_PTR]] = getelementptr inbounds double, double* [[S_P:%.+]], i{{.+}} [[L_VAL:%.+]]
  // CK11-64-DAG: [[IDX_EXT]] = sext i32 [[L_VAL:%.+]] to i64
  // CK11-DAG: [[S_P]] = load double*, double** [[P_1:%.+]],
  // CK11-DAG: [[P_1]] = getelementptr inbounds [[STRUCT_S]], [[STRUCT_S]]* [[S_ADDR:%.+]], i32 0, i32 0
  #pragma omp target update to((s->p+l)[3])
    (s->p+l)[3] += e;
  #pragma omp target update from((s->p+l)[3])
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK12 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK12 --check-prefix CK12-64
// RUN: %clang_cc1 -DCK12 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK12 --check-prefix CK12-64
// RUN: %clang_cc1 -DCK12 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK12 --check-prefix CK12-32
// RUN: %clang_cc1 -DCK12 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK12 --check-prefix CK12-32

// RUN: %clang_cc1 -DCK12 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK12 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK12 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK12 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK12

struct S {
  double *p;
  struct S *sp;
};
// CK12: [[MTYPE00:@.+]] = {{.+}}constant [3 x i64] [i64 32, i64 281474976710672, i64 17]

// CK12-LABEL: lvalue
void lvalue(struct S *s, int l, int e) {

  // CK12-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i{{.+}} [[GSIZE:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}, [3 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK12-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK12-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK12-DAG: [[GSIZE]] = getelementptr inbounds {{.+}}[[SIZE:%[^,]+]]
  //
  // CK12-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK12-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK12-DAG: [[SIZE2:%.+]] = getelementptr inbounds {{.+}}[[SIZE]], i{{.+}} 0, i{{.+}} 2
  // CK12-DAG: [[BPC2:%.+]] = bitcast i8** [[BP2]] to double***
  // CK12-DAG: [[PC2:%.+]] = bitcast i8** [[P2]] to double**
  // CK12-DAG: store double** [[P_VAL:%.+]], double*** [[BPC2]]
  // CK12-DAG: store double* [[SIX:%.+]], double** [[PC2]]
  // CK12-DAG: store i{{.+}} 8, i{{.+}}* [[SIZE2]]
  // CK12-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK12-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK12-DAG: [[SIZE1:%.+]] = getelementptr inbounds {{.+}}[[SIZE]], i{{.+}} 0, i{{.+}} 1
  // CK12-DAG: [[BPC1:%.+]] = bitcast i8** [[BP1]] to [[STRUCT_S:%.+]]***
  // CK12-DAG: [[PC1:%.+]] = bitcast i8** [[P1]] to double***
  // CK12-DAG: store [[STRUCT_S]]** [[SP:%.+]], [[STRUCT_S]]*** [[BPC1]]
  // CK12-DAG: store double** [[P_VAL:%.+]], double*** [[PC1]]
  // CK12-DAG: store i{{.+}} {{4|8}}, i{{.+}}* [[SIZE1]]
  // CK12-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK12-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK12-DAG: [[SIZE0:%.+]] = getelementptr inbounds {{.+}}[[SIZE]], i{{.+}} 0, i{{.+}} 0
  // CK12-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to [[STRUCT_S:%.+]]**
  // CK12-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to [[STRUCT_S]]***
  // CK12-DAG: store [[STRUCT_S]]* [[ZERO:%.+]], [[STRUCT_S]]** [[BPC0]]
  // CK12-DAG: store [[STRUCT_S]]** [[SP]], [[STRUCT_S]]*** [[PC0]]
  // CK12-DAG: store [[STRUCT_S]]** [[S:%.+]], [[STRUCT_S]]*** [[S_VAL:%.+]]
  // CK12-DAG: store i{{.+}} {{.+}}, i{{.+}}* [[SIZE0]]
  // CK12-DAG: [[SP]] = getelementptr inbounds [[STRUCT_S]], [[STRUCT_S]]* [[ONE:%.+]], i32 0, i32 1
  // CK12-DAG: [[ONE]] = load [[STRUCT_S]]*, [[STRUCT_S]]** [[S:%.+]],
  // CK12-DAG: [[ZERO]] = load [[STRUCT_S]]*, [[STRUCT_S]]** [[S]],
  #pragma omp target update to(*(s->sp->p))
    *(s->sp->p) = e;
  #pragma omp target update from(*(s->sp->p))
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK13 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK13 --check-prefix CK13-64
// RUN: %clang_cc1 -DCK13 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK13 --check-prefix CK13-64
// RUN: %clang_cc1 -DCK13 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK13 --check-prefix CK13-32
// RUN: %clang_cc1 -DCK13 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK13 --check-prefix CK13-32

// RUN: %clang_cc1 -DCK13 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK13 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK13 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK13 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK13

// CK13: [[SIZE00:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// CK13: [[MTYPE00:@.+]] = {{.+}}constant [1 x i64] [i64 33]

// CK13-LABEL: lvalue
void lvalue(int **BB, int a, int b) {

  // CK13-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK13-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK13-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK13-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK13-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK13-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to i32***
  // CK13-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to i32**
  // CK13-DAG: store i32** [[B_VAL1:%.+]], i32*** [[BPC0]]
  // CK13-DAG: store i32* [[ADD_PTR_2:%.+]], i32** [[PC0]]
  // CK13-64-DAG: [[ADD_PTR_2]] = getelementptr inbounds i32, i32* [[RESULT:%.+]], i64 [[IDX_EXT_1:%.+]]
  // CK13-32-DAG: [[ADD_PTR_2]] = getelementptr inbounds i32, i32* [[RESULT:%.+]], i32 [[B_ADDR:%.+]]
  // CK13-64-DAG: [[IDX_EXT_1]] = sext i32 [[B_ADDR:%.+]]
  // CK13-DAG: [[RESULT]] = load i32*, i32** [[ADD_PTR:%.+]],
  // CK13-64-DAG: [[ADD_PTR]] = getelementptr inbounds i32*, i32** [[B_VAL:%.+]], i64 [[IDX_EXT:%.+]]
  // CK13-32-DAG: [[ADD_PTR]] = getelementptr inbounds i32*, i32** [[B_VAL:%.+]], i32 [[A_ADDR:%.+]]
  // CK13-64-DAG: [[IDX_EXT]] = sext i32 [[TWO:%.+]] to i64
  // CK13-DAG: [[B_VAL]] = load i32**, i32*** [[BB_ADDR:%.+]]
  // CK13-DAG: [[B_VAL1]] = load i32**, i32*** [[BB_ADDR]]
  #pragma omp target update to(*(*(BB+a)+b))
  *(*(BB+a)+b) = 1;
  #pragma omp target update from(*(*(BB+a)+b))
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK14 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK14 --check-prefix CK14-64
// RUN: %clang_cc1 -DCK14 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK14 --check-prefix CK14-64
// RUN: %clang_cc1 -DCK14 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK14 --check-prefix CK14-32
// RUN: %clang_cc1 -DCK14 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK14 --check-prefix CK14-32

// RUN: %clang_cc1 -DCK14 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK14 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK14 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK14 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK14

// CK14: [[MTYPE00:@.+]] = private {{.*}}constant [2 x i64] [i64 32, i64 281474976710673]

struct SSA {
  double *p;
  double *&pr;
  SSA(double *&pr) : pr(pr) {}
};

struct SSB {
  double *d;
  SSA *p;
  SSA *&pr;
  SSB(SSA *&pr) : pr(pr) {}

  // CK14-LABEL: define {{.+}}foo
  void foo() {

    // CK14-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GSIZE:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
    // CK14-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK14-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
    // CK14-DAG: [[GSIZE]] = getelementptr inbounds {{.+}}[[SIZE:%[^,]+]]

    // CK14-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
    // CK14-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
    // CK14-DAG: [[SIZE1:%.+]] = getelementptr inbounds {{.+}}[[SIZE]], i{{.+}} 0, i{{.+}} 1
    // CK14-DAG: [[BPC1:%.+]] = bitcast i8** [[BP1]] to double***
    // CK14-DAG: [[PC1:%.+]] = bitcast i8** [[P1]] to double**
    // CK14-DAG: store double** [[D_VAL:%.+]], double*** [[BPC1]]
    // CK14-DAG: store double* [[ADD_PTR:%.+]], double** [[PC1]]
    // CK14-DAG: store i64 8, i64* [[SIZE1]]
    // CK14-DAG: [[ADD_PTR]] = getelementptr inbounds double, double* [[ZERO:%.+]], i{{.+}} 1
    // CK14-DAG: [[ZERO]] = load double*, double** [[D_VAL_2:%.+]]
    // CK14-DAG: [[D_VAL]] = getelementptr inbounds [[SSB:%.+]], [[SSB:%.+]]* [[THIS:%.+]], i32 0, i32 0
    // CK14-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK14-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK14-DAG: [[SIZE0:%.+]] = getelementptr inbounds {{.+}}[[SIZE]], i{{.+}} 0, i{{.+}} 0
    // CK14-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to [[SSB]]**
    // CK14-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to double***
    // CK14-DAG: store [[SSB]]* [[THIS]], [[SSB]]** [[BPC0]],
    // CK14-DAG: store double** [[D_VAL]], double*** [[PC0]]
    // CK14-DAG: store i{{.+}} [[COMPUTE_LEN:%.+]], i{{.+}}* [[SIZE0]]

    #pragma omp target update to(*(this->d+1))
    *(this->d+1) = 1;
    #pragma omp target update from(*(this->d+1))
  }
};

void lvalue_member(SSA *sap) {
  double *d;
  SSA sa(d);
  SSB sb(sap);
  sb.foo();
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK15 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK15 --check-prefix CK15-64
// RUN: %clang_cc1 -DCK15 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK15 --check-prefix CK15-64
// RUN: %clang_cc1 -DCK15 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK15 --check-prefix CK15-32
// RUN: %clang_cc1 -DCK15 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK15 --check-prefix CK15-32

// RUN: %clang_cc1 -DCK15 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK15 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK15 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK15 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK15

// CK15: [[MTYPE00:@.+]] = {{.+}}constant [2 x i64] [i64 32, i64 281474976710673]

struct SSA {
  double *p;
  double *&pr;
  SSA(double *&pr) : pr(pr) {}
};

//CK-15-LABEL: lvalue_member
void lvalue_member(SSA *sap) {

  // CK15-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GSIZE:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK15-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK15-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK15-DAG: [[GSIZE]] = getelementptr inbounds {{.+}}[[SIZE:%[^,]+]]

  // CK15-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK15-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK15-DAG: [[SIZE1:%.+]] = getelementptr inbounds {{.+}}[[SIZE]], i{{.+}} 0, i{{.+}} 1
  // CK15-DAG: [[BPC1:%.+]] = bitcast i8** [[BP1]] to double***
  // CK15-DAG: [[PC1:%.+]] = bitcast i8** [[P1]] to double**
  // CK15-DAG: store double** [[P_VAL:%.+]], double*** [[BPC1]]
  // CK15-DAG: store double* [[ADD_PTR:%.+]], double** [[PC1]]
  // CK15-DAG: store i64 {{4|8}}, i64* [[SIZE1]]
  // CK15-DAG: [[ADD_PTR]] = getelementptr inbounds double, double* [[THREE:%.+]], i{{.+}} 3
  // CK15-DAG: [[THREE]] = load double*, double** [[P_VAL_1:%.+]]
  // CK15-DAG: [[P_VAL]] = getelementptr inbounds [[SSA:%.+]], [[SSA:%.+]]* [[THIS:%.+]], i32 0, i32 0
  // CK15-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK15-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK15-DAG: [[SIZE0:%.+]] = getelementptr inbounds {{.+}}[[SIZE]], i{{.+}} 0, i{{.+}} 0
  // CK15-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to [[SSA]]**
  // CK15-DAG: store [[SSA]]* [[ZERO:%.+]], [[SSA]]** [[BPC0]],
  // CK15-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to double***
  // CK15-DAG: store double** [[P_VAL]], double*** [[PC0]],
  // CK15-DAG: store i{{.+}} [[COMPUTE_SIZE:%.+]], i{{.+}}* [[SIZE0]]
  // CK15-DAG: [[COMPUTE_SIZE]] = sdiv exact i64 [[NINE:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
  // CK15-DAG: [[NINE]] = sub i{{.+}} [[SEVEN:%.+]], [[EIGHT:%.+]]
  // CK15-DAG: [[SEVEN]] = ptrtoint i8* [[SIX:%.+]] to i64
  // CK15-DAG: [[EIGHT]] = ptrtoint i8* [[FIVE:%.+]] to i64
  // CK15-DAG: [[SIX]] = bitcast double** {{.+}} to i8*
  // CK15-DAG: [[FIVE]] = bitcast double** {{.+}} to i8*
  // CK15-DAG: [[ZERO]] = load [[SSA]]*, [[SSA]]** %{{.+}},
  #pragma omp target update to(*(3+sap->p))
  *(3+sap->p) = 1;
  #pragma omp target update from(*(3+sap->p))
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK16 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK16 --check-prefix CK16-64
// RUN: %clang_cc1 -DCK16 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK16 --check-prefix CK16-64
// RUN: %clang_cc1 -DCK16 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK16 --check-prefix CK16-32
// RUN: %clang_cc1 -DCK16 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK16 --check-prefix CK16-32

// RUN: %clang_cc1 -DCK16 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK16 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK16 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK16 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK16

// CK16: [[SIZE00:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// CK16: [[MTYPE00:@.+]] = {{.+}}constant [1 x i64] [i64 33]

//CK16-LABEL: lvalue_find_base
void lvalue_find_base(float *f, int *i) {

  // CK16-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK16-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK16-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK16-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK16-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK16-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK16-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to float**
  // CK16-DAG: store float* [[F:%.+]], float** [[BPC0]]
  // CK16-DAG: store float* [[ADD_PTR:%.+]], float** [[PC0]]
  // CK16-32-DAG: [[ADD_PTR]] = getelementptr inbounds float, float* [[THREE:%.+]], i32 [[I:%.+]]
  // CK16-64-DAG: [[ADD_PTR]] = getelementptr inbounds float, float* [[THREE:%.+]], i64 [[IDX_EXT:%.+]]
  // CK16-DAG: [[THREE]] = load float*, float** [[F_ADDR:%.+]],
  // CK16-DAG: [[F]] = load float*, float** [[F_ADDR]],
  // CK16-64-DAG: [[IDX_EXT]] = sext i32 [[I:%.+]] to i64

  #pragma omp target update to(*(*i+f))
  *(*i+f) = 1.0;
  #pragma omp target update from(*(*i+f))
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK17 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK17 --check-prefix CK17-64
// RUN: %clang_cc1 -DCK17 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK17 --check-prefix CK17-64
// RUN: %clang_cc1 -DCK17 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK17 --check-prefix CK17-32
// RUN: %clang_cc1 -DCK17 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK17 --check-prefix CK17-32

// RUN: %clang_cc1 -DCK17 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK17 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK17 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK17 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK17

// CK17: [[SIZE00:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// CK17: [[MTYPE00:@.+]] = {{.+}}constant [1 x i64] [i64 33]

struct SSA {
  int i;
  SSA *sa;
};

//CK17-LABEL: lvalue_find_base
void lvalue_find_base(float **f, SSA *sa) {

  // CK17-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK17-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK17-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK17-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK17-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK17-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to float***
  // CK17-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to float**
  // CK17-DAG: store float** [[F_VAL:%.+]], float*** [[BPC0]],
  // CK17-DAG: store float* [[ADD_PTR_4:%.+]], float** [[PC0]],
  // CK17-64-DAG: [[ADD_PTR_4]] = getelementptr inbounds float, float* [[SEVEN:%.+]], i64 [[IDX_EXT_3:%.+]]
  // CK17-64-DAG: [[IDX_EXT_3]] = sext i32 [[I_VAL:%.+]] to i64
  // CK17-32-DAG: [[ADD_PTR_4]] = getelementptr inbounds float, float* [[SEVEN:%.+]], i32 [[I_VAL:%.+]]
  // CK17-DAG: [[I_VAL]] = load i32, i32* [[I:%.+]],
  // CK17-DAG: [[SEVEN]] = load float*, float** [[ADD_PTR:%.+]],
  // CK17-64-DAG: [[ADD_PTR]] = getelementptr inbounds float*, float** [[F:%.+]], i64 [[IDX_EXT:%.+]]
  // CK17-32-DAG: [[ADD_PTR]] = getelementptr inbounds float*, float** [[F:%.+]], i32 [[ADD:%.+]]
  // CK17-64-DAG: [[IDX_EXT]] = sext i32 [[ADD:%.+]] to i64
  // CK17-DAG: [[ADD]] = add nsw i32 1, [[FIVE:%.+]]
  // CK17-DAG: [[FIVE]] = load i32, i32* [[I_2:%.+]],
  // CK17-DAG: [[I_2]] = getelementptr inbounds [[SSA:%.+]], [[SSA]]* [[FOUR:%.+]], i32 0, i32 0
  // CK17-DAG: [[FOUR]] = load [[SSA]]*, [[SSA]]** [[SSA_ADDR:%.+]],
  // CK17-DAG: [[F]] = load float**, float*** [[F_ADDR:%.+]],
  // CK17-DAG: [[F_VAL]] = load float**, float*** [[F_ADDR]],

  #pragma omp target update to(*(sa->sa->i+*(1+sa->i+f)))
  *(sa->sa->i+*(1+sa->i+f)) = 1;
  #pragma omp target update from(*(sa->sa->i+*(1+sa->i+f)))
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK18 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK18 --check-prefix CK18-64
// RUN: %clang_cc1 -DCK18 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK18 --check-prefix CK18-64
// RUN: %clang_cc1 -DCK18 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK18 --check-prefix CK18-32
// RUN: %clang_cc1 -DCK18 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK18 --check-prefix CK18-32

// RUN: %clang_cc1 -DCK18 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK18 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK18 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY18 %s
// RUN: %clang_cc1 -DCK18 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY18 %s
// SIMD-ONLY18-NOT: {{__kmpc|__tgt}}
#ifdef CK18

// CK18-DAG: [[MTYPE_TO:@.+]] = {{.+}}constant [1 x i64] [i64 33]
// CK18-DAG: [[MTYPE_FROM:@.+]] = {{.+}}constant [1 x i64] [i64 34]

//CK18-LABEL: array_shaping
void array_shaping(float *f, int sa) {

  // CK18-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE_TO]]{{.+}}, i8** null)
  // CK18-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK18-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK18-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK18-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK18-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK18-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0

  // CK18-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK18-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to float**

  // CK18-DAG: store float* [[F1:%.+]], float** [[BPC0]],
  // CK18-DAG: store float* [[F2:%.+]], float** [[PC0]],
  // CK18-DAG: store i64 [[SIZE:%.+]], i64* [[S0]],
  // CK18-DAG: [[F1]] = load float*, float** [[F_ADDR:%.+]],
  // CK18-DAG: [[F2]] = load float*, float** [[F_ADDR]],

  // CK18-64-DAG: [[SIZE]] = mul nuw i64 [[SZ1:%.+]], 4
  // CK18-64-DAG: [[SZ1]] = mul nuw i64 12, %{{.+}}
  // CK18-32-DAG: [[SIZE]] = sext i32 [[SZ1:%.+]] to i64
  // CK18-32-DAG: [[SZ1]] = mul nuw i32 [[SZ2:%.+]], 4
  // CK18-32-DAG: [[SZ2]] = mul nuw i32 12, %{{.+}}
  #pragma omp target update to(([3][sa][4])f)
  sa = 1;
  // CK18-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE_FROM]]{{.+}}, i8** null)
  // CK18-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK18-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK18-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK18-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK18-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK18-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0

  // CK18-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK18-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to float**

  // CK18-DAG: store float* [[F1:%.+]], float** [[BPC0]],
  // CK18-DAG: store float* [[F2:%.+]], float** [[PC0]],
  // CK18-DAG: store i64 [[SIZE:%.+]], i64* [[S0]],
  // CK18-DAG: [[F1]] = load float*, float** [[F_ADDR]],
  // CK18-DAG: [[F2]] = load float*, float** [[F_ADDR]],

  // CK18-64-DAG: [[SIZE]] = mul nuw i64 [[SZ1:%.+]], 5
  // CK18-64-DAG: [[SZ1]] = mul nuw i64 4, %{{.+}}
  // CK18-32-DAG: [[SIZE]] = sext i32 [[SZ1:%.+]] to i64
  // CK18-32-DAG: [[SZ1]] = mul nuw i32 [[SZ2:%.+]], 5
  // CK18-32-DAG: [[SZ2]] = mul nuw i32 4, %{{.+}}
  #pragma omp target update from(([sa][5])f)
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK19 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK19 --check-prefix CK19-64
// RUN: %clang_cc1 -DCK19 -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK19 --check-prefix CK19-64
// RUN: %clang_cc1 -DCK19 -verify -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK19 --check-prefix CK19-32
// RUN: %clang_cc1 -DCK19 -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK19 --check-prefix CK19-32

// RUN: %clang_cc1 -DCK19 -verify -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK19 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK19 -verify -fopenmp-version=51 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -DCK19 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=51 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK19

// PRESENT=0x1000 | TARGET_PARAM=0x20 | TO=0x1 = 0x1021
// CK19: [[MTYPE00:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x1021]]]

// PRESENT=0x1000 | TARGET_PARAM=0x20 | FROM=0x2 = 0x1022
// CK19: [[MTYPE01:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x1022]]]

// CK19-LABEL: _Z13check_presenti
void check_present(int arg) {
  int la;
  float lb[arg];

  // Region 00
  // CK19-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
  // CK19-DAG: store float* [[VAL0:%[^,]+]], float** [[CBP0]]
  // CK19-DAG: store float* [[VAL0]], float** [[CP0]]
  // CK19-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  #pragma omp target update to(present: lb)
  ;

  // Region 01
  // CK19-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null)
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to float**
  // CK19-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to float**
  // CK19-DAG: store float* [[VAL0:%[^,]+]], float** [[CBP0]]
  // CK19-DAG: store float* [[VAL0]], float** [[CP0]]
  // CK19-DAG: store i64 [[CSVAL0:%[^,]+]], i64* [[S0]]
  #pragma omp target update from(present: lb)
  ;
}
#endif

///==========================================================================///
// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK20 --check-prefix CK20-32
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK20 --check-prefix CK20-32

// RUN: %clang_cc1 -DCK20 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK20 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK20 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK20 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// SIMD-ONLY19-NOT: {{__kmpc|__tgt}}
#ifdef CK20

struct ST {
  int a;
  double *b;
};

// CK20: [[STRUCT_ST:%.+]] = type { i32, double* }
// CK20: [[STRUCT_DESCRIPTOR:%.+]]  = type { i64, i64, i64 }

// CK20: [[MSIZE:@.+]] = {{.+}}constant [1 x i64] [i64 3]
// CK20: [[MTYPE:@.+]] = {{.+}}constant [1 x i64] [i64 17592186044449]

// CK20-LABEL: _Z3foo
void foo(int arg) {
  ST arr[3][4];
  // CK20: [[DIMS:%.+]] = alloca [3 x [[STRUCT_DESCRIPTOR]]],
  // CK20: [[ARRAY_IDX:%.+]] = getelementptr inbounds [3 x [4 x [[STRUCT_ST]]]], [3 x [4 x [[STRUCT_ST]]]]* [[ARR:%.+]], {{.+}} 0, {{.+}} 0
  // CK20: [[ARRAY_DECAY:%.+]] = getelementptr inbounds [4 x [[STRUCT_ST]]], [4 x [[STRUCT_ST]]]* [[ARRAY_IDX]], {{.+}} 0, {{.+}} 0
  // CK20: [[ARRAY_IDX_1:%.+]] = getelementptr inbounds [[STRUCT_ST]], [[STRUCT_ST]]* [[ARRAY_DECAY]], {{.+}}
  // CK20: [[BP0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP:%.+]], {{.+}} 0, {{.+}} 0
  // CK20: [[BPC:%.+]] = bitcast i8** [[BP0]] to [3 x [4 x [[STRUCT_ST]]]]**
  // CK20: store [3 x [4 x [[STRUCT_ST]]]]* [[ARR]], [3 x [4 x [[STRUCT_ST]]]]** [[BPC]],
  // CK20: [[P0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P:%.+]], {{.+}} 0, {{.+}} 0
  // CK20: [[PC:%.+]] = bitcast i8** [[P0]] to [[STRUCT_ST]]**
  // CK20: store [[STRUCT_ST]]* [[ARRAY_IDX_1]], [[STRUCT_ST]]** [[PC]],
  // CK20: [[DIM_1:%.+]] = getelementptr inbounds [3 x [[STRUCT_DESCRIPTOR]]], [3 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 0
  // CK20: [[OFFSET:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_1]], {{.+}} 0, {{.+}} 0
  // CK20: store i64 0, i64* [[OFFSET]],
  // CK20: [[COUNT:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_1]], {{.+}} 0, {{.+}} 1
  // CK20: store i64 2, i64* [[COUNT]],
  // CK20: [[STRIDE:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_1]], {{.+}} 0, {{.+}} 2
  // CK20: store i64 {{32|64}}, i64* [[STRIDE]],
  // CK20: [[DIM_2:%.+]] = getelementptr inbounds [3 x [[STRUCT_DESCRIPTOR]]], [3 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 1
  // CK20: [[OFFSET_2:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_2]], {{.+}} 0, {{.+}} 0
  // CK20: store i64 1, i64* [[OFFSET_2]],
  // CK20: [[COUNT_2:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_2]], {{.+}} 0, {{.+}} 1
  // CK20: store i64 4, i64* [[COUNT_2]],
  // CK20: [[STRIDE_2:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_2]], {{.+}} 0, {{.+}} 2
  // CK20: store i64 {{8|16}}, i64* [[STRIDE_2]],
  // CK20: [[DIM_3:%.+]] = getelementptr inbounds [3 x [[STRUCT_DESCRIPTOR]]], [3 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 2
  // CK20: [[OFFSET_3:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_3]], {{.+}} 0, {{.+}} 0
  // CK20: store i64 0, i64* [[OFFSET_3]],
  // CK20: [[COUNT_3:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_3]], {{.+}} 0, {{.+}} 1
  // CK20: store i64 1, i64* [[COUNT_3]],
  // CK20: [[STRIDE_3:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_3]], {{.+}} 0, {{.+}} 2
  // CK20: store i64 {{8|16}}, i64* [[STRIDE_3]],
  // CK20-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MSIZE]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE]]{{.+}})
  // CK20-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
  // CK20-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK20-DAG: [[PC0:%.+]] = bitcast [3 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]] to i8*
  // CK20-DAG: [[PTRS:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* %.offload_ptrs, i32 0, i32 0
  // CK20-DAG: store i8* [[PC0]], i8** [[PTRS]],

#pragma omp target update to(arr[0:2][1:4])
  { ++arg; }
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK21 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK21 --check-prefix CK21-64
// RUN: %clang_cc1 -DCK21 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK21 --check-prefix CK21-64
// RUN: %clang_cc1 -DCK21 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK21 --check-prefix CK21-32
// RUN: %clang_cc1 -DCK21 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK21 --check-prefix CK21-32

// RUN: %clang_cc1 -DCK21 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK21 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK21 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK21 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// SIMD-ONLY19-NOT: {{__kmpc|__tgt}}
#ifdef CK21

// CK21: [[STRUCT_ST:%.+]] = type { [10 x [10 x [10 x double*]]] }
// CK21: [[STRUCT_DESCRIPTOR:%.+]]  = type { i64, i64, i64 }

// CK21: [[MTYPE:@.+]] = {{.+}}constant [2 x i64] [i64 32, i64 299067162755073]

struct ST {
  double *dptr[10][10][10];

  // CK21: _ZN2ST3fooEv
  void foo() {
    // CK21: [[DIMS:%.+]] = alloca [4 x [[STRUCT_DESCRIPTOR]]],
    // CK21: [[ARRAY_IDX:%.+]] = getelementptr inbounds [10 x [10 x [10 x double*]]], [10 x [10 x [10 x double*]]]* [[DPTR:%.+]], {{.+}} 0, {{.+}} 0
    // CK21: [[ARRAY_DECAY:%.+]] = getelementptr inbounds [10 x [10 x double*]], [10 x [10 x double*]]* [[ARRAY_IDX]], {{.+}} 0, {{.+}} 0
    // CK21: [[ARRAY_IDX_1:%.+]] = getelementptr inbounds [10 x double*], [10 x double*]* [[ARRAY_DECAY]], {{.+}} 1
    // CK21: [[ARRAY_DECAY_2:%.+]] = getelementptr inbounds [10 x double*], [10 x double*]* [[ARRAY_IDX_1]], {{.+}} 0, {{.+}} 0
    // CK21: [[ARRAY_IDX_3:%.+]] = getelementptr inbounds {{.+}}, {{.+}}* [[ARRAY_DECAY_2]], {{.+}} 0
    // CK21: [[BP0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[BP:%.+]], {{.+}} 0, {{.+}} 0
    // CK21: [[P0:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* [[P:%.+]], i{{.+}} 0, i{{.+}} 0
    // CK21: [[DIM_1:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 0
    // CK21: [[OFFSET_1:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_1]], {{.+}} 0, {{.+}} 0
    // CK21: store i64 0, i64* [[OFFSET_1]],
    // CK21: [[COUNT_1:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_1]], {{.+}} 0, {{.+}} 1
    // CK21: store i64 2, i64* [[COUNT_1]],
    // CK21: [[STRIDE_1:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_1]], {{.+}} 0, {{.+}} 2
    // CK21: store i64 {{400|800}}, i64* [[STRIDE_1]],
    // CK21: [[DIM_2:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 1
    // CK21: [[OFFSET_2:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_2]], {{.+}} 0, {{.+}} 0
    // CK21: store i64 1, i64* [[OFFSET_2]],
    // CK21: [[COUNT_2:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_2]], {{.+}} 0, {{.+}} 1
    // CK21: store i64 3, i64* [[COUNT_2]],
    // CK21: [[STRIDE_2:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_2]], {{.+}} 0, {{.+}} 2
    // CK21: store i64 {{40|80}}, i64* [[STRIDE_2]],
    // CK21: [[DIM_3:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 2
    // CK21: [[OFFSET_3:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_3]], {{.+}} 0, {{.+}} 0
    // CK21: store i64 0, i64* [[OFFSET_3]],
    // CK21: [[COUNT_3:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_3]], {{.+}} 0, {{.+}} 1
    // CK21: store i64 4, i64* [[COUNT_3]],
    // CK21: [[STRIDE_3:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_3]], {{.+}} 0, {{.+}} 2
    // CK21: store i64 {{4|8}}, i64* [[STRIDE_3]],
    // CK21: [[DIM_4:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 3
    // CK21: [[OFFSET_4:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_4]], {{.+}} 0, {{.+}} 0
    // CK21: store i64 0, i64* [[OFFSET_4]],
    // CK21: [[COUNT_4:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_4]], {{.+}} 0, {{.+}} 1
    // CK21: store i64 1, i64* [[COUNT_4]],
    // CK21: [[STRIDE_4:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_4]], {{.+}} 0, {{.+}} 2
    // CK21: store i64 {{4|8}}, i64* [[STRIDE_4]],
    // CK21-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i{{.+}}* [[GEPSZ:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE]]{{.+}})
    // CK21-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
    // CK21-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
    // CK21-DAG: [[PC0:%.+]] = bitcast [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]] to i8*
    // CK21-DAG: [[PTRS:%.+]] = getelementptr inbounds [2 x i8*], [2 x i8*]* %.offload_ptrs, i32 0, i32 0
    // CK21-DAG: store i8* [[PC0]], i8** [[PTRS]],
#pragma omp target update to(dptr[0:2][1:3][0:4])
  }
};

void bar() {
  ST st;
  st.foo();
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK22 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK22 --check-prefix CK22-64
// RUN: %clang_cc1 -DCK22 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK22 --check-prefix CK22-64
// RUN: %clang_cc1 -DCK22 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK22 --check-prefix CK22-32
// RUN: %clang_cc1 -DCK22 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK22 --check-prefix CK22-32

// RUN: %clang_cc1 -DCK22 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK22 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK22 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK22 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// SIMD-ONLY19-NOT: {{__kmpc|__tgt}}
#ifdef CK22

// CK22: [[STRUCT_DESCRIPTOR:%.+]]  = type { i64, i64, i64 }

// CK22: [[MSIZE:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// CK22: [[MTYPE:@.+]] = {{.+}}constant [1 x i64] [i64 17592186044449]

struct ST {
  // CK22: _ZN2ST3fooEPA10_Pi
  void foo(int *arr[5][10]) {
    // CK22: [[DIMS:%.+]] = alloca [4 x [[STRUCT_DESCRIPTOR]]],
    // CK22: [[ARRAY_IDX:%.+]] = getelementptr inbounds [10 x i32*], [10 x i32*]* [[ARR:%.+]], {{.+}} 0
    // CK22: [[ARRAY_DECAY:%.+]] = getelementptr inbounds [10 x i32*], [10 x i32*]* [[ARRAY_IDX]], {{.+}} 0, {{.+}} 0
    // CK22: [[ARRAY_IDX_2:%.+]] = getelementptr inbounds i32*, i32** [[ARRAY_DECAY:%.+]], {{.+}} 1
    // CK22: [[BP0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP:%.+]], {{.+}} 0, {{.+}} 0
    // CK22: [[P0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P:%.+]], i{{.+}} 0, i{{.+}} 0
    // CK22: [[DIM_1:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 0
    // CK22: [[OFFSET:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_1]], {{.+}} 0, {{.+}} 0
    // CK22: store i64 0, i64* [[OFFSET]],
    // CK22: [[COUNT:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_1]], {{.+}} 0, {{.+}} 1
    // CK22: store i64 2, i64* [[COUNT]],
    // CK22: [[STRIDE:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_1]], {{.+}} 0, {{.+}} 2
    // CK22: store i64 200, i64* [[STRIDE]],
    // CK22: [[DIM_2:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 1
    // CK22: [[OFFSET:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_2]], {{.+}} 0, {{.+}} 0
    // CK22: store i64 1, i64* [[OFFSET]],
    // CK22: [[COUNT:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_2]], {{.+}} 0, {{.+}} 1
    // CK22: store i64 3, i64* [[COUNT]],
    // CK22: [[STRIDE:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_2]], {{.+}} 0, {{.+}} 2
    // CK22: store i64 40, i64* [[STRIDE]],
    // CK22: [[DIM_3:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 2
    // CK22: [[OFFSET:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_3]], {{.+}} 0, {{.+}} 0
    // CK22: store i64 0, i64* [[OFFSET]],
    // CK22: [[COUNT:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_3]], {{.+}} 0, {{.+}} 1
    // CK22: store i64 4, i64* [[COUNT]],
    // CK22: [[STRIDE:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_3]], {{.+}} 0, {{.+}} 2
    // CK22: store i64 4, i64* [[STRIDE]],
    // CK22: [[DIM_4:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 3
    // CK22: [[OFFSET:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_4]], {{.+}} 0, {{.+}} 0
    // CK22: store i64 0, i64* [[OFFSET]],
    // CK22: [[COUNT:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_4]], {{.+}} 0, {{.+}} 1
    // CK22: store i64 1, i64* [[COUNT]],
    // CK22: [[STRIDE:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_4]], {{.+}} 0, {{.+}} 2
    // CK22: store i64 4, i64* [[STRIDE]],
    // CK22-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MSIZE]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE]]{{.+}})
    // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
    // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
    // CK22-DAG: [[PC0:%.+]] = bitcast [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]] to i8*
    // CK22-DAG: [[PTRS:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* %.offload_ptrs, i32 0, i32 0
    // CK22-DAG: store i8* [[PC0]], i8** [[PTRS]],
#pragma omp target update to(arr[0:2][1:3][0:4])
  }
};

void bar() {
  ST st;
  int *arr[5][10];
  st.foo(arr);
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK23 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -DCK23 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -DCK23 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK23 --check-prefix CK23-32
// RUN: %clang_cc1 -DCK23 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK23 --check-prefix CK23-32

// RUN: %clang_cc1 -DCK23 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK23 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK23 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK23 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// SIMD-ONLY19-NOT: {{__kmpc|__tgt}}
#ifdef CK23

// CK23: [[STRUCT_DESCRIPTOR:%.+]]  = type { i64, i64, i64 }

// CK23: [[MSIZE:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// CK23: [[MTYPE:@.+]] = {{.+}}constant [1 x i64] [i64 17592186044449]

// CK23: foo
void foo(int arg) {
  float farr[5][5][5];
  // CK23: [[ARG_ADDR:%.+]] = alloca i32,
  // CK23: [[DIMS:%.+]] = alloca [4 x [[STRUCT_DESCRIPTOR]]],
  // CK23: [[ARRAY_IDX:%.+]] = getelementptr inbounds [5 x [5 x [5 x float]]], [5 x [5 x [5 x float]]]* [[ARR:%.+]], {{.+}} 0, {{.+}} 0
  // CK23: [[ARRAY_DECAY:%.+]] = getelementptr inbounds [5 x [5 x float]], [5 x [5 x float]]* [[ARRAY_IDX]], {{.+}} 0, {{.+}} 0
  // CK23: [[ARRAY_IDX_1:%.+]] = getelementptr inbounds [5 x float], [5 x float]* [[ARRAY_DECAY]], {{.+}}
  // CK23: [[ARRAY_DECAY_2:%.+]] = getelementptr inbounds [5 x float], [5 x float]* [[ARRAY_IDX_1]], {{.+}} 0, {{.+}} 0
  // CK23: [[ARRAY_IDX_2:%.+]] = getelementptr inbounds float, float* [[ARRAY_DECAY_2]], {{.+}}
  // CK23: [[MUL:%.+]] = mul nuw i64 4,
  // CK23: [[BP0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP:%.+]], {{.+}} 0, {{.+}} 0
  // CK23: [[BPC:%.+]] = bitcast i8** [[BP0]] to [5 x [5 x [5 x float]]]**
  // CK23: store [5 x [5 x [5 x float]]]* [[ARR]], [5 x [5 x [5 x float]]]** [[BPC]],
  // CK23: [[P0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P:%.+]], {{.+}} 0, {{.+}} 0
  // CK23: [[PC:%.+]] = bitcast i8** [[P0]] to float**
  // CK23: store float* [[ARRAY_IDX_2]], float** [[PC]],
  // CK23: [[DIM_1:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 0
  // CK23: [[OFFSET:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_1]], {{.+}} 0, {{.+}} 0
  // CK23: store i64 0, i64* [[OFFSET]],
  // CK23: [[COUNT:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_1]], {{.+}} 0, {{.+}} 1
  // CK23: store i64 2, i64* [[COUNT]],
  // CK23: [[STRIDE:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_1]], {{.+}} 0, {{.+}} 2
  // CK23: store i64 200, i64* [[STRIDE]],
  // CK23: [[DIM_2:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 1
  // CK23: [[OFFSET_2:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_2]], {{.+}} 0, {{.+}} 0
  // CK23: store i64 1, i64* [[OFFSET_2]],
  // CK23: [[COUNT_2:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_2]], {{.+}} 0, {{.+}} 1
  // CK23: store i64 2, i64* [[COUNT_2]],
  // CK23: [[STRIDE_2:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_2]], {{.+}} 0, {{.+}} 2
  // CK23: store i64 20, i64* [[STRIDE_2]],
  // CK23: [[DIM_3:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 2
  // CK23: [[OFFSET_3:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_3]], {{.+}} 0, {{.+}} 0
  // CK23: store i64 0, i64* [[OFFSET_3]],
  // CK23: [[COUNT_3:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_3]], {{.+}} 0, {{.+}} 1
  // CK23: store i64 2, i64* [[COUNT_3]],
  // CK23: [[STRIDE_3:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_3]], {{.+}} 0, {{.+}} 2
  // CK23: store i64 [[MUL]], i64* [[STRIDE_3]],
  // CK23: [[DIM_4:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 3
  // CK23: [[OFFSET_4:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_4]], {{.+}} 0, {{.+}} 0
  // CK23: store i64 0, i64* [[OFFSET_4]],
  // CK23: [[COUNT_4:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_4]], {{.+}} 0, {{.+}} 1
  // CK23: store i64 1, i64* [[COUNT_4]],
  // CK23: [[STRIDE_4:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_4]], {{.+}} 0, {{.+}} 2
  // CK23: store i64 4, i64* [[STRIDE_4]],
  // CK23-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MSIZE]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE]]{{.+}})
  // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
  // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK23-DAG: [[PC0:%.+]] = bitcast [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]] to i8*
  // CK23-DAG: [[PTRS:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* %.offload_ptrs, i32 0, i32 0
  // CK23-DAG: store i8* [[PC0]], i8** [[PTRS]],
#pragma omp target update to(farr[0:2:2][1:2:1][0:2:arg])
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK24 --check-prefix CK24-32
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK24 --check-prefix CK24-32

// RUN: %clang_cc1 -DCK24 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK24 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK24 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK24 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// SIMD-ONLY19-NOT: {{__kmpc|__tgt}}
#ifdef CK24

// CK24: [[STRUCT_DESCRIPTOR:%.+]]  = type { i64, i64, i64 }

// CK24: [[MSIZE:@.+]] = {{.+}}constant [1 x i64] [i64 4]
// CK24: [[MTYPE:@.+]] = {{.+}}constant [1 x i64] [i64 17592186044449]

// CK24: foo
void foo(int arg) {
  double darr[3][4][5];
  // CK24: [[DIMS:%.+]] = alloca [4 x [[STRUCT_DESCRIPTOR]]],
  // CK24: [[ARRAY_IDX:%.+]] = getelementptr inbounds [3 x [4 x [5 x double]]], [3 x [4 x [5 x double]]]* [[ARR:%.+]], {{.+}} 0, {{.+}} 0
  // CK24: [[ARRAY_DECAY:%.+]] = getelementptr inbounds [4 x [5 x double]], [4 x [5 x double]]* [[ARRAY_IDX]], {{.+}} 0, {{.+}} 0
  // CK24: [[ARRAY_IDX_1:%.+]] = getelementptr inbounds [5 x double], [5 x double]* [[ARRAY_DECAY]], {{.+}}
  // CK24: [[ARRAY_DECAY_2:%.+]] = getelementptr inbounds [5 x double], [5 x double]* [[ARRAY_IDX_1]], {{.+}} 0, {{.+}} 0
  // CK24: [[ARRAY_IDX_2:%.+]] = getelementptr inbounds double, double* [[ARRAY_DECAY_2]], {{.+}}
  // CK24: [[MUL:%.+]] = mul nuw i64 8,
  // CK24: [[SUB:%.+]] = sub nuw i64 4, [[ARG:%.+]]
  // CK24: [[LEN:%.+]] = udiv {{.+}} [[SUB]], 1
  // CK24: [[BP0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[BP:%.+]], {{.+}} 0, {{.+}} 0
  // CK24: [[BPC:%.+]] = bitcast i8** [[BP0]] to [3 x [4 x [5 x double]]]**
  // CK24: store [3 x [4 x [5 x double]]]* [[ARR]], [3 x [4 x [5 x double]]]** [[BPC]],
  // CK24: [[P0:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* [[P:%.+]], {{.+}} 0, {{.+}} 0
  // CK24: [[PC:%.+]] = bitcast i8** [[P0]] to double**
  // CK24: store double* [[ARRAY_IDX_2]], double** [[PC]],
  // CK24: [[DIM_1:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 0
  // CK24: [[OFFSET:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_1]], {{.+}} 0, {{.+}} 0
  // CK24: store i64 0, i64* [[OFFSET]],
  // CK24: [[COUNT:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_1]], {{.+}} 0, {{.+}} 1
  // CK24: store i64 2, i64* [[COUNT]],
  // CK24: [[STRIDE:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_1]], {{.+}} 0, {{.+}} 2
  // CK24: store i64 320, i64* [[STRIDE]],
  // CK24: [[DIM_2:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 1
  // CK24: [[OFFSET_2:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_2]], {{.+}} 0, {{.+}} 0
  // CK24: store i64 [[ARG]], i64* [[OFFSET_2]],
  // CK24: [[COUNT_2:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_2]], {{.+}} 0, {{.+}} 1
  // CK24: store i64 [[LEN]], i64* [[COUNT_2]],
  // CK24: [[STRIDE_2:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_2]], {{.+}} 0, {{.+}} 2
  // CK24: store i64 40, i64* [[STRIDE_2]],
  // CK24: [[DIM_3:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 2
  // CK24: [[OFFSET_3:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_3]], {{.+}} 0, {{.+}} 0
  // CK24: store i64 0, i64* [[OFFSET_3]],
  // CK24: [[COUNT_3:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_3]], {{.+}} 0, {{.+}} 1
  // CK24: store i64 2, i64* [[COUNT_3]],
  // CK24: [[STRIDE_3:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_3]], {{.+}} 0, {{.+}} 2
  // CK24: store i64 [[MUL]], i64* [[STRIDE_3]],
  // CK24: [[DIM_4:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 3
  // CK24: [[OFFSET_4:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_4]], {{.+}} 0, {{.+}} 0
  // CK24: store i64 0, i64* [[OFFSET_4]],
  // CK24: [[COUNT_4:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_4]], {{.+}} 0, {{.+}} 1
  // CK24: store i64 1, i64* [[COUNT_4]],
  // CK24: [[STRIDE_4:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_4]], {{.+}} 0, {{.+}} 2
  // CK24: store i64 8, i64* [[STRIDE_4]],
  // CK24-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MSIZE]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE]]{{.+}})
  // CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
  // CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK24-DAG: [[PC0:%.+]] = bitcast [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]] to i8*
  // CK24-DAG: [[PTRS:%.+]] = getelementptr inbounds [1 x i8*], [1 x i8*]* %.offload_ptrs, i32 0, i32 0
  // CK24-DAG: store i8* [[PC0]], i8** [[PTRS]],
#pragma omp target update to(darr[0:2:2][arg: :1][:2:arg])
}
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK25 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK25 --check-prefix CK25-64
// RUN: %clang_cc1 -DCK25 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK25 --check-prefix CK25-64
// RUN: %clang_cc1 -DCK25 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK25 --check-prefix CK25-32
// RUN: %clang_cc1 -DCK25 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK25 --check-prefix CK25-32

// RUN: %clang_cc1 -DCK25 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK25 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK25 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// RUN: %clang_cc1 -DCK25 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY19 %s
// SIMD-ONLY19-NOT: {{__kmpc|__tgt}}
#ifdef CK25

// CK25: [[STRUCT_DESCRIPTOR:%.+]]  = type { i64, i64, i64 }

// CK25: [[MSIZE:@.+]] = {{.+}}constant [3 x i64] [i64 4, i64 4, i64 3]
// CK25: [[MTYPE:@.+]] = {{.+}}constant [3 x i64] [i64 17592186044449, i64 33, i64 17592186044449]

// CK25-LABEL: _Z3foo
void foo(int arg) {
  int arr[3][4][5], x;
  float farr[4][3];

  // CK25: [[DIMS:%.+]] = alloca [4 x [[STRUCT_DESCRIPTOR]]],
  // CK25: [[DIMS_2:%.+]] = alloca [3 x [[STRUCT_DESCRIPTOR]]],
  // CK25: [[ARRAY_IDX:%.+]] = getelementptr inbounds [3 x [4 x [5 x i32]]], [3 x [4 x [5 x i32]]]* [[ARR:%.+]], {{.+}} 0, {{.+}} 0
  // CK25: [[ARRAY_DECAY:%.+]] = getelementptr inbounds [4 x [5 x i32]], [4 x [5 x i32]]* [[ARRAY_IDX]], {{.+}} 0, {{.+}} 0
  // CK25: [[ARRAY_IDX_1:%.+]] = getelementptr inbounds [5 x i32], [5 x i32]* [[ARRAY_DECAY]], {{.+}}
  // CK25: [[ARRAY_DECAY_2:%.+]] = getelementptr inbounds [5 x i32], [5 x i32]* [[ARRAY_IDX_1]], {{.+}} 0, {{.+}} 0
  // CK25: [[ARRAY_IDX_3:%.+]] = getelementptr inbounds {{.+}}, {{.+}}* [[ARRAY_DECAY_2]], {{.+}} 1
  // CK25: [[LEN:%.+]] = sub nuw i64 4, [[ARG_ADDR:%.+]]
  // CK25: [[ARRAY_IDX_4:%.+]] = getelementptr inbounds [4 x [3 x float]], [4 x [3 x float]]* [[FARR:%.+]], {{.+}} 0, {{.+}} 0
  // CK25: [[ARRAY_DECAY_5:%.+]] = getelementptr inbounds [3 x float], [3 x float]* [[ARRAY_IDX_4]], {{.+}} 0, {{.+}} 0
  // CK25: [[ARRAY_IDX_6:%.+]] = getelementptr inbounds float, float* [[ARRAY_DECAY_5:%.+]], {{.+}} 1
  // CK25: [[BP0:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[BP:%.+]], i{{.+}} 0, i{{.+}} 0
  // CK25: [[P0:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* [[P:%.+]], i{{.+}} 0, i{{.+}} 0
  // CK25: [[DIM_1:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 0
  // CK25: [[OFFSET:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_1]], {{.+}} 0, {{.+}} 0
  // CK25: store i64 0, i64* [[OFFSET]],
  // CK25: [[COUNT:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_1]], {{.+}} 0, {{.+}} 1
  // CK25: store i64 2, i64* [[COUNT]],
  // CK25: [[STRIDE:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_1]], {{.+}} 0, {{.+}} 2
  // CK25: store i64 80, i64* [[STRIDE]],
  // CK25: [[DIM_2:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 1
  // CK25: [[OFFSET_2:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_2]], {{.+}} 0, {{.+}} 0
  // CK25: store i64 [[ARG:%.+]], i64* [[OFFSET_2]],
  // CK25: [[COUNT_2:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_2]], {{.+}} 0, {{.+}} 1
  // CK25: store i64 [[LEN]], i64* [[COUNT_2]],
  // CK25: [[STRIDE_2:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_2]], {{.+}} 0, {{.+}} 2
  // CK25: store i64 20, i64* [[STRIDE_2]],
  // CK25: [[DIM_3:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 2
  // CK25: [[OFFSET_3:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_3]], {{.+}} 0, {{.+}} 0
  // CK25: store i64 1, i64* [[OFFSET_3]],
  // CK25: [[COUNT_3:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_3]], {{.+}} 0, {{.+}} 1
  // CK25: store i64 4, i64* [[COUNT_3]],
  // CK25: [[STRIDE_3:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_3]], {{.+}} 0, {{.+}} 2
  // CK25: store i64 4, i64* [[STRIDE_3]],
  // CK25: [[DIM_4:%.+]] = getelementptr inbounds [4 x [[STRUCT_DESCRIPTOR]]], [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]], {{.+}} 0, {{.+}} 3
  // CK25: [[OFFSET_4:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_4]], {{.+}} 0, {{.+}} 0
  // CK25: store i64 0, i64* [[OFFSET_4]],
  // CK25: [[COUNT_4:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_4]], {{.+}} 0, {{.+}} 1
  // CK25: store i64 1, i64* [[COUNT_4]],
  // CK25: [[STRIDE_4:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_4]], {{.+}} 0, {{.+}} 2
  // CK25: store i64 4, i64* [[STRIDE_4]],
  // CK25: [[PC0:%.+]] = bitcast [4 x [[STRUCT_DESCRIPTOR]]]* [[DIMS]] to i8*
  // CK25: [[PTRS:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* %.offload_ptrs, i32 0, i32 0
  // CK25: store i8* [[PC0]], i8** [[PTRS]],
  // CK25: [[DIM_5:%.+]] = getelementptr inbounds [3 x [[STRUCT_DESCRIPTOR]]], [3 x [[STRUCT_DESCRIPTOR]]]* [[DIMS_2]], {{.+}} 0, {{.+}} 0
  // CK25: [[OFFSET_2_1:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_5]], {{.+}} 0, {{.+}} 0
  // CK25: store i64 0, i64* [[OFFSET_2_1]],
  // CK25: [[COUNT_2_1:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_5]], {{.+}} 0, {{.+}} 1
  // CK25: store i64 2, i64* [[COUNT_2_1]],
  // CK25: [[STRIDE_2_1:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_5]], {{.+}} 0, {{.+}} 2
  // CK25: store i64 12, i64* [[STRIDE_2_1]],
  // CK25: [[DIM_6:%.+]] = getelementptr inbounds [3 x [[STRUCT_DESCRIPTOR]]], [3 x [[STRUCT_DESCRIPTOR]]]* [[DIMS_2]], {{.+}} 0, {{.+}} 1
  // CK25: [[OFFSET_2_2:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_6]], {{.+}} 0, {{.+}} 0
  // CK25: store i64 1, i64* [[OFFSET_2_2]],
  // CK25: [[COUNT_2_2:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_6]], {{.+}} 0, {{.+}} 1
  // CK25: store i64 2, i64* [[COUNT_2_2]],
  // CK25: [[STRIDE_2_2:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_6]], {{.+}} 0, {{.+}} 2
  // CK25: store i64 4, i64* [[STRIDE_2_2]],
  // CK25: [[DIM_7:%.+]] = getelementptr inbounds [3 x [[STRUCT_DESCRIPTOR]]], [3 x [[STRUCT_DESCRIPTOR]]]* [[DIMS_2]], {{.+}} 0, {{.+}} 2
  // CK25: [[OFFSET_2_3:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_7]], {{.+}} 0, {{.+}} 0
  // CK25: store i64 0, i64* [[OFFSET_2_3]],
  // CK25: [[COUNT_2_3:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_7]], {{.+}} 0, {{.+}} 1
  // CK25: store i64 1, i64* [[COUNT_2_3]],
  // CK25: [[STRIDE_2_3:%.+]] = getelementptr inbounds [[STRUCT_DESCRIPTOR]], [[STRUCT_DESCRIPTOR]]* [[DIM_7]], {{.+}} 0, {{.+}} 2
  // CK25: store i64 4, i64* [[STRIDE_2_3]],
  // CK25: [[PC1:%.+]] = bitcast [3 x [[STRUCT_DESCRIPTOR]]]* [[DIMS_2]] to i8*
  // CK25: [[PTRS_2:%.+]] = getelementptr inbounds [3 x i8*], [3 x i8*]* %.offload_ptrs, i32 0, i32 2
  // CK25: store i8* [[PC1]], i8** [[PTRS_2]],
  // CK25-DAG: call void @__tgt_target_data_update_mapper(%struct.ident_t* @{{.+}}, i64 -1, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MSIZE]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE]]{{.+}})
  // CK25-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP]]
  // CK25-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

#pragma omp target update to(arr[0:2][arg:][1:4], x, farr[0:2][1:2])
  { ++arg; }
}

#endif
#endif
