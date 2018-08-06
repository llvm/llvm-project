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

// CK1: [[MTYPE03:@.+]] = {{.+}}constant [1 x i64] [i64 34]

// CK1-64: [[SIZE04:@.+]] = {{.+}}constant [2 x i64] [i64 sdiv exact (i64 sub (i64 ptrtoint (double** getelementptr (double*, double** getelementptr inbounds (%struct.ST, %struct.ST* @gb, i32 0, i32 1), i32 1) to i64), i64 ptrtoint (double** getelementptr inbounds (%struct.ST, %struct.ST* @gb, i32 0, i32 1) to i64)), i64 ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)), i64 24]
// CK1-32: [[SIZE04:@.+]] = {{.+}}constant [2 x i32] [i32 trunc (i64 sdiv exact (i64 sub (i64 ptrtoint (double** getelementptr (double*, double** getelementptr inbounds (%struct.ST, %struct.ST* @gb, i32 0, i32 1), i32 1) to i64), i64 ptrtoint (double** getelementptr inbounds (%struct.ST, %struct.ST* @gb, i32 0, i32 1) to i64)), i64 ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)) to i32), i32 24]
// CK1: [[MTYPE04:@.+]] = {{.+}}constant [2 x i64] [i64 32, i64 281474976710673]

// CK1-LABEL: _Z3fooi
void foo(int arg) {
  int la;
  float lb[arg];

  // Region 00
  // CK1-DAG: call void @__tgt_target_data_update_nowait(i64 [[DEV:%[^,]+]], i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}})
  // CK1-DAG: [[DEV]] = sext i32 [[DEVi32:%[^,]+]] to i64
  // CK1-DAG: [[DEVi32]] = load i32, i32* %{{[^,]+}},
  // CK1-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK1-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK1-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK1-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to [100 x double]**
  // CK1-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to [100 x double]**
  // CK1-DAG: store [100 x double]* @gc, [100 x double]** [[BPC0]]
  // CK1-DAG: store [100 x double]* @gc, [100 x double]** [[PC0]]

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target update if(1+3-5) device(arg) from(gc) nowait
  {++arg;}

  // Region 01
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  #pragma omp target update to(la) if(1+3-4)
  {++arg;}

  // Region 02
  // CK1: br i1 %{{[^,]+}}, label %[[IFTHEN:[^,]+]], label %[[IFELSE:[^,]+]]
  // CK1: [[IFTHEN]]
  // CK1-DAG: call void @__tgt_target_data_update(i64 4, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}})
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
  // CK1-DAG: call void @__tgt_target_data_update(i64 -1, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}})
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
  // CK1-DAG: store i[[sz]] [[CSVAL0:%[^,]+]], i[[sz]]* [[S0]]
  // CK1-DAG: [[CSVAL0]] = mul nuw i[[sz]] %{{[^,]+}}, 4
  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  // CK1-NOT: __tgt_target_data_end
  #pragma omp target update from(lb)
  {++arg;}

  // CK1: %{{.+}} = add nsw i32 %{{[^,]+}}, 1
  {++arg;}

  // Region 04
  // CK1-DAG: call void @__tgt_target_data_update(i64 -1, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE04]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE04]]{{.+}})
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
// CK2-DAG: call void @__tgt_target_data_update(i64 [[DEV:%[^,]+]], i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[sz:64|32]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE00]]{{.+}})
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
#endif
