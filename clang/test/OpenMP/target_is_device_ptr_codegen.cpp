// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32
// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32

// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY0 %s
// SIMD-ONLY0-NOT: {{__kmpc|__tgt}}
#ifdef CK1

double *g;

// CK1: @g ={{.*}} global double*
// CK1: [[SIZES00:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} {{8|4}}]
// CK1: [[TYPES00:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK1: [[SIZES01:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] {{8|4}}]
// CK1: [[TYPES01:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK1: [[SIZES02:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] {{8|4}}]
// CK1: [[TYPES02:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK1: [[SIZES03:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] {{8|4}}]
// CK1: [[TYPES03:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK1: [[SIZES04:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] {{8|4}}]
// CK1: [[TYPES04:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK1: [[SIZES05:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] {{8|4}}]
// CK1: [[TYPES05:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK1: [[SIZES06:@.+]] = {{.+}}constant [2 x i[[sz]]] [i[[sz]] {{8|4}}, i[[sz]] {{8|4}}]
// CK1: [[TYPES06:@.+]] = {{.+}}constant [2 x i64] [i64 288, i64 288]

// CK1-LABEL: @_Z3foo{{.*}}(
template<typename T>
void foo(float *&lr, T *&tr) {
  float *l;
  T *t;

  // CK1-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES00]]{{.+}}, {{.+}}[[TYPES00]]{{.+}}, i8** null, i8** null)
  // CK1-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK1-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double**
  // CK1-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double**
  // CK1-DAG: store double* [[VAL:%.+]], double** [[CBP1]]
  // CK1-DAG: store double* [[VAL]], double** [[CP1]]
  // CK1-DAG: [[VAL]] = load double*, double** [[ADDR:@g]],

  // CK1: call void [[KERNEL:@.+]](double* [[VAL]])
  #pragma omp target is_device_ptr(g)
  {
    ++g;
  }

  // CK1-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES01]]{{.+}}, {{.+}}[[TYPES01]]{{.+}}, i8** null, i8** null)
  // CK1-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK1-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to float**
  // CK1-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to float**
  // CK1-DAG: store float* [[VAL:%.+]], float** [[CBP1]]
  // CK1-DAG: store float* [[VAL]], float** [[CP1]]
  // CK1-DAG: [[VAL]] = load float*, float** [[ADDR:%.+]],

  // CK1: call void [[KERNEL:@.+]](float* [[VAL]])
  #pragma omp target is_device_ptr(l)
  {
    ++l;
  }

  // CK1-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES02]]{{.+}}, {{.+}}[[TYPES02]]{{.+}}, i8** null, i8** null)
  // CK1-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK1-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK1-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK1-DAG: store i32* [[VAL:%.+]], i32** [[CBP1]]
  // CK1-DAG: store i32* [[VAL]], i32** [[CP1]]
  // CK1-DAG: [[VAL]] = load i32*, i32** [[ADDR:%.+]],

  // CK1: call void [[KERNEL:@.+]](i32* [[VAL]])
  #pragma omp target is_device_ptr(t)
  {
    ++t;
  }

  // CK1-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES03]]{{.+}}, {{.+}}[[TYPES03]]{{.+}}, i8** null, i8** null)
  // CK1-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK1-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to float**
  // CK1-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to float**
  // CK1-DAG: store float* [[VAL:%.+]], float** [[CBP1]]
  // CK1-DAG: store float* [[VAL]], float** [[CP1]]
  // CK1-DAG: [[VAL]] = load float*, float** [[ADDR:%.+]],
  // CK1-DAG: [[ADDR]] = load float**, float*** [[ADDR2:%.+]],

  // CK1: call void [[KERNEL:@.+]](float* [[VAL]])
  #pragma omp target is_device_ptr(lr)
  {
    ++lr;
  }

  // CK1-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES04]]{{.+}}, {{.+}}[[TYPES04]]{{.+}}, i8** null, i8** null)
  // CK1-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK1-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK1-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK1-DAG: store i32* [[VAL:%.+]], i32** [[CBP1]]
  // CK1-DAG: store i32* [[VAL]], i32** [[CP1]]
  // CK1-DAG: [[VAL]] = load i32*, i32** [[ADDR:%.+]],
  // CK1-DAG: [[ADDR]] = load i32**, i32*** [[ADDR2:%.+]],

  // CK1: call void [[KERNEL:@.+]](i32* [[VAL]])
  #pragma omp target is_device_ptr(tr)
  {
    ++tr;
  }

  // CK1-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES05]]{{.+}}, {{.+}}[[TYPES05]]{{.+}}, i8** null, i8** null)
  // CK1-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK1-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK1-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK1-DAG: store i32* [[VAL:%.+]], i32** [[CBP1]]
  // CK1-DAG: store i32* [[VAL]], i32** [[CP1]]
  // CK1-DAG: [[VAL]] = load i32*, i32** [[ADDR:%.+]],
  // CK1-DAG: [[ADDR]] = load i32**, i32*** [[ADDR2:%.+]],

  // CK1: call void [[KERNEL:@.+]](i32* [[VAL]])
  #pragma omp target is_device_ptr(tr,lr)
  {
    ++tr;
  }

  // CK1-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 2, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES06]]{{.+}}, {{.+}}[[TYPES06]]{{.+}}, i8** null, i8** null)
  // CK1-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK1-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to i32**
  // CK1-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to i32**
  // CK1-DAG: store i32* [[VAL:%.+]], i32** [[CBP1]]
  // CK1-DAG: store i32* [[VAL]], i32** [[CP1]]
  // CK1-DAG: [[VAL]] = load i32*, i32** [[ADDR:%.+]],
  // CK1-DAG: [[ADDR]] = load i32**, i32*** [[ADDR2:%.+]],

  // CK1-DAG: [[_BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 1
  // CK1-DAG: [[_P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 1
  // CK1-DAG: [[_CBP1:%.+]] = bitcast i8** [[_BP1]] to float**
  // CK1-DAG: [[_CP1:%.+]] = bitcast i8** [[_P1]] to float**
  // CK1-DAG: store float* [[_VAL:%.+]], float** [[_CBP1]]
  // CK1-DAG: store float* [[_VAL]], float** [[_CP1]]
  // CK1-DAG: [[_VAL]] = load float*, float** [[_ADDR:%.+]],
  // CK1-DAG: [[_ADDR]] = load float**, float*** [[_ADDR2:%.+]],

  // CK1: call void [[KERNEL:@.+]](i32* [[VAL]], float* [[_VAL]])
  #pragma omp target is_device_ptr(tr,lr)
  {
    ++tr,++lr;
  }
}

void bar(float *&a, int *&b) {
  foo<int>(a,b);
}

#endif
///==========================================================================///
// RUN: %clang_cc1 -no-opaque-pointers -DCK2 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK2 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK2 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-32
// RUN: %clang_cc1 -no-opaque-pointers -DCK2 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-32

// RUN: %clang_cc1 -no-opaque-pointers -DCK2 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK2 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK2 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK2 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
#ifdef CK2

// CK2: [[ST:%.+]] = type { double*, double** }

// CK2-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l244.region_id = weak{{.*}} constant i8 0

// CK2: [[SIZE00:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} {{8|4}}]
// CK2: [[MTYPE00:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK2-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l259.region_id = weak{{.*}} constant i8 0

// CK2: [[SIZE01:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] {{8|4}}]
// CK2: [[MTYPE01:@.+]] = {{.+}}constant [1 x i64] [i64 288]

// CK2-LABEL: @.__omp_offloading_{{.*}}foo{{.*}}_l274.region_id = weak{{.*}} constant i8 0

// CK2: [[SIZE02:@.+]] = {{.+}}constant [1 x i[[sz]]] [i[[sz]] {{8|4}}]
// CK2: [[MTYPE02:@.+]] = {{.+}}constant [1 x i64] [i64 288]

template <typename T>
struct ST {
  T *a;
  double *&b;
  ST(double *&b) : a(0), b(b) {}

  // CK2-LABEL: @{{.*}}foo{{.*}}
  void foo(double *&arg) {
    int *la = 0;

    // CK2-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}}, i8** null, i8** null)
    // CK2-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK2-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK2-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK2-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK2-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
    // CK2-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[ST]]**
    // CK2-DAG: store [[ST]]* [[VAR0:%.+]], [[ST]]** [[CBP0]]
    // CK2-DAG: store [[ST]]* [[VAR0]], [[ST]]** [[CP0]]
    #pragma omp target is_device_ptr(a)
    {
      a++;
    }

    // CK2-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}}, i8** null, i8** null)
    // CK2-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK2-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK2-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK2-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK2-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
    // CK2-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[ST]]**
    // CK2-DAG: store [[ST]]* [[VAR0:%.+]], [[ST]]** [[CBP0]]
    // CK2-DAG: store [[ST]]* [[VAR0]], [[ST]]** [[CP0]]
    #pragma omp target is_device_ptr(b)
    {
      b++;
    }

    // CK2-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}}, i8** null, i8** null)
    // CK2-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK2-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK2-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK2-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK2-DAG: [[CBP0:%.+]] = bitcast i8** [[BP0]] to [[ST]]**
    // CK2-DAG: [[CP0:%.+]] = bitcast i8** [[P0]] to [[ST]]**
    // CK2-DAG: store [[ST]]* [[VAR0:%.+]], [[ST]]** [[CBP0]]
    // CK2-DAG: store [[ST]]* [[VAR0]], [[ST]]** [[CP0]]
    #pragma omp target is_device_ptr(a, b)
    {
      a++;
      b++;
    }
  }
};

void bar(double *arg){
  ST<double> A(arg);
  A.foo(arg);
  ++arg;
}
#endif
///==========================================================================///
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-32
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-32

// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK3 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY1 %s
// SIMD-ONLY1-NOT: {{__kmpc|__tgt}}
#ifdef CK3

// CK3-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[SZ:64|32]]] [i{{64|32}} {{8|4}}]
// OMP_MAP_TARGET_PARAM = 0x20 | OMP_MAP_TO = 0x1 = 0x21
// CK3-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i64] [i64 [[#0x21]]]
void bar() {
  __attribute__((aligned(64))) double *ptr;
  // CK3-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}}, i8** null, i8** null)
  // CK3-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK3-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK3-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK3-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK3-DAG: [[CBP1:%.+]] = bitcast i8** [[BP1]] to double***
  // CK3-DAG: [[CP1:%.+]] = bitcast i8** [[P1]] to double***
  // CK3-DAG: store double** [[PTR:%.+]], double*** [[CBP1]]
  // CK3-DAG: store double** [[PTR]], double*** [[CP1]]

  // CK3: call void [[KERNEL:@.+]](double** [[PTR]])
#pragma omp target is_device_ptr(ptr)
  *ptr = 0;
}
#endif
#endif
