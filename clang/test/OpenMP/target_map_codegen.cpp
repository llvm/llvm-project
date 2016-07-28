// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///
/// Implicit maps.
///

///==========================================================================///
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32
// RUN: %clang_cc1 -DCK1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32
#ifdef CK1

// CK1-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_IS_FIRST = 288
// CK1-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 288]

// CK1-LABEL: implicit_maps_integer
void implicit_maps_integer (int a){
  int i = a;

  // CK1-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
  // CK1-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK1-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK1-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK1-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK1-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK1-DAG: [[VALBP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK1-DAG: [[VALP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
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
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-32
// RUN: %clang_cc1 -DCK2 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-32
#ifdef CK2

// CK2: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_IS_FIRST = 288
// CK2: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 288]
// CK2: [[SIZES2:@.+]] = {{.+}}constant [1 x i[[sz]]] zeroinitializer
// Map types: OMP_MAP_IS_PTR = 32
// CK2: [[TYPES2:@.+]] = {{.+}}constant [1 x i32] [i32 32]

// CK2-LABEL: implicit_maps_reference
void implicit_maps_reference (int a, int *b){
  int &i = a;
  // CK2-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
  // CK2-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK2-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK2-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK2-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK2-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK2-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK2-DAG: [[VALBP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK2-DAG: [[VALP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK2-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK2-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to i32*
  // CK2-64-DAG: store i32 {{.+}}, i32* [[CADDR]],

  // CK2: call void [[KERNEL:@.+]](i[[sz]] [[VAL]])
  #pragma omp target
  {
   ++i;
  }

  int *&p = b;
  // CK2-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES2]]{{.+}}, {{.+}}[[TYPES2]]{{.+}})
  // CK2-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK2-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK2-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK2-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK2-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK2-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK2-DAG: [[VALBP]] = bitcast i32* [[VAL:%.+]] to i8*
  // CK2-DAG: [[VALP]] = bitcast i32* [[VAL]] to i8*
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
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-32
// RUN: %clang_cc1 -DCK3 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-32
#ifdef CK3

// CK3-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_IS_FIRST = 288
// CK3-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 288]

// CK3-LABEL: implicit_maps_parameter
void implicit_maps_parameter (int a){

  // CK3-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
  // CK3-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK3-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK3-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK3-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK3-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK3-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK3-DAG: [[VALBP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK3-DAG: [[VALP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
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
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK4 --check-prefix CK4-32
// RUN: %clang_cc1 -DCK4 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK4 --check-prefix CK4-32
#ifdef CK4

// CK4-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_IS_FIRST = 288
// CK4-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 288]

// CK4-LABEL: implicit_maps_nested_integer
void implicit_maps_nested_integer (int a){
  int i = a;

  // The captures in parallel are by reference. Only the capture in target is by
  // copy.

  // CK4: call void {{.+}}@__kmpc_fork_call({{.+}} [[KERNELP1:@.+]] to void (i32*, i32*, ...)*), i32* {{.+}})
  // CK4: define internal void [[KERNELP1]](i32* {{[^,]+}}, i32* {{[^,]+}}, i32* {{[^,]+}})
  #pragma omp parallel
  {
    // CK4-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
    // CK4-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
    // CK4-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
    // CK4-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
    // CK4-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
    // CK4-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
    // CK4-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
    // CK4-DAG: [[VALBP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
    // CK4-DAG: [[VALP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
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
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK5 --check-prefix CK5-32
// RUN: %clang_cc1 -DCK5 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK5 --check-prefix CK5-32
#ifdef CK5

// CK5-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_IS_FIRST = 288
// CK5-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 288]

// CK5-LABEL: implicit_maps_nested_integer_and_enum
void implicit_maps_nested_integer_and_enum (int a){
  enum Bla {
    SomeEnum = 0x09
  };

  // Using an enum should not change the mapping information.
  int  i = a;

  // CK5-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
  // CK5-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK5-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK5-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK5-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK5-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK5-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK5-DAG: [[VALBP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK5-DAG: [[VALP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
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
// RUN: %clang_cc1 -DCK6 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK6 --check-prefix CK6-64
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK6 --check-prefix CK6-64
// RUN: %clang_cc1 -DCK6 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK6 --check-prefix CK6-32
// RUN: %clang_cc1 -DCK6 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK6 --check-prefix CK6-32
#ifdef CK6
// CK6-DAG: [[GBL:@Gi]] = global i32 0
// CK6-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_IS_FIRST = 288
// CK6-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 288]

// CK6-LABEL: implicit_maps_host_global
int Gi;
void implicit_maps_host_global (int a){
  // CK6-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
  // CK6-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK6-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK6-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK6-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK6-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK6-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK6-DAG: [[VALBP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK6-DAG: [[VALP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
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
// RUN: %clang_cc1 -DCK7 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK7 --check-prefix CK7-64
// RUN: %clang_cc1 -DCK7 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK7  --check-prefix CK7-64
// RUN: %clang_cc1 -DCK7 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK7  --check-prefix CK7-32
// RUN: %clang_cc1 -DCK7 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK7  --check-prefix CK7-32
#ifdef CK7

// For a 32-bit targets, the value doesn't fit the size of the pointer,
// therefore it is passed by reference with a map 'to' specification.

// CK7-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 8]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_IS_FIRST = 288
// CK7-64-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 288]
// Map types: OMP_MAP_TO  | OMP_MAP_IS_FIRST = 33
// CK7-32-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 33]

// CK7-LABEL: implicit_maps_double
void implicit_maps_double (int a){
  double d = (double)a;

  // CK7-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
  // CK7-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK7-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK7-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK7-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0

  // CK7-64-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK7-64-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK7-64-DAG: [[VALBP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK7-64-DAG: [[VALP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK7-64-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK7-64-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to double*
  // CK7-64-64-DAG: store double {{.+}}, double* [[CADDR]],

  // CK7-32-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK7-32-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK7-32-DAG: [[VALBP]] = bitcast double* [[DECL:%.+]] to i8*
  // CK7-32-DAG: [[VALP]] = bitcast double* [[DECL]] to i8*

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
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK8
#ifdef CK8

// CK8-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// Map types: OMP_MAP_PRIVATE_VAL | OMP_MAP_IS_FIRST = 288
// CK8-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 288]

// CK8-LABEL: implicit_maps_float
void implicit_maps_float (int a){
  float f = (float)a;

  // CK8-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
  // CK8-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK8-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK8-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK8-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK8-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK8-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK8-DAG: [[VALBP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK8-DAG: [[VALP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
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
// RUN: %clang_cc1 -DCK9 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK9
// RUN: %clang_cc1 -DCK9 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK9
// RUN: %clang_cc1 -DCK9 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK9
// RUN: %clang_cc1 -DCK9 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK9
#ifdef CK9

// CK9-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 16]
// Map types: OMP_MAP_TO + OMP_MAP_FROM + OMP_MAP_IS_FIRST = 35
// CK9-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 35]

// CK9-LABEL: implicit_maps_array
void implicit_maps_array (int a){
  double darr[2] = {(double)a, (double)a};

  // CK9-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
  // CK9-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK9-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK9-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK9-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK9-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK9-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK9-DAG: [[VALBP]] = bitcast [2 x double]* [[DECL:%.+]] to i8*
  // CK9-DAG: [[VALP]] = bitcast [2 x double]* [[DECL]] to i8*

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
// CK9: {{.+}} = getelementptr inbounds [2 x double], [2 x double]* [[REF]], i[[sz]] 0, i[[sz]] 0
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK10 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK10
#ifdef CK10

// CK10-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] zeroinitializer
// Map types: OMP_MAP_IS_FIRST = 32
// CK10-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 32]

// CK10-LABEL: implicit_maps_pointer
void implicit_maps_pointer (){
  double *ddyn;

  // CK10-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
  // CK10-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK10-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK10-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK10-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK10-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK10-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK10-DAG: [[VALBP]] = bitcast double* [[PTR:%.+]] to i8*
  // CK10-DAG: [[VALP]] = bitcast double* [[PTR]] to i8*

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
// CK10: {{.+}} = getelementptr inbounds double, double* [[REF]], i[[sz]] 0

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK11 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK11
// RUN: %clang_cc1 -DCK11 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK11
// RUN: %clang_cc1 -DCK11 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK11
// RUN: %clang_cc1 -DCK11 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK11
#ifdef CK11

// CK11-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 16]
// Map types: OMP_MAP_TO + OMP_MAP_IS_FIRST = 33
// CK11-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 33]

// CK11-LABEL: implicit_maps_double_complex
void implicit_maps_double_complex (int a){
  double _Complex dc = (double)a;

  // CK11-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
  // CK11-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK11-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK11-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK11-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK11-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK11-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK11-DAG: [[VALBP]] = bitcast { double, double }* [[PTR:%.+]] to i8*
  // CK11-DAG: [[VALP]] = bitcast { double, double }* [[PTR]] to i8*

  // CK11: call void [[KERNEL:@.+]]({ double, double }* [[PTR]])
  #pragma omp target
  {
   dc *= dc;
  }
}

// CK11: define internal void [[KERNEL]]({ double, double }* {{.*}}[[ARG:%.+]])
// CK11: [[ADDR:%.+]] = alloca { double, double }*,
// CK11: store { double, double }* [[ARG]], { double, double }** [[ADDR]],
// CK11: [[REF:%.+]] = load { double, double }*, { double, double }** [[ADDR]],
// CK11: {{.+}} = getelementptr inbounds { double, double }, { double, double }* [[REF]], i32 0, i32 0
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK12 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK12 --check-prefix CK12-64
// RUN: %clang_cc1 -DCK12 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK12 --check-prefix CK12-64
// RUN: %clang_cc1 -DCK12 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK12 --check-prefix CK12-32
// RUN: %clang_cc1 -DCK12 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK12 --check-prefix CK12-32
#ifdef CK12

// For a 32-bit targets, the value doesn't fit the size of the pointer,
// therefore it is passed by reference with a map 'to' specification.

// CK12-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 8]
// Map types: OMP_MAP_PRIVATE_VAL + OMP_MAP_IS_FIRST = 288
// CK12-64-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 288]
// Map types: OMP_MAP_TO + OMP_MAP_IS_FIRST = 33
// CK12-32-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 33]

// CK12-LABEL: implicit_maps_float_complex
void implicit_maps_float_complex (int a){
  float _Complex fc = (float)a;

  // CK12-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
  // CK12-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK12-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK12-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK12-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0

  // CK12-64-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK12-64-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK12-64-DAG: [[VALBP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK12-64-DAG: [[VALP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK12-64-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK12-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to { float, float }*
  // CK12-64-DAG: store { float, float } {{.+}}, { float, float }* [[CADDR]],

  // CK12-32-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK12-32-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK12-32-DAG: [[VALBP]] = bitcast { float, float }* [[DECL:%.+]] to i8*
  // CK12-32-DAG: [[VALP]] = bitcast { float, float }* [[DECL]] to i8*

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
// RUN: %clang_cc1 -DCK13 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK13
#ifdef CK13

// We don't have a constant map size for VLAs.
// Map types:
//  - OMP_MAP_PRIVATE_VAL + OMP_MAP_IS_FIRST = 288 (vla size)
//  - OMP_MAP_PRIVATE_VAL + OMP_MAP_IS_FIRST = 288 (vla size)
//  - OMP_MAP_TO + OMP_MAP_FROM + OMP_MAP_IS_FIRST = 35
// CK13-DAG: [[TYPES:@.+]] = {{.+}}constant [3 x i32] [i32 288, i32 288, i32 35]

// CK13-LABEL: implicit_maps_variable_length_array
void implicit_maps_variable_length_array (int a){
  double vla[2][a];

  // CK13-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 3, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], i[[sz:64|32]]* [[SGEP:%[^,]+]], {{.+}}[[TYPES]]{{.+}})
  // CK13-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK13-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK13-DAG: [[SGEP]] = getelementptr inbounds {{.+}}[[SS:%[^,]+]], i32 0, i32 0

  // CK13-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK13-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK13-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[SS]], i32 0, i32 0
  // CK13-DAG: store i8* inttoptr (i[[sz]] 2 to i8*), i8** [[BP0]],
  // CK13-DAG: store i8* inttoptr (i[[sz]] 2 to i8*), i8** [[P0]],
  // CK13-DAG: store i[[sz]] {{8|4}}, i[[sz]]* [[S0]],

  // CK13-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 1
  // CK13-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 1
  // CK13-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[SS]], i32 0, i32 1
  // CK13-DAG: store i8* [[VALBP1:%.+]], i8** [[BP1]],
  // CK13-DAG: store i8* [[VALP1:%.+]], i8** [[P1]],
  // CK13-DAG: [[VALBP1]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK13-DAG: [[VALP1]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK13-DAG: store i[[sz]] {{8|4}}, i[[sz]]* [[S1]],

  // CK13-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 2
  // CK13-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 2
  // CK13-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[SS]], i32 0, i32 2
  // CK13-DAG: store i8* [[VALBP2:%.+]], i8** [[BP2]],
  // CK13-DAG: store i8* [[VALP2:%.+]], i8** [[P2]],
  // CK13-DAG: store i[[sz]] [[VALS2:%.+]], i[[sz]]* [[S2]],
  // CK13-DAG: [[VALBP2]] = bitcast double* [[DECL:%.+]] to i8*
  // CK13-DAG: [[VALP2]] = bitcast double* [[DECL]] to i8*
  // CK13-DAG: [[VALS2]] = mul nuw i[[sz]] %{{.+}}, 8

  // CK13: call void [[KERNEL:@.+]](i[[sz]] {{.+}}, i[[sz]] {{.+}}, double* [[DECL]])
  #pragma omp target
  {
    vla[1][3] += 1.0;
  }
}

// CK13: define internal void [[KERNEL]](i[[sz]] [[VLA0:%.+]], i[[sz]] [[VLA1:%.+]], double* {{.+}}[[ARG:%.+]])
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
// RUN: %clang_cc1 -DCK14 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK14 --check-prefix CK14-64
// RUN: %clang_cc1 -DCK14 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK14 --check-prefix CK14-64
// RUN: %clang_cc1 -DCK14 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK14 --check-prefix CK14-32
// RUN: %clang_cc1 -DCK14 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK14 --check-prefix CK14-32
#ifdef CK14

// CK14-DAG: [[ST:%.+]] = type { i32, double }
// CK14-DAG: [[SIZES:@.+]] = {{.+}}constant [2 x i[[sz:64|32]]] [i{{64|32}} {{16|12}}, i{{64|32}} 4]
// Map types:
// - OMP_MAP_TO + OMP_MAP_FROM + OMP_MAP_IS_FIRST = 35
// - OMP_MAP_PRIVATE_VAL + OMP_MAP_IS_FIRST = 288
// CK14-DAG: [[TYPES:@.+]] = {{.+}}constant [2 x i32] [i32 35, i32 288]

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

// CK14-LABEL: implicit_maps_class
void implicit_maps_class (int a){
  SSS sss(a, (double)a);

  // CK14: define {{.*}}void @{{.+}}foo{{.+}}([[ST]]* {{[^,]+}}, i32 {{[^,]+}})
  // CK14-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 2, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
  // CK14-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK14-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0

  // CK14-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK14-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK14-DAG: store i8* [[VALBP0:%.+]], i8** [[BP0]],
  // CK14-DAG: store i8* [[VALP0:%.+]], i8** [[P0]],
  // CK14-DAG: [[VALBP0]] = bitcast [[ST]]* [[DECL:%.+]] to i8*
  // CK14-DAG: [[VALP0]] = bitcast [[ST]]* [[DECL]] to i8*

  // CK14-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 1
  // CK14-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 1
  // CK14-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK14-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK14-DAG: [[VALBP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK14-DAG: [[VALP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
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
// RUN: %clang_cc1 -DCK15 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK15 --check-prefix CK15-64
// RUN: %clang_cc1 -DCK15 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK15 --check-prefix CK15-64
// RUN: %clang_cc1 -DCK15 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK15 --check-prefix CK15-32
// RUN: %clang_cc1 -DCK15 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK15 --check-prefix CK15-32
#ifdef CK15

// CK15: [[ST:%.+]] = type { i32, double, i32* }
// CK15: [[SIZES:@.+]] = {{.+}}constant [2 x i[[sz:64|32]]] [i{{64|32}} {{24|16}}, i{{64|32}} 4]
// Map types:
// - OMP_MAP_TO + OMP_MAP_FROM + OMP_MAP_IS_FIRST = 35
// - OMP_MAP_PRIVATE_VAL + OMP_MAP_IS_FIRST = 288
// CK15: [[TYPES:@.+]] = {{.+}}constant [2 x i32] [i32 35, i32 288]

// CK15: [[SIZES2:@.+]] = {{.+}}constant [2 x i[[sz]]] [i{{64|32}} {{24|16}}, i{{64|32}} 4]
// Map types:
// - OMP_MAP_TO + OMP_MAP_FROM + OMP_MAP_IS_FIRST = 35
// - OMP_MAP_PRIVATE_VAL + OMP_MAP_IS_FIRST = 288
// CK15: [[TYPES2:@.+]] = {{.+}}constant [2 x i32] [i32 35, i32 288]

template<int x>
class SSST {
public:
  int a;
  double b;
  int &r;

  void foo(int c) {
    #pragma omp target
    {
      a += c + x;
      b += (double)(c + x);
      r += x;
    }
  }
  template<int y>
  void bar(int c) {
    #pragma omp target
    {
      a += c + x + y;
      b += (double)(c + x + y);
      r += x + y;
    }
  }

  SSST(int a, double b, int &r) : a(a), b(b), r(r) {}
};

// CK15-LABEL: implicit_maps_templated_class
void implicit_maps_templated_class (int a){
  SSST<123> ssst(a, (double)a, a);

  // CK15: define {{.*}}void @{{.+}}foo{{.+}}([[ST]]* {{[^,]+}}, i32 {{[^,]+}})
  // CK15-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 2, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
  // CK15-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK15-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0

  // CK15-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK15-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK15-DAG: store i8* [[VALBP0:%.+]], i8** [[BP0]],
  // CK15-DAG: store i8* [[VALP0:%.+]], i8** [[P0]],
  // CK15-DAG: [[VALBP0]] = bitcast [[ST]]* [[DECL:%.+]] to i8*
  // CK15-DAG: [[VALP0]] = bitcast [[ST]]* [[DECL]] to i8*

  // CK15-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 1
  // CK15-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 1
  // CK15-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK15-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK15-DAG: [[VALBP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK15-DAG: [[VALP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK15-DAG: [[VAL]] = load i[[sz]], i[[sz]]* [[ADDR:%.+]],
  // CK15-64-DAG: [[CADDR:%.+]] = bitcast i[[sz]]* [[ADDR]] to i32*
  // CK15-64-DAG: store i32 {{.+}}, i32* [[CADDR]],

  // CK15: call void [[KERNEL:@.+]]([[ST]]* [[DECL]], i[[sz]] {{.+}})
  ssst.foo(456);

  // CK15: define {{.*}}void @{{.+}}bar{{.+}}([[ST]]* {{[^,]+}}, i32 {{[^,]+}})
  // CK15-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 2, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES2]]{{.+}}, {{.+}}[[TYPES2]]{{.+}})
  // CK15-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK15-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0

  // CK15-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK15-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK15-DAG: store i8* [[VALBP0:%.+]], i8** [[BP0]],
  // CK15-DAG: store i8* [[VALP0:%.+]], i8** [[P0]],
  // CK15-DAG: [[VALBP0]] = bitcast [[ST]]* [[DECL:%.+]] to i8*
  // CK15-DAG: [[VALP0]] = bitcast [[ST]]* [[DECL]] to i8*

  // CK15-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 1
  // CK15-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 1
  // CK15-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK15-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK15-DAG: [[VALBP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK15-DAG: [[VALP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
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
// RUN: %clang_cc1 -DCK16 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK16 --check-prefix CK16-64
// RUN: %clang_cc1 -DCK16 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK16 --check-prefix CK16-64
// RUN: %clang_cc1 -DCK16 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK16 --check-prefix CK16-32
// RUN: %clang_cc1 -DCK16 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK16 --check-prefix CK16-32
#ifdef CK16

// CK16-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// Map types:
// - OMP_MAP_PRIVATE_VAL + OMP_MAP_IS_FIRST = 288
// CK16-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 288]

template<int y>
int foo(int d) {
  int res = d;
  #pragma omp target
  {
    res += y;
  }
  return res;
}
// CK16-LABEL: implicit_maps_templated_function
void implicit_maps_templated_function (int a){
  int i = a;

  // CK16: define {{.*}}i32 @{{.+}}foo{{.+}}(i32 {{[^,]+}})
  // CK16-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
  // CK16-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK16-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0

  // CK16-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK16-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK16-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK16-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK16-DAG: [[VALBP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK16-DAG: [[VALP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
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
// RUN: %clang_cc1 -DCK17 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK17
// RUN: %clang_cc1 -DCK17 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK17
// RUN: %clang_cc1 -DCK17 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK17
// RUN: %clang_cc1 -DCK17 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK17
#ifdef CK17

// CK17-DAG: [[ST:%.+]] = type { i32, double }
// CK17-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} {{16|12}}]
// Map types: OMP_MAP_TO + OMP_MAP_FROM + OMP_MAP_IS_FIRST = 35
// CK17-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 35]

class SSS {
public:
  int a;
  double b;
};

// CK17-LABEL: implicit_maps_struct
void implicit_maps_struct (int a){
  SSS s = {a, (double)a};

  // CK17-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
  // CK17-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK17-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK17-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK17-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK17-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK17-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK17-DAG: [[VALBP]] = bitcast [[ST]]* [[DECL:%.+]] to i8*
  // CK17-DAG: [[VALP]] = bitcast [[ST]]* [[DECL]] to i8*

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
// RUN: %clang_cc1 -DCK18 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK18 --check-prefix CK18-64
// RUN: %clang_cc1 -DCK18 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK18 --check-prefix CK18-64
// RUN: %clang_cc1 -DCK18 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK18 --check-prefix CK18-32
// RUN: %clang_cc1 -DCK18 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK18 --check-prefix CK18-32
#ifdef CK18

// CK18-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// Map types:
// - OMP_MAP_PRIVATE_VAL + OMP_MAP_IS_FIRST = 288
// CK18-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 288]

template<typename T>
int foo(T d) {
  #pragma omp target
  {
    d += (T)1;
  }
  return d;
}
// CK18-LABEL: implicit_maps_template_type_capture
void implicit_maps_template_type_capture (int a){
  int i = a;

  // CK18: define {{.*}}i32 @{{.+}}foo{{.+}}(i32 {{[^,]+}})
  // CK18-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZES]]{{.+}}, {{.+}}[[TYPES]]{{.+}})
  // CK18-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK18-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0

  // CK18-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK18-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK18-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK18-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK18-DAG: [[VALBP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
  // CK18-DAG: [[VALP]] = inttoptr i[[sz]] [[VAL:%.+]] to i8*
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
// RUN: %clang_cc1 -DCK19 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK19 --check-prefix CK19-64
// RUN: %clang_cc1 -DCK19 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK19 --check-prefix CK19-64
// RUN: %clang_cc1 -DCK19 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK19 --check-prefix CK19-32
// RUN: %clang_cc1 -DCK19 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK19 --check-prefix CK19-32
#ifdef CK19

// CK19: [[SIZE00:@.+]] = private {{.*}}constant [1 x i[[Z:64|32]]] [i[[Z:64|32]] 4]
// CK19: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i32] [i32 32]

// CK19: [[SIZE01:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 400]
// CK19: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i32] [i32 33]

// CK19: [[SIZE02:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 240]
// CK19: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i32] [i32 34]

// CK19: [[SIZE03:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 240]
// CK19: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK19: [[SIZE04:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 400]
// CK19: [[MTYPE04:@.+]] = private {{.*}}constant [1 x i32] [i32 32]

// CK19: [[SIZE05:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 4]
// CK19: [[MTYPE05:@.+]] = private {{.*}}constant [1 x i32] [i32 33]

// CK19: [[MTYPE06:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK19: [[MTYPE07:@.+]] = private {{.*}}constant [1 x i32] [i32 32]

// CK19: [[SIZE08:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 4]
// CK19: [[MTYPE08:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK19: [[SIZE09:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] {{8|4}}]
// CK19: [[MTYPE09:@.+]] = private {{.*}}constant [1 x i32] [i32 34]

// CK19: [[SIZE10:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 240]
// CK19: [[MTYPE10:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK19: [[SIZE11:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 240]
// CK19: [[MTYPE11:@.+]] = private {{.*}}constant [1 x i32] [i32 32]

// CK19: [[SIZE12:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 4]
// CK19: [[MTYPE12:@.+]] = private {{.*}}constant [1 x i32] [i32 33]

// CK19: [[MTYPE13:@.+]] = private {{.*}}constant [1 x i32] [i32 32]

// CK19: [[MTYPE14:@.+]] = private {{.*}}constant [1 x i32] [i32 33]

// CK19: [[SIZE15:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 4]
// CK19: [[MTYPE15:@.+]] = private {{.*}}constant [1 x i32] [i32 34]

// CK19: [[MTYPE16:@.+]] = private {{.*}}constant [2 x i32] [i32 288, i32 33]

// CK19: [[SIZE17:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] 240]
// CK19: [[MTYPE17:@.+]] = private {{.*}}constant [2 x i32] [i32 288, i32 34]

// CK19: [[SIZE18:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] 240]
// CK19: [[MTYPE18:@.+]] = private {{.*}}constant [2 x i32] [i32 288, i32 35]

// CK19: [[MTYPE19:@.+]] = private {{.*}}constant [2 x i32] [i32 288, i32 32]

// CK19: [[SIZE20:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] 4]
// CK19: [[MTYPE20:@.+]] = private {{.*}}constant [2 x i32] [i32 288, i32 33]

// CK19: [[MTYPE21:@.+]] = private {{.*}}constant [2 x i32] [i32 288, i32 35]

// CK19: [[SIZE22:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] 4]
// CK19: [[MTYPE22:@.+]] = private {{.*}}constant [2 x i32] [i32 288, i32 35]

// CK19: [[SIZE23:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 4]
// CK19: [[MTYPE23:@.+]] = private {{.*}}constant [1 x i32] [i32 39]

// CK19: [[SIZE24:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 480]
// CK19: [[MTYPE24:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK19: [[SIZE25:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 16]
// CK19: [[MTYPE25:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK19: [[SIZE26:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 24]
// CK19: [[MTYPE26:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK19: [[SIZE27:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 4]
// CK19: [[MTYPE27:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK19: [[SIZE28:@.+]] = private {{.*}}constant [3 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] {{8|4}}, i[[Z]] 16]
// CK19: [[MTYPE28:@.+]] = private {{.*}}constant [3 x i32] [i32 35, i32 19, i32 19]

// CK19: [[SIZE29:@.+]] = private {{.*}}constant [3 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] {{8|4}}, i[[Z]] 4]
// CK19: [[MTYPE29:@.+]] = private {{.*}}constant [3 x i32] [i32 35, i32 19, i32 19]

// CK19: [[MTYPE30:@.+]] = private {{.*}}constant [4 x i32] [i32 288, i32 288, i32 288, i32 35]

// CK19: [[SIZE31:@.+]] = private {{.*}}constant [4 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] {{8|4}}, i[[Z]] {{8|4}}, i[[Z]] 40]
// CK19: [[MTYPE31:@.+]] = private {{.*}}constant [4 x i32] [i32 288, i32 288, i32 288, i32 35]

// CK19: [[SIZE32:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 13728]
// CK19: [[MTYPE32:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK19: [[SIZE33:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 13728]
// CK19: [[MTYPE33:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK19: [[SIZE34:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 13728]
// CK19: [[MTYPE34:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK19: [[MTYPE35:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK19: [[SIZE36:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 208]
// CK19: [[MTYPE36:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK19: [[MTYPE37:@.+]] = private {{.*}}constant [3 x i32] [i32 288, i32 288, i32 35]

// CK19: [[MTYPE38:@.+]] = private {{.*}}constant [3 x i32] [i32 288, i32 288, i32 35]

// CK19: [[MTYPE39:@.+]] = private {{.*}}constant [3 x i32] [i32 288, i32 288, i32 35]

// CK19: [[MTYPE40:@.+]] = private {{.*}}constant [3 x i32] [i32 288, i32 288, i32 35]

// CK19: [[SIZE41:@.+]] = private {{.*}}constant [3 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] {{8|4}}, i[[Z]] 208]
// CK19: [[MTYPE41:@.+]] = private {{.*}}constant [3 x i32] [i32 288, i32 288, i32 35]

// CK19: [[SIZE42:@.+]] = private {{.*}}constant [3 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] {{8|4}}, i[[Z]] 104]
// CK19: [[MTYPE42:@.+]] = private {{.*}}constant [3 x i32] [i32 35, i32 19, i32 19]

// CK19: [[MTYPE43:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK19-LABEL: explicit_maps_single
void explicit_maps_single (int ii){
  // Map of a scalar.
  int a = ii;

  // Region 00
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast i32* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32* [[VAR0]] to i8*

  // CK19: call void [[CALL00:@.+]](i32* {{[^,]+}})
  #pragma omp target map(alloc:a)
  {
    ++a;
  }

  // Map of an array.
  int arra[100];

  // Region 01
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast [100 x i32]* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast [100 x i32]* [[VAR0]] to i8*

  // CK19: call void [[CALL01:@.+]]([100 x i32]* {{[^,]+}})
  #pragma omp target map(to:arra)
  {
    arra[50]++;
  }

  // Region 02
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast [100 x i32]* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} 20

  // CK19: call void [[CALL02:@.+]]([100 x i32]* {{[^,]+}})
  #pragma omp target map(from:arra[20:60])
  {
    arra[50]++;
  }

  // Region 03
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast [100 x i32]* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} 0

  // CK19: call void [[CALL03:@.+]]([100 x i32]* {{[^,]+}})
  #pragma omp target map(tofrom:arra[:60])
  {
    arra[50]++;
  }

  // Region 04
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE04]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE04]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast [100 x i32]* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} 0

  // CK19: call void [[CALL04:@.+]]([100 x i32]* {{[^,]+}})
  #pragma omp target map(alloc:arra[:])
  {
    arra[50]++;
  }

  // Region 05
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE05]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE05]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast [100 x i32]* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} 15

  // CK19: call void [[CALL05:@.+]]([100 x i32]* {{[^,]+}})
  #pragma omp target map(to:arra[15])
  {
    arra[15]++;
  }

  // Region 06
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[Z]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE06]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: store i{{.+}} [[CSVAL0:%[^,]+]], i{{.+}}* [[S0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast [100 x i32]* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[CSVAL0]] = mul nuw i{{.+}} %{{.*}}, 4
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} %{{.*}}

  // CK19: call void [[CALL06:@.+]]([100 x i32]* {{[^,]+}})
  #pragma omp target map(tofrom:arra[ii:ii+23])
  {
    arra[50]++;
  }

  // Region 07
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[Z]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE07]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: store i{{.+}} [[CSVAL0:%[^,]+]], i{{.+}}* [[S0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast [100 x i32]* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[CSVAL0]] = mul nuw i{{.+}} %{{.*}}, 4
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} 0

  // CK19: call void [[CALL07:@.+]]([100 x i32]* {{[^,]+}})
  #pragma omp target map(alloc:arra[:ii])
  {
    arra[50]++;
  }

  // Region 08
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE08]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE08]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast [100 x i32]* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[100 x i32]* [[VAR0]], i{{.+}} 0, i{{.+}} %{{.*}}

  // CK19: call void [[CALL08:@.+]]([100 x i32]* {{[^,]+}})
  #pragma omp target map(tofrom:arra[ii])
  {
    arra[15]++;
  }

  // Map of a pointer.
  int *pa;

  // Region 09
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE09]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE09]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast i32** [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32** [[VAR0]] to i8*

  // CK19: call void [[CALL09:@.+]](i32** {{[^,]+}})
  #pragma omp target map(from:pa)
  {
    pa[50]++;
  }

  // Region 10
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE10]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE10]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast i32* [[RVAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 20
  // CK19-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK19: call void [[CALL10:@.+]](i32* {{[^,]+}})
  #pragma omp target map(tofrom:pa[20:60])
  {
    pa[50]++;
  }

  // Region 11
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE11]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE11]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast i32* [[RVAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 0
  // CK19-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK19: call void [[CALL11:@.+]](i32* {{[^,]+}})
  #pragma omp target map(alloc:pa[:60])
  {
    pa[50]++;
  }

  // Region 12
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE12]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE12]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast i32* [[RVAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 15
  // CK19-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK19: call void [[CALL12:@.+]](i32* {{[^,]+}})
  #pragma omp target map(to:pa[15])
  {
    pa[15]++;
  }

  // Region 13
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[Z]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE13]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: store i{{.+}} [[CSVAL0:%[^,]+]], i{{.+}}* [[S0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast i32* [[RVAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[CSVAL0]] = mul nuw i{{.+}} %{{.*}}, 4
  // CK19-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} %{{.*}}
  // CK19-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK19: call void [[CALL13:@.+]](i32* {{[^,]+}})
  #pragma omp target map(alloc:pa[ii-23:ii])
  {
    pa[50]++;
  }

  // Region 14
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[Z]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE14]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: store i{{.+}} [[CSVAL0:%[^,]+]], i{{.+}}* [[S0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast i32* [[RVAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[CSVAL0]] = mul nuw i{{.+}} %{{.*}}, 4
  // CK19-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 0
  // CK19-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK19: call void [[CALL14:@.+]](i32* {{[^,]+}})
  #pragma omp target map(to:pa[:ii])
  {
    pa[50]++;
  }

  // Region 15
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE15]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE15]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast i32* [[RVAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} %{{.*}}
  // CK19-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK19: call void [[CALL15:@.+]](i32* {{[^,]+}})
  #pragma omp target map(from:pa[ii+12])
  {
    pa[15]++;
  }

  // Map of a variable-size array.
  int va[ii];

  // Region 16
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[Z]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE16]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: store i{{.+}} {{8|4}}, i{{.+}}* [[S0]]
  // CK19-DAG: [[CBPVAL0]] = inttoptr i[[Z]] %{{.+}} to i8*
  // CK19-DAG: [[CPVAL0]] = inttoptr i[[Z]] %{{.+}}to i8*

  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
  // CK19-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
  // CK19-DAG: store i{{.+}} [[CSVAL1:%[^,]+]], i{{.+}}* [[S1]]
  // CK19-DAG: [[CBPVAL1]] = bitcast i32* [[VAR1:%.+]] to i8*
  // CK19-DAG: [[CPVAL1]] = bitcast i32* [[VAR1]] to i8*
  // CK19-DAG: [[CSVAL1]] = mul nuw i{{.+}} %{{.*}}, 4

  // CK19: call void [[CALL16:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  #pragma omp target map(to:va)
  {
   va[50]++;
  }

  // Region 17
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE17]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE17]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = inttoptr i[[Z]] %{{.+}} to i8*
  // CK19-DAG: [[CPVAL0]] = inttoptr i[[Z]] %{{.+}}to i8*

  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
  // CK19-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
  // CK19-DAG: [[CBPVAL1]] = bitcast i32* [[VAR1:%.+]] to i8*
  // CK19-DAG: [[CPVAL1]] = bitcast i32* [[SEC1:%.+]] to i8*
  // CK19-DAG: [[SEC1]] = getelementptr {{.*}}i32* [[VAR1]], i{{.+}} 20

  // CK19: call void [[CALL17:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  #pragma omp target map(from:va[20:60])
  {
   va[50]++;
  }

  // Region 18
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE18]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE18]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = inttoptr i[[Z]] %{{.+}} to i8*
  // CK19-DAG: [[CPVAL0]] = inttoptr i[[Z]] %{{.+}}to i8*

  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
  // CK19-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
  // CK19-DAG: [[CBPVAL1]] = bitcast i32* [[VAR1:%.+]] to i8*
  // CK19-DAG: [[CPVAL1]] = bitcast i32* [[SEC1:%.+]] to i8*
  // CK19-DAG: [[SEC1]] = getelementptr {{.*}}i32* [[VAR1]], i{{.+}} 0

  // CK19: call void [[CALL18:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  #pragma omp target map(tofrom:va[:60])
  {
   va[50]++;
  }

  // Region 19
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[Z]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE19]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: store i{{.+}} {{8|4}}, i{{.+}}* [[S0]]
  // CK19-DAG: [[CBPVAL0]] = inttoptr i[[Z]] %{{.+}} to i8*
  // CK19-DAG: [[CPVAL0]] = inttoptr i[[Z]] %{{.+}}to i8*

  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
  // CK19-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
  // CK19-DAG: store i{{.+}} [[CSVAL1:%[^,]+]], i{{.+}}* [[S1]]
  // CK19-DAG: [[CBPVAL1]] = bitcast i32* [[VAR1:%.+]] to i8*
  // CK19-DAG: [[CPVAL1]] = bitcast i32* [[SEC1:%.+]] to i8*
  // CK19-DAG: [[CSVAL1]] = mul nuw i{{.+}} %{{.*}}, 4
  // CK19-DAG: [[SEC1]] = getelementptr {{.*}}i32* [[VAR1]], i{{.+}} 0

  // CK19: call void [[CALL19:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  #pragma omp target map(alloc:va[:])
  {
   va[50]++;
  }

  // Region 20
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE20]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE20]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = inttoptr i[[Z]] %{{.+}} to i8*
  // CK19-DAG: [[CPVAL0]] = inttoptr i[[Z]] %{{.+}}to i8*

  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
  // CK19-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
  // CK19-DAG: [[CBPVAL1]] = bitcast i32* [[VAR1:%.+]] to i8*
  // CK19-DAG: [[CPVAL1]] = bitcast i32* [[SEC1:%.+]] to i8*
  // CK19-DAG: [[SEC1]] = getelementptr {{.*}}i32* [[VAR1]], i{{.+}} 15

  // CK19: call void [[CALL20:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  #pragma omp target map(to:va[15])
  {
   va[15]++;
  }

  // Region 21
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[Z]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE21]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: store i{{.+}} {{8|4}}, i{{.+}}* [[S0]]
  // CK19-DAG: [[CBPVAL0]] = inttoptr i[[Z]] %{{.+}} to i8*
  // CK19-DAG: [[CPVAL0]] = inttoptr i[[Z]] %{{.+}}to i8*

  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
  // CK19-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
  // CK19-DAG: store i{{.+}} [[CSVAL1:%[^,]+]], i{{.+}}* [[S1]]
  // CK19-DAG: [[CBPVAL1]] = bitcast i32* [[VAR1:%.+]] to i8*
  // CK19-DAG: [[CPVAL1]] = bitcast i32* [[SEC1:%.+]] to i8*
  // CK19-DAG: [[CSVAL1]] = mul nuw i{{.+}} %{{.*}}, 4
  // CK19-DAG: [[SEC1]] = getelementptr {{.*}}i32* [[VAR1]], i{{.+}} %{{.+}}

  // CK19: call void [[CALL21:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  #pragma omp target map(tofrom:va[ii:ii+23])
  {
   va[50]++;
  }

  // Region 22
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE22]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE22]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = inttoptr i[[Z]] %{{.+}} to i8*
  // CK19-DAG: [[CPVAL0]] = inttoptr i[[Z]] %{{.+}}to i8*

  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
  // CK19-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
  // CK19-DAG: [[CBPVAL1]] = bitcast i32* [[VAR1:%.+]] to i8*
  // CK19-DAG: [[CPVAL1]] = bitcast i32* [[SEC1:%.+]] to i8*
  // CK19-DAG: [[SEC1]] = getelementptr {{.*}}i32* [[VAR1]], i{{.+}} %{{.+}}

  // CK19: call void [[CALL22:@.+]](i{{.+}} {{[^,]+}}, i32* {{[^,]+}})
  #pragma omp target map(tofrom:va[ii])
  {
   va[15]++;
  }

  // Always.
  // Region 23
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE23]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE23]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast i32* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32* [[VAR0]] to i8*

  // CK19: call void [[CALL23:@.+]](i32* {{[^,]+}})
  #pragma omp target map(always, tofrom: a)
  {
   a++;
  }

  // Multidimensional arrays.
  int marr[4][5][6];
  int ***mptr;

  // Region 24
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE24]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE24]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast [4 x [5 x [6 x i32]]]* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast [4 x [5 x [6 x i32]]]* [[VAR0]] to i8*

  // CK19: call void [[CALL24:@.+]]([4 x [5 x [6 x i32]]]* {{[^,]+}})
  #pragma omp target map(tofrom: marr)
  {
   marr[1][2][3]++;
  }

  // Region 25
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE25]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE25]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast [4 x [5 x [6 x i32]]]* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[6 x i32]* [[SEC00:[^,]+]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[SEC00]] = getelementptr {{.*}}[5 x [6 x i32]]* [[SEC000:[^,]+]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[SEC000]] = getelementptr {{.*}}[4 x [5 x [6 x i32]]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

  // CK19: call void [[CALL25:@.+]]([4 x [5 x [6 x i32]]]* {{[^,]+}})
  #pragma omp target map(tofrom: marr[1][2][2:4])
  {
   marr[1][2][3]++;
  }

  // Region 26
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE26]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE26]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast [4 x [5 x [6 x i32]]]* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[6 x i32]* [[SEC00:[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[SEC00]] = getelementptr {{.*}}[5 x [6 x i32]]* [[SEC000:[^,]+]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[SEC000]] = getelementptr {{.*}}[4 x [5 x [6 x i32]]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

  // CK19: call void [[CALL26:@.+]]([4 x [5 x [6 x i32]]]* {{[^,]+}})
  #pragma omp target map(tofrom: marr[1][2][:])
  {
   marr[1][2][3]++;
  }

  // Region 27
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE27]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE27]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast [4 x [5 x [6 x i32]]]* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}[6 x i32]* [[SEC00:[^,]+]], i{{.+}} 0, i{{.+}} 3
  // CK19-DAG: [[SEC00]] = getelementptr {{.*}}[5 x [6 x i32]]* [[SEC000:[^,]+]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[SEC000]] = getelementptr {{.*}}[4 x [5 x [6 x i32]]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

  // CK19: call void [[CALL27:@.+]]([4 x [5 x [6 x i32]]]* {{[^,]+}})
  #pragma omp target map(tofrom: marr[1][2][3])
  {
   marr[1][2][3]++;
  }

  // Region 28
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[SIZE28]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE28]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast i32*** [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32*** [[SEC0:%.+]] to i8*
  // CK19-DAG: [[VAR0]] = load i32***, i32**** [[PTR:%[^,]+]],
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32*** [[SEC00:[^,]+]], i{{.+}} 1
  // CK19-DAG: [[SEC00]] = load i32***, i32**** [[PTR]],

  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
  // CK19-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
  // CK19-DAG: [[CBPVAL1]] = bitcast i32*** [[SEC0]] to i8*
  // CK19-DAG: [[CPVAL1]] = bitcast i32** [[SEC1:%.+]] to i8*
  // CK19-DAG: [[SEC1]] = getelementptr {{.*}}i32** [[SEC11:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC11]] = load i32**, i32*** [[SEC111:%[^,]+]],
  // CK19-DAG: [[SEC111]] = getelementptr {{.*}}i32*** [[SEC1111:[^,]+]], i{{.+}} 1
  // CK19-DAG: [[SEC1111]] = load i32***, i32**** [[PTR]],

  // CK19-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: store i8* [[CBPVAL2:%[^,]+]], i8** [[BP2]]
  // CK19-DAG: store i8* [[CPVAL2:%[^,]+]], i8** [[P2]]
  // CK19-DAG: [[CBPVAL2]] = bitcast i32** [[SEC1]] to i8*
  // CK19-DAG: [[CPVAL2]] = bitcast i32* [[SEC2:%.+]] to i8*
  // CK19-DAG: [[SEC2]] = getelementptr {{.*}}i32* [[SEC22:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC22]] = load i32*, i32** [[SEC222:%[^,]+]],
  // CK19-DAG: [[SEC222]] = getelementptr {{.*}}i32** [[SEC2222:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC2222]] = load i32**, i32*** [[SEC22222:%[^,]+]],
  // CK19-DAG: [[SEC22222]] = getelementptr {{.*}}i32*** [[SEC222222:[^,]+]], i{{.+}} 1
  // CK19-DAG: [[SEC222222]] = load i32***, i32**** [[PTR]],

  // CK19: call void [[CALL28:@.+]](i32*** {{[^,]+}})
  #pragma omp target map(tofrom: mptr[1][2][2:4])
  {
    mptr[1][2][3]++;
  }

  // Region 29
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[SIZE29]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE29]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast i32*** [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast i32*** [[SEC0:%.+]] to i8*
  // CK19-DAG: [[VAR0]] = load i32***, i32**** [[PTR:%[^,]+]],
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}i32*** [[SEC00:[^,]+]], i{{.+}} 1
  // CK19-DAG: [[SEC00]] = load i32***, i32**** [[PTR]],

  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
  // CK19-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
  // CK19-DAG: [[CBPVAL1]] = bitcast i32*** [[SEC0]] to i8*
  // CK19-DAG: [[CPVAL1]] = bitcast i32** [[SEC1:%.+]] to i8*
  // CK19-DAG: [[SEC1]] = getelementptr {{.*}}i32** [[SEC11:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC11]] = load i32**, i32*** [[SEC111:%[^,]+]],
  // CK19-DAG: [[SEC111]] = getelementptr {{.*}}i32*** [[SEC1111:[^,]+]], i{{.+}} 1
  // CK19-DAG: [[SEC1111]] = load i32***, i32**** [[PTR]],

  // CK19-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: store i8* [[CBPVAL2:%[^,]+]], i8** [[BP2]]
  // CK19-DAG: store i8* [[CPVAL2:%[^,]+]], i8** [[P2]]
  // CK19-DAG: [[CBPVAL2]] = bitcast i32** [[SEC1]] to i8*
  // CK19-DAG: [[CPVAL2]] = bitcast i32* [[SEC2:%.+]] to i8*
  // CK19-DAG: [[SEC2]] = getelementptr {{.*}}i32* [[SEC22:[^,]+]], i{{.+}} 3
  // CK19-DAG: [[SEC22]] = load i32*, i32** [[SEC222:%[^,]+]],
  // CK19-DAG: [[SEC222]] = getelementptr {{.*}}i32** [[SEC2222:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC2222]] = load i32**, i32*** [[SEC22222:%[^,]+]],
  // CK19-DAG: [[SEC22222]] = getelementptr {{.*}}i32*** [[SEC222222:[^,]+]], i{{.+}} 1
  // CK19-DAG: [[SEC222222]] = load i32***, i32**** [[PTR]],

  // CK19: call void [[CALL29:@.+]](i32*** {{[^,]+}})
  #pragma omp target map(tofrom: mptr[1][2][3])
  {
    mptr[1][2][3]++;
  }

  // Multidimensional VLA.
  double mva[23][ii][ii+5];

  // Region 30
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 4, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[Z]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[4 x i{{.+}}]* [[MTYPE30]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* inttoptr (i[[Z]] 23 to i8*), i8** [[BP0]]
  // CK19-DAG: store i8* inttoptr (i[[Z]] 23 to i8*), i8** [[P0]]
  // CK19-DAG: store i[[Z]] {{8|4}}, i[[Z]]* [[S0]]
  //
  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
  // CK19-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
  // CK19-DAG: store i[[Z]] {{8|4}}, i[[Z]]* [[S1]]
  // CK19-DAG: [[CBPVAL1]] = inttoptr i[[Z]] [[VAR1:%.+]] to i8*
  // CK19-DAG: [[CPVAL1]] = inttoptr i[[Z]] [[VAR11:%.+]] to i8*
  // CK19-64-DAG: [[VAR1]] = zext i32 %{{[^,]+}} to i64
  // CK19-64-DAG: [[VAR11]] = zext i32 %{{[^,]+}} to i64
  //
  // CK19-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: store i8* [[CBPVAL2:%[^,]+]], i8** [[BP2]]
  // CK19-DAG: store i8* [[CPVAL2:%[^,]+]], i8** [[P2]]
  // CK19-DAG: store i[[Z]] {{8|4}}, i[[Z]]* [[S2]]
  // CK19-DAG: [[CBPVAL2]] = inttoptr i[[Z]] [[VAR2:%.+]] to i8*
  // CK19-DAG: [[CPVAL2]] = inttoptr i[[Z]] [[VAR22:%.+]] to i8*
  // CK19-64-DAG: [[VAR2]] = zext i32 %{{[^,]+}} to i64
  // CK19-64-DAG: [[VAR22]] = zext i32 %{{[^,]+}} to i64
  //
  // CK19-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
  // CK19-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
  // CK19-DAG: [[S3:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 3
  // CK19-DAG: store i8* [[CBPVAL3:%[^,]+]], i8** [[BP3]]
  // CK19-DAG: store i8* [[CPVAL3:%[^,]+]], i8** [[P3]]
  // CK19-DAG: store i[[Z]] [[CSVAL3:%[^,]+]], i[[Z]]* [[S3]]
  // CK19-DAG: [[CBPVAL3]] = bitcast double* [[VAR3:%.+]] to i8*
  // CK19-DAG: [[CPVAL3]] = bitcast double* [[VAR3]] to i8*
  // CK19-DAG: [[CSVAL3]] = mul nuw i[[Z]] %{{[^,]+}}, {{8|4}}

  // CK19: call void [[CALL30:@.+]](i[[Z]] 23, i[[Z]] %{{[^,]+}}, i[[Z]] %{{[^,]+}}, double* %{{[^,]+}})
  #pragma omp target map(tofrom: mva)
  {
    mva[1][2][3]++;
  }

  // Region 31
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 4, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[4 x i{{.+}}]* [[SIZE31]], {{.+}}getelementptr {{.+}}[4 x i{{.+}}]* [[MTYPE31]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  //
  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* inttoptr (i[[Z]] 23 to i8*), i8** [[BP0]]
  // CK19-DAG: store i8* inttoptr (i[[Z]] 23 to i8*), i8** [[P0]]
  //
  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
  // CK19-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
  // CK19-DAG: [[CBPVAL1]] = inttoptr i[[Z]] [[VAR1:%.+]] to i8*
  // CK19-DAG: [[CPVAL1]] = inttoptr i[[Z]] [[VAR11:%.+]] to i8*
  //
  // CK19-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: store i8* [[CBPVAL2:%[^,]+]], i8** [[BP2]]
  // CK19-DAG: store i8* [[CPVAL2:%[^,]+]], i8** [[P2]]
  // CK19-DAG: [[CBPVAL2]] = inttoptr i[[Z]] [[VAR2:%.+]] to i8*
  // CK19-DAG: [[CPVAL2]] = inttoptr i[[Z]] [[VAR22:%.+]] to i8*
  //
  // CK19-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
  // CK19-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
  // CK19-DAG: store i8* [[CBPVAL3:%[^,]+]], i8** [[BP3]]
  // CK19-DAG: store i8* [[CPVAL3:%[^,]+]], i8** [[P3]]
  // CK19-DAG: [[CBPVAL3]] = bitcast double* [[VAR3:%.+]] to i8*
  // CK19-DAG: [[CPVAL3]] = bitcast double* [[SEC3:%.+]] to i8*
  // CK19-DAG: [[SEC3]] = getelementptr {{.*}}double* [[SEC33:%.+]], i[[Z]] 0
  // CK19-DAG: [[SEC33]] = getelementptr {{.*}}double* [[SEC333:%.+]], i[[Z]] [[IDX3:%.+]]
  // CK19-DAG: [[IDX3]] = mul nsw i[[Z]] %{{[^,]+}}, %{{[^,]+}}
  // CK19-DAG: [[SEC333]] = getelementptr {{.*}}double* [[VAR3]], i[[Z]] [[IDX33:%.+]]
  // CK19-DAG: [[IDX33]] = mul nsw i[[Z]] 1, %{{[^,]+}}

  // CK19: call void [[CALL31:@.+]](i[[Z]] 23, i[[Z]] %{{[^,]+}}, i[[Z]] %{{[^,]+}}, double* %{{[^,]+}})
  #pragma omp target map(tofrom: mva[1][ii-2][:5])
  {
    mva[1][2][3]++;
  }

  // Multidimensional array sections.
  double marras[11][12][13];
  double mvlaas[11][ii][13];
  double ***mptras;

  // Region 32
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE32]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE32]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast [11 x [12 x [13 x double]]]* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast [11 x [12 x [13 x double]]]* [[VAR0]] to i8*

  // CK19: call void [[CALL32:@.+]]([11 x [12 x [13 x double]]]* {{[^,]+}})
  #pragma omp target map(marras)
  {
    marras[1][2][3]++;
  }

  // Region 33
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE33]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE33]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast [11 x [12 x [13 x double]]]* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast [12 x [13 x double]]* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[SEC0]] = getelementptr {{.+}}[11 x [12 x [13 x double]]]* [[VAR0]], i[[Z]] 0, i[[Z]] 0

  // CK19: call void [[CALL33:@.+]]([11 x [12 x [13 x double]]]* {{[^,]+}})
  #pragma omp target map(marras[:])
  {
    marras[1][2][3]++;
  }

  // Region 34
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE34]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE34]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast [11 x [12 x [13 x double]]]* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast [12 x [13 x double]]* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[SEC0]] = getelementptr {{.+}}[11 x [12 x [13 x double]]]* [[VAR0]], i[[Z]] 0, i[[Z]] 0

  // CK19: call void [[CALL34:@.+]]([11 x [12 x [13 x double]]]* {{[^,]+}})
  #pragma omp target map(marras[:][:][:])
  {
    marras[1][2][3]++;
  }

  // Region 35
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[Z]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE35]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0

  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: store i[[Z]] [[CSVAL0:%[^,]+]], i[[Z]]* [[S0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast [11 x [12 x [13 x double]]]* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast [13 x double]* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[SEC0]] = getelementptr {{.+}}[12 x [13 x double]]* [[SEC00:%[^,]+]], i[[Z]] 0, i[[Z]] 0
  // CK19-DAG: [[SEC00]] = getelementptr {{.+}}[11 x [12 x [13 x double]]]* [[VAR0]], i[[Z]] 0, i[[Z]] 1
  // CK19-DAG: [[CSVAL0]] = mul nuw i[[Z]] %{{[^,]+}}, 104

  // CK19: call void [[CALL35:@.+]]([11 x [12 x [13 x double]]]* {{[^,]+}})
  #pragma omp target map(marras[1][:ii][:])
  {
    marras[1][2][3]++;
  }

  // Region 36
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE36]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE36]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast [11 x [12 x [13 x double]]]* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast [13 x double]* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[SEC0]] = getelementptr {{.+}}[13 x double]* [[SEC00:%[^,]+]], i{{.+}} 0
  // CK19-DAG: [[SEC00]] = getelementptr {{.+}}[12 x [13 x double]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[SEC000]] = getelementptr {{.+}}[11 x [12 x [13 x double]]]* [[VAR0]], i{{.+}} 0, i{{.+}} 0

  // CK19: call void [[CALL36:@.+]]([11 x [12 x [13 x double]]]* {{[^,]+}})
  #pragma omp target map(marras[:1][:2][:13])
  {
    marras[1][2][3]++;
  }

  // Region 37
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[Z]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE37]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* inttoptr (i[[Z]] 11 to i8*), i8** [[BP0]]
  // CK19-DAG: store i8* inttoptr (i[[Z]] 11 to i8*), i8** [[P0]]
  // CK19-DAG: store i[[Z]] {{8|4}}, i[[Z]]* [[S0]]
  //
  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
  // CK19-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
  // CK19-DAG: store i[[Z]] {{8|4}}, i[[Z]]* [[S1]]
  // CK19-DAG: [[CBPVAL1]] = inttoptr i[[Z]] [[VAR1:%.+]] to i8*
  // CK19-DAG: [[CPVAL1]] = inttoptr i[[Z]] [[VAR11:%.+]] to i8*
  //
  // CK19-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: store i8* [[CBPVAL2:%[^,]+]], i8** [[BP2]]
  // CK19-DAG: store i8* [[CPVAL2:%[^,]+]], i8** [[P2]]
  // CK19-DAG: store i[[Z]] [[CSVAL2:%[^,]+]], i[[Z]]* [[S2]]
  // CK19-DAG: [[CBPVAL2]] = bitcast [13 x double]* [[VAR2:%.+]] to i8*
  // CK19-DAG: [[CPVAL2]] = bitcast [13 x double]* [[VAR2]] to i8*
  // CK19-DAG: [[CSVAL2]] = mul nuw i[[Z]] %{{[^,]+}}, 104

  // CK19: call void [[CALL37:@.+]](i[[Z]] 11, i[[Z]] %{{[^,]+}}, [13 x double]* %{{[^,]+}})
  #pragma omp target map(mvlaas)
  {
    mvlaas[1][2][3]++;
  }

  // Region 38
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[Z]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE38]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* inttoptr (i[[Z]] 11 to i8*), i8** [[BP0]]
  // CK19-DAG: store i8* inttoptr (i[[Z]] 11 to i8*), i8** [[P0]]
  // CK19-DAG: store i[[Z]] {{8|4}}, i[[Z]]* [[S0]]
  //
  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
  // CK19-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
  // CK19-DAG: store i[[Z]] {{8|4}}, i[[Z]]* [[S1]]
  // CK19-DAG: [[CBPVAL1]] = inttoptr i[[Z]] [[VAR1:%.+]] to i8*
  // CK19-DAG: [[CPVAL1]] = inttoptr i[[Z]] [[VAR11:%.+]] to i8*
  //
  // CK19-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: store i8* [[CBPVAL2:%[^,]+]], i8** [[BP2]]
  // CK19-DAG: store i8* [[CPVAL2:%[^,]+]], i8** [[P2]]
  // CK19-DAG: store i[[Z]] [[CSVAL2:%[^,]+]], i[[Z]]* [[S2]]
  // CK19-DAG: [[CBPVAL2]] = bitcast [13 x double]* [[VAR2:%.+]] to i8*
  // CK19-DAG: [[CPVAL2]] = bitcast [13 x double]* [[SEC2:%.+]] to i8*
  // CK19-DAG: [[SEC2]] = getelementptr {{.+}}[13 x double]* [[VAR2]], i[[Z]] [[SEC22:%[^,]+]]
  // CK19-DAG: [[SEC22]] = mul nsw i[[Z]] 0, %{{[^,]+}}
  // CK19-DAG: [[CSVAL2]] = mul nuw i[[Z]] %{{[^,]+}}, 104

  // CK19: call void [[CALL38:@.+]](i[[Z]] 11, i[[Z]] %{{[^,]+}}, [13 x double]* %{{[^,]+}})
  #pragma omp target map(mvlaas[:])
  {
    mvlaas[1][2][3]++;
  }

  // Region 39
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[Z]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE39]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* inttoptr (i[[Z]] 11 to i8*), i8** [[BP0]]
  // CK19-DAG: store i8* inttoptr (i[[Z]] 11 to i8*), i8** [[P0]]
  // CK19-DAG: store i[[Z]] {{8|4}}, i[[Z]]* [[S0]]
  //
  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
  // CK19-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
  // CK19-DAG: store i[[Z]] {{8|4}}, i[[Z]]* [[S1]]
  // CK19-DAG: [[CBPVAL1]] = inttoptr i[[Z]] [[VAR1:%.+]] to i8*
  // CK19-DAG: [[CPVAL1]] = inttoptr i[[Z]] [[VAR11:%.+]] to i8*
  //
  // CK19-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: store i8* [[CBPVAL2:%[^,]+]], i8** [[BP2]]
  // CK19-DAG: store i8* [[CPVAL2:%[^,]+]], i8** [[P2]]
  // CK19-DAG: store i[[Z]] [[CSVAL2:%[^,]+]], i[[Z]]* [[S2]]
  // CK19-DAG: [[CBPVAL2]] = bitcast [13 x double]* [[VAR2:%.+]] to i8*
  // CK19-DAG: [[CPVAL2]] = bitcast [13 x double]* [[SEC2:%.+]] to i8*
  // CK19-DAG: [[SEC2]] = getelementptr {{.+}}[13 x double]* [[VAR2]], i[[Z]] [[SEC22:%[^,]+]]
  // CK19-DAG: [[SEC22]] = mul nsw i[[Z]] 0, %{{[^,]+}}
  // CK19-DAG: [[CSVAL2]] = mul nuw i[[Z]] %{{[^,]+}}, 104

  // CK19: call void [[CALL39:@.+]](i[[Z]] 11, i[[Z]] %{{[^,]+}}, [13 x double]* %{{[^,]+}})
  #pragma omp target map(mvlaas[:][:][:])
  {
    mvlaas[1][2][3]++;
  }

  // Region 40
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[Z]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE40]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* inttoptr (i[[Z]] 11 to i8*), i8** [[BP0]]
  // CK19-DAG: store i8* inttoptr (i[[Z]] 11 to i8*), i8** [[P0]]
  // CK19-DAG: store i[[Z]] {{8|4}}, i[[Z]]* [[S0]]
  //
  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
  // CK19-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
  // CK19-DAG: store i[[Z]] {{8|4}}, i[[Z]]* [[S1]]
  // CK19-DAG: [[CBPVAL1]] = inttoptr i[[Z]] [[VAR1:%.+]] to i8*
  // CK19-DAG: [[CPVAL1]] = inttoptr i[[Z]] [[VAR11:%.+]] to i8*
  //
  // CK19-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: store i8* [[CBPVAL2:%[^,]+]], i8** [[BP2]]
  // CK19-DAG: store i8* [[CPVAL2:%[^,]+]], i8** [[P2]]
  // CK19-DAG: store i[[Z]] [[CSVAL2:%[^,]+]], i[[Z]]* [[S2]]
  // CK19-DAG: [[CBPVAL2]] = bitcast [13 x double]* [[VAR2:%.+]] to i8*
  // CK19-DAG: [[CPVAL2]] = bitcast [13 x double]* [[SEC2:%.+]] to i8*
  // CK19-DAG: [[SEC2]] = getelementptr {{.+}}[13 x double]* [[SEC22:%[^,]+]], i[[Z]] 0
  // CK19-DAG: [[SEC22]] = getelementptr {{.+}}[13 x double]* [[VAR2]], i[[Z]] [[SEC222:%[^,]+]]
  // CK19-DAG: [[SEC222]] = mul nsw i[[Z]] 1, %{{[^,]+}}

  // CK19: call void [[CALL40:@.+]](i[[Z]] 11, i[[Z]] %{{[^,]+}}, [13 x double]* %{{[^,]+}})
  #pragma omp target map(mvlaas[1][:ii][:])
  {
    mvlaas[1][2][3]++;
  }

  // Region 41
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[SIZE41]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE41]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  //
  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* inttoptr (i[[Z]] 11 to i8*), i8** [[BP0]]
  // CK19-DAG: store i8* inttoptr (i[[Z]] 11 to i8*), i8** [[P0]]
  //
  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
  // CK19-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
  // CK19-DAG: [[CBPVAL1]] = inttoptr i[[Z]] [[VAR1:%.+]] to i8*
  // CK19-DAG: [[CPVAL1]] = inttoptr i[[Z]] [[VAR11:%.+]] to i8*
  //
  // CK19-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: store i8* [[CBPVAL2:%[^,]+]], i8** [[BP2]]
  // CK19-DAG: store i8* [[CPVAL2:%[^,]+]], i8** [[P2]]
  // CK19-DAG: [[CBPVAL2]] = bitcast [13 x double]* [[VAR2:%.+]] to i8*
  // CK19-DAG: [[CPVAL2]] = bitcast [13 x double]* [[SEC2:%.+]] to i8*
  // CK19-DAG: [[SEC2]] = getelementptr {{.+}}[13 x double]* [[SEC22:%[^,]+]], i[[Z]] 0
  // CK19-DAG: [[SEC22]] = getelementptr {{.+}}[13 x double]* [[VAR2]], i[[Z]] [[SEC222:%[^,]+]]
  // CK19-DAG: [[SEC222]] = mul nsw i[[Z]] 0, %{{[^,]+}}

  // CK19: call void [[CALL41:@.+]](i[[Z]] 11, i[[Z]] %{{[^,]+}}, [13 x double]* %{{[^,]+}})
  #pragma omp target map(mvlaas[:1][:2][:13])
  {
    mvlaas[1][2][3]++;
  }

  // Region 42
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 3, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[SIZE42]], {{.+}}getelementptr {{.+}}[3 x i{{.+}}]* [[MTYPE42]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast double*** [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast double*** [[SEC0:%.+]] to i8*
  // CK19-DAG: [[VAR0]] = load double***, double**** [[PTR:%[^,]+]],
  // CK19-DAG: [[SEC0]] = getelementptr {{.*}}double*** [[SEC00:[^,]+]], i{{.+}} 0
  // CK19-DAG: [[SEC00]] = load double***, double**** [[PTR]],

  // CK19-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK19-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
  // CK19-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
  // CK19-DAG: [[CBPVAL1]] = bitcast double*** [[SEC0]] to i8*
  // CK19-DAG: [[CPVAL1]] = bitcast double** [[SEC1:%.+]] to i8*
  // CK19-DAG: [[SEC1]] = getelementptr {{.*}}double** [[SEC11:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC11]] = load double**, double*** [[SEC111:%[^,]+]],
  // CK19-DAG: [[SEC111]] = getelementptr {{.*}}double*** [[SEC1111:[^,]+]], i{{.+}} 0
  // CK19-DAG: [[SEC1111]] = load double***, double**** [[PTR]],

  // CK19-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK19-DAG: store i8* [[CBPVAL2:%[^,]+]], i8** [[BP2]]
  // CK19-DAG: store i8* [[CPVAL2:%[^,]+]], i8** [[P2]]
  // CK19-DAG: [[CBPVAL2]] = bitcast double** [[SEC1]] to i8*
  // CK19-DAG: [[CPVAL2]] = bitcast double* [[SEC2:%.+]] to i8*
  // CK19-DAG: [[SEC2]] = getelementptr {{.*}}double* [[SEC22:[^,]+]], i{{.+}} 0
  // CK19-DAG: [[SEC22]] = load double*, double** [[SEC222:%[^,]+]],
  // CK19-DAG: [[SEC222]] = getelementptr {{.*}}double** [[SEC2222:[^,]+]], i{{.+}} 2
  // CK19-DAG: [[SEC2222]] = load double**, double*** [[SEC22222:%[^,]+]],
  // CK19-DAG: [[SEC22222]] = getelementptr {{.*}}double*** [[SEC222222:[^,]+]], i{{.+}} 0
  // CK19-DAG: [[SEC222222]] = load double***, double**** [[PTR]],

  // CK19: call void [[CALL42:@.+]](double*** {{[^,]+}})
  #pragma omp target map(mptras[:1][2][:13])
  {
    mptras[1][2][3]++;
  }

  // Region 43 - the memory is not contiguous for this map - will map the whole last dimension.
  // CK19-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i[[Z]]* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE43]]{{.+}})
  // CK19-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK19-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK19-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  //
  // CK19-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK19-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0

  // CK19-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK19-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK19-DAG: store i[[Z]] [[CSVAL0:%[^,]+]], i[[Z]]* [[S0]]
  // CK19-DAG: [[CBPVAL0]] = bitcast [11 x [12 x [13 x double]]]* [[VAR0:%.+]] to i8*
  // CK19-DAG: [[CPVAL0]] = bitcast [13 x double]* [[SEC0:%.+]] to i8*
  // CK19-DAG: [[SEC0]] = getelementptr {{.+}}[12 x [13 x double]]* [[SEC00:%[^,]+]], i[[Z]] 0, i[[Z]] 0
  // CK19-DAG: [[SEC00]] = getelementptr {{.+}}[11 x [12 x [13 x double]]]* [[VAR0]], i[[Z]] 0, i[[Z]] 1
  // CK19-DAG: [[CSVAL0]] = mul nuw i[[Z]] %{{[^,]+}}, 104

  // CK19: call void [[CALL43:@.+]]([11 x [12 x [13 x double]]]* {{[^,]+}})
  #pragma omp target map(marras[1][:ii][1:])
  {
    marras[1][2][3]++;
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

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK20 --check-prefix CK20-64
// RUN: %clang_cc1 -DCK20 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK20 --check-prefix CK20-32
// RUN: %clang_cc1 -DCK20 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK20 --check-prefix CK20-32
#ifdef CK20

// CK20: [[SIZE00:@.+]] = private {{.*}}constant [1 x i[[Z:64|32]]] [i[[Z:64|32]] 4]
// CK20: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i32] [i32 33]

// CK20: [[SIZE01:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 20]
// CK20: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i32] [i32 33]

// CK20: [[SIZE02:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 4]
// CK20: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i32] [i32 34]

// CK20: [[SIZE03:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 12]
// CK20: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i32] [i32 34]

// CK20-LABEL: explicit_maps_references_and_function_args
void explicit_maps_references_and_function_args (int a, float b, int (&c)[10], float *d){

  int &aa = a;
  float &bb = b;
  int (&cc)[10] = c;
  float *&dd = d;

  // Region 00
  // CK20-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}})
  // CK20-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK20-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK20-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK20-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK20-DAG: [[CBPVAL0]] = bitcast i32* [[RVAR0:%.+]] to i8*
  // CK20-DAG: [[CPVAL0]] = bitcast i32* [[RVAR00:%.+]] to i8*
  // CK20-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK20-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK20: call void [[CALL00:@.+]](i32* {{[^,]+}})
  #pragma omp target map(to:aa)
  {
    aa += 1;
  }

  // Region 01
  // CK20-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}})
  // CK20-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK20-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK20-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK20-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK20-DAG: [[CBPVAL0]] = bitcast [10 x i32]* [[RVAR0:%.+]] to i8*
  // CK20-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK20-DAG: [[SEC0]] = getelementptr {{.*}}[10 x i32]* [[RVAR00:%.+]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[RVAR0]] = load [10 x i32]*, [10 x i32]** [[VAR0:%[^,]+]]
  // CK20-DAG: [[RVAR00]] = load [10 x i32]*, [10 x i32]** [[VAR0]]

  // CK20: call void [[CALL01:@.+]]([10 x i32]* {{[^,]+}})
  #pragma omp target map(to:cc[:5])
  {
    cc[3] += 1;
  }

  // Region 02
  // CK20-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}})
  // CK20-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK20-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK20-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK20-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK20-DAG: [[CBPVAL0]] = bitcast float* [[VAR0:%.+]] to i8*
  // CK20-DAG: [[CPVAL0]] = bitcast float* [[VAR0]] to i8*

  // CK20: call void [[CALL02:@.+]](float* {{[^,]+}})
  #pragma omp target map(from:b)
  {
    b += 1.0f;
  }

  // Region 03
  // CK20-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}})
  // CK20-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK20-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK20-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK20-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK20-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK20-DAG: [[CBPVAL0]] = bitcast float* [[RVAR0:%.+]] to i8*
  // CK20-DAG: [[CPVAL0]] = bitcast float* [[SEC0:%.+]] to i8*
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
// RUN: %clang_cc1 -DCK21 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK21 --check-prefix CK21-64
// RUN: %clang_cc1 -DCK21 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK21 --check-prefix CK21-64
// RUN: %clang_cc1 -DCK21 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK21 --check-prefix CK21-32
// RUN: %clang_cc1 -DCK21 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK21 --check-prefix CK21-32
#ifdef CK21
// CK21: [[ST:%.+]] = type { i32, i32, float* }

// CK21: [[SIZE00:@.+]] = private {{.*}}constant [1 x i[[Z:64|32]]] [i[[Z:64|32]] 4]
// CK21: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK21: [[SIZE01:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 492]
// CK21: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK21: [[SIZE02:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] 500]
// CK21: [[MTYPE02:@.+]] = private {{.*}}constant [2 x i32] [i32 34, i32 18]

// CK21: [[SIZE03:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 492]
// CK21: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i32] [i32 34]

// CK21: [[SIZE04:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 4]
// CK21: [[MTYPE04:@.+]] = private {{.*}}constant [1 x i32] [i32 34]

// CK21: [[SIZE05:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] 4, i[[Z]] 4]
// CK21: [[MTYPE05:@.+]] = private {{.*}}constant [2 x i32] [i32 35, i32 3]

// CK21-LABEL: explicit_maps_template_args_and_members

template <int X, typename T>
struct CC {
  T A;
  int A2;
  float *B;

  int foo(T arg) {
    float la[X];
    T *lb;

    // Region 00
    // CK21-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}})
    // CK21-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK21-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
    // CK21-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
    // CK21-DAG: [[CBPVAL0]] = bitcast [[ST]]* [[VAR0:%.+]] to i8*
    // CK21-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
    // CK21-DAG: [[SEC0]] = getelementptr {{.*}}[[ST]]* [[VAR0:%.+]], i{{.+}} 0, i{{.+}} 0

    // CK21: call void [[CALL00:@.+]]([[ST]]* {{[^,]+}})
    #pragma omp target map(A)
    {
      A += 1;
    }
    
    // Region 01
    // CK21-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}})
    // CK21-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK21-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
    // CK21-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
    // CK21-DAG: [[CBPVAL0]] = bitcast i32* [[RVAR0:%.+]] to i8*
    // CK21-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
    // CK21-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
    // CK21-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 0
    // CK21-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

    // CK21: call void [[CALL01:@.+]](i32* {{[^,]+}})
    #pragma omp target map(lb[:X])
    {
      lb[4] += 1;
    }
    
    // Region 02
    // CK21-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE02]]{{.+}})
    // CK21-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK21-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
    // CK21-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
    // CK21-DAG: [[CBPVAL0]] = bitcast [[ST]]* [[VAR0:%.+]] to i8*
    // CK21-DAG: [[CPVAL0]] = bitcast float** [[SEC0:%.+]] to i8*
    // CK21-DAG: [[SEC0]] = getelementptr {{.*}}[[ST]]* [[VAR0]], i{{.+}} 0, i{{.+}} 2
    
    // CK21-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
    // CK21-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
    // CK21-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
    // CK21-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
    // CK21-DAG: [[CBPVAL1]] = bitcast float** [[SEC0]] to i8*
    // CK21-DAG: [[CPVAL1]] = bitcast float* [[SEC1:%.+]] to i8*
    // CK21-DAG: [[SEC1]] = getelementptr {{.*}}float* [[RVAR1:%[^,]+]], i{{.+}} 123
    // CK21-DAG: [[RVAR1]] = load float*, float** [[SEC1_:%[^,]+]]
    // CK21-DAG: [[SEC1_]] = getelementptr {{.*}}[[ST]]* [[VAR0]], i{{.+}} 0, i{{.+}} 2

    // CK21: call void [[CALL02:@.+]]([[ST]]* {{[^,]+}})
    #pragma omp target map(from:B[X:X+2])
    {
      B[2] += 1.0f;
    }
    
    // Region 03
    // CK21-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}})
    // CK21-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK21-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
    // CK21-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
    // CK21-DAG: [[CBPVAL0]] = bitcast [123 x float]* [[VAR0:%.+]] to i8*
    // CK21-DAG: [[CPVAL0]] = bitcast [123 x float]* [[VAR0]] to i8*

    // CK21: call void [[CALL03:@.+]]([123 x float]* {{[^,]+}})
    #pragma omp target map(from:la)
    {
      la[3] += 1.0f;
    }
    
    // Region 04
    // CK21-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE04]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE04]]{{.+}})
    // CK21-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK21-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
    // CK21-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
    // CK21-DAG: [[CBPVAL0]] = bitcast i32* [[VAR0:%.+]] to i8*
    // CK21-DAG: [[CPVAL0]] = bitcast i32* [[VAR0]] to i8*

    // CK21: call void [[CALL04:@.+]](i32* {{[^,]+}})
    #pragma omp target map(from:arg)
    {
      arg +=1;
    }
    
    // Make sure the extra flag is passed to the second map.
    // Region 05
    // CK21-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE05]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE05]]{{.+}})
    // CK21-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK21-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK21-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK21-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
    // CK21-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
    // CK21-DAG: [[CBPVAL0]] = bitcast [[ST]]* [[VAR0:%.+]] to i8*
    // CK21-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
    // CK21-DAG: [[SEC0]] = getelementptr {{.*}}[[ST]]* [[VAR0]], i{{.+}} 0, i{{.+}} 0

    // CK21-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
    // CK21-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
    // CK21-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
    // CK21-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
    // CK21-DAG: [[CBPVAL1]] = bitcast [[ST]]* [[VAR1:%.+]] to i8*
    // CK21-DAG: [[CPVAL1]] = bitcast i32* [[SEC1:%.+]] to i8*
    // CK21-DAG: [[SEC1]] = getelementptr {{.*}}[[ST]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

    // CK21: call void [[CALL05:@.+]]([[ST]]* {{[^,]+}})
    #pragma omp target map(A, A2)
    {
      A += 1;
      A2 += 1;
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
// RUN: %clang_cc1 -DCK22 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK22 --check-prefix CK22-64
// RUN: %clang_cc1 -DCK22 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK22 --check-prefix CK22-64
// RUN: %clang_cc1 -DCK22 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK22 --check-prefix CK22-32
// RUN: %clang_cc1 -DCK22 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK22 --check-prefix CK22-32
#ifdef CK22

// CK22-DAG: [[ST:%.+]] = type { float }
// CK22-DAG: [[STT:%.+]] = type { i32 }

// CK22: [[SIZE00:@.+]] = private {{.*}}constant [1 x i[[Z:64|32]]] [i[[Z:64|32]] 4]
// CK22: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK22: [[SIZE01:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 400]
// CK22: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK22: [[SIZE02:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] {{8|4}}]
// CK22: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK22: [[SIZE03:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 16]
// CK22: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK22: [[SIZE04:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 20]
// CK22: [[MTYPE04:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK22: [[SIZE05:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 4]
// CK22: [[MTYPE05:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK22: [[SIZE06:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 400]
// CK22: [[MTYPE06:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK22: [[SIZE07:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] {{8|4}}]
// CK22: [[MTYPE07:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK22: [[SIZE08:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 16]
// CK22: [[MTYPE08:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK22: [[SIZE09:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 20]
// CK22: [[MTYPE09:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK22: [[SIZE10:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 4]
// CK22: [[MTYPE10:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK22: [[SIZE11:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 400]
// CK22: [[MTYPE11:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK22: [[SIZE12:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] {{8|4}}]
// CK22: [[MTYPE12:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK22: [[SIZE13:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 16]
// CK22: [[MTYPE13:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK22: [[SIZE14:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 20]
// CK22: [[MTYPE14:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

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

// CK22-LABEL: explicit_maps_globals
int explicit_maps_globals(void){
  // Region 00
  // CK22-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}})
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: store i8* bitcast (i32* @a to i8*), i8** [[BP0]]
  // CK22-DAG: store i8* bitcast (i32* @a to i8*), i8** [[P0]]

  // CK22: call void [[CALL00:@.+]](i32* {{[^,]+}})
  #pragma omp target map(a)
  { a+=1; }
  
  // Region 01
  // CK22-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}})
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: store i8* bitcast ([100 x i32]* @c to i8*), i8** [[BP0]]
  // CK22-DAG: store i8* bitcast ([100 x i32]* @c to i8*), i8** [[P0]]

  // CK22: call void [[CALL01:@.+]]([100 x i32]* {{[^,]+}})
  #pragma omp target map(c)
  { c[3]+=1; }
  
  // Region 02
  // CK22-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}})
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: store i8* bitcast (i32** @d to i8*), i8** [[BP0]]
  // CK22-DAG: store i8* bitcast (i32** @d to i8*), i8** [[P0]]

  // CK22: call void [[CALL02:@.+]](i32** {{[^,]+}})
  #pragma omp target map(d)
  { d[3]+=1; }
    
  // Region 03
  // CK22-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}})
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: store i8* bitcast ([100 x i32]* @c to i8*), i8** [[BP0]]
  // CK22-DAG: store i8* bitcast (i32* getelementptr inbounds ([100 x i32], [100 x i32]* @c, i{{.+}} 0, i{{.+}} 1) to i8*), i8** [[P0]]

  // CK22: call void [[CALL03:@.+]]([100 x i32]* {{[^,]+}})
  #pragma omp target map(c[1:4])
  { c[3]+=1; }
  
  // Region 04
  // CK22-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE04]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE04]]{{.+}})
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK22-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK22-DAG: [[CBPVAL0]] = bitcast i32* [[RVAR0:%.+]] to i8*
  // CK22-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK22-DAG: [[RVAR0]] = load i32*, i32** @d
  // CK22-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 2
  // CK22-DAG: [[RVAR00]] = load i32*, i32** @d

  // CK22: call void [[CALL04:@.+]](i32* {{[^,]+}})
  #pragma omp target map(d[2:5])
  { d[3]+=1; }
  
  // Region 05
  // CK22-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE05]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE05]]{{.+}})
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: store i8* bitcast ([[ST]]* @sa to i8*), i8** [[BP0]]
  // CK22-DAG: store i8* bitcast ([[ST]]* @sa to i8*), i8** [[P0]]

  // CK22: call void [[CALL05:@.+]]([[ST]]* {{[^,]+}})
  #pragma omp target map(sa)
  { sa.fa+=1; }
  
  // Region 06
  // CK22-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE06]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE06]]{{.+}})
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: store i8* bitcast ([100 x [[ST]]]* @sc to i8*), i8** [[BP0]]
  // CK22-DAG: store i8* bitcast ([100 x [[ST]]]* @sc to i8*), i8** [[P0]]

  // CK22: call void [[CALL06:@.+]]([100 x [[ST]]]* {{[^,]+}})
  #pragma omp target map(sc)
  { sc[3].fa+=1; }
  
  // Region 07
  // CK22-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE07]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE07]]{{.+}})
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: store i8* bitcast ([[ST]]** @sd to i8*), i8** [[BP0]]
  // CK22-DAG: store i8* bitcast ([[ST]]** @sd to i8*), i8** [[P0]]

  // CK22: call void [[CALL07:@.+]]([[ST]]** {{[^,]+}})
  #pragma omp target map(sd)
  { sd[3].fa+=1; }
  
  // Region 08
  // CK22-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE08]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE08]]{{.+}})
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: store i8* bitcast ([100 x [[ST]]]* @sc to i8*), i8** [[BP0]]
  // CK22-DAG: store i8* bitcast ([[ST]]* getelementptr inbounds ([100 x [[ST]]], [100 x [[ST]]]* @sc, i{{.+}} 0, i{{.+}} 1) to i8*), i8** [[P0]]

  // CK22: call void [[CALL08:@.+]]([100 x [[ST]]]* {{[^,]+}})
  #pragma omp target map(sc[1:4])
  { sc[3].fa+=1; }
  
  // Region 09
  // CK22-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE09]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE09]]{{.+}})
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK22-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK22-DAG: [[CBPVAL0]] = bitcast [[ST]]* [[RVAR0:%.+]] to i8*
  // CK22-DAG: [[CPVAL0]] = bitcast [[ST]]* [[SEC0:%.+]] to i8*
  // CK22-DAG: [[RVAR0]] = load [[ST]]*, [[ST]]** @sd
  // CK22-DAG: [[SEC0]] = getelementptr {{.*}}[[ST]]* [[RVAR00:%.+]], i{{.+}} 2
  // CK22-DAG: [[RVAR00]] = load [[ST]]*, [[ST]]** @sd

  // CK22: call void [[CALL09:@.+]]([[ST]]* {{[^,]+}})
  #pragma omp target map(sd[2:5])
  { sd[3].fa+=1; }
  
  // Region 10
  // CK22-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE10]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE10]]{{.+}})
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: store i8* bitcast ([[STT]]* @sta to i8*), i8** [[BP0]]
  // CK22-DAG: store i8* bitcast ([[STT]]* @sta to i8*), i8** [[P0]]

  // CK22: call void [[CALL10:@.+]]([[STT]]* {{[^,]+}})
  #pragma omp target map(sta)
  { sta.fa+=1; }
  
  // Region 11
  // CK22-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE11]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE11]]{{.+}})
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: store i8* bitcast ([100 x [[STT]]]* @stc to i8*), i8** [[BP0]]
  // CK22-DAG: store i8* bitcast ([100 x [[STT]]]* @stc to i8*), i8** [[P0]]

  // CK22: call void [[CALL11:@.+]]([100 x [[STT]]]* {{[^,]+}})
  #pragma omp target map(stc)
  { stc[3].fa+=1; }
  
  // Region 12
  // CK22-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE12]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE12]]{{.+}})
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: store i8* bitcast ([[STT]]** @std to i8*), i8** [[BP0]]
  // CK22-DAG: store i8* bitcast ([[STT]]** @std to i8*), i8** [[P0]]

  // CK22: call void [[CALL12:@.+]]([[STT]]** {{[^,]+}})
  #pragma omp target map(std)
  { std[3].fa+=1; }
  
  // Region 13
  // CK22-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE13]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE13]]{{.+}})
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: store i8* bitcast ([100 x [[STT]]]* @stc to i8*), i8** [[BP0]]
  // CK22-DAG: store i8* bitcast ([[STT]]* getelementptr inbounds ([100 x [[STT]]], [100 x [[STT]]]* @stc, i{{.+}} 0, i{{.+}} 1) to i8*), i8** [[P0]]

  // CK22: call void [[CALL13:@.+]]([100 x [[STT]]]* {{[^,]+}})
  #pragma omp target map(stc[1:4])
  { stc[3].fa+=1; }
  
  // Region 14
  // CK22-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE14]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE14]]{{.+}})
  // CK22-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK22-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK22-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK22-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK22-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK22-DAG: [[CBPVAL0]] = bitcast [[STT]]* [[RVAR0:%.+]] to i8*
  // CK22-DAG: [[CPVAL0]] = bitcast [[STT]]* [[SEC0:%.+]] to i8*
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
// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK23 --check-prefix CK23-64
// RUN: %clang_cc1 -std=c++11 -DCK23 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK23 --check-prefix CK23-32
// RUN: %clang_cc1 -std=c++11 -DCK23 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK23 --check-prefix CK23-32
#ifdef CK23

// CK23: [[SIZE00:@.+]] = private {{.*}}constant [1 x i[[Z:64|32]]] [i[[Z:64|32]] 4]
// CK23: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK23: [[SIZE01:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 4]
// CK23: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK23: [[SIZE02:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 400]
// CK23: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK23: [[SIZE03:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] {{8|4}}]
// CK23: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK23: [[SIZE04:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 16]
// CK23: [[MTYPE04:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK23: [[SIZE05:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 16]
// CK23: [[MTYPE05:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK23-LABEL: explicit_maps_inside_captured
int explicit_maps_inside_captured(int a){
  float b;
  float c[100];
  float *d;
  
  // CK23: call void @{{.*}}explicit_maps_inside_captured{{.*}}([[SA:%.+]]* {{.*}}) 
  // CK23: define {{.*}}explicit_maps_inside_captured{{.*}}
  [&](void){
    // Region 00
    // CK23-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}})
    // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
    // CK23-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
    // CK23-DAG: [[CBPVAL0]] = bitcast i32* [[VAR0:%.+]] to i8*
    // CK23-DAG: [[CPVAL0]] = bitcast i32* [[VAR00:%.+]] to i8*
    // CK23-DAG: [[VAR0]] = load i32*, i32** [[CAP0:%[^,]+]]
    // CK23-DAG: [[CAP0]] = getelementptr inbounds [[SA]], [[SA]]
    // CK23-DAG: [[VAR00]] = load i32*, i32** [[CAP00:%[^,]+]]
    // CK23-DAG: [[CAP00]] = getelementptr inbounds [[SA]], [[SA]]

    // CK23: call void [[CALL00:@.+]](i32* {{[^,]+}})
    #pragma omp target map(a)
      { a+=1; }
    // Region 01
    // CK23-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}})
    // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
    // CK23-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
    // CK23-DAG: [[CBPVAL0]] = bitcast float* [[VAR0:%.+]] to i8*
    // CK23-DAG: [[CPVAL0]] = bitcast float* [[VAR00:%.+]] to i8*
    // CK23-DAG: [[VAR0]] = load float*, float** [[CAP0:%[^,]+]]
    // CK23-DAG: [[CAP0]] = getelementptr inbounds [[SA]], [[SA]]
    // CK23-DAG: [[VAR00]] = load float*, float** [[CAP00:%[^,]+]]
    // CK23-DAG: [[CAP00]] = getelementptr inbounds [[SA]], [[SA]]

    // CK23: call void [[CALL01:@.+]](float* {{[^,]+}})
    #pragma omp target map(b)
      { b+=1; }
    // Region 02
    // CK23-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}})
    // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
    // CK23-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
    // CK23-DAG: [[CBPVAL0]] = bitcast [100 x float]* [[VAR0:%.+]] to i8*
    // CK23-DAG: [[CPVAL0]] = bitcast [100 x float]* [[VAR00:%.+]] to i8*
    // CK23-DAG: [[VAR0]] = load [100 x float]*, [100 x float]** [[CAP0:%[^,]+]]
    // CK23-DAG: [[CAP0]] = getelementptr inbounds [[SA]], [[SA]]
    // CK23-DAG: [[VAR00]] = load [100 x float]*, [100 x float]** [[CAP00:%[^,]+]]
    // CK23-DAG: [[CAP00]] = getelementptr inbounds [[SA]], [[SA]]

    // CK23: call void [[CALL02:@.+]]([100 x float]* {{[^,]+}})
    #pragma omp target map(c)
      { c[3]+=1; }
      
    // Region 03
    // CK23-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}})
    // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
    // CK23-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
    // CK23-DAG: [[CBPVAL0]] = bitcast float** [[VAR0:%.+]] to i8*
    // CK23-DAG: [[CPVAL0]] = bitcast float** [[VAR00:%.+]] to i8*
    // CK23-DAG: [[VAR0]] = load float**, float*** [[CAP0:%[^,]+]]
    // CK23-DAG: [[CAP0]] = getelementptr inbounds [[SA]], [[SA]]
    // CK23-DAG: [[VAR00]] = load float**, float*** [[CAP00:%[^,]+]]
    // CK23-DAG: [[CAP00]] = getelementptr inbounds [[SA]], [[SA]]

    // CK23: call void [[CALL03:@.+]](float** {{[^,]+}})
    #pragma omp target map(d)
      { d[3]+=1; }
    // Region 04
    // CK23-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE04]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE04]]{{.+}})
    // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
    // CK23-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
    // CK23-DAG: [[CBPVAL0]] = bitcast [100 x float]* [[VAR0:%.+]] to i8*
    // CK23-DAG: [[CPVAL0]] = bitcast float* [[SEC0:%.+]] to i8*
    // CK23-DAG: [[SEC0]] = getelementptr {{.*}}[100 x float]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 2
    // CK23-DAG: [[VAR0]] = load [100 x float]*, [100 x float]** [[CAP0:%[^,]+]]
    // CK23-DAG: [[CAP0]] = getelementptr inbounds [[SA]], [[SA]]
    // CK23-DAG: [[VAR00]] = load [100 x float]*, [100 x float]** [[CAP00:%[^,]+]]
    // CK23-DAG: [[CAP00]] = getelementptr inbounds [[SA]], [[SA]]
    
    // CK23: call void [[CALL04:@.+]]([100 x float]* {{[^,]+}})
    #pragma omp target map(c[2:4])
      { c[3]+=1; }
      
    // Region 05
    // CK23-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE05]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE05]]{{.+}})
    // CK23-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK23-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK23-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK23-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
    // CK23-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
    // CK23-DAG: [[CBPVAL0]] = bitcast float* [[RVAR0:%.+]] to i8*
    // CK23-DAG: [[CPVAL0]] = bitcast float* [[SEC0:%.+]] to i8*
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
// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK24 --check-prefix CK24-64
// RUN: %clang_cc1 -DCK24 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK24 --check-prefix CK24-32
// RUN: %clang_cc1 -DCK24 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK24 --check-prefix CK24-32
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

// CK24: [[SIZE01:@.+]] = private {{.*}}constant [1 x i[[Z:64|32]]] [i[[Z:64|32]] 4]
// CK24: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK24: [[SIZE02:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] {{56|48}}]
// CK24: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK24: [[SIZE03:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 4]
// CK24: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK24: [[SIZE04:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 20]
// CK24: [[MTYPE04:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK24: [[SIZE05:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] {{3560|2880}}]
// CK24: [[MTYPE05:@.+]] = private {{.*}}constant [2 x i32] [i32 35, i32 19]

// CK24: [[SIZE06:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 4]
// CK24: [[MTYPE06:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK24: [[SIZE07:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] 4]
// CK24: [[MTYPE07:@.+]] = private {{.*}}constant [2 x i32] [i32 35, i32 19]

// CK24: [[SIZE08:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] 4]
// CK24: [[MTYPE08:@.+]] = private {{.*}}constant [2 x i32] [i32 35, i32 19]

// CK24: [[SIZE09:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] 4]
// CK24: [[MTYPE09:@.+]] = private {{.*}}constant [2 x i32] [i32 35, i32 19]

// CK24: [[SIZE10:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 8]
// CK24: [[MTYPE10:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK24: [[SIZE11:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] {{8|4}}]
// CK24: [[MTYPE11:@.+]] = private {{.*}}constant [2 x i32] [i32 35, i32 19]

// CK24: [[SIZE12:@.+]] = private {{.*}}constant [4 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] {{8|4}}, i[[Z]] {{8|4}}, i[[Z]] 4]
// CK24: [[MTYPE12:@.+]] = private {{.*}}constant [4 x i32] [i32 35, i32 19, i32 19, i32 19]

// CK24: [[SIZE13:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 4]
// CK24: [[MTYPE13:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK24: [[SIZE14:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] {{56|48}}]
// CK24: [[MTYPE14:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK24: [[SIZE15:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 4]
// CK24: [[MTYPE15:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK24: [[SIZE16:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 20]
// CK24: [[MTYPE16:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK24: [[SIZE17:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] {{3560|2880}}]
// CK24: [[MTYPE17:@.+]] = private {{.*}}constant [2 x i32] [i32 35, i32 19]

// CK24: [[SIZE18:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 4]
// CK24: [[MTYPE18:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK24: [[SIZE19:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] 4]
// CK24: [[MTYPE19:@.+]] = private {{.*}}constant [2 x i32] [i32 35, i32 19]

// CK24: [[SIZE20:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] 4]
// CK24: [[MTYPE20:@.+]] = private {{.*}}constant [2 x i32] [i32 35, i32 19]

// CK24: [[SIZE21:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] 4]
// CK24: [[MTYPE21:@.+]] = private {{.*}}constant [2 x i32] [i32 35, i32 19]

// CK24: [[SIZE22:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] {{8|4}}]
// CK24: [[MTYPE22:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK24: [[SIZE23:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] {{8|4}}]
// CK24: [[MTYPE23:@.+]] = private {{.*}}constant [2 x i32] [i32 35, i32 19]

// CK24: [[SIZE24:@.+]] = private {{.*}}constant [4 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] {{8|4}}, i[[Z]] {{8|4}}, i[[Z]] 4]
// CK24: [[MTYPE24:@.+]] = private {{.*}}constant [4 x i32] [i32 35, i32 19, i32 19, i32 19]

// CK24-LABEL: explicit_maps_struct_fields
int explicit_maps_struct_fields(int a){
  SC s;
  SC *p;

// Region 01
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 0

// CK24: call void [[CALL01:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(s.a)
  { s.a++; }
  
// Region 02
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast [[SA]]* [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SB]]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

// CK24: call void [[CALL02:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(s.s.s)
  { s.a++; }
  
// Region 03
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SA]]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SB]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

// CK24: call void [[CALL03:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(s.s.s.a)
  { s.a++; }

// Region 04
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE04]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE04]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[10 x i32]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 3

// CK24: call void [[CALL04:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(s.b[:5])
  { s.a++; }
  
// Region 05
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE05]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE05]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast [[SB]]** [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
// CK24-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
// CK24-DAG: [[CBPVAL1]] = bitcast [[SB]]** [[SEC0]] to i8*
// CK24-DAG: [[CPVAL1]] = bitcast [[SB]]* [[SEC1:%.+]] to i8*
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[[SB]]* [[SEC11:%[^,]+]], i{{.+}} 0
// CK24-DAG: [[SEC11]] = load [[SB]]*, [[SB]]** [[SEC111:%[^,]+]],
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 2

// CK24: call void [[CALL05:@.+]]([[SC]]* {{[^,]+}})  
#pragma omp target map(s.p[:5])
  { s.a++; }

// Region 06
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE06]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE06]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SA]]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[10 x [[SA]]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}[[SB]]* [[SEC0000:%[^,]+]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[SEC0000]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

// CK24: call void [[CALL06:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(s.s.sa[3].a)
  { s.a++; }
  
// Region 07
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE07]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE07]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast [[SA]]** [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[10 x [[SA]]*]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SB]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
// CK24-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
// CK24-DAG: [[CBPVAL1]] = bitcast [[SA]]** [[SEC0]] to i8*
// CK24-DAG: [[CPVAL1]] = bitcast i32* [[SEC1:%.+]] to i8*
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[[SA]]* [[SEC11:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC11]] = load [[SA]]*, [[SA]]** [[SEC111:%[^,]+]],
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}[10 x [[SA]]*]* [[SEC1111:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC1111]] = getelementptr {{.*}}[[SB]]* [[SEC11111:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC11111]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

// CK24: call void [[CALL07:@.+]]([[SC]]* {{[^,]+}}) 
#pragma omp target map(s.s.sp[3]->a)
  { s.a++; }

// Region 08
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE08]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE08]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast [[SB]]** [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
// CK24-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
// CK24-DAG: [[CBPVAL1]] = bitcast [[SB]]** [[SEC0]] to i8*
// CK24-DAG: [[CPVAL1]] = bitcast i32* [[SEC1:%.+]] to i8*
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[[SB]]* [[SEC11:%[^,]+]], i{{.+}} 0
// CK24-DAG: [[SEC11]] = load [[SB]]*, [[SB]]** [[SEC111:%[^,]+]],
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 2

// CK24: call void [[CALL08:@.+]]([[SC]]* {{[^,]+}}) 
#pragma omp target map(s.p->a)
  { s.a++; }
  
// Region 09
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE09]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE09]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast [[SA]]** [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SB]]* [[SEC00:[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
// CK24-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
// CK24-DAG: [[CBPVAL1]] = bitcast [[SA]]** [[SEC0]] to i8*
// CK24-DAG: [[CPVAL1]] = bitcast i32* [[SEC1:%.+]] to i8*
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[[SA]]* [[SEC11:%[^,]+]], i{{.+}} 0
// CK24-DAG: [[SEC11]] = load [[SA]]*, [[SA]]** [[SEC111:%[^,]+]],
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}[[SB]]* [[SEC1111:[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC1111]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

// CK24: call void [[CALL09:@.+]]([[SC]]* {{[^,]+}}) 
#pragma omp target map(s.s.p->a)
  { s.a++; }

// Region 10
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE10]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE10]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[10 x i32]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SA]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}[[SB]]* [[SEC0000:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC0000]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

// CK24: call void [[CALL10:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(s.s.s.b[:2])
  { s.a++; }
  
// Region 11
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE11]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE11]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast [[SA]]** [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SB]]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
// CK24-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
// CK24-DAG: [[CBPVAL1]] = bitcast [[SA]]** [[SEC0]] to i8*
// CK24-DAG: [[CPVAL1]] = bitcast i32* [[SEC1:%.+]] to i8*
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[10 x i32]* [[SEC11:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC11]] = getelementptr {{.*}}[[SA]]* [[SEC111:%[^,]+]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[SEC111]] = load [[SA]]*, [[SA]]** [[SEC1111:%[^,]+]],
// CK24-DAG: [[SEC1111]] = getelementptr {{.*}}[[SB]]* [[SEC11111:%[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC11111]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 1

// CK24: call void [[CALL11:@.+]]([[SC]]* {{[^,]+}}) 
#pragma omp target map(s.s.p->b[:2])
  { s.a++; }

// Region 12
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 4, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[4 x i{{.+}}]* [[SIZE12]], {{.+}}getelementptr {{.+}}[4 x i{{.+}}]* [[MTYPE12]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast [[SB]]** [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
// CK24-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
// CK24-DAG: [[CBPVAL1]] = bitcast [[SB]]** [[SEC0]] to i8*
// CK24-DAG: [[CPVAL1]] = bitcast [[SA]]** [[SEC1:%.+]] to i8*
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[[SB]]* [[SEC11:%[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC11]] = load [[SB]]*, [[SB]]** [[SEC111:%[^,]+]],
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: store i8* [[CBPVAL2:%[^,]+]], i8** [[BP2]]
// CK24-DAG: store i8* [[CPVAL2:%[^,]+]], i8** [[P2]]
// CK24-DAG: [[CBPVAL2]] = bitcast [[SA]]** [[SEC1]] to i8*
// CK24-DAG: [[CPVAL2]] = bitcast [[SA]]** [[SEC2:%.+]] to i8*
// CK24-DAG: [[SEC2]] = getelementptr {{.*}}[[SA]]* [[SEC22:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC22]] = load [[SA]]*, [[SA]]** [[SEC222:%[^,]+]],
// CK24-DAG: [[SEC222]] = getelementptr {{.*}}[[SB]]* [[SEC2222:%[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC2222]] = load [[SB]]*, [[SB]]** [[SEC22222:%[^,]+]],
// CK24-DAG: [[SEC22222]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: store i8* [[CBPVAL3:%[^,]+]], i8** [[BP3]]
// CK24-DAG: store i8* [[CPVAL3:%[^,]+]], i8** [[P3]]
// CK24-DAG: [[CBPVAL3]] = bitcast [[SA]]** [[SEC2]] to i8*
// CK24-DAG: [[CPVAL3]] = bitcast i32* [[SEC3:%.+]] to i8*
// CK24-DAG: [[SEC3]] = getelementptr {{.*}}[[SA]]* [[SEC33:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC33]] = load [[SA]]*, [[SA]]** [[SEC333:%[^,]+]],
// CK24-DAG: [[SEC333]] = getelementptr {{.*}}[[SA]]* [[SEC3333:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC3333]] = load [[SA]]*, [[SA]]** [[SEC33333:%[^,]+]],
// CK24-DAG: [[SEC33333]] = getelementptr {{.*}}[[SB]]* [[SEC333333:%[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC333333]] = load [[SB]]*, [[SB]]** [[SEC3333333:%[^,]+]],
// CK24-DAG: [[SEC3333333]] = getelementptr {{.*}}[[SC]]* [[VAR0]], i{{.+}} 0, i{{.+}} 2

// CK24: call void [[CALL12:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(s.p->p->p->a)
  { s.a++; }

//
// Same thing but starting from a pointer.
//
// Region 13
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE13]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE13]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 0

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL13:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->a)
  { p->a++; }
  
// Region 14
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE14]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE14]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast [[SA]]* [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SB]]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL14:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->s.s)
  { p->a++; }
  
// Region 15
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE15]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE15]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SA]]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SB]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL15:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->s.s.a)
  { p->a++; }

// Region 16
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE16]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE16]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[10 x i32]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 3

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL16:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->b[:5])
  { p->a++; }
  
// Region 17
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE17]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE17]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast [[SB]]** [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
// CK24-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
// CK24-DAG: [[CBPVAL1]] = bitcast [[SB]]** [[SEC0]] to i8*
// CK24-DAG: [[CPVAL1]] = bitcast [[SB]]* [[SEC1:%.+]] to i8*
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
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE18]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE18]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SA]]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[10 x [[SA]]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}[[SB]]* [[SEC0000:%[^,]+]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[SEC0000]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL18:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->s.sa[3].a)
  { p->a++; }
  
// Region 19
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE19]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE19]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast [[SA]]** [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[10 x [[SA]]*]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SB]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
// CK24-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
// CK24-DAG: [[CBPVAL1]] = bitcast [[SA]]** [[SEC0]] to i8*
// CK24-DAG: [[CPVAL1]] = bitcast i32* [[SEC1:%.+]] to i8*
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
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE20]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE20]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast [[SB]]** [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
// CK24-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
// CK24-DAG: [[CBPVAL1]] = bitcast [[SB]]** [[SEC0]] to i8*
// CK24-DAG: [[CPVAL1]] = bitcast i32* [[SEC1:%.+]] to i8*
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
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE21]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE21]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast [[SA]]** [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SB]]* [[SEC00:[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
// CK24-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
// CK24-DAG: [[CBPVAL1]] = bitcast [[SA]]** [[SEC0]] to i8*
// CK24-DAG: [[CPVAL1]] = bitcast i32* [[SEC1:%.+]] to i8*
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
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE22]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE22]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[10 x i32]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SA]]* [[SEC000:%[^,]+]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[SEC000]] = getelementptr {{.*}}[[SB]]* [[SEC0000:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC0000]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[VAR0]] = load [[SC]]*, [[SC]]** %{{.+}}
// CK24-DAG: [[VAR00]] = load [[SC]]*, [[SC]]** %{{.+}}

// CK24: call void [[CALL22:@.+]]([[SC]]* {{[^,]+}})
#pragma omp target map(p->s.s.b[:2])
  { p->a++; }
  
// Region 23
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE23]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE23]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast [[SA]]** [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SB]]* [[SEC00:%[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC00]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 1

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
// CK24-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
// CK24-DAG: [[CBPVAL1]] = bitcast [[SA]]** [[SEC0]] to i8*
// CK24-DAG: [[CPVAL1]] = bitcast i32* [[SEC1:%.+]] to i8*
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
// CK24-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 4, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[4 x i{{.+}}]* [[SIZE24]], {{.+}}getelementptr {{.+}}[4 x i{{.+}}]* [[MTYPE24]]{{.+}})
// CK24-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
// CK24-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

// CK24-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
// CK24-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
// CK24-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
// CK24-DAG: [[CBPVAL0]] = bitcast [[SC]]* [[VAR0:%.+]] to i8*
// CK24-DAG: [[CPVAL0]] = bitcast [[SB]]** [[SEC0:%.+]] to i8*
// CK24-DAG: [[SEC0]] = getelementptr {{.*}}[[SC]]* [[VAR00:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
// CK24-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
// CK24-DAG: [[CBPVAL1]] = bitcast [[SB]]** [[SEC0]] to i8*
// CK24-DAG: [[CPVAL1]] = bitcast [[SA]]** [[SEC1:%.+]] to i8*
// CK24-DAG: [[SEC1]] = getelementptr {{.*}}[[SB]]* [[SEC11:%[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC11]] = load [[SB]]*, [[SB]]** [[SEC111:%[^,]+]],
// CK24-DAG: [[SEC111]] = getelementptr {{.*}}[[SC]]* [[VAR000:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
// CK24-DAG: store i8* [[CBPVAL2:%[^,]+]], i8** [[BP2]]
// CK24-DAG: store i8* [[CPVAL2:%[^,]+]], i8** [[P2]]
// CK24-DAG: [[CBPVAL2]] = bitcast [[SA]]** [[SEC1]] to i8*
// CK24-DAG: [[CPVAL2]] = bitcast [[SA]]** [[SEC2:%.+]] to i8*
// CK24-DAG: [[SEC2]] = getelementptr {{.*}}[[SA]]* [[SEC22:%[^,]+]], i{{.+}} 0, i{{.+}} 1
// CK24-DAG: [[SEC22]] = load [[SA]]*, [[SA]]** [[SEC222:%[^,]+]],
// CK24-DAG: [[SEC222]] = getelementptr {{.*}}[[SB]]* [[SEC2222:%[^,]+]], i{{.+}} 0, i{{.+}} 4
// CK24-DAG: [[SEC2222]] = load [[SB]]*, [[SB]]** [[SEC22222:%[^,]+]],
// CK24-DAG: [[SEC22222]] = getelementptr {{.*}}[[SC]]* [[VAR0000:%.+]], i{{.+}} 0, i{{.+}} 2

// CK24-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
// CK24-DAG: store i8* [[CBPVAL3:%[^,]+]], i8** [[BP3]]
// CK24-DAG: store i8* [[CPVAL3:%[^,]+]], i8** [[P3]]
// CK24-DAG: [[CBPVAL3]] = bitcast [[SA]]** [[SEC2]] to i8*
// CK24-DAG: [[CPVAL3]] = bitcast i32* [[SEC3:%.+]] to i8*
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
// CK24: define {{.+}}[[CALL02]]
// CK24: define {{.+}}[[CALL03]]
// CK24: define {{.+}}[[CALL04]]
// CK24: define {{.+}}[[CALL05]]
// CK24: define {{.+}}[[CALL06]]
// CK24: define {{.+}}[[CALL07]]
// CK24: define {{.+}}[[CALL08]]
// CK24: define {{.+}}[[CALL09]]
// CK24: define {{.+}}[[CALL10]]
// CK24: define {{.+}}[[CALL11]]
// CK24: define {{.+}}[[CALL12]]
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
// RUN: %clang_cc1 -DCK25 -std=c++11 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK25 --check-prefix CK25-64
// RUN: %clang_cc1 -DCK25 -std=c++11 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK25 --check-prefix CK25-64
// RUN: %clang_cc1 -DCK25 -std=c++11 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK25 --check-prefix CK25-32
// RUN: %clang_cc1 -DCK25 -std=c++11 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK25 --check-prefix CK25-32
#ifdef CK25
// CK25: [[ST:%.+]] = type { i32, float }
// CK25: [[CA00:%.+]] = type { [[ST]]* }
// CK25: [[CA01:%.+]] = type { i32* }

// CK25: [[SIZE00:@.+]] = private {{.*}}constant [1 x i[[Z:64|32]]] [i[[Z:64|32]] 4]
// CK25: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i32] [i32 33]

// CK25: [[SIZE01:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 4]
// CK25: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i32] [i32 33]

// CK25-LABEL: explicit_maps_with_inner_lambda

template <int X, typename T>
struct CC {
  T A;
  float B;

  int foo(T arg) {
    // Region 00
    // CK25-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}})
    // CK25-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK25-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK25-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK25-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK25-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
    // CK25-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
    // CK25-DAG: [[CBPVAL0]] = bitcast [[ST]]* [[VAR0:%.+]] to i8*
    // CK25-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
    // CK25-DAG: [[SEC0]] = getelementptr {{.*}}[[ST]]* [[VAR0:%.+]], i{{.+}} 0, i{{.+}} 0

    // CK25: call void [[CALL00:@.+]]([[ST]]* {{[^,]+}})
    #pragma omp target map(to:A)
    {
      [&]() {
        A += 1;
      }();
    }

    // Region 01
    // CK25-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}})
    // CK25-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK25-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK25-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK25-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK25-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
    // CK25-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
    // CK25-DAG: [[CBPVAL0]] = bitcast i32* [[VAR0:%.+]] to i8*
    // CK25-DAG: [[CPVAL0]] = bitcast i32* [[VAR0]] to i8*

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
// CK25: call void {{.*}}[[LAMBDA]]{{.*}}([[CA01]]* [[CA]])
#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK26 -std=c++11 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK26 --check-prefix CK26-64
// RUN: %clang_cc1 -DCK26 -std=c++11 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK26 --check-prefix CK26-64
// RUN: %clang_cc1 -DCK26 -std=c++11 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK26 --check-prefix CK26-32
// RUN: %clang_cc1 -DCK26 -std=c++11 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -std=c++11 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK26 --check-prefix CK26-32
#ifdef CK26
// CK26: [[ST:%.+]] = type { i32, float*, i32, float* }

// CK26: [[SIZE00:@.+]] = private {{.*}}constant [2 x i[[Z:64|32]]] [i[[Z:64|32]] {{32|16}}, i[[Z:64|32]] 4]
// CK26: [[MTYPE00:@.+]] = private {{.*}}constant [2 x i32] [i32 35, i32 35]

// CK26: [[SIZE01:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] {{32|16}}, i[[Z]] 4]
// CK26: [[MTYPE01:@.+]] = private {{.*}}constant [2 x i32] [i32 35, i32 35]

// CK26: [[SIZE02:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] {{32|16}}, i[[Z]] 4]
// CK26: [[MTYPE02:@.+]] = private {{.*}}constant [2 x i32] [i32 35, i32 35]

// CK26: [[SIZE03:@.+]] = private {{.*}}constant [2 x i[[Z]]] [i[[Z]] {{32|16}}, i[[Z]] 4]
// CK26: [[MTYPE03:@.+]] = private {{.*}}constant [2 x i32] [i32 35, i32 35]

// CK26-LABEL: explicit_maps_with_private_class_members

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
      // CK26-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE00]]{{.+}})
      // CK26-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
      // CK26-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

      // CK26-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
      // CK26-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
      // CK26-DAG: [[CBPVAL0]] = bitcast [[ST]]* [[VAR0:%.+]] to i8*
      // CK26-DAG: [[CPVAL0]] = bitcast [[ST]]* [[VAR0]] to i8*

      // CK26-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
      // CK26-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
      // CK26-DAG: [[CBPVAL1]] = bitcast i32* [[VAR1:%.+]] to i8*
      // CK26-DAG: [[CPVAL1]] = bitcast i32* [[SEC1:%.+]] to i8*
      // CK26-DAG: [[VAR1]] = load i32*, i32** [[PVT:%.+]],
      // CK26-DAG: [[SEC1]] = load i32*, i32** [[PVT]],

      // CK26: call void [[CALL00:@.+]]([[ST]]* {{[^,]+}}, i32* {{[^,]+}})
      #pragma omp target map(fA)
      {
        ++fA;
      }

      // Region 01
      // CK26-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE01]]{{.+}})
      // CK26-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
      // CK26-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

      // CK26-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
      // CK26-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
      // CK26-DAG: [[CBPVAL0]] = bitcast [[ST]]* [[VAR0:%.+]] to i8*
      // CK26-DAG: [[CPVAL0]] = bitcast [[ST]]* [[VAR0]] to i8*

      // CK26-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
      // CK26-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
      // CK26-DAG: [[CBPVAL1]] = bitcast float* [[VAR1:%.+]] to i8*
      // CK26-DAG: [[CPVAL1]] = bitcast float* [[SEC1:%.+]] to i8*
      // CK26-DAG: [[VAR1]] = load float*, float** [[PVT:%.+]],
      // CK26-DAG: [[SEC1]] = load float*, float** [[PVT]],

      // CK26: call void [[CALL01:@.+]]([[ST]]* {{[^,]+}}, float* {{[^,]+}})
      #pragma omp target map(fB)
      {
        fB += 1.0;
      }

      // Region 02
      // CK26-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE02]]{{.+}})
      // CK26-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
      // CK26-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

      // CK26-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
      // CK26-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
      // CK26-DAG: [[CBPVAL0]] = bitcast [[ST]]* [[VAR0:%.+]] to i8*
      // CK26-DAG: [[CPVAL0]] = bitcast [[ST]]* [[VAR0]] to i8*

      // CK26-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
      // CK26-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
      // CK26-DAG: [[CBPVAL1]] = bitcast i32* [[VAR1:%.+]] to i8*
      // CK26-DAG: [[CPVAL1]] = bitcast i32* [[SEC1:%.+]] to i8*
      // CK26-DAG: [[VAR1]] = load i32*, i32** [[PVT:%.+]],
      // CK26-DAG: [[SEC1]] = load i32*, i32** [[PVT]],

      // CK26: call void [[CALL02:@.+]]([[ST]]* {{[^,]+}}, i32* {{[^,]+}})
      #pragma omp target map(pA)
      {
        ++pA;
      }

      // Region 01
      // CK26-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 2, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[2 x i{{.+}}]* [[MTYPE03]]{{.+}})
      // CK26-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
      // CK26-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

      // CK26-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
      // CK26-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
      // CK26-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
      // CK26-DAG: [[CBPVAL0]] = bitcast [[ST]]* [[VAR0:%.+]] to i8*
      // CK26-DAG: [[CPVAL0]] = bitcast [[ST]]* [[VAR0]] to i8*

      // CK26-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
      // CK26-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
      // CK26-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
      // CK26-DAG: [[CBPVAL1]] = bitcast float* [[VAR1:%.+]] to i8*
      // CK26-DAG: [[CPVAL1]] = bitcast float* [[SEC1:%.+]] to i8*
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
// CK26: [[VAL:%.+]] = load i32, i32* [[ADDR]],
// CK26: add nsw i32 [[VAL]], 1

// CK26: define {{.+}}[[CALL01]]({{.*}}float*{{.*}}[[PVTARG:%.+]])
// CK26: store float* [[PVTARG]], float** [[PVTADDR:%.+]],
// CK26: [[ADDR:%.+]] = load float*, float** [[PVTADDR]],
// CK26: [[VAL:%.+]] = load float, float* [[ADDR]],
// CK26: [[EXT:%.+]] = fpext float [[VAL]] to double
// CK26: fadd double [[EXT]], 1.000000e+00

// CK26: define {{.+}}[[CALL02]]({{.*}}i32*{{.*}}[[PVTARG:%.+]])
// CK26: store i32* [[PVTARG]], i32** [[PVTADDR:%.+]],
// CK26: [[ADDR:%.+]] = load i32*, i32** [[PVTADDR]],
// CK26: [[VAL:%.+]] = load i32, i32* [[ADDR]],
// CK26: add nsw i32 [[VAL]], 1

// CK26: define {{.+}}[[CALL03]]({{.*}}float*{{.*}}[[PVTARG:%.+]])
// CK26: store float* [[PVTARG]], float** [[PVTADDR:%.+]],
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
// RUN: %clang_cc1 -DCK27 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK27 --check-prefix CK27-64
// RUN: %clang_cc1 -DCK27 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK27 --check-prefix CK27-64
// RUN: %clang_cc1 -DCK27 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK27 --check-prefix CK27-32
// RUN: %clang_cc1 -DCK27 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK27 --check-prefix CK27-32
#ifdef CK27

// CK27: [[SIZE00:@.+]] = private {{.*}}constant [1 x i[[Z:64|32]]] zeroinitializer
// CK27: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i32] [i32 32]

// CK27: [[SIZE01:@.+]] = private {{.*}}constant [1 x i[[Z]]] zeroinitializer
// CK27: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK27: [[SIZE02:@.+]] = private {{.*}}constant [1 x i[[Z]]] zeroinitializer
// CK27: [[MTYPE02:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK27: [[SIZE03:@.+]] = private {{.*}}constant [1 x i[[Z]]] zeroinitializer
// CK27: [[MTYPE03:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK27: [[SIZE05:@.+]] = private {{.*}}constant [1 x i[[Z]]] zeroinitializer
// CK27: [[MTYPE05:@.+]] = private {{.*}}constant [1 x i32] [i32 32]

// CK27: [[SIZE07:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 4]
// CK27: [[MTYPE07:@.+]] = private {{.*}}constant [1 x i32] [i32 288]

// CK27: [[SIZE09:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 40]
// CK27: [[MTYPE09:@.+]] = private {{.*}}constant [1 x i32] [i32 161]

// CK27-LABEL: zero_size_section_and_private_maps
void zero_size_section_and_private_maps (int ii){

  // Map of a pointer.
  int *pa;

  // Region 00
  // CK27-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}})
  // CK27-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK27-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK27-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK27-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK27-DAG: [[CBPVAL0]] = bitcast i32* [[VAR0:%.+]] to i8*
  // CK27-DAG: [[CPVAL0]] = bitcast i32* [[VAR0]] to i8*

  // CK27: call void [[CALL00:@.+]](i32* {{[^,]+}})
  #pragma omp target
  {
    pa[50]++;
  }

  // Region 01
  // CK27-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}})
  // CK27-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK27-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK27-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK27-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK27-DAG: [[CBPVAL0]] = bitcast i32* [[RVAR0:%.+]] to i8*
  // CK27-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK27-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK27-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 0
  // CK27-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK27: call void [[CALL01:@.+]](i32* {{[^,]+}})
  #pragma omp target map(pa[:0])
  {
    pa[50]++;
  }

  // Region 02
  // CK27-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE02]]{{.+}})
  // CK27-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK27-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK27-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK27-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK27-DAG: [[CBPVAL0]] = bitcast i32* [[RVAR0:%.+]] to i8*
  // CK27-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
  // CK27-DAG: [[RVAR0]] = load i32*, i32** [[VAR0:%[^,]+]]
  // CK27-DAG: [[SEC0]] = getelementptr {{.*}}i32* [[RVAR00:%.+]], i{{.+}} 0
  // CK27-DAG: [[RVAR00]] = load i32*, i32** [[VAR0]]

  // CK27: call void [[CALL02:@.+]](i32* {{[^,]+}})
  #pragma omp target map(pa[0:0])
  {
    pa[50]++;
  }

  // Region 03
  // CK27-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE03]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE03]]{{.+}})
  // CK27-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK27-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK27-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK27-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK27-DAG: [[CBPVAL0]] = bitcast i32* [[RVAR0:%.+]] to i8*
  // CK27-DAG: [[CPVAL0]] = bitcast i32* [[SEC0:%.+]] to i8*
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
  // CK27: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 0, i8** null, i8** null, i{{64|32}}* null, i32* null)
  // CK27: call void [[CALL04:@.+]]()
  #pragma omp target private(pvtPtr)
  {
    pvtPtr[5]++;
  }

  // Region 05
  // CK27-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE05]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE05]]{{.+}})
  // CK27-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK27-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK27-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK27-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK27-DAG: [[CBPVAL0]] = bitcast i32* [[VAR0:%.+]] to i8*
  // CK27-DAG: [[CPVAL0]] = bitcast i32* [[VAR0]] to i8*

  // CK27: call void [[CALL05:@.+]](i32* {{[^,]+}})
  #pragma omp target firstprivate(pvtPtr)
  {
    pvtPtr[5]++;
  }

  // Region 06
  // CK27: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 0, i8** null, i8** null, i{{64|32}}* null, i32* null)
  // CK27: call void [[CALL06:@.+]]()
  #pragma omp target private(pvtScl)
  {
    pvtScl++;
  }

  // Region 07
  // CK27-DAG: call i32 @__tgt_target(i32 {{.+}}, i8* {{.+}}, i32 1, i8** [[BPGEP:%[0-9]+]], i8** [[PGEP:%[0-9]+]], {{.+}}[[SIZE07]]{{.+}}, {{.+}}[[MTYPE07]]{{.+}})
  // CK27-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BPS:%[^,]+]], i32 0, i32 0
  // CK27-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[PS:%[^,]+]], i32 0, i32 0
  // CK27-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BPS]], i32 0, i32 0
  // CK27-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[PS]], i32 0, i32 0
  // CK27-DAG: store i8* [[VALBP:%.+]], i8** [[BP1]],
  // CK27-DAG: store i8* [[VALP:%.+]], i8** [[P1]],
  // CK27-DAG: [[VALBP]] = inttoptr i[[Z]] [[VAL:%.+]] to i8*
  // CK27-DAG: [[VALP]] = inttoptr i[[Z]] [[VAL:%.+]] to i8*
  // CK27-DAG: [[VAL]] = load i[[Z]], i[[Z]]* [[ADDR:%.+]],
  // CK27-64-DAG: [[CADDR:%.+]] = bitcast i[[Z]]* [[ADDR]] to i32*
  // CK27-64-DAG: store i32 {{.+}}, i32* [[CADDR]],

  // CK27: call void [[CALL07:@.+]](i[[Z]] [[VAL]])
  #pragma omp target firstprivate(pvtScl)
  {
    pvtScl++;
  }

  // Region 08
  // CK27: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 0, i8** null, i8** null, i{{64|32}}* null, i32* null)
  // CK27: call void [[CALL08:@.+]]()
  #pragma omp target private(pvtArr)
  {
    pvtArr[5]++;
  }

  // Region 09
  // CK27-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE09]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE09]]{{.+}})
  // CK27-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK27-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK27-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK27-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK27-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK27-DAG: [[CBPVAL0]] = bitcast [10 x i32]* [[VAR0:%.+]] to i8*
  // CK27-DAG: [[CPVAL0]] = bitcast [10 x i32]* [[VAR0]] to i8*

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
// RUN: %clang_cc1 -DCK28 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK28 --check-prefix CK28-64
// RUN: %clang_cc1 -DCK28 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK28 --check-prefix CK28-64
// RUN: %clang_cc1 -DCK28 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK28 --check-prefix CK28-32
// RUN: %clang_cc1 -DCK28 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK28 --check-prefix CK28-32
#ifdef CK28

// CK28: [[SIZE00:@.+]] = private {{.*}}constant [1 x i[[Z:64|32]]] [i[[Z:64|32]] {{8|4}}]
// CK28: [[MTYPE00:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK28: [[SIZE01:@.+]] = private {{.*}}constant [1 x i[[Z]]] [i[[Z]] 400]
// CK28: [[MTYPE01:@.+]] = private {{.*}}constant [1 x i32] [i32 35]

// CK28-LABEL: explicit_maps_pointer_references
void explicit_maps_pointer_references (int *p){
  int *&a = p;

  // Region 00
  // CK28-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE00]]{{.+}})
  // CK28-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK28-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK28-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK28-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK28-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK28-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK28-DAG: [[CBPVAL0]] = bitcast i32** [[VAR0:%.+]] to i8*
  // CK28-DAG: [[CPVAL0]] = bitcast i32** [[VAR1:%.+]] to i8*
  // CK28-DAG: [[VAR0]] = load i32**, i32*** [[VAR00:%.+]],
  // CK28-DAG: [[VAR1]] = load i32**, i32*** [[VAR11:%.+]],

  // CK28: call void [[CALL00:@.+]](i32** {{[^,]+}})
  #pragma omp target map(a)
  {
    ++a;
  }

  // Region 01
  // CK28-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 1, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[1 x i{{.+}}]* [[MTYPE01]]{{.+}})
  // CK28-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK28-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

  // CK28-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK28-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK28-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
  // CK28-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
  // CK28-DAG: [[CBPVAL0]] = bitcast i32* [[VAR0:%.+]] to i8*
  // CK28-DAG: [[CPVAL0]] = bitcast i32* [[VAR1:%.+]] to i8*
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
// RUN: %clang_cc1 -DCK29 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK29 --check-prefix CK29-64
// RUN: %clang_cc1 -DCK29 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK29 --check-prefix CK29-64
// RUN: %clang_cc1 -DCK29 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK29 --check-prefix CK29-32
// RUN: %clang_cc1 -DCK29 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK29 --check-prefix CK29-32
#ifdef CK29

// CK29: [[SSA:%.+]] = type { double*, double** }
// CK29: [[SSB:%.+]]  = type { [[SSA]]*, [[SSA]]** }

// CK29: [[SIZE00:@.+]] = private {{.*}}constant [4 x i[[Z:64|32]]] [i[[Z:64|32]] {{8|4}}, i[[Z:64|32]] {{8|4}}, i[[Z:64|32]] {{8|4}}, i[[Z:64|32]] 80]
// CK29: [[MTYPE00:@.+]] = private {{.*}}constant [4 x i32] [i32 35, i32 16, i32 19, i32 19]

// CK29: [[SIZE01:@.+]] = private {{.*}}constant [4 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] {{8|4}}, i[[Z]] {{8|4}}, i[[Z]] 80]
// CK29: [[MTYPE01:@.+]] = private {{.*}}constant [4 x i32] [i32 32, i32 19, i32 19, i32 19]

// CK29: [[SIZE02:@.+]] = private {{.*}}constant [5 x i[[Z]]] [i[[Z]] {{8|4}}, i[[Z]] {{8|4}}, i[[Z]] {{8|4}}, i[[Z]] {{8|4}}, i[[Z]] 80]
// CK29: [[MTYPE02:@.+]] = private {{.*}}constant [5 x i32] [i32 32, i32 19, i32 16, i32 19, i32 19]

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
    // CK29-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 4, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[4 x i{{.+}}]* [[SIZE00]], {{.+}}getelementptr {{.+}}[4 x i{{.+}}]* [[MTYPE00]]{{.+}})

    // CK29-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK29-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK29-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
    // CK29-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
    // CK29-DAG: [[CBPVAL0]] = bitcast [[SSB]]* [[VAR0:%.+]] to i8*
    // CK29-DAG: [[CPVAL0]] = bitcast [[SSA]]** [[VAR00:%.+]] to i8*
    // CK29-DAG: [[VAR0]] = load [[SSB]]*, [[SSB]]** %
    // CK29-DAG: [[VAR00]] = getelementptr inbounds [[SSB]], [[SSB]]* [[VAR0]], i32 0, i32 0

    // CK29-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
    // CK29-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
    // CK29-DAG: [[CBPVAL1]] = bitcast [[SSA]]** [[VAR00]] to i8*
    // CK29-DAG: [[CPVAL1]] = bitcast double*** [[VAR1:%.+]] to i8*
    // CK29-DAG: [[VAR1]] = getelementptr inbounds [[SSA]], [[SSA]]* %{{.+}}, i32 0, i32 1

    // CK29-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
    // CK29-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
    // CK29-DAG: store i8* [[CBPVAL2:%[^,]+]], i8** [[BP2]]
    // CK29-DAG: store i8* [[CPVAL2:%[^,]+]], i8** [[P2]]
    // CK29-DAG: [[CBPVAL2]] = bitcast double*** [[VAR1]] to i8*
    // CK29-DAG: [[CPVAL2]] = bitcast double** [[VAR2:%.+]] to i8*
    // CK29-DAG: [[VAR2]] = load double**, double*** [[VAR22:%.+]],
    // CK29-DAG: [[VAR22]] = getelementptr inbounds [[SSA]], [[SSA]]* %{{.+}}, i32 0, i32 1

    // CK29-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
    // CK29-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
    // CK29-DAG: store i8* [[CBPVAL3:%[^,]+]], i8** [[BP3]]
    // CK29-DAG: store i8* [[CPVAL3:%[^,]+]], i8** [[P3]]
    // CK29-DAG: [[CBPVAL3]] = bitcast double** [[VAR2]] to i8*
    // CK29-DAG: [[CPVAL3]] = bitcast double* [[VAR3:%.+]] to i8*
    // CK29-DAG: [[VAR3]] = getelementptr inbounds double, double* [[VAR33:%.+]], i{{.+}} 0
    // CK29-DAG: [[VAR33]] = load double*, double** %{{.+}},

    // CK29: call void [[CALL00:@.+]]([[SSB]]* {{[^,]+}})
    #pragma omp target map(p->pr[:10])
    {
      p->pr++;
    }

    // Region 01
    // CK29-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 4, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[4 x i{{.+}}]* [[SIZE01]], {{.+}}getelementptr {{.+}}[4 x i{{.+}}]* [[MTYPE01]]{{.+}})

    // CK29-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK29-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK29-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
    // CK29-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
    // CK29-DAG: [[CBPVAL0]] = bitcast [[SSB]]* [[VAR0:%.+]] to i8*
    // CK29-DAG: [[CPVAL0]] = bitcast [[SSA]]*** [[VAR00:%.+]] to i8*
    // CK29-DAG: [[VAR00]] = getelementptr inbounds [[SSB]], [[SSB]]* [[VAR0]], i32 0, i32 1

    // CK29-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
    // CK29-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
    // CK29-DAG: [[CBPVAL1]] = bitcast [[SSA]]*** [[VAR00]] to i8*
    // CK29-DAG: [[CPVAL1]] = bitcast [[SSA]]** [[VAR1:%.+]] to i8*
    // CK29-DAG: [[VAR1]] = load [[SSA]]**, [[SSA]]*** [[VAR00]],

    // CK29-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
    // CK29-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
    // CK29-DAG: store i8* [[CBPVAL2:%[^,]+]], i8** [[BP2]]
    // CK29-DAG: store i8* [[CPVAL2:%[^,]+]], i8** [[P2]]
    // CK29-DAG: [[CBPVAL2]] = bitcast [[SSA]]** [[VAR1]] to i8*
    // CK29-DAG: [[CPVAL2]] = bitcast double** [[VAR2:%.+]] to i8*
    // CK29-DAG: [[VAR2]] = getelementptr inbounds [[SSA]], [[SSA]]* %{{.+}}, i32 0, i32 0

    // CK29-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
    // CK29-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
    // CK29-DAG: store i8* [[CBPVAL3:%[^,]+]], i8** [[BP3]]
    // CK29-DAG: store i8* [[CPVAL3:%[^,]+]], i8** [[P3]]
    // CK29-DAG: [[CBPVAL3]] = bitcast double** [[VAR2]] to i8*
    // CK29-DAG: [[CPVAL3]] = bitcast double* [[VAR3:%.+]] to i8*
    // CK29-DAG: [[VAR3]] = getelementptr inbounds double, double* [[VAR33:%.+]], i{{.+}} 0
    // CK29-DAG: [[VAR33]] = load double*, double** %{{.+}},

    // CK29: call void [[CALL00:@.+]]([[SSB]]* {{[^,]+}})
    #pragma omp target map(pr->p[:10])
    {
      pr->p++;
    }

    // Region 02
    // CK29-DAG: call i32 @__tgt_target(i32 {{[^,]+}}, i8* {{[^,]+}}, i32 5, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], {{.+}}getelementptr {{.+}}[5 x i{{.+}}]* [[SIZE02]], {{.+}}getelementptr {{.+}}[5 x i{{.+}}]* [[MTYPE02]]{{.+}})

    // CK29-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
    // CK29-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]

    // CK29-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
    // CK29-DAG: store i8* [[CBPVAL0:%[^,]+]], i8** [[BP0]]
    // CK29-DAG: store i8* [[CPVAL0:%[^,]+]], i8** [[P0]]
    // CK29-DAG: [[CBPVAL0]] = bitcast [[SSB]]* [[VAR0:%.+]] to i8*
    // CK29-DAG: [[CPVAL0]] = bitcast [[SSA]]*** [[VAR00:%.+]] to i8*
    // CK29-DAG: [[VAR00]] = getelementptr inbounds [[SSB]], [[SSB]]* [[VAR0]], i32 0, i32 1

    // CK29-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
    // CK29-DAG: store i8* [[CBPVAL1:%[^,]+]], i8** [[BP1]]
    // CK29-DAG: store i8* [[CPVAL1:%[^,]+]], i8** [[P1]]
    // CK29-DAG: [[CBPVAL1]] = bitcast [[SSA]]*** [[VAR00]] to i8*
    // CK29-DAG: [[CPVAL1]] = bitcast [[SSA]]** [[VAR1:%.+]] to i8*
    // CK29-DAG: [[VAR1]] = load [[SSA]]**, [[SSA]]*** [[VAR00]],

    // CK29-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
    // CK29-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
    // CK29-DAG: store i8* [[CBPVAL2:%[^,]+]], i8** [[BP2]]
    // CK29-DAG: store i8* [[CPVAL2:%[^,]+]], i8** [[P2]]
    // CK29-DAG: [[CBPVAL2]] = bitcast [[SSA]]** [[VAR1]] to i8*
    // CK29-DAG: [[CPVAL2]] = bitcast double*** [[VAR2:%.+]] to i8*
    // CK29-DAG: [[VAR2]] = getelementptr inbounds [[SSA]], [[SSA]]* %{{.+}}, i32 0, i32 1

    // CK29-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
    // CK29-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
    // CK29-DAG: store i8* [[CBPVAL3:%[^,]+]], i8** [[BP3]]
    // CK29-DAG: store i8* [[CPVAL3:%[^,]+]], i8** [[P3]]
    // CK29-DAG: [[CBPVAL3]] = bitcast double*** [[VAR2]] to i8*
    // CK29-DAG: [[CPVAL3]] = bitcast double** [[VAR3:%.+]] to i8*
    // CK29-DAG: [[VAR3]] = load double**, double*** [[VAR2]],

    // CK29-DAG: [[BP4:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 4
    // CK29-DAG: [[P4:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 4
    // CK29-DAG: store i8* [[CBPVAL4:%[^,]+]], i8** [[BP4]]
    // CK29-DAG: store i8* [[CPVAL4:%[^,]+]], i8** [[P4]]
    // CK29-DAG: [[CBPVAL4]] = bitcast double** [[VAR3]] to i8*
    // CK29-DAG: [[CPVAL4]] = bitcast double* [[VAR4:%.+]] to i8*
    // CK29-DAG: [[VAR4]] = getelementptr inbounds double, double* [[VAR44:%.+]], i{{.+}} 0
    // CK29-DAG: [[VAR44]] = load double*, double**

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
#endif
