// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///
/// Implicit maps.
///

///==========================================================================///
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-64
// RUN: %clang_cc1 -DCK1 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32
// RUN: %clang_cc1 -DCK1 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK1 --check-prefix CK1-32
#ifdef CK1

// CK1-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// Map types: OMP_MAP_BYCOPY = 128
// CK1-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 128]

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
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-64
// RUN: %clang_cc1 -DCK2 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-32
// RUN: %clang_cc1 -DCK2 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK2 --check-prefix CK2-32
#ifdef CK2

// CK2-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// Map types: OMP_MAP_BYCOPY = 128
// CK2-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 128]

// CK2-LABEL: implicit_maps_integer_reference
void implicit_maps_integer_reference (int a){
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

#endif
///==========================================================================///
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-64
// RUN: %clang_cc1 -DCK3 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-32
// RUN: %clang_cc1 -DCK3 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK3 --check-prefix CK3-32
#ifdef CK3

// CK3-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// Map types: OMP_MAP_BYCOPY = 128
// CK3-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 128]

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
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK4 --check-prefix CK4-64
// RUN: %clang_cc1 -DCK4 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK4 --check-prefix CK4-32
// RUN: %clang_cc1 -DCK4 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK4 --check-prefix CK4-32
#ifdef CK4

// CK4-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// Map types: OMP_MAP_BYCOPY = 128
// CK4-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 128]

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
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK5 --check-prefix CK5-64
// RUN: %clang_cc1 -DCK5 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK5 --check-prefix CK5-32
// RUN: %clang_cc1 -DCK5 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK5 --check-prefix CK5-32
#ifdef CK5

// CK5-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// Map types: OMP_MAP_BYCOPY = 128
// CK5-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 128]

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
// RUN: %clang_cc1 -DCK6 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK6 --check-prefix CK6-64
// RUN: %clang_cc1 -DCK6 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK6 --check-prefix CK6-64
// RUN: %clang_cc1 -DCK6 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK6 --check-prefix CK6-32
// RUN: %clang_cc1 -DCK6 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK6 --check-prefix CK6-32
#ifdef CK6
// CK6-DAG: [[GBL:@Gi]] = global i32 0
// CK6-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// Map types: OMP_MAP_BYCOPY = 128
// CK6-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 128]

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
// RUN: %clang_cc1 -DCK7 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK7 --check-prefix CK7-64
// RUN: %clang_cc1 -DCK7 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK7  --check-prefix CK7-64
// RUN: %clang_cc1 -DCK7 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK7  --check-prefix CK7-32
// RUN: %clang_cc1 -DCK7 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK7  --check-prefix CK7-32
#ifdef CK7

// For a 32-bit targets, the value doesn't fit the size of the pointer,
// therefore it is passed by reference with a map 'to' specification.

// CK7-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 8]
// Map types: OMP_MAP_BYCOPY = 128
// CK7-64-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 128]
// Map types: OMP_MAP_TO = 1
// CK7-32-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 1]

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
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK8
// RUN: %clang_cc1 -DCK8 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK8
#ifdef CK8

// CK8-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// Map types: OMP_MAP_BYCOPY = 128
// CK8-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 128]

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
// RUN: %clang_cc1 -DCK9 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK9
// RUN: %clang_cc1 -DCK9 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK9
// RUN: %clang_cc1 -DCK9 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK9
// RUN: %clang_cc1 -DCK9 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK9
#ifdef CK9

// CK9-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 16]
// Map types: OMP_MAP_TO + OMP_MAP_FROM = 2 + 1
// CK9-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 3]

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
// RUN: %clang_cc1 -DCK10 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK10
// RUN: %clang_cc1 -DCK10 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK10
#ifdef CK10

// CK10-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} {{8|4}}]
// Map types: OMP_MAP_BYCOPY | OMP_MAP_PTR = 128 + 32
// CK10-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 160]

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
// RUN: %clang_cc1 -DCK11 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK11
// RUN: %clang_cc1 -DCK11 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK11
// RUN: %clang_cc1 -DCK11 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK11
// RUN: %clang_cc1 -DCK11 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK11
#ifdef CK11

// CK11-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 16]
// Map types: OMP_MAP_TO = 1
// CK11-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 1]

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
// RUN: %clang_cc1 -DCK12 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK12 --check-prefix CK12-64
// RUN: %clang_cc1 -DCK12 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK12 --check-prefix CK12-64
// RUN: %clang_cc1 -DCK12 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK12 --check-prefix CK12-32
// RUN: %clang_cc1 -DCK12 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK12 --check-prefix CK12-32
#ifdef CK12

// For a 32-bit targets, the value doesn't fit the size of the pointer,
// therefore it is passed by reference with a map 'to' specification.

// CK12-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 8]
// Map types: OMP_MAP_BYCOPY = 128
// CK12-64-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 128]
// Map types: OMP_MAP_TO = 1
// CK12-32-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 1]

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
// RUN: %clang_cc1 -DCK13 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK13
// RUN: %clang_cc1 -DCK13 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK13
#ifdef CK13

// We don't have a constant map size for VLAs.
// Map types:
//  - OMP_MAP_BYCOPY = 128 (vla size)
//  - OMP_MAP_BYCOPY = 128 (vla size)
//  - OMP_MAP_TO + OMP_MAP_FROM = 2 + 1
// CK13-DAG: [[TYPES:@.+]] = {{.+}}constant [3 x i32] [i32 128, i32 128, i32 3]

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
// RUN: %clang_cc1 -DCK14 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK14 --check-prefix CK14-64
// RUN: %clang_cc1 -DCK14 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK14 --check-prefix CK14-64
// RUN: %clang_cc1 -DCK14 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK14 --check-prefix CK14-32
// RUN: %clang_cc1 -DCK14 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK14 --check-prefix CK14-32
#ifdef CK14

// CK14-DAG: [[ST:%.+]] = type { i32, double }
// CK14-DAG: [[SIZES:@.+]] = {{.+}}constant [2 x i[[sz:64|32]]] [i{{64|32}} {{16|12}}, i{{64|32}} 4]
// Map types:
// - OMP_MAP_TO | OMP_MAP_FROM = 1 + 2
// - OMP_MAP_BYCOPY = 128
// CK14-DAG: [[TYPES:@.+]] = {{.+}}constant [2 x i32] [i32 3, i32 128]

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
// RUN: %clang_cc1 -DCK15 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK15 --check-prefix CK15-64
// RUN: %clang_cc1 -DCK15 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK15 --check-prefix CK15-64
// RUN: %clang_cc1 -DCK15 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK15 --check-prefix CK15-32
// RUN: %clang_cc1 -DCK15 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK15 --check-prefix CK15-32
#ifdef CK15

// CK15: [[ST:%.+]] = type { i32, double, i32* }
// CK15: [[SIZES:@.+]] = {{.+}}constant [2 x i[[sz:64|32]]] [i{{64|32}} {{24|16}}, i{{64|32}} 4]
// Map types:
// - OMP_MAP_TO | OMP_MAP_FROM = 1 + 2
// - OMP_MAP_BYCOPY = 128
// CK15: [[TYPES:@.+]] = {{.+}}constant [2 x i32] [i32 3, i32 128]

// CK15: [[SIZES2:@.+]] = {{.+}}constant [2 x i[[sz]]] [i{{64|32}} {{24|16}}, i{{64|32}} 4]
// Map types:
// - OMP_MAP_TO | OMP_MAP_FROM = 1 + 2
// - OMP_MAP_BYCOPY = 128
// CK15: [[TYPES2:@.+]] = {{.+}}constant [2 x i32] [i32 3, i32 128]

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
// RUN: %clang_cc1 -DCK16 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK16 --check-prefix CK16-64
// RUN: %clang_cc1 -DCK16 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK16 --check-prefix CK16-64
// RUN: %clang_cc1 -DCK16 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK16 --check-prefix CK16-32
// RUN: %clang_cc1 -DCK16 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK16 --check-prefix CK16-32
#ifdef CK16

// CK16-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// Map types:
// - OMP_MAP_BYCOPY = 128
// CK16-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 128]

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
// RUN: %clang_cc1 -DCK17 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK17
// RUN: %clang_cc1 -DCK17 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK17
// RUN: %clang_cc1 -DCK17 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK17
// RUN: %clang_cc1 -DCK17 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK17
#ifdef CK17

// CK17-DAG: [[ST:%.+]] = type { i32, double }
// CK17-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} {{16|12}}]
// Map types: OMP_MAP_TO + OMP_MAP_FROM = 2 + 1
// CK17-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 3]

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
// RUN: %clang_cc1 -DCK18 -verify -fopenmp -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK18 --check-prefix CK18-64
// RUN: %clang_cc1 -DCK18 -fopenmp -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK18 --check-prefix CK18-64
// RUN: %clang_cc1 -DCK18 -verify -fopenmp -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK18 --check-prefix CK18-32
// RUN: %clang_cc1 -DCK18 -fopenmp -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK18 --check-prefix CK18-32
#ifdef CK18

// CK18-DAG: [[SIZES:@.+]] = {{.+}}constant [1 x i[[sz:64|32]]] [i{{64|32}} 4]
// Map types:
// - OMP_MAP_BYCOPY = 128
// CK18-DAG: [[TYPES:@.+]] = {{.+}}constant [1 x i32] [i32 128]

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
#endif
