// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -no-opaque-pointers -DCK35 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK35 --check-prefix CK35-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK35 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK35 --check-prefix CK35-64
// RUN: %clang_cc1 -no-opaque-pointers -DCK35 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK35 --check-prefix CK35-32
// RUN: %clang_cc1 -no-opaque-pointers -DCK35 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK35 --check-prefix CK35-32

// RUN: %clang_cc1 -no-opaque-pointers -DCK35 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK35 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK35 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// RUN: %clang_cc1 -no-opaque-pointers -DCK35 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -no-opaque-pointers -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// SIMD-ONLY32-NOT: {{__kmpc|__tgt}}
#ifdef CK35

class S {
public:
  S(double &b) : b(b) {}
  int a;
  double &b;
  void foo();
};

// CK35-DAG: [[SIZE_TO:@.+]] = private {{.*}}constant [4 x i64] [i64 0, i64 0, i64 0, i64 8]
// TARGET_PARAM = 0x20
// MEMBER_OF_1 | TO = 0x1000000000001
// MEMBER_OF_1 | PTR_AND_OBJ | TO = 0x1000000000011
// CK35-DAG: [[MTYPE_TO:@.+]] = {{.+}}constant [4 x i64] [i64 [[#0x20]], i64 [[#0x1000000000001]], i64 [[#0x1000000000001]], i64 [[#0x1000000000011]]]
// CK35-DAG: [[SIZE_FROM:@.+]] = private {{.*}}constant [2 x i64] [i64 0, i64 8]
// TARGET_PARAM = 0x20
// MEMBER_OF_1 | PTR_AND_OBJ | FROM = 0x1000000000012
// CK35-DAG: [[MTYPE_FROM:@.+]] = {{.+}}constant [2 x i64] [i64 [[#0x20]], i64 [[#0x1000000000012]]]

void ref_map() {
  double b;
  S s(b);

  // CK35-DAG: call i32 @__tgt_target_kernel(%struct.ident_t* @{{.+}}, i64 -1, i32 -1, i32 0, i8* @.{{.+}}.region_id, %struct.__tgt_kernel_arguments* [[ARGS:%.+]])
  // CK35-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
  // CK35-DAG: store i8** [[BPGEP:%.+]], i8*** [[BPARG]]
  // CK35-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
  // CK35-DAG: store i8** [[PGEP:%.+]], i8*** [[PARG]]
  // CK35-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
  // CK35-DAG: store i64* [[SIZES:%.+]], i64** [[SARG]]
  // CK35-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK35-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK35-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // pass TARGET_PARAM {&s, &s, ((void*)(&s+1)-(void*)&s)}

  // CK35-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK35-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK35-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0

  // CK35-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to %class.S**
  // CK35-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to %class.S**

  // CK35-DAG: store %class.S* [[S_ADDR:%.+]], %class.S** [[BPC0]],
  // CK35-DAG: store %class.S* [[S_ADDR]], %class.S** [[PC0]],
  // CK35-DAG: store i64 [[S_SIZE:%.+]], i64* [[S0]],

  // CK35-DAG: [[S_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
  // CK35-DAG: [[SZ]] = sub i64 [[S_1_INTPTR:%.+]], [[S_INTPTR:%.+]]
  // CK35-DAG: [[S_1_INTPTR]] = ptrtoint i8* [[S_1_VOID:%.+]] to i64
  // CK35-DAG: [[S_INTPTR]] = ptrtoint i8* [[S_VOID:%.+]] to i64
  // CK35-DAG: [[S_1_VOID]] = bitcast %class.S* [[S_1:%.+]] to i8*
  // CK35-DAG: [[S_VOID]] = bitcast %class.S* [[S_ADDR]] to i8*
  // CK35-DAG: [[S_1]] = getelementptr %class.S, %class.S* [[S_ADDR]], i32 1

  // pass MEMBER_OF_1 | TO {&s, &s, ((void*)(&s.a+1)-(void*)&s)} to copy the data of s.a.

  // CK35-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK35-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK35-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1

  // CK35-DAG: [[BPC1:%.+]] = bitcast i8** [[BP1]] to %class.S**
  // CK35-DAG: [[PC1:%.+]] = bitcast i8** [[P1]] to %class.S**

  // CK35-DAG: store %class.S* [[S_ADDR]], %class.S** [[BPC1]],
  // CK35-DAG: store %class.S* [[S_ADDR]], %class.S** [[PC1]],
  // CK35-DAG: store i64 [[A_SIZE:%.+]], i64* [[S1]],

  // CK35-DAG: [[A_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
  // CK35-DAG: [[SZ]] = sub i64 [[B_BEGIN_INTPTR:%.+]], [[S_INTPTR:%.+]]
  // CK35-DAG: [[S_INTPTR]] = ptrtoint i8* [[S_VOID:%.+]] to i64
  // CK35-DAG: [[B_BEGIN_INTPTR]] = ptrtoint i8* [[B_BEGIN_VOID:%.+]] to i64
  // CK35-DAG: [[S_VOID]] = bitcast %class.S* [[S_ADDR]] to i8*
  // CK35-DAG: [[B_BEGIN_VOID]] = bitcast double** [[B_ADDR:%.+]] to i8*
  // CK35-DAG: [[B_ADDR]] = getelementptr inbounds %class.S, %class.S* [[S_ADDR]], i32 0, i32 1

  // pass MEMBER_OF_1 | TO {&s, &s.b+1, ((void*)(&s+1)-(void*)(&s.b+1))} to copy the data of remainder of s.

  // CK35-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK35-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK35-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2

  // CK35-DAG: [[BPC2:%.+]] = bitcast i8** [[BP2]] to %class.S**
  // CK35-DAG: [[PC2:%.+]] = bitcast i8** [[P2]] to double***

  // CK35-DAG: store %class.S* [[S_ADDR]], %class.S** [[BPC2]],
  // CK35-DAG: store double** [[B_END:%.+]], double*** [[PC2]],
  // CK35-DAG: store i64 [[REM_SIZE:%.+]], i64* [[S2]],

  // CK35-DAG: [[B_END]] = getelementptr double*, double** [[B_ADDR]], i{{.+}} 1

  // CK35-DAG: [[REM_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
  // CK35-DAG: [[SZ]] = sub i64 [[S_END_INTPTR:%.+]], [[B_END_INTPTR:%.+]]
  // CK35-DAG: [[B_END_INTPTR]] = ptrtoint i8* [[B_END_VOID:%.+]] to i64
  // CK35-DAG: [[S_END_INTPTR]] = ptrtoint i8* [[S_END_VOID:%.+]] to i64
  // CK35-DAG: [[B_END_VOID]] = bitcast double** [[B_END]] to i8*
  // CK35-DAG: [[S_END_VOID]] = getelementptr i8, i8* [[S_LAST:%.+]], i{{.+}} 1
  // CK35-64-DAG: [[S_LAST]] = getelementptr i8, i8* [[S_VOIDPTR:%.+]], i64 15
  // CK35-32-DAG: [[S_LAST]] = getelementptr i8, i8* [[S_VOIDPTR:%.+]], i32 7
  // CK35-DAG: [[S_VOIDPTR]] = bitcast %class.S* [[S_ADDR]] to i8*

  // pass MEMBER_OF_1 | PTR_AND_OBJ | TO {&s, &s.b, 8|4} to copy the data of s.b.

  // CK35-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
  // CK35-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3

  // CK35-DAG: [[BPC3:%.+]] = bitcast i8** [[BP3]] to %class.S**
  // CK35-DAG: [[PC3:%.+]] = bitcast i8** [[P3]] to double**

  // CK35-DAG: store %class.S* [[S_ADDR]], %class.S** [[BPC3]],
  // CK35-DAG: store double* [[B_ADDR:%.+]], double** [[PC3]],

  // CK35-DAG: [[B_ADDR]] = load double*, double** [[B_REF:%.+]],
  // CK35-DAG: [[B_REF]] = getelementptr inbounds %class.S, %class.S* [[S_ADDR]], i32 0, i32 1

  #pragma omp target map(to: s, s.b)
  s.foo();

  // CK35 : call void

  // CK35-DAG: call i32 @__tgt_target_kernel(%struct.ident_t* @{{.+}}, i64 -1, i32 -1, i32 0, i8* @.{{.+}}.region_id, %struct.__tgt_kernel_arguments* [[ARGS:%.+]])
  // CK35-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
  // CK35-DAG: store i8** [[BPGEP:%.+]], i8*** [[BPARG]]
  // CK35-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
  // CK35-DAG: store i8** [[PGEP:%.+]], i8*** [[PARG]]
  // CK35-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
  // CK35-DAG: store i64* [[SIZES:%.+]], i64** [[SARG]]
  // CK35-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK35-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK35-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // pass TARGET_PARAM {&s, &s.b, ((void*)(&s.b+1)-(void*)&s.b)}

  // CK35-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK35-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK35-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0

  // CK35-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to %class.S**
  // CK35-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to double***

  // CK35-DAG: store %class.S* [[S_ADDR]], %class.S** [[BPC0]],
  // CK35-DAG: store double** [[SB_ADDR:%.+]], double*** [[PC0]],
  // CK35-DAG: store i64 [[B_SIZE:%.+]], i64* [[S0]],

  // CK35-DAG: [[B_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
  // CK35-DAG: [[SZ]] = sub i64 [[SB_1_INTPTR:%.+]], [[SB_INTPTR:%.+]]
  // CK35-DAG: [[SB_1_INTPTR]] = ptrtoint i8* [[SB_1_VOID:%.+]] to i64
  // CK35-DAG: [[SB_INTPTR]] = ptrtoint i8* [[SB_VOID:%.+]] to i64
  // CK35-DAG: [[SB_1_VOID]] = bitcast double** [[SB_1:%.+]] to i8*
  // CK35-DAG: [[SB_VOID]] = bitcast double** [[SB_ADDR:%.+]] to i8*
  // CK35-DAG: [[SB_ADDR]] = getelementptr inbounds %class.S, %class.S* [[S_ADDR]], i32 0, i32 1
  // CK35-DAG: [[SB_1]] = getelementptr double*, double** [[SB_ADDR]], i{{.+}} 1

  // pass MEMBER_OF_1 | PTR_AND_OBJ | FROM {&s, &s.b, 8|4} to copy the data of s.c.

  // CK35-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK35-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1

  // CK35-DAG: [[BPC1:%.+]] = bitcast i8** [[BP1]] to %class.S**
  // CK35-DAG: [[PC1:%.+]] = bitcast i8** [[P1]] to double**

  // CK35-DAG: store %class.S* [[S_ADDR]], %class.S** [[BPC1]],
  // CK35-DAG: store double* [[B_ADDR:%.+]], double** [[PC1]],

  // CK35-DAG: [[B_ADDR]] = load double*, double** [[SB_ADDR]],

  #pragma omp target map(from: s.b)
  s.foo();
}

#endif // CK35
#endif
