// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK34 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK34 --check-prefix CK34-64
// RUN: %clang_cc1 -DCK34 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK34 --check-prefix CK34-64
// RUN: %clang_cc1 -DCK34 -verify -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK34 --check-prefix CK34-32
// RUN: %clang_cc1 -DCK34 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK34 --check-prefix CK34-32

// RUN: %clang_cc1 -DCK34 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// RUN: %clang_cc1 -DCK34 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// RUN: %clang_cc1 -DCK34 -verify -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// RUN: %clang_cc1 -DCK34 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-version=50 -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// SIMD-ONLY32-NOT: {{__kmpc|__tgt}}
#ifdef CK34

class C {
public:
  int a;
  double *b;
};

#pragma omp declare mapper(C s) map(s.a, s.b[0:2])

class S {
  int a;
  C c;
  int b;
public:
  void foo();
};

// TARGET_PARAM = 0x20
// MEMBER_OF_1 | TO = 0x1000000000001
// MEMBER_OF_1 | IMPLICIT | TO = 0x1000000000201
// CK34-DAG: [[MTYPE_TO:@.+]] = {{.+}}constant [4 x i64] [i64 [[#0x20]], i64 [[#0x1000000000001]], i64 [[#0x1000000000001]], i64 [[#0x1000000000201]]]
// TARGET_PARAM = 0x20
// MEMBER_OF_1 | FROM = 0x1000000000002
// MEMBER_OF_1 | IMPLICIT | FROM = 0x1000000000202
// CK34-DAG: [[MTYPE_FROM:@.+]] = {{.+}}constant [4 x i64] [i64 [[#0x20]], i64 [[#0x1000000000002]], i64 [[#0x1000000000002]], i64 [[#0x1000000000202]]]

void default_mapper() {
  S s;

  // CK34-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{.+}}, i32 4, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[4 x i{{.+}}]* [[MTYPE_TO]]{{.+}}, i8** null, i8** [[GEPMF:%.+]])
  // CK34-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK34-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK34-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  // CK34-DAG: [[GEPMF]] = bitcast [4 x i8*]* [[MF:%.+]] to i8**

  // pass TARGET_PARAM {&s, &s, ((void*)(&s+1)-(void*)&s)}

  // CK34-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK34-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK34-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK34-DAG: [[MF0:%.+]] = getelementptr inbounds {{.+}}[[MF]], i{{.+}} 0, i{{.+}} 0

  // CK34-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to %class.S**
  // CK34-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to %class.S**

  // CK34-DAG: store %class.S* [[S_ADDR:%.+]], %class.S** [[BPC0]],
  // CK34-DAG: store %class.S* [[S_ADDR]], %class.S** [[PC0]],
  // CK34-DAG: store i64 [[S_SIZE:%.+]], i64* [[S0]],
  // CK34-DAG: store i8* null, i8** [[MF0]],

  // CK34-DAG: [[S_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
  // CK34-DAG: [[SZ]] = sub i64 [[S_1_INTPTR:%.+]], [[S_INTPTR:%.+]]
  // CK34-DAG: [[S_1_INTPTR]] = ptrtoint i8* [[S_1_VOID:%.+]] to i64
  // CK34-DAG: [[S_INTPTR]] = ptrtoint i8* [[S_VOID:%.+]] to i64
  // CK34-DAG: [[S_1_VOID]] = bitcast %class.S* [[S_1:%.+]] to i8*
  // CK34-DAG: [[S_VOID]] = bitcast %class.S* [[S_ADDR]] to i8*
  // CK34-DAG: [[S_1]] = getelementptr %class.S, %class.S* [[S_ADDR]], i32 1

  // pass MEMBER_OF_1 | TO {&s, &s, ((void*)(&s.a+1)-(void*)&s)} to copy the data of s.a.

  // CK34-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK34-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK34-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK34-DAG: [[MF1:%.+]] = getelementptr inbounds {{.+}}[[MF]], i{{.+}} 0, i{{.+}} 1

  // CK34-DAG: [[BPC1:%.+]] = bitcast i8** [[BP1]] to %class.S**
  // CK34-DAG: [[PC1:%.+]] = bitcast i8** [[P1]] to %class.S**

  // CK34-DAG: store %class.S* [[S_ADDR]], %class.S** [[BPC1]],
  // CK34-DAG: store %class.S* [[S_ADDR]], %class.S** [[PC1]],
  // CK34-DAG: store i64 [[A_SIZE:%.+]], i64* [[S1]],
  // CK34-DAG: store i8* null, i8** [[MF1]],

  // CK34-DAG: [[A_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
  // CK34-DAG: [[SZ]] = sub i64 [[C_BEGIN_INTPTR:%.+]], [[S_INTPTR:%.+]]
  // CK34-DAG: [[S_INTPTR]] = ptrtoint i8* [[S_VOID:%.+]] to i64
  // CK34-DAG: [[C_BEGIN_INTPTR]] = ptrtoint i8* [[C_BEGIN_VOID:%.+]] to i64
  // CK34-DAG: [[S_VOID]] = bitcast %class.S* [[S_ADDR]] to i8*
  // CK34-DAG: [[C_BEGIN_VOID]] = bitcast %class.C* [[C_ADDR:%.+]] to i8*
  // CK34-64-DAG: [[C_ADDR]] = getelementptr inbounds %class.S, %class.S* [[S_ADDR]], i32 0, i32 2
  // CK34-32-DAG: [[C_ADDR]] = getelementptr inbounds %class.S, %class.S* [[S_ADDR]], i32 0, i32 1

  // pass MEMBER_OF_1 | TO {&s, &s.c+1, ((void*)(&s)+31+1-(void*)(&s.c+1))} to copy the data of s.b.

  // CK34-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK34-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK34-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK34-DAG: [[MF2:%.+]] = getelementptr inbounds {{.+}}[[MF]], i{{.+}} 0, i{{.+}} 2

  // CK34-DAG: [[BPC2:%.+]] = bitcast i8** [[BP2]] to %class.S**
  // CK34-DAG: [[PC2:%.+]] = bitcast i8** [[P2]] to %class.C**

  // CK34-DAG: store %class.S* [[S_ADDR]], %class.S** [[BPC2]],
  // CK34-DAG: store %class.C* [[C_END:%.+]], %class.C** [[PC2]],
  // CK34-DAG: store i64 [[B_SIZE:%.+]], i64* [[S2]],
  // CK34-DAG: store i8* null, i8** [[MF2]],

  // CK34-DAG: [[C_END]] = getelementptr %class.C, %class.C* [[C_ADDR]], i{{.+}} 1

  // CK34-DAG: [[B_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
  // CK34-DAG: [[SZ]] = sub i64 [[S_END_INTPTR:%.+]], [[C_END_INTPTR:%.+]]
  // CK34-DAG: [[C_END_INTPTR]] = ptrtoint i8* [[C_END_VOID:%.+]] to i64
  // CK34-DAG: [[S_END_INTPTR]] = ptrtoint i8* [[S_END_VOID:%.+]] to i64
  // CK34-DAG: [[C_END_VOID]] = bitcast %class.C* [[C_END]] to i8*
  // CK34-DAG: [[S_END_VOID]] = getelementptr i8, i8* [[S_LAST:%.+]], i{{.+}} 1
  // CK34-64-DAG: [[S_LAST]] = getelementptr i8, i8* [[S_VOID:%.+]], i64 31
  // CK34-32-DAG: [[S_LAST]] = getelementptr i8, i8* [[S_VOID:%.+]], i32 15
  // CK34-DAG: [[S_VOID]] = bitcast %class.S* [[S_ADDR]] to i8*

  // pass MEMBER_OF_1 | TO | IMPLICIT | MAPPER {&s, &s.c, 16} to copy the data of s.c.

  // CK34-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
  // CK34-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
  // CK34-DAG: [[S3:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 3
  // CK34-DAG: [[MF3:%.+]] = getelementptr inbounds {{.+}}[[MF]], i{{.+}} 0, i{{.+}} 3

  // CK34-DAG: [[BPC3:%.+]] = bitcast i8** [[BP3]] to %class.S**
  // CK34-DAG: [[PC3:%.+]] = bitcast i8** [[P3]] to %class.C**

  // CK34-DAG: store %class.S* [[S_ADDR]], %class.S** [[BPC3]],
  // CK34-DAG: store %class.C* [[C_ADDR:%.+]], %class.C** [[PC3]],
  // CK34-64-DAG: store i64 16, i64* [[S3]],
  // CK34-32-DAG: store i64 8, i64* [[S3]],
  // CK34-DAG: store i8* bitcast (void (i8*, i8*, i8*, i64, i64, i8*)* [[C_DEFAULT_MAPPER:@.+]] to i8*), i8** [[MF3]],

  // CK34-64-DAG: [[C_ADDR]] = getelementptr inbounds %class.S, %class.S* [[S_ADDR]], i32 0, i32 2
  // CK34-32-DAG: [[C_ADDR]] = getelementptr inbounds %class.S, %class.S* [[S_ADDR]], i32 0, i32 1

  #pragma omp target map(to: s)
  s.foo();

  // CK34 : call void

  // CK34-DAG: call i32 @__tgt_target_mapper(%struct.ident_t* @{{.+}}, i64 -1, i8* @{{.+}}, i32 4, i8** [[GEPBP:%.+]], i8** [[GEPP:%.+]], i64* [[GEPS:%.+]], {{.+}}getelementptr {{.+}}[4 x i{{.+}}]* [[MTYPE_FROM]]{{.+}}, i8** null, i8** [[GEPMF:%.+]])
  // CK34-DAG: [[GEPBP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK34-DAG: [[GEPP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK34-DAG: [[GEPS]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]
  // CK34-DAG: [[GEPMF]] = bitcast [4 x i8*]* [[MF:%.+]] to i8**

  // pass TARGET_PARAM {&s, &s, ((void*)(&s+1)-(void*)&s)}

  // CK34-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK34-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK34-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK34-DAG: [[MF0:%.+]] = getelementptr inbounds {{.+}}[[MF]], i{{.+}} 0, i{{.+}} 0

  // CK34-DAG: [[BPC0:%.+]] = bitcast i8** [[BP0]] to %class.S**
  // CK34-DAG: [[PC0:%.+]] = bitcast i8** [[P0]] to %class.S**

  // CK34-DAG: store %class.S* [[S_ADDR]], %class.S** [[BPC0]],
  // CK34-DAG: store %class.S* [[S_ADDR]], %class.S** [[PC0]],
  // CK34-DAG: store i64 [[S_SIZE:%.+]], i64* [[S0]],
  // CK34-DAG: store i8* null, i8** [[MF0]],

  // CK34-DAG: [[S_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
  // CK34-DAG: [[SZ]] = sub i64 [[S_1_INTPTR:%.+]], [[S_INTPTR:%.+]]
  // CK34-DAG: [[S_1_INTPTR]] = ptrtoint i8* [[S_1_VOID:%.+]] to i64
  // CK34-DAG: [[S_INTPTR]] = ptrtoint i8* [[S_VOID:%.+]] to i64
  // CK34-DAG: [[S_1_VOID]] = bitcast %class.S* [[S_1:%.+]] to i8*
  // CK34-DAG: [[S_VOID]] = bitcast %class.S* [[S_ADDR]] to i8*
  // CK34-DAG: [[S_1]] = getelementptr %class.S, %class.S* [[S_ADDR]], i32 1

  // pass MEMBER_OF_1 | FROM {&s, &s, ((void*)(&s.a+1)-(void*)&s)} to copy the data of s.a.

  // CK34-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK34-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK34-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK34-DAG: [[MF1:%.+]] = getelementptr inbounds {{.+}}[[MF]], i{{.+}} 0, i{{.+}} 1

  // CK34-DAG: [[BPC1:%.+]] = bitcast i8** [[BP1]] to %class.S**
  // CK34-DAG: [[PC1:%.+]] = bitcast i8** [[P1]] to %class.S**

  // CK34-DAG: store %class.S* [[S_ADDR]], %class.S** [[BPC1]],
  // CK34-DAG: store %class.S* [[S_ADDR]], %class.S** [[PC1]],
  // CK34-DAG: store i64 [[A_SIZE:%.+]], i64* [[S1]],
  // CK34-DAG: store i8* null, i8** [[MF1]],

  // CK34-DAG: [[A_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
  // CK34-DAG: [[SZ]] = sub i64 [[C_BEGIN_INTPTR:%.+]], [[S_INTPTR:%.+]]
  // CK34-DAG: [[S_INTPTR]] = ptrtoint i8* [[S_VOID:%.+]] to i64
  // CK34-DAG: [[C_BEGIN_INTPTR]] = ptrtoint i8* [[C_BEGIN_VOID:%.+]] to i64
  // CK34-DAG: [[S_VOID]] = bitcast %class.S* [[S_ADDR]] to i8*
  // CK34-DAG: [[C_BEGIN_VOID]] = bitcast %class.C* [[C_ADDR:%.+]] to i8*
  // CK34-64-DAG: [[C_ADDR]] = getelementptr inbounds %class.S, %class.S* [[S_ADDR]], i32 0, i32 2
  // CK34-32-DAG: [[C_ADDR]] = getelementptr inbounds %class.S, %class.S* [[S_ADDR]], i32 0, i32 1

  // pass MEMBER_OF_1 | FROM {&s, &s.c+1, ((void*)(&s)+31+1-(void*)(&s.c+1))} to copy the data of s.b.

  // CK34-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK34-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK34-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK34-DAG: [[MF2:%.+]] = getelementptr inbounds {{.+}}[[MF]], i{{.+}} 0, i{{.+}} 2

  // CK34-DAG: [[BPC2:%.+]] = bitcast i8** [[BP2]] to %class.S**
  // CK34-DAG: [[PC2:%.+]] = bitcast i8** [[P2]] to %class.C**

  // CK34-DAG: store %class.S* [[S_ADDR]], %class.S** [[BPC2]],
  // CK34-DAG: store %class.C* [[C_END:%.+]], %class.C** [[PC2]],
  // CK34-DAG: store i64 [[B_SIZE:%.+]], i64* [[S2]],
  // CK34-DAG: store i8* null, i8** [[MF2]],

  // CK34-DAG: [[C_END]] = getelementptr %class.C, %class.C* [[C_ADDR]], i{{.+}} 1

  // CK34-DAG: [[B_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (i8* getelementptr (i8, i8* null, i32 1) to i64)
  // CK34-DAG: [[SZ]] = sub i64 [[S_END_INTPTR:%.+]], [[C_END_INTPTR:%.+]]
  // CK34-DAG: [[C_END_INTPTR]] = ptrtoint i8* [[C_END_VOID:%.+]] to i64
  // CK34-DAG: [[S_END_INTPTR]] = ptrtoint i8* [[S_END_VOID:%.+]] to i64
  // CK34-DAG: [[C_END_VOID]] = bitcast %class.C* [[C_END]] to i8*
  // CK34-DAG: [[S_END_VOID]] = getelementptr i8, i8* [[S_LAST:%.+]], i{{.+}} 1
  // CK34-64-DAG: [[S_LAST]] = getelementptr i8, i8* [[S_VOID:%.+]], i64 31
  // CK34-32-DAG: [[S_LAST]] = getelementptr i8, i8* [[S_VOID:%.+]], i32 15
  // CK34-DAG: [[S_VOID]] = bitcast %class.S* [[S_ADDR]] to i8*

  // pass MEMBER_OF_1 | FROM | IMPLICIT | MAPPER {&s, &s.c, 16} to copy the data of s.c.

  // CK34-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
  // CK34-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
  // CK34-DAG: [[S3:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 3
  // CK34-DAG: [[MF3:%.+]] = getelementptr inbounds {{.+}}[[MF]], i{{.+}} 0, i{{.+}} 3

  // CK34-DAG: [[BPC3:%.+]] = bitcast i8** [[BP3]] to %class.S**
  // CK34-DAG: [[PC3:%.+]] = bitcast i8** [[P3]] to %class.C**

  // CK34-DAG: store %class.S* [[S_ADDR]], %class.S** [[BPC3]],
  // CK34-DAG: store %class.C* [[C_ADDR:%.+]], %class.C** [[PC3]],
  // CK34-64-DAG: store i64 16, i64* [[S3]],
  // CK34-32-DAG: store i64 8, i64* [[S3]],
  // CK34-DAG: store i8* bitcast (void (i8*, i8*, i8*, i64, i64, i8*)* [[C_DEFAULT_MAPPER]] to i8*), i8** [[MF3]],

  // CK34-64-DAG: [[C_ADDR]] = getelementptr inbounds %class.S, %class.S* [[S_ADDR]], i32 0, i32 2
  // CK34-32-DAG: [[C_ADDR]] = getelementptr inbounds %class.S, %class.S* [[S_ADDR]], i32 0, i32 1

  #pragma omp target map(from: s)
  s.foo();
}

#endif // CK34
#endif
