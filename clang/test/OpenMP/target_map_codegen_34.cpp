// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK34 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK34 --check-prefix CK34-64
// RUN: %clang_cc1 -DCK34 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK34 --check-prefix CK34-64
// RUN: %clang_cc1 -DCK34 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK34 --check-prefix CK34-32
// RUN: %clang_cc1 -DCK34 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK34 --check-prefix CK34-32

// RUN: %clang_cc1 -DCK34 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// RUN: %clang_cc1 -DCK34 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// RUN: %clang_cc1 -DCK34 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// RUN: %clang_cc1 -DCK34 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY32 %s
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

// CK34-DAG: [[SIZE_TO:@.+]] = private {{.*}}constant [4 x i64] [i64 0, i64 0, i64 0, i64 {{16|8}}]
// TARGET_PARAM = 0x20
// MEMBER_OF_1 | TO = 0x1000000000001
// MEMBER_OF_1 | IMPLICIT | TO = 0x1000000000201
// CK34-DAG: [[MTYPE_TO:@.+]] = {{.+}}constant [4 x i64] [i64 [[#0x20]], i64 [[#0x1000000000001]], i64 [[#0x1000000000001]], i64 [[#0x1000000000201]]]
// CK34-DAG: [[SIZE_FROM:@.+]] = private {{.*}}constant [4 x i64] [i64 0, i64 0, i64 0, i64 {{16|8}}]
// TARGET_PARAM = 0x20
// MEMBER_OF_1 | FROM = 0x1000000000002
// MEMBER_OF_1 | IMPLICIT | FROM = 0x1000000000202
// CK34-DAG: [[MTYPE_FROM:@.+]] = {{.+}}constant [4 x i64] [i64 [[#0x20]], i64 [[#0x1000000000002]], i64 [[#0x1000000000002]], i64 [[#0x1000000000202]]]

void default_mapper() {
  S s;

  // CK34-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
  // CK34-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
  // CK34-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
  // CK34-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
  // CK34-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
  // CK34-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
  // CK34-DAG: store ptr [[SIZES:%.+]], ptr [[SARG]]
  // CK34-DAG: [[MARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 7
  // CK34-DAG: store ptr [[MF:%.+]], ptr [[MARG]]
  // CK34-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK34-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK34-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // pass TARGET_PARAM {&s, &s, ((void*)(&s+1)-(void*)&s)}

  // CK34-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK34-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK34-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK34-DAG: [[MF0:%.+]] = getelementptr inbounds {{.+}}[[MF]], i{{.+}} 0, i{{.+}} 0

  // CK34-DAG: store ptr [[S_ADDR:%.+]], ptr [[BP0]],
  // CK34-DAG: store ptr [[S_ADDR]], ptr [[P0]],
  // CK34-DAG: store i64 [[S_SIZE:%.+]], ptr [[S0]],
  // CK34-DAG: store ptr null, ptr [[MF0]],

  // CK34-DAG: [[S_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
  // CK34-DAG: [[SZ]] = sub i64 [[S_1_INTPTR:%.+]], [[S_INTPTR:%.+]]
  // CK34-DAG: [[S_1_INTPTR]] = ptrtoint ptr [[S_1:%.+]] to i64
  // CK34-DAG: [[S_INTPTR]] = ptrtoint ptr [[S_ADDR]] to i64
  // CK34-DAG: [[S_1]] = getelementptr %class.S, ptr [[S_ADDR]], i32 1

  // pass MEMBER_OF_1 | TO {&s, &s, ((void*)(&s.a+1)-(void*)&s)} to copy the data of s.a.

  // CK34-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK34-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK34-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK34-DAG: [[MF1:%.+]] = getelementptr inbounds {{.+}}[[MF]], i{{.+}} 0, i{{.+}} 1

  // CK34-DAG: store ptr [[S_ADDR]], ptr [[BP1]],
  // CK34-DAG: store ptr [[S_ADDR]], ptr [[P1]],
  // CK34-DAG: store i64 [[A_SIZE:%.+]], ptr [[S1]],
  // CK34-DAG: store ptr null, ptr [[MF1]],

  // CK34-DAG: [[A_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
  // CK34-DAG: [[SZ]] = sub i64 [[C_BEGIN_INTPTR:%.+]], [[S_INTPTR:%.+]]
  // CK34-DAG: [[S_INTPTR]] = ptrtoint ptr [[S_ADDR]] to i64
  // CK34-DAG: [[C_BEGIN_INTPTR]] = ptrtoint ptr [[C_ADDR:%.+]] to i64
  // CK34-64-DAG: [[C_ADDR]] = getelementptr inbounds nuw %class.S, ptr [[S_ADDR]], i32 0, i32 2
  // CK34-32-DAG: [[C_ADDR]] = getelementptr inbounds nuw %class.S, ptr [[S_ADDR]], i32 0, i32 1

  // pass MEMBER_OF_1 | TO {&s, &s.c+1, ((void*)(&s)+31+1-(void*)(&s.c+1))} to copy the data of s.b.

  // CK34-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK34-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK34-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK34-DAG: [[MF2:%.+]] = getelementptr inbounds {{.+}}[[MF]], i{{.+}} 0, i{{.+}} 2

  // CK34-DAG: store ptr [[S_ADDR]], ptr [[BP2]],
  // CK34-DAG: store ptr [[C_END:%.+]], ptr [[P2]],
  // CK34-DAG: store i64 [[B_SIZE:%.+]], ptr [[S2]],
  // CK34-DAG: store ptr null, ptr [[MF2]],

  // CK34-DAG: [[C_END]] = getelementptr %class.C, ptr [[C_ADDR]], i{{.+}} 1

  // CK34-DAG: [[B_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
  // CK34-DAG: [[SZ]] = sub i64 [[S_END_INTPTR:%.+]], [[C_END_INTPTR:%.+]]
  // CK34-DAG: [[C_END_INTPTR]] = ptrtoint ptr [[C_END]] to i64
  // CK34-DAG: [[S_END_INTPTR]] = ptrtoint ptr [[S_END_VOID:%.+]] to i64
  // CK34-DAG: [[S_END_VOID]] = getelementptr i8, ptr [[S_LAST:%.+]], i{{.+}} 1
  // CK34-64-DAG: [[S_LAST]] = getelementptr i8, ptr [[S_ADDR]], i64 31
  // CK34-32-DAG: [[S_LAST]] = getelementptr i8, ptr [[S_ADDR]], i32 15

  // pass MEMBER_OF_1 | TO | IMPLICIT | MAPPER {&s, &s.c, 16} to copy the data of s.c.

  // CK34-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
  // CK34-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
  // CK34-DAG: [[MF3:%.+]] = getelementptr inbounds {{.+}}[[MF]], i{{.+}} 0, i{{.+}} 3

  // CK34-DAG: store ptr [[S_ADDR]], ptr [[BP3]],
  // CK34-DAG: store ptr [[C_ADDR:%.+]], ptr [[P3]],
  // CK34-DAG: store ptr [[C_DEFAULT_MAPPER:@.+]], ptr [[MF3]],

  // CK34-64-DAG: [[C_ADDR]] = getelementptr inbounds nuw %class.S, ptr [[S_ADDR]], i32 0, i32 2
  // CK34-32-DAG: [[C_ADDR]] = getelementptr inbounds nuw %class.S, ptr [[S_ADDR]], i32 0, i32 1

  #pragma omp target map(to: s)
  s.foo();

  // CK34 : call void

  // CK34-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
  // CK34-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
  // CK34-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
  // CK34-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
  // CK34-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
  // CK34-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
  // CK34-DAG: store ptr [[SIZES:%.+]], ptr [[SARG]]
  // CK34-DAG: [[MARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 7
  // CK34-DAG: store ptr [[MF:%.+]], ptr [[MARG]]
  // CK34-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK34-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK34-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // pass TARGET_PARAM {&s, &s, ((void*)(&s+1)-(void*)&s)}

  // CK34-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK34-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK34-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0
  // CK34-DAG: [[MF0:%.+]] = getelementptr inbounds {{.+}}[[MF]], i{{.+}} 0, i{{.+}} 0

  // CK34-DAG: store ptr [[S_ADDR]], ptr [[BP0]],
  // CK34-DAG: store ptr [[S_ADDR]], ptr [[P0]],
  // CK34-DAG: store i64 [[S_SIZE:%.+]], ptr [[S0]],
  // CK34-DAG: store ptr null, ptr [[MF0]],

  // CK34-DAG: [[S_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
  // CK34-DAG: [[SZ]] = sub i64 [[S_1_INTPTR:%.+]], [[S_INTPTR:%.+]]
  // CK34-DAG: [[S_1_INTPTR]] = ptrtoint ptr [[S_1:%.+]] to i64
  // CK34-DAG: [[S_INTPTR]] = ptrtoint ptr [[S_ADDR]] to i64
  // CK34-DAG: [[S_1]] = getelementptr %class.S, ptr [[S_ADDR]], i32 1

  // pass MEMBER_OF_1 | FROM {&s, &s, ((void*)(&s.a+1)-(void*)&s)} to copy the data of s.a.

  // CK34-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK34-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK34-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1
  // CK34-DAG: [[MF1:%.+]] = getelementptr inbounds {{.+}}[[MF]], i{{.+}} 0, i{{.+}} 1

  // CK34-DAG: store ptr [[S_ADDR]], ptr [[BP1]],
  // CK34-DAG: store ptr [[S_ADDR]], ptr [[P1]],
  // CK34-DAG: store i64 [[A_SIZE:%.+]], ptr [[S1]],
  // CK34-DAG: store ptr null, ptr [[MF1]],

  // CK34-DAG: [[A_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
  // CK34-DAG: [[SZ]] = sub i64 [[C_BEGIN_INTPTR:%.+]], [[S_INTPTR:%.+]]
  // CK34-DAG: [[S_INTPTR]] = ptrtoint ptr [[S_ADDR]] to i64
  // CK34-DAG: [[C_BEGIN_INTPTR]] = ptrtoint ptr [[C_ADDR:%.+]] to i64
  // CK34-64-DAG: [[C_ADDR]] = getelementptr inbounds nuw %class.S, ptr [[S_ADDR]], i32 0, i32 2
  // CK34-32-DAG: [[C_ADDR]] = getelementptr inbounds nuw %class.S, ptr [[S_ADDR]], i32 0, i32 1

  // pass MEMBER_OF_1 | FROM {&s, &s.c+1, ((void*)(&s)+31+1-(void*)(&s.c+1))} to copy the data of s.b.

  // CK34-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK34-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK34-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2
  // CK34-DAG: [[MF2:%.+]] = getelementptr inbounds {{.+}}[[MF]], i{{.+}} 0, i{{.+}} 2

  // CK34-DAG: store ptr [[S_ADDR]], ptr [[BP2]],
  // CK34-DAG: store ptr [[C_END:%.+]], ptr [[P2]],
  // CK34-DAG: store i64 [[B_SIZE:%.+]], ptr [[S2]],
  // CK34-DAG: store ptr null, ptr [[MF2]],

  // CK34-DAG: [[C_END]] = getelementptr %class.C, ptr [[C_ADDR]], i{{.+}} 1

  // CK34-DAG: [[B_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
  // CK34-DAG: [[SZ]] = sub i64 [[S_END_INTPTR:%.+]], [[C_END_INTPTR:%.+]]
  // CK34-DAG: [[C_END_INTPTR]] = ptrtoint ptr [[C_END]] to i64
  // CK34-DAG: [[S_END_INTPTR]] = ptrtoint ptr [[S_END_VOID:%.+]] to i64
  // CK34-DAG: [[S_END_VOID]] = getelementptr i8, ptr [[S_LAST:%.+]], i{{.+}} 1
  // CK34-64-DAG: [[S_LAST]] = getelementptr i8, ptr [[S_ADDR]], i64 31
  // CK34-32-DAG: [[S_LAST]] = getelementptr i8, ptr [[S_ADDR]], i32 15

  // pass MEMBER_OF_1 | FROM | IMPLICIT | MAPPER {&s, &s.c, 16} to copy the data of s.c.

  // CK34-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
  // CK34-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3
  // CK34-DAG: [[MF3:%.+]] = getelementptr inbounds {{.+}}[[MF]], i{{.+}} 0, i{{.+}} 3

  // CK34-DAG: store ptr [[S_ADDR]], ptr [[BP3]],
  // CK34-DAG: store ptr [[C_ADDR:%.+]], ptr [[P3]],
  // CK34-DAG: store ptr [[C_DEFAULT_MAPPER]], ptr [[MF3]],

  // CK34-64-DAG: [[C_ADDR]] = getelementptr inbounds nuw %class.S, ptr [[S_ADDR]], i32 0, i32 2
  // CK34-32-DAG: [[C_ADDR]] = getelementptr inbounds nuw %class.S, ptr [[S_ADDR]], i32 0, i32 1

  #pragma omp target map(from: s)
  s.foo();
}

#endif // CK34
#endif
