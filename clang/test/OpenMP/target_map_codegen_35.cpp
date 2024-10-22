// expected-no-diagnostics
#ifndef HEADER
#define HEADER

///==========================================================================///
// RUN: %clang_cc1 -DCK35 -verify -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck %s --check-prefix CK35 --check-prefix CK35-64
// RUN: %clang_cc1 -DCK35 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK35 --check-prefix CK35-64
// RUN: %clang_cc1 -DCK35 -verify -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck %s  --check-prefix CK35 --check-prefix CK35-32
// RUN: %clang_cc1 -DCK35 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck %s  --check-prefix CK35 --check-prefix CK35-32

// RUN: %clang_cc1 -DCK35 -verify -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// RUN: %clang_cc1 -DCK35 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -std=c++11 -triple powerpc64le-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=powerpc64le-ibm-linux-gnu -x c++ -triple powerpc64le-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// RUN: %clang_cc1 -DCK35 -verify -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -emit-llvm %s -o - | FileCheck --check-prefix SIMD-ONLY32 %s
// RUN: %clang_cc1 -DCK35 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -std=c++11 -triple i386-unknown-unknown -emit-pch -o %t %s
// RUN: %clang_cc1 -fopenmp-simd -fopenmp-targets=i386-pc-linux-gnu -x c++ -triple i386-unknown-unknown -std=c++11 -include-pch %t -verify %s -emit-llvm -o - | FileCheck --check-prefix SIMD-ONLY32 %s
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

  // CK35-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
  // CK35-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
  // CK35-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
  // CK35-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
  // CK35-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
  // CK35-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
  // CK35-DAG: store ptr [[SIZES:%.+]], ptr [[SARG]]
  // CK35-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK35-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK35-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // pass TARGET_PARAM {&s, &s, ((ptr)(&s+1)-(ptr)&s)}

  // CK35-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK35-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK35-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0


  // CK35-DAG: store ptr [[S_ADDR:%.+]], ptr [[BP0]],
  // CK35-DAG: store ptr [[S_ADDR]], ptr [[P0]],
  // CK35-DAG: store i64 [[S_SIZE:%.+]], ptr [[S0]],

  // CK35-DAG: [[S_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
  // CK35-DAG: [[SZ]] = sub i64 [[S_1_INTPTR:%.+]], [[S_INTPTR:%.+]]
  // CK35-DAG: [[S_1_INTPTR]] = ptrtoint ptr [[S_1_VOID:%.+]] to i64
  // CK35-DAG: [[S_INTPTR]] = ptrtoint ptr [[S_VOID:%.+]] to i64
  // CK35-DAG: [[S_1:%.+]] = getelementptr %class.S, ptr [[S_ADDR]], i32 1

  // pass MEMBER_OF_1 | TO {&s, &s, ((ptr)(&s.a+1)-(ptr)&s)} to copy the data of s.a.

  // CK35-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK35-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1
  // CK35-DAG: [[S1:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 1


  // CK35-DAG: store ptr [[S_ADDR]], ptr [[BP1]],
  // CK35-DAG: store ptr [[S_ADDR]], ptr [[P1]],
  // CK35-DAG: store i64 [[A_SIZE:%.+]], ptr [[S1]],

  // CK35-DAG: [[A_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
  // CK35-DAG: [[SZ]] = sub i64 [[B_BEGIN_INTPTR:%.+]], [[S_INTPTR:%.+]]
  // CK35-DAG: [[S_INTPTR]] = ptrtoint ptr [[S_VOID:%.+]] to i64
  // CK35-DAG: [[B_BEGIN_INTPTR]] = ptrtoint ptr [[B_BEGIN_VOID:%.+]] to i64
  // CK35-DAG: [[B_ADDR:%.+]] = getelementptr inbounds nuw %class.S, ptr [[S_ADDR]], i32 0, i32 1

  // pass MEMBER_OF_1 | TO {&s, &s.b+1, ((ptr)(&s+1)-(ptr)(&s.b+1))} to copy the data of remainder of s.

  // CK35-DAG: [[BP2:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 2
  // CK35-DAG: [[P2:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 2
  // CK35-DAG: [[S2:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 2


  // CK35-DAG: store ptr [[S_ADDR]], ptr [[BP2]],
  // CK35-DAG: store ptr [[B_END:%.+]], ptr [[P2]],
  // CK35-DAG: store i64 [[REM_SIZE:%.+]], ptr [[S2]],

  // CK35-DAG: [[B_END]] = getelementptr ptr, ptr [[B_ADDR]], i{{.+}} 1

  // CK35-DAG: [[REM_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
  // CK35-DAG: [[SZ]] = sub i64 [[S_END_INTPTR:%.+]], [[B_END_INTPTR:%.+]]
  // CK35-DAG: [[B_END_INTPTR]] = ptrtoint ptr [[B_END_VOID:%.+]] to i64
  // CK35-DAG: [[S_END_INTPTR]] = ptrtoint ptr [[S_END_VOID:%.+]] to i64
  // CK35-DAG: [[S_END_VOID]] = getelementptr i8, ptr [[S_LAST:%.+]], i{{.+}} 1
  // CK35-64-DAG: [[S_LAST]] = getelementptr i8, ptr [[S_VOIDPTR:%.+]], i64 15
  // CK35-32-DAG: [[S_LAST]] = getelementptr i8, ptr [[S_VOIDPTR:%.+]], i32 7

  // pass MEMBER_OF_1 | PTR_AND_OBJ | TO {&s, &s.b, 8|4} to copy the data of s.b.

  // CK35-DAG: [[BP3:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 3
  // CK35-DAG: [[P3:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 3


  // CK35-DAG: store ptr [[S_ADDR]], ptr [[BP3]],
  // CK35-DAG: store ptr [[B_ADDR:%.+]], ptr [[P3]],

  // CK35-DAG: [[B_ADDR]] = load ptr, ptr [[B_REF:%.+]],
  // CK35-DAG: [[B_REF]] = getelementptr inbounds nuw %class.S, ptr [[S_ADDR]], i32 0, i32 1

  #pragma omp target map(to: s, s.b)
  s.foo();

  // CK35 : call void

  // CK35-DAG: call i32 @__tgt_target_kernel(ptr @{{.+}}, i64 -1, i32 -1, i32 0, ptr @.{{.+}}.region_id, ptr [[ARGS:%.+]])
  // CK35-DAG: [[BPARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 2
  // CK35-DAG: store ptr [[BPGEP:%.+]], ptr [[BPARG]]
  // CK35-DAG: [[PARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 3
  // CK35-DAG: store ptr [[PGEP:%.+]], ptr [[PARG]]
  // CK35-DAG: [[SARG:%.+]] = getelementptr inbounds {{.+}}[[ARGS]], i32 0, i32 4
  // CK35-DAG: store ptr [[SIZES:%.+]], ptr [[SARG]]
  // CK35-DAG: [[BPGEP]] = getelementptr inbounds {{.+}}[[BP:%[^,]+]]
  // CK35-DAG: [[PGEP]] = getelementptr inbounds {{.+}}[[P:%[^,]+]]
  // CK35-DAG: [[SIZES]] = getelementptr inbounds {{.+}}[[S:%[^,]+]]

  // pass TARGET_PARAM {&s, &s.b, ((ptr)(&s.b+1)-(ptr)&s.b)}

  // CK35-DAG: [[BP0:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 0
  // CK35-DAG: [[P0:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 0
  // CK35-DAG: [[S0:%.+]] = getelementptr inbounds {{.+}}[[S]], i{{.+}} 0, i{{.+}} 0


  // CK35-DAG: store ptr [[S_ADDR]], ptr [[BP0]],
  // CK35-DAG: store ptr [[SB_ADDR:%.+]], ptr [[P0]],
  // CK35-DAG: store i64 [[B_SIZE:%.+]], ptr [[S0]],

  // CK35-DAG: [[B_SIZE]] = sdiv exact i64 [[SZ:%.+]], ptrtoint (ptr getelementptr (i8, ptr null, i32 1) to i64)
  // CK35-DAG: [[SZ]] = sub i64 [[SB_1_INTPTR:%.+]], [[SB_INTPTR:%.+]]
  // CK35-DAG: [[SB_1_INTPTR]] = ptrtoint ptr [[SB_1_VOID:%.+]] to i64
  // CK35-DAG: [[SB_INTPTR]] = ptrtoint ptr [[SB_VOID:%.+]] to i64
  // CK35-DAG: [[SB_ADDR:%.+]] = getelementptr inbounds nuw %class.S, ptr [[S_ADDR]], i32 0, i32 1
  // CK35-DAG: [[SB_1:%.+]] = getelementptr ptr, ptr [[SB_ADDR]], i{{.+}} 1

  // pass MEMBER_OF_1 | PTR_AND_OBJ | FROM {&s, &s.b, 8|4} to copy the data of s.c.

  // CK35-DAG: [[BP1:%.+]] = getelementptr inbounds {{.+}}[[BP]], i{{.+}} 0, i{{.+}} 1
  // CK35-DAG: [[P1:%.+]] = getelementptr inbounds {{.+}}[[P]], i{{.+}} 0, i{{.+}} 1


  // CK35-DAG: store ptr [[S_ADDR]], ptr [[BP1]],
  // CK35-DAG: store ptr [[B_ADDR:%.+]], ptr [[P1]],

  // CK35-DAG: [[B_ADDR]] = load ptr, ptr [[SB_ADDR]],

  #pragma omp target map(from: s.b)
  s.foo();
}

#endif // CK35
#endif
