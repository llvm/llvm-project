// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t-cir.ll %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -emit-llvm %s -o %t.ll
// RUN: FileCheck --check-prefix=OGCG --input-file=%t.ll %s

// LLVM: @[[STR1_GLOBAL:.*]] = dso_local global [2 x i8] c"1\00"
// LLVM: @[[STR2_GLOBAL:.*]] = dso_local global [1 x i8] zeroinitializer
// LLVM: @[[STR3_GLOBAL:.*]] = dso_local global [2 x i8] zeroinitializer

// OGCG: @[[STR1_GLOBAL:.*]] = private unnamed_addr constant [2 x i8] c"1\00"
// OGCG: @[[STR2_GLOBAL:.*]] = private unnamed_addr constant [1 x i8] zeroinitializer
// OGCG: @[[STR3_GLOBAL:.*]] = private unnamed_addr constant [2 x i8] zeroinitializer

char *f1() {
  return "1";
}

// CIR: cir.global external @[[STR1_GLOBAL:.*]] = #cir.const_array<"1\00" : !cir.array<!s8i x 2>> : !cir.array<!s8i x 2>
// CIR: cir.func @f1()
// CIR:   %[[STR:.*]] = cir.get_global @[[STR1_GLOBAL]] : !cir.ptr<!cir.array<!s8i x 2>>

// LLVM: define ptr @f1()
// LLVM:   store ptr @[[STR1_GLOBAL]], ptr {{.*}}

// OGCG: define {{.*}}ptr @f1()
// OGCG:   ret ptr @[[STR1_GLOBAL]]

char *f2() {
  return "";
}

// CIR: cir.global external @[[STR2_GLOBAL:.*]] = #cir.zero : !cir.array<!s8i x 1>
// CIR: cir.func @f2()
// CIR:   %[[STR2:.*]] = cir.get_global @[[STR2_GLOBAL]] : !cir.ptr<!cir.array<!s8i x 1>>

// LLVM: define ptr @f2()
// LLVM:   store ptr @[[STR2_GLOBAL]], ptr {{.*}}

// OGCG: define {{.*}}ptr @f2()
// OGCG:   ret ptr @[[STR2_GLOBAL]]

char *f3() {
  return "\00";
}

// CIR: cir.global external @[[STR3_GLOBAL:.*]] = #cir.zero : !cir.array<!s8i x 2>
// CIR: cir.func @f3()
// CIR:   %[[STR3:.*]] = cir.get_global @[[STR3_GLOBAL]] : !cir.ptr<!cir.array<!s8i x 2>>

// LLVM: define ptr @f3()
// LLVM:   store ptr @[[STR3_GLOBAL]], ptr {{.*}}

// OGCG: define {{.*}}ptr @f3()
// OGCG:   ret ptr @[[STR3_GLOBAL]]
