// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

int f1(int i);

int f1(int i) {
  i;
  return i;
}

//      CIR: module
// CIR-NEXT: cir.func @f1(%arg0: !cir.int<s, 32> loc({{.*}})) -> !cir.int<s, 32>
// CIR-NEXT:   %[[I_PTR:.*]] = cir.alloca !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>, ["i", init] {alignment = 4 : i64}
// CIR-NEXT:   cir.store %arg0, %[[I_PTR]] : !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>
// CIR-NEXT:   %[[I_IGNORED:.*]] = cir.load %[[I_PTR]] : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CIR-NEXT:   %[[I:.*]] = cir.load %[[I_PTR]] : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CIR-NEXT:   cir.return %[[I]] : !cir.int<s, 32>

//      LLVM: define i32 @f1(i32 %[[I:.*]])
// LLVM-NEXT:   %[[I_PTR:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT:   store i32 %[[I]], ptr %[[I_PTR]], align 4
// LLVM-NEXT:   %[[I_IGNORED:.*]] = load i32, ptr %[[I_PTR]], align 4
// LLVM-NEXT:   %[[I:.*]] = load i32, ptr %[[I_PTR]], align 4
// LLVM-NEXT:   ret i32 %[[I]]

//      OGCG: define{{.*}} i32 @f1(i32 noundef %[[I:.*]])
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[I_PTR:.*]] = alloca i32, align 4
// OGCG-NEXT:   store i32 %[[I]], ptr %[[I_PTR]], align 4
// OGCG-NEXT:   %[[I_IGNORED:.*]] = load i32, ptr %[[I_PTR]], align 4
// OGCG-NEXT:   %[[I:.*]] = load i32, ptr %[[I_PTR]], align 4
// OGCG-NEXT:   ret i32 %[[I]]

int f2(void) { return 3; }

//      CIR: cir.func @f2() -> !cir.int<s, 32>
// CIR-NEXT:   %[[THREE:.*]] = cir.const #cir.int<3> : !cir.int<s, 32>
// CIR-NEXT:   cir.return %[[THREE]] : !cir.int<s, 32>

//      LLVM: define i32 @f2()
// LLVM-NEXT:   ret i32 3

//      OGCG: define{{.*}} i32 @f2()
// OGCG-NEXT: entry:
// OGCG-NEXT:   ret i32 3

int f3(void) {
  int i = 3;
  return i;
}

//      CIR: cir.func @f3() -> !cir.int<s, 32>
// CIR-NEXT:   %[[I_PTR:.*]] = cir.alloca !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>, ["i", init] {alignment = 4 : i64}
// CIR-NEXT:   %[[THREE:.*]] = cir.const #cir.int<3> : !cir.int<s, 32>
// CIR-NEXT:   cir.store %[[THREE]], %[[I_PTR]] : !cir.int<s, 32>, !cir.ptr<!cir.int<s, 32>>
// CIR-NEXT:   %[[I:.*]] = cir.load %[[I_PTR]] : !cir.ptr<!cir.int<s, 32>>, !cir.int<s, 32>
// CIR-NEXT:   cir.return %[[I]] : !cir.int<s, 32>

//      LLVM: define i32 @f3()
// LLVM-NEXT:   %[[I_PTR:.*]] = alloca i32, i64 1, align 4
// LLVM-NEXT:   store i32 3, ptr %[[I_PTR]], align 4
// LLVM-NEXT:   %[[I:.*]] = load i32, ptr %[[I_PTR]], align 4
// LLVM-NEXT:   ret i32 %[[I]]

//      OGCG: define{{.*}} i32 @f3
// OGCG-NEXT: entry:
// OGCG-NEXT:   %[[I_PTR:.*]] = alloca i32, align 4
// OGCG-NEXT:   store i32 3, ptr %[[I_PTR]], align 4
// OGCG-NEXT:   %[[I:.*]] = load i32, ptr %[[I_PTR]], align 4
// OGCG-NEXT:   ret i32 %[[I]]
