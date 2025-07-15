// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

int foo() {
  int e = (int){1};
  return e;
}

// CIR: %[[RET:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR: %[[INIT:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["e", init]
// CIR: %[[COMPOUND:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, [".compoundliteral", init]
// CIR: %[[VALUE:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store{{.*}} %[[VALUE]], %[[COMPOUND]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[COMPOUND]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store{{.*}}  %[[TMP]], %[[INIT]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP_2:.*]] = cir.load{{.*}} %[[INIT]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store %[[TMP_2]], %[[RET]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP_3:.*]] = cir.load %[[RET]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return %[[TMP_3]] : !s32i

// LLVM: %[[RET:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[INIT:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[COMPOUND:.*]] = alloca i32, i64 1, align 4
// LLVM: store i32 1, ptr %[[COMPOUND]], align 4
// LLVM: %[[TMP:.*]] = load i32, ptr %[[COMPOUND]], align 4
// LLVM: store i32 %[[TMP]], ptr %[[INIT]], align 4
// LLVM: %[[TMP_2:.*]] = load i32, ptr %[[INIT]], align 4
// LLVM: store i32 %[[TMP_2]], ptr %[[RET]], align 4
// LLVM: %[[TMP_3:.*]] = load i32, ptr %[[RET]], align 4
// LLVM: ret i32 %[[TMP_3]]

// OGCG: %[[INIT:.*]] = alloca i32, align 4
// OGCG: %[[COMPOUND:.*]] = alloca i32, align 4
// OGCG: store i32 1, ptr %[[COMPOUND]], align 4
// OGCG: %[[TMP:.*]] = load i32, ptr %[[COMPOUND]], align 4
// OGCG: store i32 %[[TMP]], ptr %[[INIT]], align 4
// OGCG: %[[TMP_2:.*]] = load i32, ptr %[[INIT]], align 4
// OGCG: ret i32 %[[TMP_2]]
