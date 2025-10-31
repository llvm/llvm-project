// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++2c -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++2c -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++2c -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

int pack_indexing(auto... p) { return p...[0]; }

// CIR: %[[P_0:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["p", init]
// CIR: %[[P_1:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["p", init]
// CIR: %[[P_2:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["p", init]
// CIR: %[[RET_VAL:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR: %[[RESULT:.*]] = cir.load{{.*}} %[[P_0]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store %[[RESULT]], %[[RET_VAL]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP:.*]] = cir.load %[[RET_VAL]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return %[[TMP]] : !s32i

// LLVM: %[[P_0:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[P_1:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[P_2:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[RET_VAL:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[RESULT:.*]] = load i32, ptr %[[P_0]], align 4
// LLVM: store i32 %[[RESULT]], ptr %[[RET_VAL]], align 4
// LLVM: %[[TMP:.*]] = load i32, ptr %[[RET_VAL]], align 4
// LLVM: ret i32 %[[TMP]]

// OGCG-DAG: %[[P_0:.*]] = alloca i32, align 4
// OGCG-DAG: %[[P_1:.*]] = alloca i32, align 4
// OGCG-DAG: %[[P_2:.*]] = alloca i32, align 4
// OGCG-DAG: %[[RESULT:.*]] = load i32, ptr %[[P_0]], align 4
// OGCG-DAG-NEXT: ret i32 %[[RESULT]]

int foo() { return pack_indexing(1, 2, 3); }

// CIR: %[[RET_VAL:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__retval"]
// CIR: %[[RESULT:.*]] = cir.call @_Z13pack_indexingIJiiiEEiDpT_({{.*}}, {{.*}}, {{.*}}) : (!s32i, !s32i, !s32i) -> !s32i
// CIR: cir.store %[[RESULT]], %[[RET_VAL]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[TMP:.*]] = cir.load %[[RET_VAL]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.return %[[TMP]] : !s32i

// LLVM: %[[RET_VAL:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[RESULT:.*]] = call i32 @_Z13pack_indexingIJiiiEEiDpT_(i32 1, i32 2, i32 3)
// LLVM: store i32 %[[RESULT]], ptr %[[RET_VAL]], align 4
// LLVM: %[[TMP:.*]] = load i32, ptr %[[RET_VAL]], align 4
// LLVM: ret i32 %[[TMP]]

// OGCG-DAG: %[[CALL:.*]] = call noundef i32 @_Z13pack_indexingIJiiiEEiDpT_(i32 noundef 1, i32 noundef 2, i32 noundef 3)
// OGCG-DAG-NEXT: ret i32 %[[RESULT]]
