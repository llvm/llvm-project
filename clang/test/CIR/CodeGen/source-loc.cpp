// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void line_column() {
  unsigned int a = __builtin_LINE();
  unsigned int b = __builtin_COLUMN();
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["a", init]
// CIR: %[[B_ADDR:.*]] = cir.alloca !u32i, !cir.ptr<!u32i>, ["b", init]
// CIR: %[[CONST_9:.*]] = cir.const #cir.int<9> : !u32i
// CIR: cir.store {{.*}} %[[CONST_9]], %[[A_ADDR]] : !u32i, !cir.ptr<!u32i>
// CIR: %[[CONST_20:.*]] = cir.const #cir.int<20> : !u32i
// CIR: cir.store {{.*}} %[[CONST_20]], %[[B_ADDR]] : !u32i, !cir.ptr<!u32i>

// LLVM: %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: store i32 9, ptr %[[A_ADDR]], align 4
// LLVM: store i32 20, ptr %[[B_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[B_ADDR:.*]] = alloca i32, align 4
// OGCG: store i32 9, ptr %[[A_ADDR]], align 4
// OGCG: store i32 20, ptr %[[B_ADDR]], align 4

void function_file() {
  const char *a = __builtin_FUNCTION();
  const char *b = __builtin_FILE();
  const char *c = __builtin_FILE_NAME();
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>, ["a", init]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>, ["b", init]
// CIR: %[[C_ADDR:.*]] = cir.alloca !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>, ["c", init]
// CIR: %[[FUNC__GV:.*]] = cir.const #cir.global_view<@".str"> : !cir.ptr<!s8i>
// CIR: cir.store {{.*}} %[[FUNC__GV]], %[[A_ADDR]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CIR: %[[FILE_PATH_GV:.*]] = cir.const #cir.global_view<@".str.1"> : !cir.ptr<!s8i>
// CIR: cir.store {{.*}} %[[FILE_PATH_GV]], %[[B_ADDR]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CIR: %[[FILE_GV:.*]] = cir.const #cir.global_view<@".str.2"> : !cir.ptr<!s8i>
// CIR: cir.store {{.*}} %[[FILE_GV]], %[[C_ADDR]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>

// LLVM: %[[A_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM: %[[B_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM: %[[C_ADDR:.*]] = alloca ptr, i64 1, align 8
// LLVM: store ptr @.str, ptr %[[A_ADDR]], align 8
// LLVM: store ptr @.str.1, ptr %[[B_ADDR]], align 8
// LLVM: store ptr @.str.2, ptr %[[C_ADDR]], align 8

// OGCG: %[[A_ADDR:.*]] = alloca ptr, align 8
// OGCG: %[[B_ADDR:.*]] = alloca ptr, align 8
// OGCG: %[[C_ADDR:.*]] = alloca ptr, align 8
// OGCG: store ptr @.str, ptr %[[A_ADDR]], align 8
// OGCG: store ptr @.str.1, ptr %[[B_ADDR]], align 8
// OGCG: store ptr @.str.2, ptr %[[C_ADDR]], align 8
