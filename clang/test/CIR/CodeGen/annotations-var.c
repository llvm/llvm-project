// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// LLVM-DAG: @.str.annotation = private unnamed_addr constant [15 x i8] c"localvar_ann_0\00", section "llvm.metadata"
// LLVM-DAG: @.str.1.annotation = private unnamed_addr constant [{{[0-9]+}} x i8] c"{{.*}}annotations-var.c\00", section "llvm.metadata"
// LLVM-DAG: @.str.2.annotation = private unnamed_addr constant [15 x i8] c"localvar_ann_1\00", section "llvm.metadata"

void local(void) {
    int localvar __attribute__((annotate("localvar_ann_0"))) __attribute__((annotate("localvar_ann_1"))) = 3;
// CIR-LABEL: @local
// CIR: %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["localvar", init] [#cir.annotation<name = "localvar_ann_0", args = []>, #cir.annotation<name = "localvar_ann_1", args = []>]

// LLVM-LABEL: @local
// LLVM: %[[ALLOC:.*]] = alloca i32
// LLVM: call void @llvm.var.annotation.p0.p0(ptr %[[ALLOC]], ptr @.str.annotation, ptr @.str.1.annotation, i32 11, ptr null)
// LLVM: call void @llvm.var.annotation.p0.p0(ptr %[[ALLOC]], ptr @.str.2.annotation, ptr @.str.1.annotation, i32 11, ptr null)
}
