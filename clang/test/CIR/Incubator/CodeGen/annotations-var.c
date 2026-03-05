// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

// CIR-DAG:  cir.global external @globalvar = #cir.int<3> : !s32i [#cir.annotation<name = "globalvar_ann_0", args = []>] {alignment = 4 : i64}
// CIR-DAG:  cir.global external @globalvar2 = #cir.int<2> : !s32i [#cir.annotation<name = "common_ann", args = ["os", 21 : i32]>] {alignment = 4 : i64}

// LLVM-DAG: @.str.annotation = private unnamed_addr constant [15 x i8] c"localvar_ann_0\00", section "llvm.metadata"
// LLVM-DAG: @.str.1.annotation = private unnamed_addr constant [{{[0-9]+}} x i8] c"{{.*}}annotations-var.c\00", section "llvm.metadata"
// LLVM-DAG: @.str.2.annotation = private unnamed_addr constant [15 x i8] c"localvar_ann_1\00", section "llvm.metadata"
// LLVM-DAG: @.str.3.annotation = private unnamed_addr constant [11 x i8] c"common_ann\00", section "llvm.metadata"
// LLVM-DAG: @.str.annotation.arg = private unnamed_addr constant [3 x i8] c"os\00", align 1
// LLVM-DAG: @.args.annotation = private unnamed_addr constant { ptr, i32 } { ptr @.str.annotation.arg, i32 21 }, section "llvm.metadata"
// LLVM-DAG: @.str.4.annotation = private unnamed_addr constant [16 x i8] c"globalvar_ann_0\00", section "llvm.metadata"
// LLVM-DAG: @llvm.global.annotations = appending global [2 x { ptr, ptr, ptr, i32, ptr }]
// LLVM-DAG-SAME: [{ ptr, ptr, ptr, i32, ptr } { ptr @globalvar, ptr @.str.4.annotation, ptr @.str.1.annotation, i32 18, ptr null }, { ptr, ptr, ptr, i32, ptr }
// LLVM-DAG-SAME: { ptr @globalvar2, ptr @.str.3.annotation, ptr @.str.1.annotation, i32 19, ptr @.args.annotation }], section "llvm.metadata"

int globalvar __attribute__((annotate("globalvar_ann_0"))) = 3;
int globalvar2 __attribute__((annotate("common_ann", "os", 21))) = 2;
void local(void) {
    int localvar __attribute__((annotate("localvar_ann_0"))) __attribute__((annotate("localvar_ann_1"))) = 3;
    int localvar2 __attribute__((annotate("localvar_ann_0"))) = 3;
    int localvar3 __attribute__((annotate("common_ann", "os", 21)))  = 3;
// CIR-LABEL: @local
// CIR: %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["localvar", init] [#cir.annotation<name = "localvar_ann_0", args = []>, #cir.annotation<name = "localvar_ann_1", args = []>]
// CIR: %1 = cir.alloca !s32i, !cir.ptr<!s32i>, ["localvar2", init] [#cir.annotation<name = "localvar_ann_0", args = []>]
// CIR: %2 = cir.alloca !s32i, !cir.ptr<!s32i>, ["localvar3", init] [#cir.annotation<name = "common_ann", args = ["os", 21 : i32]>]


// LLVM-LABEL: @local
// LLVM: %[[ALLOC:.*]] = alloca i32
// LLVM: call void @llvm.var.annotation.p0.p0(ptr %[[ALLOC]], ptr @.str.annotation, ptr @.str.1.annotation, i32 23, ptr null)
// LLVM: call void @llvm.var.annotation.p0.p0(ptr %[[ALLOC]], ptr @.str.2.annotation, ptr @.str.1.annotation, i32 23, ptr null)
// LLVM: %[[ALLOC2:.*]] = alloca i32
// LLVM: call void @llvm.var.annotation.p0.p0(ptr %[[ALLOC2]], ptr @.str.annotation, ptr @.str.1.annotation, i32 24, ptr null)
// LLVM: %[[ALLOC3:.*]] = alloca i32
// LLVM: call void @llvm.var.annotation.p0.p0(ptr %[[ALLOC3]], ptr @.str.3.annotation, 
// LLVM-SAME: ptr @.str.1.annotation, i32 25, ptr @.args.annotation)
}
