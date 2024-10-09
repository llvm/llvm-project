// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR

void local(void) {
    int localvar __attribute__((annotate("localvar_ann_0"))) __attribute__((annotate("localvar_ann_1"))) = 3;
// CIR-LABEL: @local
// CIR: %0 = cir.alloca !s32i, !cir.ptr<!s32i>, ["localvar", init] [#cir.annotation<name = "localvar_ann_0", args = []>, #cir.annotation<name = "localvar_ann_1", args = []>]
}