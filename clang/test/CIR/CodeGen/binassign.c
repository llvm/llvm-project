// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void binary_assign(void) {
    bool b;
    char c;
    float f;
    int i;

    b = true;
    c = 65;
    f = 3.14f;
    i = 42;
}

// CIR-LABEL: cir.func{{.*}} @binary_assign()
// CIR:         %[[B:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["b"]
// CIR:         %[[C:.*]] = cir.alloca !s8i, !cir.ptr<!s8i>, ["c"]
// CIR:         %[[F:.*]] = cir.alloca !cir.float, !cir.ptr<!cir.float>, ["f"]
// CIR:         %[[I:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["i"]
// CIR:         %[[TRUE:.*]] = cir.const #true
// CIR:         cir.store{{.*}} %[[TRUE]], %[[B]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR:         %[[CHAR_INI_INIT:.*]] = cir.const #cir.int<65> : !s32i
// CIR:         %[[CHAR_VAL:.*]] = cir.cast integral %[[CHAR_INI_INIT]] : !s32i -> !s8i
// CIR:         cir.store{{.*}} %[[CHAR_VAL]], %[[C]] : !s8i, !cir.ptr<!s8i>
// CIR:         %[[FLOAT_VAL:.*]] = cir.const #cir.fp<3.140000e+00> : !cir.float
// CIR:         cir.store{{.*}} %[[FLOAT_VAL]], %[[F]] : !cir.float, !cir.ptr<!cir.float>
// CIR:         %[[INT_VAL:.*]] = cir.const #cir.int<42> : !s32i
// CIR:         cir.store{{.*}} %[[INT_VAL]], %[[I]] : !s32i, !cir.ptr<!s32i>
// CIR:         cir.return

// LLVM-LABEL: define {{.*}}void @binary_assign(){{.*}} {
// LLVM:         %[[B_PTR:.*]] = alloca i8
// LLVM:         %[[C_PTR:.*]] = alloca i8
// LLVM:         %[[F_PTR:.*]] = alloca float
// LLVM:         %[[I_PTR:.*]] = alloca i32
// LLVM:         store i8 1, ptr %[[B_PTR]]
// LLVM:         store i8 65, ptr %[[C_PTR]]
// LLVM:         store float 0x40091EB860000000, ptr %[[F_PTR]]
// LLVM:         store i32 42, ptr %[[I_PTR]]
// LLVM:         ret void

// OGCG-LABEL: define {{.*}}void @binary_assign()
// OGCG:         %[[B_PTR:.*]] = alloca i8
// OGCG:         %[[C_PTR:.*]] = alloca i8
// OGCG:         %[[F_PTR:.*]] = alloca float
// OGCG:         %[[I_PTR:.*]] = alloca i32
// OGCG:         store i8 1, ptr %[[B_PTR]]
// OGCG:         store i8 65, ptr %[[C_PTR]]
// OGCG:         store float 0x40091EB860000000, ptr %[[F_PTR]]
// OGCG:         store i32 42, ptr %[[I_PTR]]
// OGCG:         ret void
