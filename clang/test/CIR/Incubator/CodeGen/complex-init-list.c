// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM

void foo() {
   int _Complex c = (int _Complex){1, 2};
}

// CIR: %[[INIT:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, ["c", init]
// CIR: %[[COMPOUND:.*]] = cir.alloca !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>, [".compoundliteral", init]
// CIR: %[[COMPLEX:.*]] = cir.const #cir.complex<#cir.int<1> : !s32i, #cir.int<2> : !s32i> : !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[COMPLEX]], %[[COMPOUND]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[COMPOUND]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: cir.store{{.*}} %[[TMP]], %[[INIT]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM: %[[INIT:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[COMPOUND:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: store { i32, i32 } { i32 1, i32 2 }, ptr %[[COMPOUND]], align 4
// LLVM: %[[TMP:.*]] = load { i32, i32 }, ptr %[[COMPOUND:.*]], align 4
// LLVM: store { i32, i32 } %[[TMP]], ptr %[[INIT]], align 4

void foo2(float a, float b) {
  float _Complex c = (float _Complex){a, b};
}

// CIR: %[[INIT:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, ["c", init]
// CIR: %[[COMPOUND:.*]] = cir.alloca !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>, [".compoundliteral", init]
// CIR: %[[A:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[B:.*]] = cir.load{{.*}} {{.*}} : !cir.ptr<!cir.float>, !cir.float
// CIR: %[[COMPLEX:.*]] = cir.complex.create %[[A]], %[[B]] : !cir.float -> !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[COMPLEX]], %[[COMPOUND]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR: %[[TMP:.*]] = cir.load{{.*}} %[[COMPOUND]] : !cir.ptr<!cir.complex<!cir.float>>, !cir.complex<!cir.float>
// CIR: cir.store{{.*}} %[[TMP]], %[[INIT]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: %[[INIT:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[COMPOUND:.*]] = alloca { float, float }, i64 1, align 4
// LLVM: %[[A:.*]] = load float, ptr {{.*}}, align 4
// LLVM: %[[B:.*]] = load float, ptr {{.*}}, align 4
// LLVM: %[[INSERT:.*]] = insertvalue { float, float } {{.*}}, float %[[A]], 0
// LLVM: %[[INSERT_2:.*]] = insertvalue { float, float } %[[INSERT]], float %[[B]], 1
// LLVM: store { float, float } %[[INSERT_2]], ptr %[[COMPOUND]], align 4
// LLVM: %[[TMP:.*]] = load { float, float }, ptr %[[COMPOUND]], align 4
// LLVM: store { float, float } %[[TMP]], ptr %[[INIT]], align 4
