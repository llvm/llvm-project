// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void function_may_throw();

void function_no_except() noexcept;

void no_except() {
  bool a = noexcept(1);
  bool b = noexcept(function_may_throw());
  bool c = noexcept(function_no_except());
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["a", init]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["b", init]
// CIR: %[[C_ADDR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["c", init]
// CIR: %[[CONST_TRUE:.*]] = cir.const #true
// CIR: cir.store {{.*}} %[[CONST_TRUE]], %[[A_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR: %[[CONST_FALSE:.*]] = cir.const #false
// CIR: cir.store {{.*}} %[[CONST_FALSE]], %[[B_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR: %[[CONST_TRUE:.*]] = cir.const #true
// CIR: cir.store {{.*}} %[[CONST_TRUE]], %[[C_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM: %[[A_ADDR:.*]] = alloca i8, i64 1, align 1
// LLVM: %[[B_ADDR:.*]] = alloca i8, i64 1, align 1
// LLVM: %[[C_ADDR:.*]] = alloca i8, i64 1, align 1
// LLVM: store i8 1, ptr %[[A_ADDR]], align 1
// LLVM: store i8 0, ptr %[[B_ADDR]], align 1
// LLVM: store i8 1, ptr %[[C_ADDR]], align 1

// OGCG: %[[A_ADDR:.*]] = alloca i8, align 1
// OGCG: %[[B_ADDR:.*]] = alloca i8, align 1
// OGCG: %[[C_ADDR:.*]] = alloca i8, align 1
// OGCG: store i8 1, ptr %[[A_ADDR]], align 1
// OGCG: store i8 0, ptr %[[B_ADDR]], align 1
// OGCG: store i8 1, ptr %[[C_ADDR]], align 1
