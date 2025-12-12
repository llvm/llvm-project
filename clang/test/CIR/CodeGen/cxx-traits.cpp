// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void expression_trait_expr() {
  bool a = __is_lvalue_expr(0);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["a", init]
// CIR: %[[CONST_FALSE:.*]] = cir.const #false
// CIR: cir.store {{.*}} %[[CONST_FALSE]], %[[A_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM: %[[A_ADDR:.*]] = alloca i8, i64 1, align 1
// LLVM: store i8 0, ptr %[[A_ADDR]], align 1

// OGCG: %[[A_ADDR:.*]] = alloca i8, align 1
// OGCG: store i8 0, ptr %[[A_ADDR]], align 1

void type_trait_expr() {
  enum E {};
  bool a = __is_enum(E);
  bool b = __is_same(int, float);
  bool c = __is_constructible(int, int, int, int);
  bool d = __is_array(int);
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["a", init]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["b", init]
// CIR: %[[C_ADDR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["c", init]
// CIR: %[[D_ADDR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["d", init]
// CIR: %[[CONST_TRUE:.*]] = cir.const #true
// CIR: cir.store {{.*}} %[[CONST_TRUE]], %[[A_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR: %[[CONST_FALSE:.*]] = cir.const #false
// CIR: cir.store {{.*}} %[[CONST_FALSE]], %[[B_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR: %[[CONST_FALSE:.*]] = cir.const #false
// CIR: cir.store {{.*}} %[[CONST_FALSE]], %[[C_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>
// CIR: %[[CONST_FALSE:.*]] = cir.const #false
// CIR: cir.store {{.*}} %[[CONST_FALSE]], %[[D_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM: %[[A_ADDR:.*]] = alloca i8, i64 1, align 1
// LLVM: %[[B_ADDR:.*]] = alloca i8, i64 1, align 1
// LLVM: %[[C_ADDR:.*]] = alloca i8, i64 1, align 1
// LLVM: %[[D_ADDR:.*]] = alloca i8, i64 1, align 1
// LLVM: store i8 1, ptr %[[A_ADDR]], align 1
// LLVM: store i8 0, ptr %[[B_ADDR]], align 1
// LLVM: store i8 0, ptr %[[C_ADDR]], align 1
// LLVM: store i8 0, ptr %[[D_ADDR]], align 1

// OGCG: %[[A_ADDR:.*]] = alloca i8, align 1
// OGCG: %[[B_ADDR:.*]] = alloca i8, align 1
// OGCG: %[[C_ADDR:.*]] = alloca i8, align 1
// OGCG: %[[D_ADDR:.*]] = alloca i8, align 1
// OGCG: store i8 1, ptr %[[A_ADDR]], align 1
// OGCG: store i8 0, ptr %[[B_ADDR]], align 1
// OGCG: store i8 0, ptr %[[C_ADDR]], align 1
// OGCG: store i8 0, ptr %[[D_ADDR]], align 1
