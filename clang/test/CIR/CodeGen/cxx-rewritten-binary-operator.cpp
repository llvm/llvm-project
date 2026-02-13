// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

struct HasOpEq {
  bool operator==(const HasOpEq &) const;
};

void cxx_rewritten_binary_operator_scalar_expr() {
  HasOpEq a;
  HasOpEq b;
  bool neq = a != b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !rec_HasOpEq, !cir.ptr<!rec_HasOpEq>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !rec_HasOpEq, !cir.ptr<!rec_HasOpEq>, ["b"]
// CIR: %[[NEQ_ADDR:.*]] = cir.alloca !cir.bool, !cir.ptr<!cir.bool>, ["neq", init]
// CIR: %[[EQ:.*]] = cir.call @_ZNK7HasOpEqeqERKS_(%[[A_ADDR]], %[[B_ADDR]]) : (!cir.ptr<!rec_HasOpEq>, !cir.ptr<!rec_HasOpEq>) -> !cir.bool
// CIR: %[[NEQ:.*]] = cir.unary(not, %[[EQ]]) : !cir.bool, !cir.bool
// CIR: cir.store{{.*}} %[[NEQ]], %[[NEQ_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM: %[[A_ADDR:.*]] = alloca %struct.HasOpEq, i64 1, align 1
// LLVM: %[[B_ADDR:.*]] = alloca %struct.HasOpEq, i64 1, align 1
// LLVM: %[[NEQ_ADDR:.*]] = alloca i8, i64 1, align 1
// LLVM: %[[EQ:.*]] = call i1 @_ZNK7HasOpEqeqERKS_(ptr %[[A_ADDR]], ptr %[[B_ADDR]])
// LLVM: %[[NEQ_I1:.*]] = xor i1 %[[EQ]], true
// LLVM: %[[NEQ:.*]] = zext i1 %[[NEQ_I1]] to i8
// LLVM: store i8 %[[NEQ]], ptr %[[NEQ_ADDR]], align 1

// OGCG: %[[A_ADDR:.*]] = alloca %struct.HasOpEq, align 1
// OGCG: %[[B_ADDR:.*]] = alloca %struct.HasOpEq, align 1
// OGCG: %[[NEQ_ADDR:.*]] = alloca i8, align 1
// OGCG: %[[EQ:.*]] = call {{.*}} zeroext i1 @_ZNK7HasOpEqeqERKS_(ptr {{.*}} %[[A_ADDR]], ptr {{.*}} %[[B_ADDR]])
// OGCG: %[[NEQ_I1:.*]] = xor i1 %[[EQ]], true
// OGCG: %[[NEQ:.*]] = zext i1 %[[NEQ_I1]] to i8
// OGCG: store i8 %[[NEQ]], ptr %[[NEQ_ADDR]], align 1
