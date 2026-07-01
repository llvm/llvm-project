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

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!rec_HasOpEq>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!rec_HasOpEq>
// CIR: %[[NEQ_ADDR:.*]] = cir.alloca "neq" {{.*}} init : !cir.ptr<!cir.bool>
// CIR: %[[EQ:.*]] = cir.call @_ZNK7HasOpEqeqERKS_(%[[A_ADDR]], %[[B_ADDR]]) : (!cir.ptr<!rec_HasOpEq> {{.*}}, !cir.ptr<!rec_HasOpEq> {{.*}}) -> (!cir.bool{{.*}})
// CIR: %[[NEQ:.*]] = cir.not %[[EQ]] : !cir.bool
// CIR: cir.store{{.*}} %[[NEQ]], %[[NEQ_ADDR]] : !cir.bool, !cir.ptr<!cir.bool>

// LLVM: %[[A_ADDR:.*]] = alloca %struct.HasOpEq, i64 1, align 1
// LLVM: %[[B_ADDR:.*]] = alloca %struct.HasOpEq, i64 1, align 1
// LLVM: %[[NEQ_ADDR:.*]] = alloca i8, i64 1, align 1
// LLVM: %[[EQ:.*]] = call {{.*}} i1 @_ZNK7HasOpEqeqERKS_(ptr {{.*}} %[[A_ADDR]], ptr {{.*}} %[[B_ADDR]])
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

struct SpaceshipComplexResult {
  int _Complex operator<(int) const { return {}; }
};

struct ComplexItem {
  SpaceshipComplexResult operator<=>(const ComplexItem &) const { return {}; }
};

void cxx_rewritten_binary_operator_complex_expr() {
  ComplexItem a;
  ComplexItem b;
  int _Complex r = a < b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!rec_ComplexItem>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!rec_ComplexItem>
// CIR: %[[R_ADDR:.*]] = cir.alloca "r" {{.*}} init : !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[TMP_ADDR:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_SpaceshipComplexResult>
// CIR: %[[OP_RESULT:.*]] = cir.call @_ZNK11ComplexItemssERKS_(%[[A_ADDR]], %[[B_ADDR]]) : (!cir.ptr<!rec_ComplexItem> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, !cir.ptr<!rec_ComplexItem> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}) -> !rec_SpaceshipComplexResult
// CIR: cir.store {{.*}} %[[OP_RESULT]], %[[TMP_ADDR]] : !rec_SpaceshipComplexResult, !cir.ptr<!rec_SpaceshipComplexResult>
// CIR: %[[CONST_0:.*]] = cir.const #cir.int<0> : !s32i
// CIR: %[[RESULT:.*]] = cir.call @_ZNK22SpaceshipComplexResultltEi(%[[TMP_ADDR]], %[[CONST_0]]) : (!cir.ptr<!rec_SpaceshipComplexResult> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, !s32i {llvm.noundef}) -> (!cir.complex<!s32i> {llvm.noundef})
// CIR: cir.store {{.*}} %[[RESULT]], %[[R_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// The difference between LLVM and OGCG is due to missing ABI lowering.

// LLVM: %[[A_ADDR:.*]] = alloca %struct.ComplexItem, i64 1, align 1
// LLVM: %[[B_ADDR:.*]] = alloca %struct.ComplexItem, i64 1, align 1
// LLVM: %[[R_ADDR:.*]] = alloca { i32, i32 }, i64 1, align 4
// LLVM: %[[TMP_ADDR:.*]] = alloca %struct.SpaceshipComplexResult, i64 1, align 1
// LLVM: %[[OP_RESULT:.*]] = call %struct.SpaceshipComplexResult @_ZNK11ComplexItemssERKS_(ptr noundef nonnull align 1 dereferenceable(1) %[[A_ADDR]], ptr noundef nonnull align 1 dereferenceable(1) %[[B_ADDR]])
// LLVM: store %struct.SpaceshipComplexResult %[[OP_RESULT]], ptr %[[TMP_ADDR]], align 1
// LLVM: %[[RESULT:.*]] = call noundef { i32, i32 } @_ZNK22SpaceshipComplexResultltEi(ptr noundef nonnull align 1 dereferenceable(1) %[[TMP_ADDR]], i32 noundef 0)
// LLVM: store { i32, i32 } %[[RESULT]], ptr %[[R_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca %struct.ComplexItem, align 1
// OGCG: %[[B_ADDR:.*]] = alloca %struct.ComplexItem, align 1
// OGCG: %[[R_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[TMP_ADDR:.*]] = alloca %struct.SpaceshipComplexResult, align 1
// OGCG: %[[TMP_RESULT_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: call void @_ZNK11ComplexItemssERKS_(ptr noundef nonnull align 1 dereferenceable(1) %[[A_ADDR]], ptr noundef nonnull align 1 dereferenceable(1) %[[B_ADDR]])
// OGCG: %[[RESULT:.*]] = call noundef i64 @_ZNK22SpaceshipComplexResultltEi(ptr noundef nonnull align 1 dereferenceable(1) %[[TMP_ADDR]], i32 noundef 0)
// OGCG: store i64 %[[RESULT]], ptr %[[TMP_RESULT_ADDR]], align 4
// OGCG: %[[RESULT_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[TMP_RESULT_ADDR]], i32 0, i32 0
// OGCG: %[[RESULT_REAL:.*]] = load i32, ptr %[[RESULT_REAL_PTR]], align 4
// OGCG: %[[RESULT_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[TMP_RESULT_ADDR]], i32 0, i32 1
// OGCG: %[[RESULT_IMAG:.*]] = load i32, ptr %[[RESULT_IMAG_PTR]], align 4
// OGCG: %[[R_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[R_ADDR]], i32 0, i32 0
// OGCG: %[[R_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[R_ADDR]], i32 0, i32 1
// OGCG: store i32 %[[RESULT_REAL]], ptr %[[R_REAL_PTR]], align 4
// OGCG: store i32 %[[RESULT_IMAG]], ptr %[[R_IMAG_PTR]], align 4

struct Result {
  int value;
};

struct SpaceshipResult {
  Result operator<(int) const { return {}; }
};

struct Item {
  SpaceshipResult operator<=>(const Item &) const { return {}; }
};

void cxx_rewritten_binary_operator_aggr_expr() {
  Item a;
  Item b;
  Result r = a < b;
}

// CIR: %[[A_ADDR:.*]] = cir.alloca "a" {{.*}} : !cir.ptr<!rec_Item>
// CIR: %[[B_ADDR:.*]] = cir.alloca "b" {{.*}} : !cir.ptr<!rec_Item>
// CIR: %[[R_ADDR:.*]] = cir.alloca "r" {{.*}} init : !cir.ptr<!rec_Result>
// CIR: %[[TMP_ADDR:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<!rec_SpaceshipResult>
// CIR: %[[OP_RESULT:.*]] = cir.call @_ZNK4ItemssERKS_(%[[A_ADDR]], %[[B_ADDR]]) : (!cir.ptr<!rec_Item> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, !cir.ptr<!rec_Item> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}) -> !rec_SpaceshipResult
// CIR: cir.store {{.*}} %[[OP_RESULT]], %[[TMP_ADDR]] : !rec_SpaceshipResult, !cir.ptr<!rec_SpaceshipResult>
// CIR: %[[CONST_0:.*]] = cir.const #cir.int<0> : !s32i
// CIR: %[[RESULT:.*]] = cir.call @_ZNK15SpaceshipResultltEi(%[[TMP_ADDR]], %[[CONST_0]]) : (!cir.ptr<!rec_SpaceshipResult> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, !s32i {llvm.noundef}) -> !rec_Result
// CIR: cir.store {{.*}} %[[RESULT]], %[[R_ADDR]] : !rec_Result, !cir.ptr<!rec_Result>

// The difference between LLVM and OGCG is due to missing ABI lowering.

// LLVM: %[[A_ADDR:.*]] = alloca %struct.Item, i64 1, align 1
// LLVM: %[[B_ADDR:.*]] = alloca %struct.Item, i64 1, align 1
// LLVM: %[[R_ADDR:.*]] = alloca %struct.Result, i64 1, align 4
// LLVM: %[[TMP_ADDR:.*]] = alloca %struct.SpaceshipResult, i64 1, align 1
// LLVM: %[[OP_RESULT:.*]] = call %struct.SpaceshipResult @_ZNK4ItemssERKS_(ptr noundef nonnull align 1 dereferenceable(1) %[[A_ADDR]], ptr noundef nonnull align 1 dereferenceable(1) %[[B_ADDR]])
// LLVM: store %struct.SpaceshipResult %[[OP_RESULT]], ptr %[[TMP_ADDR]], align 1
// LLVM: %[[RESULT:.*]] = call %struct.Result @_ZNK15SpaceshipResultltEi(ptr noundef nonnull align 1 dereferenceable(1) %[[TMP_ADDR]], i32 noundef 0)
// LLVM: store %struct.Result %[[RESULT]], ptr %[[R_ADDR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca %struct.Item, align 1
// OGCG: %[[B_ADDR:.*]] = alloca %struct.Item, align 1
// OGCG: %[[R_ADDR:.*]] = alloca %struct.Result, align 4
// OGCG: %[[TMP_ADDR:.*]] = alloca %struct.SpaceshipResult, align 1
// OGCG: call void @_ZNK4ItemssERKS_(ptr noundef nonnull align 1 dereferenceable(1) %[[A_ADDR]], ptr noundef nonnull align 1 dereferenceable(1) %[[B_ADDR]])
// OGCG: %[[RESULT:.*]] = call i32 @_ZNK15SpaceshipResultltEi(ptr noundef nonnull align 1 dereferenceable(1) %[[TMP_ADDR]], i32 noundef 0)
// OGCG: %[[R_ADDR_PTR:.*]] = getelementptr inbounds nuw %struct.Result, ptr %[[R_ADDR:.*]], i32 0, i32 0
// OGCG: store i32 %[[RESULT:.*]], ptr %[[R_ADDR_PTR]], align 4
