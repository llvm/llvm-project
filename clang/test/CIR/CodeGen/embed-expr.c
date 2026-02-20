// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c23 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

void embed_expr_on_scalar_with_constants() {
  int a[3] = {
      1,
      2,
#embed __FILE__
  };
}

// CIR-DAG: cir.global "private" constant cir_private @[[EMBED_A:.*]] = #cir.const_array<[#cir.int<1> : !s32i, #cir.int<2> : !s32i, #cir.int<47> : !s32i]> : !cir.array<!s32i x 3>

// CIR: %[[A_ADDR:.*]] = cir.alloca !cir.array<!s32i x 3>, !cir.ptr<!cir.array<!s32i x 3>>, ["a", init]
// CIR: %[[ARRAY:.*]] = cir.get_global @[[EMBED_A]] : !cir.ptr<!cir.array<!s32i x 3>>
// CIR: cir.copy %[[ARRAY]] to %[[A_ADDR]] : !cir.ptr<!cir.array<!s32i x 3>>

// LLVM: %[[A_ADDR:.*]] = alloca [3 x i32], i64 1, align 4
// LLVM: call void @llvm.memcpy{{.*}}(ptr %[[A_ADDR]], ptr @[[EMBED_A:.*]], i64 12, i1 false)

// OGCG: %[[A_ADDR:.*]] = alloca [3 x i32], align 4
// OGCG: call void @llvm.memcpy.p0.p0.i64(ptr align 4 %[[A_ADDR]], ptr align 4 @__const.embed_expr_on_scalar_with_constants.a, i64 12, i1 false)

void embed_expr_on_scalar_with_non_constants() {
  int a;
  int b[3] = {
      a,
      a,
#embed __FILE__
  };
}

// CIR: %[[A_ADDR:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["a"]
// CIR: %[[B_ADDR:.*]] = cir.alloca !cir.array<!s32i x 3>, !cir.ptr<!cir.array<!s32i x 3>>, ["b", init]
// CIR: %[[B_PTR:.*]] = cir.cast array_to_ptrdecay %[[B_ADDR]] : !cir.ptr<!cir.array<!s32i x 3>> -> !cir.ptr<!s32i>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store {{.*}} %[[TMP_A]], %[[B_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[CONST_1:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[B_ELEM_1_PTR:.*]] = cir.ptr_stride %[[B_PTR]], %[[CONST_1]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR: %[[TMP_A:.*]] = cir.load {{.*}} %[[A_ADDR]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.store {{.*}} %[[TMP_A]], %[[B_ELEM_1_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[CONST_2:.*]] = cir.const #cir.int<2> : !s64i
// CIR: %[[B_ELEM_2_PTR:.*]] = cir.ptr_stride %[[B_PTR]], %[[CONST_2]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR: %[[CONST_47:.*]] = cir.const #cir.int<47> : !s32i
// CIR: cir.store {{.*}} %[[CONST_47]], %[[B_ELEM_2_PTR]] : !s32i, !cir.ptr<!s32i>

// LLVM: %[[A_ADDR:.*]] = alloca i32, i64 1, align 4
// LLVM: %[[B_ADDR:.*]] = alloca [3 x i32], i64 1, align 4
// LLVM: %[[B_ELEM_0_PTR:.*]] = getelementptr i32, ptr %[[B_ADDR]], i32 0
// LLVM: %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM: store i32 %[[TMP_A]], ptr %[[B_ELEM_0_PTR]], align 4
// LLVM: %[[B_ELEM_1_PTR:.*]] = getelementptr i32, ptr %[[B_ELEM_0_PTR]], i64 1
// LLVM: %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// LLVM: store i32 %[[TMP_A]], ptr %[[B_ELEM_1_PTR]], align 4
// LLVM: %[[B_ELEM_2_PTR:.*]] = getelementptr i32, ptr %[[B_ELEM_0_PTR]], i64 2
// LLVM: store i32 47, ptr %[[B_ELEM_2_PTR]], align 4

// OGCG: %[[A_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[B_ADDR:.*]] = alloca [3 x i32], align 4
// OGCG: %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG: store i32 %[[TMP_A]], ptr %[[B_ADDR]], align 4
// OGCG: %[[B_ELEM_1_PTR:.*]] = getelementptr inbounds i32, ptr %[[B_ADDR]], i64 1
// OGCG: %[[TMP_A:.*]] = load i32, ptr %[[A_ADDR]], align 4
// OGCG: store i32 %[[TMP_A]], ptr %[[B_ELEM_1_PTR]], align 4
// OGCG: %[[B_ELEM_2_PTR:.*]] = getelementptr inbounds i32, ptr %[[B_ADDR]], i64 2
// OGCG: store i32 47, ptr %[[B_ELEM_2_PTR]], align 4
