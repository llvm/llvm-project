// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

namespace std {
template <class _E> class initializer_list {
  const _E *__begin_;
  const _E *__end_;
};
} // namespace std

void initalizer_list_with_two_pointers_layout() {
  std::initializer_list<int> a = {10, 20, 30};
}

// CIR: %[[ARR_ADDR:.*]] = cir.alloca !cir.array<!s32i x 3>, !cir.ptr<!cir.array<!s32i x 3>>, ["ref.tmp0"]
// CIR: %[[A_ADDR:.*]] = cir.alloca !rec_std3A3Ainitializer_list3Cint3E, !cir.ptr<!rec_std3A3Ainitializer_list3Cint3E>, ["a", init]
// CIR: %[[ARR_PTR:.*]] = cir.cast array_to_ptrdecay %[[ARR_ADDR]] : !cir.ptr<!cir.array<!s32i x 3>> -> !cir.ptr<!s32i>
// CIR: %[[CONST_S32_10:.*]] = cir.const #cir.int<10> : !s32i
// CIR: cir.store {{.*}} %[[CONST_S32_10]], %[[ARR_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[CONST_S64_1:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ARR_ELEM_1_PTR:.*]] = cir.ptr_stride %[[ARR_PTR]], %[[CONST_S64_1]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR: %[[CONST_S32_20:.*]] = cir.const #cir.int<20> : !s32i
// CIR: cir.store {{.*}} %[[CONST_S32_20]], %[[ARR_ELEM_1_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[CONST_S64_2:.*]] = cir.const #cir.int<2> : !s64i
// CIR: %[[ARR_ELEM_2_PTR:.*]] = cir.ptr_stride %[[ARR_PTR]], %[[CONST_S64_2]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR: %[[CONST_S32_30:.*]] = cir.const #cir.int<30> : !s32i
// CIR: cir.store {{.*}} %[[CONST_S32_30]], %[[ARR_ELEM_2_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[BEGIN_PTR:.*]] = cir.get_member %[[A_ADDR]][0] {name = "__begin_"} : !cir.ptr<!rec_std3A3Ainitializer_list3Cint3E> -> !cir.ptr<!cir.ptr<!s32i>
// CIR: %[[ARR_BEGIN_PTR:.*]] = cir.cast bitcast %[[BEGIN_PTR]] : !cir.ptr<!cir.ptr<!s32i>> -> !cir.ptr<!cir.ptr<!cir.array<!s32i x 3>>>
// CIR: cir.store {{.*}} %[[ARR_ADDR]], %[[ARR_BEGIN_PTR]] : !cir.ptr<!cir.array<!s32i x 3>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 3>>>
// CIR: %[[CONST_U64_3:.*]] = cir.const #cir.int<3> : !u64i
// CIR: %[[END_PTR:.*]] = cir.get_member %[[A_ADDR]][1] {name = "__end_"} : !cir.ptr<!rec_std3A3Ainitializer_list3Cint3E> -> !cir.ptr<!cir.ptr<!s32i>>
// CIR: %[[ARR_END:.*]] = cir.ptr_stride %[[ARR_ADDR]], %[[CONST_U64_3]] : (!cir.ptr<!cir.array<!s32i x 3>>, !u64i) -> !cir.ptr<!cir.array<!s32i x 3>>
// CIR: %[[ARR_END_PTR:.*]] = cir.cast bitcast %[[END_PTR]] : !cir.ptr<!cir.ptr<!s32i>> -> !cir.ptr<!cir.ptr<!cir.array<!s32i x 3>>>
// CIR: cir.store {{.*}} %[[ARR_END]], %[[ARR_END_PTR]] : !cir.ptr<!cir.array<!s32i x 3>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 3>>>

// LLVM: %[[ARR_ADDR:.*]] = alloca [3 x i32], i64 1, align 4
// LLVM: %[[A_ADDR:.*]] = alloca %"class.std::initializer_list<int>", i64 1, align 8
// LLVM: %[[ARR_ELEM_0_PTR:.*]] = getelementptr i32, ptr %[[ARR_ADDR]], i32 0
// LLVM: store i32 10, ptr %[[ARR_ELEM_0_PTR]], align 4
// LLVM: %[[ARR_ELEM_1_PTR:.*]] = getelementptr i32, ptr %[[ARR_ELEM_0_PTR]], i64 1
// LLVM: store i32 20, ptr %[[ARR_ELEM_1_PTR]], align 4
// LLVM: %[[ARR_ELEM_2_PTR:.*]] = getelementptr i32, ptr %[[ARR_ELEM_0_PTR]], i64 2
// LLVM: store i32 30, ptr %[[ARR_ELEM_2_PTR]], align 4
// LLVM: %[[BEGIN_PTR:.*]] = getelementptr inbounds nuw %"class.std::initializer_list<int>", ptr %[[A_ADDR]], i32 0, i32 0
// LLVM: store ptr %[[ARR_ADDR]], ptr %[[BEGIN_PTR]], align 8
// LLVM: %[[END_PTR:.*]] = getelementptr inbounds nuw %"class.std::initializer_list<int>", ptr %[[A_ADDR]], i32 0, i32 1
// LLVM: %[[ARR_END:.*]] = getelementptr [3 x i32], ptr %[[ARR_ADDR]], i64 3
// LLVM: store ptr %[[ARR_END]], ptr %[[END_PTR]], align 8
// LLVM: ret void

// OGCG: %[[A_ADDR:.*]] = alloca %"class.std::initializer_list", align 8
// OGCG: %[[ARR_ADDR:.*]] = alloca [3 x i32], align 4
// OGCG: store i32 10, ptr %[[ARR_ADDR]], align 4
// OGCG: %[[ARR_ELEM_1_PTR:.*]] = getelementptr inbounds i32, ptr %[[ARR_ADDR]], i64 1
// OGCG: store i32 20, ptr %[[ARR_ELEM_1_PTR]], align 4
// OGCG: %[[ARR_ELEM_2_PTR:.*]] = getelementptr inbounds i32, ptr %[[ARR_ADDR]], i64 2
// OGCG: store i32 30, ptr %[[ARR_ELEM_2_PTR]], align 4
// OGCG: %[[BEGIN_PTR:.*]] = getelementptr inbounds nuw %"class.std::initializer_list", ptr %[[A_ADDR]], i32 0, i32 0
// OGCG: store ptr %[[ARR_ADDR]], ptr %[[BEGIN_PTR]], align 8
// OGCG: %[[END_PTR:.*]] = getelementptr inbounds nuw %"class.std::initializer_list", ptr %[[A_ADDR]], i32 0, i32 1
// OGCG: %[[ARR_END:.*]] = getelementptr inbounds [3 x i32], ptr %[[ARR_ADDR]], i64 0, i64 3
// OGCG: store ptr %[[ARR_END]], ptr %[[END_PTR]], align 8
