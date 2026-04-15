// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -Wno-unused-value -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=OGCG

namespace std {
template <typename T> class initializer_list {
  const T *data;
  __SIZE_TYPE__ size;

public:
  initializer_list();
};
} // namespace std

struct Vector {
  Vector(std::initializer_list<int>);
  ~Vector();
};

void init_vec_using_initalizer_list() {
  Vector vec = {0, 1, 2};
}

// CIR: %[[VEC_ADDR:.*]] = cir.alloca !rec_Vector, !cir.ptr<!rec_Vector>, ["vec", init]
// CIR: %[[AGG_ADDR:.*]] = cir.alloca !rec_std3A3Ainitializer_list3Cint3E, !cir.ptr<!rec_std3A3Ainitializer_list3Cint3E>, ["agg.tmp0"]
// CIR: %[[INIT_LIST_ADDR:.*]] = cir.alloca !cir.array<!s32i x 3>, !cir.ptr<!cir.array<!s32i x 3>>, ["ref.tmp0"]
// CIR: %[[INIT_LIST_PTR:.*]] = cir.cast array_to_ptrdecay %[[INIT_LIST_ADDR]] : !cir.ptr<!cir.array<!s32i x 3>> -> !cir.ptr<!s32i>
// CIR: %[[CONST_S32_0:.*]] = cir.const #cir.int<0> : !s32i
// CIR: cir.store {{.*}} %[[CONST_S32_0]], %[[INIT_LIST_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[CONST_S64_1:.*]] = cir.const #cir.int<1> : !s64i
// CIR: %[[ELEM_1_PTR:.*]] = cir.ptr_stride %[[INIT_LIST_PTR]], %[[CONST_S64_1]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR: %[[CONST_S32_1:.*]] = cir.const #cir.int<1> : !s32i
// CIR: cir.store {{.*}} %[[CONST_S32_1]], %[[ELEM_1_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[CONST_S64_2:.*]] = cir.const #cir.int<2> : !s64i
// CIR: %[[ELEM_2_PTR:.*]] = cir.ptr_stride %[[INIT_LIST_PTR]], %[[CONST_S64_2]] : (!cir.ptr<!s32i>, !s64i) -> !cir.ptr<!s32i>
// CIR: %[[CONST_S32_2:.*]] = cir.const #cir.int<2> : !s32i
// CIR: cir.store {{.*}} %[[CONST_S32_2]], %[[ELEM_2_PTR]] : !s32i, !cir.ptr<!s32i>
// CIR: %[[DATA_PTR:.*]] = cir.get_member %[[AGG_ADDR]][0] {name = "data"} : !cir.ptr<!rec_std3A3Ainitializer_list3Cint3E> -> !cir.ptr<!cir.ptr<!s32i>>
// CIR: %[[DATA_ARR_PTR:.*]] = cir.cast bitcast %[[DATA_PTR]] : !cir.ptr<!cir.ptr<!s32i>> -> !cir.ptr<!cir.ptr<!cir.array<!s32i x 3>>>
// CIR: cir.store {{.*}} %[[INIT_LIST_ADDR]], %[[DATA_ARR_PTR]] : !cir.ptr<!cir.array<!s32i x 3>>, !cir.ptr<!cir.ptr<!cir.array<!s32i x 3>>>
// CIR: %[[CONST_U64_3:.*]] = cir.const #cir.int<3> : !u64i
// CIR: %[[SIZE_PTR:.*]] = cir.get_member %[[AGG_ADDR]][1] {name = "size"} : !cir.ptr<!rec_std3A3Ainitializer_list3Cint3E> -> !cir.ptr<!u64i>
// CIR: cir.store {{.*}} %[[CONST_U64_3]], %[[SIZE_PTR]] : !u64i, !cir.ptr<!u64i>
// CIR: %[[TMP_AGG:.*]] = cir.load {{.*}} %[[AGG_ADDR]] : !cir.ptr<!rec_std3A3Ainitializer_list3Cint3E>, !rec_std3A3Ainitializer_list3Cint3E
// CIR: cir.call @_ZN6VectorC1ESt16initializer_listIiE(%[[VEC_ADDR]], %[[TMP_AGG]]) : (!cir.ptr<!rec_Vector> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}, !rec_std3A3Ainitializer_list3Cint3E) -> ()
// CIR: cir.cleanup.scope {
// CIR:   cir.yield
// CIR: } cleanup normal {
// CIR:   cir.call @_ZN6VectorD1Ev(%[[VEC_ADDR]]) nothrow : (!cir.ptr<!rec_Vector> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}) -> ()
// CIR:   cir.yield
// CIR: }

// LLVM:   %[[VEC_ADDR:.*]] = alloca %struct.Vector, i64 1, align 1
// LLVM:   %[[AGG_ADDR:.*]] = alloca %"class.std::initializer_list<int>", i64 1, align 8
// LLVM:   %[[INIT_LIST_ADDR:.*]] = alloca [3 x i32], i64 1, align 4
// LLVM:   %[[INIT_LIST_PTR:.*]] = getelementptr i32, ptr %[[INIT_LIST_ADDR]], i32 0
// LLVM:   store i32 0, ptr %[[INIT_LIST_PTR]], align 4
// LLVM:   %[[ELEM_1_PTR:.*]] = getelementptr i32, ptr %[[INIT_LIST_PTR]], i64 1
// LLVM:   store i32 1, ptr %[[ELEM_1_PTR]], align 4
// LLVM:   %[[ELEM_2_PTR:.*]] = getelementptr i32, ptr %[[INIT_LIST_PTR]], i64 2
// LLVM:   store i32 2, ptr %[[ELEM_2_PTR]], align 4
// LLVM:   %[[DATA_PTR:.*]] = getelementptr %"class.std::initializer_list<int>", ptr %[[AGG_ADDR]], i32 0, i32 0
// LLVM:   store ptr %[[INIT_LIST_ADDR]], ptr %[[DATA_PTR]], align 8
// LLVM:   %[[SIZE_PTR:.*]] = getelementptr %"class.std::initializer_list<int>", ptr %[[AGG_ADDR]], i32 0, i32 1
// LLVM:   store i64 3, ptr %[[SIZE_PTR]], align 8
// LLVM:   %[[TMP_AGG:.*]] = load %"class.std::initializer_list<int>", ptr %[[AGG_ADDR]], align 8
// LLVM:   call void @_ZN6VectorC1ESt16initializer_listIiE(ptr noundef nonnull align 1 dereferenceable(1) %[[VEC_ADDR]], %"class.std::initializer_list<int>" %[[TMP_AGG]])
// LLVM:   br label %[[SCOPE_CONT:.*]]
// LLVM: [[SCOPE_CONT]]:
// LLVM:   br label %[[CLEANUP_START:.*]]
// LLVM: [[CLEANUP_START]]:
// LLVM:   call void @_ZN6VectorD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %[[VEC_ADDR]])
// LLVM:   br label %[[CLEANUP_CONT:.*]]
// LLVM: [[CLEANUP_CONT]]:
// LLVM:   br label %[[DONE:.*]]
// LLVM: [[DONE]]:
// LLVM:   ret void

// OGCG: %[[VEC_ADDR:.*]] = alloca %struct.Vector, align 1
// OGCG: %[[AGG_ADDR:.*]] = alloca %"class.std::initializer_list", align 8
// OGCG: %[[INIT_LIST_ADDR:.*]] = alloca [3 x i32], align 4
// OGCG: store i32 0, ptr %[[INIT_LIST_ADDR]], align 4
// OGCG: %[[ELEM_1_PTR:.*]] = getelementptr inbounds i32, ptr %[[INIT_LIST_ADDR]], i64 1
// OGCG: store i32 1, ptr %[[ELEM_1_PTR]], align 4
// OGCG: %[[ELEM_2_PTR:.*]] = getelementptr inbounds i32, ptr %[[INIT_LIST_ADDR]], i64 2
// OGCG: store i32 2, ptr %[[ELEM_2_PTR]], align 4
// OGCG: %[[DATA_PTR:.*]] = getelementptr inbounds nuw %"class.std::initializer_list", ptr %[[AGG_ADDR]], i32 0, i32 0
// OGCG: store ptr %ref.tmp, ptr %[[DATA_PTR]], align 8
// OGCG: %[[SIZE_PTR:.*]] = getelementptr inbounds nuw %"class.std::initializer_list", ptr %[[AGG_ADDR]], i32 0, i32 1
// OGCG: store i64 3, ptr %[[SIZE_PTR]], align 8
// OGCG: %[[DATA_PTR:.*]] = getelementptr inbounds nuw { ptr, i64 }, ptr %[[AGG_ADDR]], i32 0, i32 0
// OGCG: %[[TMP_DATA:.*]] = load ptr, ptr %[[DATA_PTR]], align 8
// OGCG: %[[SIZE_PTR:.*]] = getelementptr inbounds nuw { ptr, i64 }, ptr %[[AGG_ADDR]], i32 0, i32 1
// OGCG: %[[TMP_SIZE:.*]] = load i64, ptr %[[SIZE_PTR]], align 8
// OGCG: call void @_ZN6VectorC1ESt16initializer_listIiE(ptr noundef nonnull align 1 dereferenceable(1) %[[VEC_ADDR]], ptr %[[TMP_DATA]], i64 %[[TMP_SIZE]])
// OGCG: call void @_ZN6VectorD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %[[VEC_ADDR]])
