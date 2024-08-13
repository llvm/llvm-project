// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM

namespace std {
template <class b> class initializer_list {
  const b *array_start;
  const b *array_end;
};
template <class b>
void f(initializer_list<b>) {;}
void test() {
  f({"xy","uv"});
}
} // namespace std

// CIR: [[INITLIST_TYPE:!.*]] = !cir.struct<class "std::initializer_list<const char *>" {!cir.ptr<!cir.ptr<!cir.int<s, 8>>>, !cir.ptr<!cir.ptr<!cir.int<s, 8>>>}>
// CIR: cir.func linkonce_odr @_ZSt1fIPKcEvSt16initializer_listIT_E(%arg0: [[INITLIST_TYPE]]
// CIR: [[LOCAL:%.*]] = cir.alloca [[INITLIST_TYPE]], !cir.ptr<[[INITLIST_TYPE]]>,
// CIR: cir.store %arg0, [[LOCAL]] : [[INITLIST_TYPE]], !cir.ptr<[[INITLIST_TYPE]]>
// CIR: cir.return

// CIR: cir.global "private" constant internal dsolocal [[STR_XY:@.*]] = #cir.const_array<"xy\00" : !cir.array<!s8i x 3>> : !cir.array<!s8i x 3>
// CIR: cir.global "private" constant internal dsolocal [[STR_UV:@.*]] = #cir.const_array<"uv\00" : !cir.array<!s8i x 3>> : !cir.array<!s8i x 3>

// CIR: cir.func @_ZSt4testv()
// CIR: cir.scope {
// CIR: [[INITLIST_LOCAL:%.*]] = cir.alloca [[INITLIST_TYPE]], !cir.ptr<[[INITLIST_TYPE]]>,  
// CIR: [[LOCAL_ELEM_ARRAY:%.*]] = cir.alloca !cir.array<!cir.ptr<!s8i> x 2>, !cir.ptr<!cir.array<!cir.ptr<!s8i> x 2>>,
// CIR: [[FIRST_ELEM_PTR:%.*]] = cir.cast(array_to_ptrdecay, [[LOCAL_ELEM_ARRAY]] : !cir.ptr<!cir.array<!cir.ptr<!s8i> x 2>>), !cir.ptr<!cir.ptr<!s8i>>
// CIR: [[XY_CHAR_ARRAY:%.*]] = cir.get_global [[STR_XY]]  : !cir.ptr<!cir.array<!s8i x 3>>
// CIR: [[STR_XY_PTR:%.*]] = cir.cast(array_to_ptrdecay, [[XY_CHAR_ARRAY]] : !cir.ptr<!cir.array<!s8i x 3>>), !cir.ptr<!s8i>
// CIR:  cir.store [[STR_XY_PTR]], [[FIRST_ELEM_PTR]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CIR: [[ONE:%.*]] = cir.const #cir.int<1>
// CIR: [[NEXT_ELEM_PTR:%.*]] = cir.ptr_stride([[FIRST_ELEM_PTR]] : !cir.ptr<!cir.ptr<!s8i>>, [[ONE]] : !s64i), !cir.ptr<!cir.ptr<!s8i>>
// CIR: [[UV_CHAR_ARRAY:%.*]] = cir.get_global [[STR_UV]]  : !cir.ptr<!cir.array<!s8i x 3>>
// CIR: [[STR_UV_PTR:%.*]] = cir.cast(array_to_ptrdecay, [[UV_CHAR_ARRAY]] : !cir.ptr<!cir.array<!s8i x 3>>), !cir.ptr<!s8i>
// CIR:  cir.store [[STR_UV_PTR]], [[NEXT_ELEM_PTR]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CIR: [[START_FLD_PTR:%.*]] = cir.get_member [[INITLIST_LOCAL]][0] {name = "array_start"} : !cir.ptr<[[INITLIST_TYPE]]> -> !cir.ptr<!cir.ptr<!cir.ptr<!s8i>>>
// CIR: [[START_FLD_PTR_AS_PTR_2_CHAR_ARRAY:%.*]] = cir.cast(bitcast, [[START_FLD_PTR]] : !cir.ptr<!cir.ptr<!cir.ptr<!s8i>>>), !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!s8i> x 2>>>
// CIR: cir.store [[LOCAL_ELEM_ARRAY]], [[START_FLD_PTR_AS_PTR_2_CHAR_ARRAY]] : !cir.ptr<!cir.array<!cir.ptr<!s8i> x 2>>, !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!s8i> x 2>>>
// CIR: [[ELEM_ARRAY_LEN:%.*]] = cir.const #cir.int<2>
// CIR: [[END_FLD_PTR:%.*]] = cir.get_member [[INITLIST_LOCAL]][1] {name = "array_end"} : !cir.ptr<[[INITLIST_TYPE]]> -> !cir.ptr<!cir.ptr<!cir.ptr<!s8i>>>
// CIR: [[LOCAL_ELEM_ARRAY_END:%.*]] = cir.ptr_stride([[LOCAL_ELEM_ARRAY]] : !cir.ptr<!cir.array<!cir.ptr<!s8i> x 2>>, [[ELEM_ARRAY_LEN]] : !u64i), !cir.ptr<!cir.array<!cir.ptr<!s8i> x 2>>
// CIR: [[END_FLD_PTR_AS_PTR_2_CHAR_ARRAY:%.*]] = cir.cast(bitcast, [[END_FLD_PTR]] : !cir.ptr<!cir.ptr<!cir.ptr<!s8i>>>), !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!s8i> x 2>>>
// CIR: cir.store [[LOCAL_ELEM_ARRAY_END]], [[END_FLD_PTR_AS_PTR_2_CHAR_ARRAY]] : !cir.ptr<!cir.array<!cir.ptr<!s8i> x 2>>, !cir.ptr<!cir.ptr<!cir.array<!cir.ptr<!s8i> x 2>>>
// CIR: [[ARG:%.*]] = cir.load [[INITLIST_LOCAL]] : !cir.ptr<[[INITLIST_TYPE]]>, [[INITLIST_TYPE]]
// CIR: cir.call @_ZSt1fIPKcEvSt16initializer_listIT_E([[ARG]]) : ([[INITLIST_TYPE]]) -> ()
// CIR: }
// CIR: cir.return
// CIR: }

// LLVM: %"class.std::initializer_list<const char *>" = type { ptr, ptr }

// LLVM: @.str = internal constant [3 x i8] c"xy\00"
// LLVM: @.str1 = internal constant [3 x i8] c"uv\00"

// LLVM: define linkonce_odr void @_ZSt1fIPKcEvSt16initializer_listIT_E(%"class.std::initializer_list<const char *>" [[ARG0:%.*]])
// LLVM: [[LOCAL_PTR:%.*]] = alloca %"class.std::initializer_list<const char *>", i64 1, align 8, 
// LLVM: store %"class.std::initializer_list<const char *>" [[ARG0]], ptr [[LOCAL_PTR]], align 8,
// LLVM: ret void,
// LLVM: }

// LLVM: define dso_local void @_ZSt4testv()
// LLVM: br label %[[SCOPE_START:.*]],  
// LLVM: [[SCOPE_START]]: ; preds = %0
// LLVM:  [[INIT_STRUCT:%.*]] = alloca %"class.std::initializer_list<const char *>", i64 1, align 8,
// LLVM:  [[ELEM_ARRAY_PTR:%.*]] = alloca [2 x ptr], i64 1, align 8,
// LLVM:  [[PTR_FIRST_ELEM:%.*]] = getelementptr ptr, ptr [[ELEM_ARRAY_PTR]], i32 0,
// LLVM:  store ptr @.str, ptr [[PTR_FIRST_ELEM]], align 8,
// LLVM:  [[PTR_SECOND_ELEM:%.*]] = getelementptr ptr, ptr [[PTR_FIRST_ELEM]], i64 1,
// LLVM:  store ptr @.str1, ptr [[PTR_SECOND_ELEM]], align 8,
// LLVM:  [[INIT_START_FLD_PTR:%.*]] = getelementptr %"class.std::initializer_list<const char *>", ptr [[INIT_STRUCT]], i32 0, i32 0,
// LLVM:  [[INIT_END_FLD_PTR:%.*]] = getelementptr %"class.std::initializer_list<const char *>", ptr [[INIT_STRUCT]], i32 0, i32 1,
// LLVM:  [[ELEM_ARRAY_END:%.*]] = getelementptr [2 x ptr], ptr [[ELEM_ARRAY_PTR]], i64 2,
// LLVM:  store ptr [[ELEM_ARRAY_END]], ptr [[INIT_END_FLD_PTR]], align 8,
// LLVM:  [[ARG2PASS:%.*]] = load %"class.std::initializer_list<const char *>", ptr [[INIT_STRUCT]], align 8,
// LLVM:  call void @_ZSt1fIPKcEvSt16initializer_listIT_E(%"class.std::initializer_list<const char *>" [[ARG2PASS]]),
// LLVM:  br label %[[SCOPE_END:.*]],
// LLVM: [[SCOPE_END]]: ; preds = %[[SCOPE_START]]
// LLVM:  ret void
