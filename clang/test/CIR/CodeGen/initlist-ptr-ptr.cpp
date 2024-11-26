// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir -clangir-disable-passes
// RUN: FileCheck --check-prefix=BEFORE --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -triple aarch64-none-linux-android21 -fclangir -emit-llvm -fno-clangir-call-conv-lowering %s -o %t.ll
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

// BEFORE: [[INITLIST_TYPE:!.*]] = !cir.struct<class "std::initializer_list<const char *>" {!cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!cir.ptr<!s8i>>} #cir.record.decl.ast>
// BEFORE: %0 = cir.alloca [[INITLIST_TYPE]], !cir.ptr<[[INITLIST_TYPE]]>,
// BEFORE: %1 = cir.get_global @".str" : !cir.ptr<!cir.array<!s8i x 3>>
// BEFORE: %2 = cir.cast(array_to_ptrdecay, %1 : !cir.ptr<!cir.array<!s8i x 3>>), !cir.ptr<!s8i>
// BEFORE: %3 = cir.get_global @".str.1" : !cir.ptr<!cir.array<!s8i x 3>> 
// BEFORE: %4 = cir.cast(array_to_ptrdecay, %3 : !cir.ptr<!cir.array<!s8i x 3>>), !cir.ptr<!s8i>
// BEFORE: cir.std.initializer_list %0 (%2, %4 : !cir.ptr<!s8i>, !cir.ptr<!s8i>) : !cir.ptr<[[INITLIST_TYPE]]>
// BEFORE: %5 = cir.load %0 : !cir.ptr<[[INITLIST_TYPE]]>, [[INITLIST_TYPE]]
// BEFORE: cir.call @_ZSt1fIPKcEvSt16initializer_listIT_E(%5) : ([[INITLIST_TYPE]]) -> ()

// CIR: [[INITLIST_TYPE:!.*]] = !cir.struct<class "std::initializer_list<const char *>" {!cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!cir.ptr<!s8i>>}>
// CIR: cir.func linkonce_odr @_ZSt1fIPKcEvSt16initializer_listIT_E(%arg0: [[INITLIST_TYPE]]
// CIR: [[LOCAL:%.*]] = cir.alloca [[INITLIST_TYPE]], !cir.ptr<[[INITLIST_TYPE]]>,
// CIR: cir.store %arg0, [[LOCAL]] : [[INITLIST_TYPE]], !cir.ptr<[[INITLIST_TYPE]]>
// CIR: cir.return

// CIR: cir.global "private" constant cir_private dsolocal [[STR_XY:@.*]] = #cir.const_array<"xy\00" : !cir.array<!s8i x 3>> : !cir.array<!s8i x 3>
// CIR: cir.global "private" constant cir_private dsolocal [[STR_UV:@.*]] = #cir.const_array<"uv\00" : !cir.array<!s8i x 3>> : !cir.array<!s8i x 3>

// CIR: cir.func @_ZSt4testv()
// CIR: cir.scope {
// CIR: [[INITLIST_LOCAL:%.*]] = cir.alloca [[INITLIST_TYPE]], !cir.ptr<[[INITLIST_TYPE]]>,  
// CIR: [[XY_CHAR_ARRAY:%.*]] = cir.get_global [[STR_XY]]  : !cir.ptr<!cir.array<!s8i x 3>>
// CIR: [[STR_XY_PTR:%.*]] = cir.cast(array_to_ptrdecay, [[XY_CHAR_ARRAY]] : !cir.ptr<!cir.array<!s8i x 3>>), !cir.ptr<!s8i>
// CIR: [[UV_CHAR_ARRAY:%.*]] = cir.get_global [[STR_UV]]  : !cir.ptr<!cir.array<!s8i x 3>>
// CIR: [[STR_UV_PTR:%.*]] = cir.cast(array_to_ptrdecay, [[UV_CHAR_ARRAY]] : !cir.ptr<!cir.array<!s8i x 3>>), !cir.ptr<!s8i>
// CIR: [[LOCAL_ELEM_ARRAY:%.*]] = cir.alloca !cir.array<!cir.ptr<!s8i> x 2>, !cir.ptr<!cir.array<!cir.ptr<!s8i> x 2>>,
// CIR: [[ELEM_BEGIN:%.*]] = cir.cast(array_to_ptrdecay, [[LOCAL_ELEM_ARRAY]] : !cir.ptr<!cir.array<!cir.ptr<!s8i> x 2>>), !cir.ptr<!cir.ptr<!s8i>>
// CIR:  cir.store [[STR_XY_PTR]], [[ELEM_BEGIN]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CIR: [[ONE:%.*]] = cir.const #cir.int<1>
// CIR: [[NEXT_ELEM_PTR:%.*]] = cir.ptr_stride([[ELEM_BEGIN]] : !cir.ptr<!cir.ptr<!s8i>>, [[ONE]] : !u64i), !cir.ptr<!cir.ptr<!s8i>>
// CIR:  cir.store [[STR_UV_PTR]], [[NEXT_ELEM_PTR]] : !cir.ptr<!s8i>, !cir.ptr<!cir.ptr<!s8i>>
// CIR: [[START_FLD_PTR:%.*]] = cir.get_member [[INITLIST_LOCAL]][0] {name = "array_start"} : !cir.ptr<[[INITLIST_TYPE]]> -> !cir.ptr<!cir.ptr<!cir.ptr<!s8i>>>
// CIR: cir.store [[ELEM_BEGIN]], [[START_FLD_PTR]] : !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s8i>>>
// CIR: [[ELEM_ARRAY_LEN:%.*]] = cir.const #cir.int<2>
// CIR: [[ELEM_END:%.*]] = cir.ptr_stride([[ELEM_BEGIN]] : !cir.ptr<!cir.ptr<!s8i>>, [[ELEM_ARRAY_LEN]] : !u64i), !cir.ptr<!cir.ptr<!s8i>>
// CIR: [[END_FLD_PTR:%.*]] = cir.get_member [[INITLIST_LOCAL]][1] {name = "array_end"} : !cir.ptr<[[INITLIST_TYPE]]> -> !cir.ptr<!cir.ptr<!cir.ptr<!s8i>>>
// CIR: cir.store [[ELEM_END]], [[END_FLD_PTR]] : !cir.ptr<!cir.ptr<!s8i>>, !cir.ptr<!cir.ptr<!cir.ptr<!s8i>>>
// CIR: [[ARG:%.*]] = cir.load [[INITLIST_LOCAL]] : !cir.ptr<[[INITLIST_TYPE]]>, [[INITLIST_TYPE]]
// CIR: cir.call @_ZSt1fIPKcEvSt16initializer_listIT_E([[ARG]]) : ([[INITLIST_TYPE]]) -> ()
// CIR: }
// CIR: cir.return
// CIR: }

// LLVM: %"class.std::initializer_list<const char *>" = type { ptr, ptr }

// LLVM: @.str = private constant [3 x i8] c"xy\00"
// LLVM: @.str.1 = private constant [3 x i8] c"uv\00"

// LLVM: define linkonce_odr void @_ZSt1fIPKcEvSt16initializer_listIT_E(%"class.std::initializer_list<const char *>" [[ARG0:%.*]])
// LLVM: [[LOCAL_PTR:%.*]] = alloca %"class.std::initializer_list<const char *>", i64 1, align 8, 
// LLVM: store %"class.std::initializer_list<const char *>" [[ARG0]], ptr [[LOCAL_PTR]], align 8,
// LLVM: ret void,
// LLVM: }

// LLVM: define dso_local void @_ZSt4testv()
// LLVM:  [[INIT_STRUCT:%.*]] = alloca %"class.std::initializer_list<const char *>", i64 1, align 8,
// LLVM:  [[ELEM_ARRAY_PTR:%.*]] = alloca [2 x ptr], i64 1, align 8,
// LLVM: br label %[[SCOPE_START:.*]],  
// LLVM: [[SCOPE_START]]: ; preds = %0
// LLVM:  [[PTR_FIRST_ELEM:%.*]] = getelementptr ptr, ptr [[ELEM_ARRAY_PTR]], i32 0,
// LLVM:  store ptr @.str, ptr [[PTR_FIRST_ELEM]], align 8,
// LLVM:  [[PTR_SECOND_ELEM:%.*]] = getelementptr ptr, ptr [[PTR_FIRST_ELEM]], i64 1,
// LLVM:  store ptr @.str.1, ptr [[PTR_SECOND_ELEM]], align 8,
// LLVM:  [[INIT_START_FLD_PTR:%.*]] = getelementptr %"class.std::initializer_list<const char *>", ptr [[INIT_STRUCT]], i32 0, i32 0,
// LLVM:  store ptr [[PTR_FIRST_ELEM]], ptr [[INIT_START_FLD_PTR]], align 8,
// LLVM:  [[ELEM_ARRAY_END:%.*]] = getelementptr ptr, ptr [[PTR_FIRST_ELEM]], i64 2,
// LLVM:  [[INIT_END_FLD_PTR:%.*]] = getelementptr %"class.std::initializer_list<const char *>", ptr [[INIT_STRUCT]], i32 0, i32 1,
// LLVM:  store ptr [[ELEM_ARRAY_END]], ptr [[INIT_END_FLD_PTR]], align 8,
// LLVM:  [[ARG2PASS:%.*]] = load %"class.std::initializer_list<const char *>", ptr [[INIT_STRUCT]], align 8,
// LLVM:  call void @_ZSt1fIPKcEvSt16initializer_listIT_E(%"class.std::initializer_list<const char *>" [[ARG2PASS]]),
// LLVM:  br label %[[SCOPE_END:.*]],
// LLVM: [[SCOPE_END]]: ; preds = %[[SCOPE_START]]
// LLVM:  ret void
