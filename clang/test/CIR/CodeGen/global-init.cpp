// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir -mmlir --mlir-print-ir-before=cir-lowering-prepare %s -o %t.cir 2> %t-before.cir
// RUN: FileCheck --input-file=%t-before.cir %s --check-prefix=CIR-BEFORE-LPP
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
// RUN: %clang_cc1 -std=c++17 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=OGCG

// Declarations that appear before global-specific definitions

// CIR: module @{{.*}} attributes {
// CIR-SAME: cir.global_ctors = [#cir.global_ctor<"_GLOBAL__sub_I_[[FILENAME:.*]]", 65535>]

// LLVM: @__dso_handle = external hidden global i8
// LLVM: @needsCtor = global %struct.NeedsCtor zeroinitializer, align 1
// LLVM: @needsDtor = global %struct.NeedsDtor zeroinitializer, align 1
// LLVM: @needsCtorDtor = global %struct.NeedsCtorDtor zeroinitializer, align 1
// LLVM: @arrDtor = global [16 x %struct.ArrayDtor] zeroinitializer, align 16
// LLVM: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_[[FILENAME:.*]], ptr null }]

// OGCG: @needsCtor = global %struct.NeedsCtor zeroinitializer, align 1
// OGCG: @needsDtor = global %struct.NeedsDtor zeroinitializer, align 1
// OGCG: @__dso_handle = external hidden global i8
// OGCG: @needsCtorDtor = global %struct.NeedsCtorDtor zeroinitializer, align 1
// OGCG: @arrDtor = global [16 x %struct.ArrayDtor] zeroinitializer, align 16
// OGCG: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_[[FILENAME:.*]], ptr null }]

struct NeedsCtor {
  NeedsCtor();
};

NeedsCtor needsCtor;

// CIR-BEFORE-LPP: cir.global external @needsCtor = ctor : !rec_NeedsCtor {
// CIR-BEFORE-LPP:   %[[THIS:.*]] = cir.get_global @needsCtor : !cir.ptr<!rec_NeedsCtor>
// CIR-BEFORE-LPP:   cir.call @_ZN9NeedsCtorC1Ev(%[[THIS]]) : (!cir.ptr<!rec_NeedsCtor>) -> ()

// CIR: cir.global external @needsCtor = #cir.zero : !rec_NeedsCtor
// CIR: cir.func internal private @__cxx_global_var_init() {
// CIR:   %0 = cir.get_global @needsCtor : !cir.ptr<!rec_NeedsCtor>
// CIR:   cir.call @_ZN9NeedsCtorC1Ev(%0) : (!cir.ptr<!rec_NeedsCtor>) -> ()

// LLVM: define internal void @__cxx_global_var_init()
// LLVM:   call void @_ZN9NeedsCtorC1Ev(ptr @needsCtor)

// OGCG: define internal void @__cxx_global_var_init() {{.*}} section ".text.startup" {
// OGCG:   call void @_ZN9NeedsCtorC1Ev(ptr noundef nonnull align 1 dereferenceable(1) @needsCtor)


struct NeedsDtor {
  ~NeedsDtor();
};

NeedsDtor needsDtor;

// CIR-BEFORE-LPP: cir.global external @needsDtor = #cir.zero : !rec_NeedsDtor dtor {
// CIR-BEFORE-LPP:   %[[THIS:.*]] = cir.get_global @needsDtor : !cir.ptr<!rec_NeedsDtor>
// CIR-BEFORE-LPP:   cir.call @_ZN9NeedsDtorD1Ev(%[[THIS]]) : (!cir.ptr<!rec_NeedsDtor>) -> ()

// CIR: cir.global external @needsDtor = #cir.zero : !rec_NeedsDtor
// CIR: cir.func internal private @__cxx_global_var_init.1() {
// CIR:   %[[OBJ:.*]] = cir.get_global @needsDtor : !cir.ptr<!rec_NeedsDtor>
// CIR:   %[[DTOR:.*]] = cir.get_global @_ZN9NeedsDtorD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_NeedsDtor>)>>
// CIR:   %[[DTOR_CAST:.*]] = cir.cast bitcast %[[DTOR]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_NeedsDtor>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:   %[[OBJ_CAST:.*]] = cir.cast bitcast %[[OBJ]] : !cir.ptr<!rec_NeedsDtor> -> !cir.ptr<!void>
// CIR:   %[[HANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<i8>
// CIR:   cir.call @__cxa_atexit(%[[DTOR_CAST]], %[[OBJ_CAST]], %[[HANDLE]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>) -> ()

// LLVM: define internal void @__cxx_global_var_init.1() {
// LLVM:   call void @__cxa_atexit(ptr @_ZN9NeedsDtorD1Ev, ptr @needsDtor, ptr @__dso_handle)

// OGCG: define internal void @__cxx_global_var_init.1() {{.*}} section ".text.startup" {
// OGCG:   %{{.*}} = call i32 @__cxa_atexit(ptr @_ZN9NeedsDtorD1Ev, ptr @needsDtor, ptr @__dso_handle)

struct NeedsCtorDtor {
  NeedsCtorDtor();
  ~NeedsCtorDtor();
};

NeedsCtorDtor needsCtorDtor;

// CIR-BEFORE-LPP: cir.global external @needsCtorDtor = ctor : !rec_NeedsCtorDtor {
// CIR-BEFORE-LPP:   %[[THIS:.*]] = cir.get_global @needsCtorDtor : !cir.ptr<!rec_NeedsCtorDtor>
// CIR-BEFORE-LPP:   cir.call @_ZN13NeedsCtorDtorC1Ev(%[[THIS]]) : (!cir.ptr<!rec_NeedsCtorDtor>) -> ()
// CIR-BEFORE-LPP: } dtor {
// CIR-BEFORE-LPP:   %[[THIS:.*]] = cir.get_global @needsCtorDtor : !cir.ptr<!rec_NeedsCtorDtor>
// CIR-BEFORE-LPP:   cir.call @_ZN13NeedsCtorDtorD1Ev(%[[THIS]]) : (!cir.ptr<!rec_NeedsCtorDtor>) -> ()

// CIR: cir.global external @needsCtorDtor = #cir.zero : !rec_NeedsCtorDtor
// CIR: cir.func internal private @__cxx_global_var_init.2() {
// CIR:   %[[OBJ:.*]] = cir.get_global @needsCtorDtor : !cir.ptr<!rec_NeedsCtorDtor>
// CIR:   cir.call @_ZN13NeedsCtorDtorC1Ev(%[[OBJ]]) : (!cir.ptr<!rec_NeedsCtorDtor>) -> ()
// CIR:   %[[OBJ:.*]] = cir.get_global @needsCtorDtor : !cir.ptr<!rec_NeedsCtorDtor>
// CIR:   %[[DTOR:.*]] = cir.get_global @_ZN13NeedsCtorDtorD1Ev : !cir.ptr<!cir.func<(!cir.ptr<!rec_NeedsCtorDtor>)>>
// CIR:   %[[DTOR_CAST:.*]] = cir.cast bitcast %[[DTOR]] : !cir.ptr<!cir.func<(!cir.ptr<!rec_NeedsCtorDtor>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:   %[[OBJ_CAST:.*]] = cir.cast bitcast %[[OBJ]] : !cir.ptr<!rec_NeedsCtorDtor> -> !cir.ptr<!void>
// CIR:   %[[HANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<i8>
// CIR:   cir.call @__cxa_atexit(%[[DTOR_CAST]], %[[OBJ_CAST]], %[[HANDLE]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>) -> ()

// LLVM: define internal void @__cxx_global_var_init.2() {
// LLVM:   call void @_ZN13NeedsCtorDtorC1Ev(ptr @needsCtorDtor)
// LLVM:   call void @__cxa_atexit(ptr @_ZN13NeedsCtorDtorD1Ev, ptr @needsCtorDtor, ptr @__dso_handle)

// OGCG: define internal void @__cxx_global_var_init.2() {{.*}} section ".text.startup" {
// OGCG:   call void @_ZN13NeedsCtorDtorC1Ev(ptr noundef nonnull align 1 dereferenceable(1) @needsCtorDtor)
// OGCG:   %{{.*}} = call i32 @__cxa_atexit(ptr @_ZN13NeedsCtorDtorD1Ev, ptr @needsCtorDtor, ptr @__dso_handle)

float num;
float _Complex a = {num, num};

// CIR-BEFORE-LPP: cir.global external @num = #cir.fp<0.000000e+00> : !cir.float
// CIR-BEFORE-LPP: cir.global external @a = ctor : !cir.complex<!cir.float> {
// CIR-BEFORE-LPP:  %[[THIS:.*]] = cir.get_global @a : !cir.ptr<!cir.complex<!cir.float>>
// CIR-BEFORE-LPP:  %[[NUM:.*]] = cir.get_global @num : !cir.ptr<!cir.float>
// CIR-BEFORE-LPP:  %[[REAL:.*]] = cir.load{{.*}} %[[NUM]] : !cir.ptr<!cir.float>, !cir.float
// CIR-BEFORE-LPP:  %[[NUM:.*]] = cir.get_global @num : !cir.ptr<!cir.float>
// CIR-BEFORE-LPP:  %[[IMAG:.*]] = cir.load{{.*}} %[[NUM]] : !cir.ptr<!cir.float>, !cir.float
// CIR-BEFORE-LPP:  %[[COMPLEX_VAL:.*]] = cir.complex.create %[[REAL]], %[[IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR-BEFORE-LPP:  cir.store{{.*}} %[[COMPLEX_VAL:.*]], %[[THIS]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>
// CIR-BEFORE-LPP: }

// CIR:  cir.global external @num = #cir.fp<0.000000e+00> : !cir.float
// CIR:  cir.global external @a = #cir.zero : !cir.complex<!cir.float>
// CIR:  cir.func internal private @__cxx_global_var_init.3()
// CIR:   %[[A_ADDR:.*]] = cir.get_global @a : !cir.ptr<!cir.complex<!cir.float>>
// CIR:   %[[NUM:.*]] = cir.get_global @num : !cir.ptr<!cir.float>
// CIR:   %[[REAL:.*]] = cir.load{{.*}} %[[NUM]] : !cir.ptr<!cir.float>, !cir.float
// CIR:   %[[NUM:.*]] = cir.get_global @num : !cir.ptr<!cir.float>
// CIR:   %[[IMAG:.*]] = cir.load{{.*}} %[[NUM]] : !cir.ptr<!cir.float>, !cir.float
// CIR:   %[[COMPLEX_VAL:.*]] = cir.complex.create %[[REAL]], %[[IMAG]] : !cir.float -> !cir.complex<!cir.float>
// CIR:   cir.store{{.*}} %[[COMPLEX_VAL]], %[[A_ADDR]] : !cir.complex<!cir.float>, !cir.ptr<!cir.complex<!cir.float>>

// LLVM: define internal void @__cxx_global_var_init.3()
// LLVM:   %[[REAL:.*]] = load float, ptr @num, align 4
// LLVM:   %[[IMAG:.*]] = load float, ptr @num, align 4
// LLVM:   %[[TMP_COMPLEX_VAL:.*]] = insertvalue { float, float } {{.*}}, float %[[REAL]], 0
// LLVM:   %[[COMPLEX_VAL:.*]] = insertvalue { float, float } %[[TMP_COMPLEX_VAL]], float %[[IMAG]], 1
// LLVM:   store { float, float } %[[COMPLEX_VAL]], ptr @a, align 4

// OGCG: define internal void @__cxx_global_var_init.3() {{.*}} section ".text.startup"
// OGCG:   %[[REAL:.*]] = load float, ptr @num, align 4
// OGCG:   %[[IMAG:.*]] = load float, ptr @num, align 4
// OGCG:   store float %[[REAL]], ptr @a, align 4
// OGCG:   store float %[[IMAG]], ptr getelementptr inbounds nuw ({ float, float }, ptr @a, i32 0, i32 1), align 4

float fp;
int i = (int)fp;

// CIR-BEFORE-LPP: cir.global external @i = ctor : !s32i {
// CIR-BEFORE-LPP:   %[[I:.*]] = cir.get_global @i : !cir.ptr<!s32i>
// CIR-BEFORE-LPP:   %[[FP:.*]] = cir.get_global @fp : !cir.ptr<!cir.float>
// CIR-BEFORE-LPP:   %[[FP_VAL:.*]] = cir.load{{.*}} %[[FP]] : !cir.ptr<!cir.float>, !cir.float
// CIR-BEFORE-LPP:   %[[FP_I32:.*]] = cir.cast float_to_int %[[FP_VAL]] : !cir.float -> !s32i
// CIR-BEFORE-LPP:   cir.store{{.*}} %[[FP_I32]], %[[I]] : !s32i, !cir.ptr<!s32i>
// CIR-BEFORE-LPP: }

// CIR: cir.func internal private @__cxx_global_var_init.4()
// CIR:   %[[I_ADDR:.*]] = cir.get_global @i : !cir.ptr<!s32i>
// CIR:   %[[FP_ADDR:.*]] = cir.get_global @fp : !cir.ptr<!cir.float>
// CIR:   %[[TMP_FP:.*]] = cir.load{{.*}} %[[FP_ADDR]] : !cir.ptr<!cir.float>, !cir.float
// CIR:   %[[FP_I32:.*]] = cir.cast float_to_int %[[TMP_FP]] : !cir.float -> !s32i
// CIR:   cir.store{{.*}} %[[FP_I32]], %[[I_ADDR]] : !s32i, !cir.ptr<!s32i>

// LLVM: define internal void @__cxx_global_var_init.4()
// LLVM:   %[[TMP_FP:.*]] = load float, ptr @fp, align 4
// LLVM:   %[[FP_I32:.*]] = fptosi float %[[TMP_FP]] to i32
// LLVM:   store i32 %[[FP_I32]], ptr @i, align 4

// OGCG: define internal void @__cxx_global_var_init.4() {{.*}} section ".text.startup"
// OGCG:   %[[TMP_FP:.*]] = load float, ptr @fp, align 4
// OGCG:   %[[FP_I32:.*]] = fptosi float %[[TMP_FP]] to i32
// OGCG:   store i32 %[[FP_I32]], ptr @i, align 4

struct ArrayDtor {
  ~ArrayDtor();
};

ArrayDtor arrDtor[16];

// CIR-BEFORE-LPP:      cir.global external @arrDtor = #cir.zero : !cir.array<!rec_ArrayDtor x 16>
// CIR-BEFORE-LPP-SAME:   dtor {
// CIR-BEFORE-LPP:          %[[THIS:.*]] = cir.get_global @arrDtor : !cir.ptr<!cir.array<!rec_ArrayDtor x 16>>
// CIR-BEFORE-LPP:          cir.array.dtor %[[THIS]] : !cir.ptr<!cir.array<!rec_ArrayDtor x 16>> {
// CIR-BEFORE-LPP:          ^bb0(%[[ELEM:.*]]: !cir.ptr<!rec_ArrayDtor>):
// CIR-BEFORE-LPP:            cir.call @_ZN9ArrayDtorD1Ev(%[[ELEM]]) nothrow : (!cir.ptr<!rec_ArrayDtor>) -> ()
// CIR-BEFORE-LPP:            cir.yield
// CIR-BEFORE-LPP:          }
// CIR-BEFORE-LPP:        }

// CIR: cir.global external @arrDtor = #cir.zero : !cir.array<!rec_ArrayDtor x 16>
// CIR: cir.func internal private @__cxx_global_array_dtor(%[[ARR_ARG:.*]]: !cir.ptr<!void> {{.*}}) {
// CIR:   %[[CONST15:.*]] = cir.const #cir.int<15> : !u64i
// CIR:   %[[BEGIN:.*]] = cir.cast array_to_ptrdecay %[[ARR_ARG]] : !cir.ptr<!void> -> !cir.ptr<!rec_ArrayDtor>
// CIR:   %[[END:.*]] = cir.ptr_stride %[[BEGIN]], %[[CONST15]] : (!cir.ptr<!rec_ArrayDtor>, !u64i) -> !cir.ptr<!rec_ArrayDtor>
// CIR:   %[[CUR_ADDR:.*]] = cir.alloca !cir.ptr<!rec_ArrayDtor>, !cir.ptr<!cir.ptr<!rec_ArrayDtor>>, ["__array_idx"]
// CIR:   cir.store %[[END]], %[[CUR_ADDR]] : !cir.ptr<!rec_ArrayDtor>, !cir.ptr<!cir.ptr<!rec_ArrayDtor>>
// CIR:   cir.do {
// CIR:     %[[CUR:.*]] = cir.load %[[CUR_ADDR]] : !cir.ptr<!cir.ptr<!rec_ArrayDtor>>, !cir.ptr<!rec_ArrayDtor>
// CIR:     cir.call @_ZN9ArrayDtorD1Ev(%[[CUR]]) nothrow : (!cir.ptr<!rec_ArrayDtor>) -> ()
// CIR:     %[[NEG_ONE:.*]] = cir.const #cir.int<-1> : !s64i
// CIR:     %[[NEXT:.*]] = cir.ptr_stride %[[CUR]], %[[NEG_ONE]] : (!cir.ptr<!rec_ArrayDtor>, !s64i) -> !cir.ptr<!rec_ArrayDtor>
// CIR:     cir.store %[[NEXT]], %[[CUR_ADDR]] : !cir.ptr<!rec_ArrayDtor>, !cir.ptr<!cir.ptr<!rec_ArrayDtor>>
// CIR:     cir.yield
// CIR:   } while {
// CIR:     %[[CUR:.*]] = cir.load %[[CUR_ADDR]] : !cir.ptr<!cir.ptr<!rec_ArrayDtor>>, !cir.ptr<!rec_ArrayDtor>
// CIR:     %[[CMP:.*]] = cir.cmp(ne, %[[CUR]], %[[BEGIN]]) : !cir.ptr<!rec_ArrayDtor>, !cir.bool
// CIR:     cir.condition(%[[CMP]])
// CIR:   }
// CIR:   cir.return
// CIR: }
//
// CIR: cir.func internal private @__cxx_global_var_init.5() {
// CIR:   %[[ARR:.*]] = cir.get_global @arrDtor : !cir.ptr<!cir.array<!rec_ArrayDtor x 16>>
// CIR:   %[[DTOR:.*]] = cir.get_global @__cxx_global_array_dtor : !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:   %[[DTOR_CAST:.*]] = cir.cast bitcast %[[DTOR]] : !cir.ptr<!cir.func<(!cir.ptr<!void>)>> -> !cir.ptr<!cir.func<(!cir.ptr<!void>)>>
// CIR:   %[[ARR_CAST:.*]] = cir.cast bitcast %[[ARR]] : !cir.ptr<!cir.array<!rec_ArrayDtor x 16>> -> !cir.ptr<!void>
// CIR:   %[[HANDLE:.*]] = cir.get_global @__dso_handle : !cir.ptr<i8>
// CIR:   cir.call @__cxa_atexit(%[[DTOR_CAST]], %[[ARR_CAST]], %[[HANDLE]]) : (!cir.ptr<!cir.func<(!cir.ptr<!void>)>>, !cir.ptr<!void>, !cir.ptr<i8>) -> ()

// LLVM: define internal void @__cxx_global_array_dtor(ptr %[[ARR_ARG:.*]]) {
// LLVM:   %[[BEGIN:.*]] = getelementptr %struct.ArrayDtor, ptr %[[ARR_ARG]], i32 0
// LLVM:   %[[END:.*]] = getelementptr %struct.ArrayDtor, ptr %[[BEGIN]], i64 15
// LLVM:   %[[CUR_ADDR:.*]] = alloca ptr
// LLVM:   store ptr %[[END]], ptr %[[CUR_ADDR]]
// LLVM:   br label %[[LOOP_BODY:.*]]
// LLVM: [[LOOP_COND:.*]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[CUR_ADDR]]
// LLVM:   %[[CMP:.*]] = icmp ne ptr %[[CUR]], %[[BEGIN]]
// LLVM:   br i1 %[[CMP]], label %[[LOOP_BODY]], label %[[LOOP_END:.*]]
// LLVM: [[LOOP_BODY]]:
// LLVM:   %[[CUR:.*]] = load ptr, ptr %[[CUR_ADDR]]
// LLVM:   call void @_ZN9ArrayDtorD1Ev(ptr %[[CUR]]) #0
// LLVM:   %[[PREV:.*]] = getelementptr %struct.ArrayDtor, ptr %[[CUR]], i64 -1
// LLVM:   store ptr %[[PREV]], ptr %[[CUR_ADDR]]
// LLVM:   br label %[[LOOP_COND]]
// LLVM: [[LOOP_END]]:
// LLVM:   ret void
// LLVM: }
//
// LLVM: define internal void @__cxx_global_var_init.5() {
// LLVM:   call void @__cxa_atexit(ptr @__cxx_global_array_dtor, ptr @arrDtor, ptr @__dso_handle)

// Note: OGCG defines these functions in reverse order of CIR->LLVM.
// Note also: OGCG doesn't pass the address of the array to the destructor function.
//            Instead, it uses the global directly in the helper function.

// OGCG: define internal void @__cxx_global_var_init.5() {{.*}} section ".text.startup" {
// OGCG:   call i32 @__cxa_atexit(ptr @__cxx_global_array_dtor, ptr null, ptr @__dso_handle)

// OGCG: define internal void @__cxx_global_array_dtor(ptr noundef %[[ARG:.*]]) {{.*}} section ".text.startup" {
// OGCG: entry:
// OGCG:   %[[UNUSED_ADDR:.*]] = alloca ptr
// OGCG:   store ptr %[[ARG]], ptr %[[UNUSED_ADDR]]
// OGCG:   br label %[[LOOP_BODY:.*]]
// OGCG: [[LOOP_BODY]]:
// OGCG:   %[[PREV:.*]] = phi ptr [ getelementptr inbounds (%struct.ArrayDtor, ptr @arrDtor, i64 16), %entry ], [ %[[CUR:.*]], %[[LOOP_BODY]] ]
// OGCG:   %[[CUR]] = getelementptr inbounds %struct.ArrayDtor, ptr %[[PREV]], i64 -1
// OGCG:   call void @_ZN9ArrayDtorD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %[[CUR]])
// OGCG:   %[[DONE:.*]] = icmp eq ptr %[[CUR]], @arrDtor
// OGCG:   br i1 %[[DONE]], label %[[LOOP_END:.*]], label %[[LOOP_BODY]]
// OGCG: [[LOOP_END]]:
// OGCG:   ret void
// OGCG: }

// Common init function for all globals with default priority

// CIR: cir.func private @_GLOBAL__sub_I_[[FILENAME:.*]]() {
// CIR:   cir.call @__cxx_global_var_init() : () -> ()
// CIR:   cir.call @__cxx_global_var_init.1() : () -> ()
// CIR:   cir.call @__cxx_global_var_init.2() : () -> ()
// CIR:   cir.call @__cxx_global_var_init.3() : () -> ()
// CIR:   cir.call @__cxx_global_var_init.4() : () -> ()
// CIR:   cir.call @__cxx_global_var_init.5() : () -> ()

// LLVM: define void @_GLOBAL__sub_I_[[FILENAME]]()
// LLVM:   call void @__cxx_global_var_init()
// LLVM:   call void @__cxx_global_var_init.1()
// LLVM:   call void @__cxx_global_var_init.2()
// LLVM:   call void @__cxx_global_var_init.3()
// LLVM:   call void @__cxx_global_var_init.4()
// LLVM:   call void @__cxx_global_var_init.5()

// OGCG: define internal void @_GLOBAL__sub_I_[[FILENAME]]() {{.*}} section ".text.startup" {
// OGCG:   call void @__cxx_global_var_init()
// OGCG:   call void @__cxx_global_var_init.1()
// OGCG:   call void @__cxx_global_var_init.2()
// OGCG:   call void @__cxx_global_var_init.3()
// OGCG:   call void @__cxx_global_var_init.4()
// OGCG:   call void @__cxx_global_var_init.5()
