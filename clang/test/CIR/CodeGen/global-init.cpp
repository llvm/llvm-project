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
// LLVM: @llvm.global_ctors = appending global [1 x { i32, ptr, ptr }] [{ i32, ptr, ptr } { i32 65535, ptr @_GLOBAL__sub_I_[[FILENAME:.*]], ptr null }]

// OGCG: @needsCtor = global %struct.NeedsCtor zeroinitializer, align 1
// OGCG: @needsDtor = global %struct.NeedsDtor zeroinitializer, align 1
// OGCG: @__dso_handle = external hidden global i8
// OGCG: @needsCtorDtor = global %struct.NeedsCtorDtor zeroinitializer, align 1
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
// CIR-BEFORE-LPP:   %0 = cir.get_global @i : !cir.ptr<!s32i>
// CIR-BEFORE-LPP:   %1 = cir.get_global @fp : !cir.ptr<!cir.float>
// CIR-BEFORE-LPP:   %2 = cir.load{{.*}} %1 : !cir.ptr<!cir.float>, !cir.float
// CIR-BEFORE-LPP:   %3 = cir.cast float_to_int %2 : !cir.float -> !s32i
// CIR-BEFORE-LPP:   cir.store{{.*}} %3, %0 : !s32i, !cir.ptr<!s32i>
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

// Common init function for all globals with default priority

// CIR: cir.func private @_GLOBAL__sub_I_[[FILENAME:.*]]() {
// CIR:   cir.call @__cxx_global_var_init() : () -> ()
// CIR:   cir.call @__cxx_global_var_init.1() : () -> ()
// CIR:   cir.call @__cxx_global_var_init.2() : () -> ()
// CIR:   cir.call @__cxx_global_var_init.3() : () -> ()
// CIR:   cir.call @__cxx_global_var_init.4() : () -> ()

// LLVM: define void @_GLOBAL__sub_I_[[FILENAME]]()
// LLVM:   call void @__cxx_global_var_init()
// LLVM:   call void @__cxx_global_var_init.1()
// LLVM:   call void @__cxx_global_var_init.2()
// LLVM:   call void @__cxx_global_var_init.3()
// LLVM:   call void @__cxx_global_var_init.4()

// OGCG: define internal void @_GLOBAL__sub_I_[[FILENAME]]() {{.*}} section ".text.startup" {
// OGCG:   call void @__cxx_global_var_init()
// OGCG:   call void @__cxx_global_var_init.1()
// OGCG:   call void @__cxx_global_var_init.2()
// OGCG:   call void @__cxx_global_var_init.3()
// OGCG:   call void @__cxx_global_var_init.4()
