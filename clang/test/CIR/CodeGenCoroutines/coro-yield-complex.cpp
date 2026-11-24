// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-llvm -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s -check-prefix=LLVM
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes %s -o %t.og.ll
// RUN: FileCheck --input-file=%t.og.ll %s -check-prefix=OGCG

#include "Inputs/coroutine.h"

struct ComplexYieldAwaiter {
  bool await_ready() { return false; }
  void await_suspend(std::coroutine_handle<>) {}
  _Complex double await_resume() { return {}; }
};

struct Task {
  struct promise_type {
    Task get_return_object() { return {}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    ComplexYieldAwaiter yield_value(int) { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};

// CIR-LABEL: cir.func coroutine {{.*}} @_Z18test_complex_yieldv
// CIR: %[[VALUE:.*]] = cir.alloca "value" align(8) init : !cir.ptr<!cir.complex<!cir.double>>
// CIR: %[[RESUME_ADDR:.*]] = cir.alloca "__coawait_resume_rval" align(8) : !cir.ptr<!cir.complex<!cir.double>>
// CIR: cir.await(yield, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR:   %[[ABI_RESULT:.*]] = cir.call @_ZN19ComplexYieldAwaiter12await_resumeEv
// CIR:   cir.store %[[ABI_RESULT]], %[[COERCE_ADDR:[0-9]+]] : !rec_anon_struct, !cir.ptr<!rec_anon_struct>
// CIR:   %[[COERCE_CAST:.*]] = cir.cast bitcast %[[COERCE_ADDR]] : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.complex<!cir.double>>
// CIR:   %[[RESULT:.*]] = cir.load %[[COERCE_CAST]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CIR:   cir.store %[[RESULT]], %[[RESUME_ADDR]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
// CIR:   cir.yield
// CIR: },)
// CIR: %[[RESUMED:.*]] = cir.load %[[RESUME_ADDR]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CIR: cir.store align(8) %[[RESUMED]], %[[VALUE]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>

// LLVM-LABEL: define dso_local void @_Z18test_complex_yieldv()
// LLVM: %[[COERCE_ADDR:[0-9]+]] = alloca { double, double }, align 8
// LLVM: %[[VALUE:.*]] = alloca { double, double }, align 8
// LLVM: %[[RESUME_ADDR:.*]] = alloca { double, double }, align 8
// LLVM: %[[RESULT:.*]] = call noundef { double, double } @_ZN19ComplexYieldAwaiter12await_resumeEv
// LLVM: store { double, double } %[[RESULT]], ptr %[[COERCE_ADDR]], align 8
// LLVM: %[[COERCED:.*]] = load { double, double }, ptr %[[COERCE_ADDR]], align 8
// LLVM: store { double, double } %[[COERCED]], ptr %[[RESUME_ADDR]], align 8
// LLVM: %[[RESUMED:.*]] = load { double, double }, ptr %[[RESUME_ADDR]], align 8
// LLVM: store { double, double } %[[RESUMED]], ptr %[[VALUE]], align 8

// OGCG-LABEL: define dso_local void @_Z18test_complex_yieldv()
// OGCG: %[[OG_VALUE:.*]] = alloca { double, double }, align 8
// OGCG: %[[OG_RESULT:.*]] = call noundef { double, double } @_ZN19ComplexYieldAwaiter12await_resumeEv
// OGCG: %[[OG_REAL:.*]] = extractvalue { double, double } %[[OG_RESULT]], 0
// OGCG: store double %[[OG_REAL]], ptr %[[OG_REAL_TMP:.*]], align 8
// OGCG: %[[OG_IMAG:.*]] = extractvalue { double, double } %[[OG_RESULT]], 1
// OGCG: store double %[[OG_IMAG]], ptr %[[OG_IMAG_TMP:.*]], align 8
// OGCG: %[[OG_RESUMED_REAL:.*]] = load double, ptr %[[OG_REAL_TMP]], align 8
// OGCG: %[[OG_RESUMED_IMAG:.*]] = load double, ptr %[[OG_IMAG_TMP]], align 8
// OGCG: %[[OG_VALUE_REAL:.*]] = getelementptr {{.*}}, ptr %[[OG_VALUE]], i32 0, i32 0
// OGCG: %[[OG_VALUE_IMAG:.*]] = getelementptr {{.*}}, ptr %[[OG_VALUE]], i32 0, i32 1
// OGCG: store double %[[OG_RESUMED_REAL]], ptr %[[OG_VALUE_REAL]], align 8
// OGCG: store double %[[OG_RESUMED_IMAG]], ptr %[[OG_VALUE_IMAG]], align 8
Task test_complex_yield() {
  _Complex double value = co_yield 0;
}

// CIR-LABEL: cir.func coroutine {{.*}} @_Z23discarded_complex_yieldv
// CIR: %[[DISCARD_RESUME_ADDR:.*]] = cir.alloca "__coawait_resume_rval" align(8) : !cir.ptr<!cir.complex<!cir.double>>
// CIR: cir.await(yield, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR:   %[[DISCARD_ABI_RESULT:.*]] = cir.call @_ZN19ComplexYieldAwaiter12await_resumeEv
// CIR:   cir.store %[[DISCARD_ABI_RESULT]], %[[DISCARD_COERCE_ADDR:[0-9]+]] : !rec_anon_struct, !cir.ptr<!rec_anon_struct>
// CIR:   %[[DISCARD_COERCE_CAST:.*]] = cir.cast bitcast %[[DISCARD_COERCE_ADDR]] : !cir.ptr<!rec_anon_struct> -> !cir.ptr<!cir.complex<!cir.double>>
// CIR:   %[[DISCARD_RESULT:.*]] = cir.load %[[DISCARD_COERCE_CAST]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>
// CIR:   cir.store %[[DISCARD_RESULT]], %[[DISCARD_RESUME_ADDR]] : !cir.complex<!cir.double>, !cir.ptr<!cir.complex<!cir.double>>
// CIR:   cir.yield
// CIR: },)
// CIR: cir.load %[[DISCARD_RESUME_ADDR]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.complex<!cir.double>

// LLVM-LABEL: define dso_local void @_Z23discarded_complex_yieldv()
// LLVM: %[[DISCARD_COERCE_ADDR:[0-9]+]] = alloca { double, double }, align 8
// LLVM: %[[DISCARD_RESUME_ADDR:[0-9]+]] = alloca { double, double }, align 8
// LLVM: %[[DISCARD_RESULT:.*]] = call noundef { double, double } @_ZN19ComplexYieldAwaiter12await_resumeEv
// LLVM: store { double, double } %[[DISCARD_RESULT]], ptr %[[DISCARD_COERCE_ADDR]], align 8
// LLVM: %[[DISCARD_COERCED:.*]] = load { double, double }, ptr %[[DISCARD_COERCE_ADDR]], align 8
// LLVM: store { double, double } %[[DISCARD_COERCED]], ptr %[[DISCARD_RESUME_ADDR]], align 8
// LLVM: load { double, double }, ptr %[[DISCARD_RESUME_ADDR]], align 8

// OGCG-LABEL: define dso_local void @_Z23discarded_complex_yieldv()
// OGCG: %[[OG_DISCARD_RESULT:.*]] = call noundef { double, double } @_ZN19ComplexYieldAwaiter12await_resumeEv
// OGCG: extractvalue { double, double } %[[OG_DISCARD_RESULT]], 0
// OGCG: extractvalue { double, double } %[[OG_DISCARD_RESULT]], 1
Task discarded_complex_yield() {
  co_yield 0;
}

// CIR-LABEL: cir.func coroutine {{.*}} @_Z9int_yieldv
// CIR: %[[INT_COERCE_ADDR:.*]] = cir.alloca "coerce" align(8) : !cir.ptr<!u64i>
// CIR: %[[INT_VALUE:.*]] = cir.alloca "value" align(4) init : !cir.ptr<!cir.complex<!s32i>>
// CIR: %[[INT_RESUME_ADDR:.*]] = cir.alloca "__coawait_resume_rval" align(4) : !cir.ptr<!cir.complex<!s32i>>
// CIR: cir.await(yield, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR:   %[[INT_ABI_RESULT:.*]] = cir.call @_ZN26IntegerComplexYieldAwaiter12await_resumeEv
// CIR:   cir.store %[[INT_ABI_RESULT]], %[[INT_COERCE_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR:   %[[INT_COERCE_CAST:.*]] = cir.cast bitcast %[[INT_COERCE_ADDR]] : !cir.ptr<!u64i> -> !cir.ptr<!cir.complex<!s32i>>
// CIR:   %[[INT_RESULT:.*]] = cir.load %[[INT_COERCE_CAST]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR:   cir.store %[[INT_RESULT]], %[[INT_RESUME_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
// CIR:   cir.yield
// CIR: },)
// CIR: %[[INT_RESUMED:.*]] = cir.load %[[INT_RESUME_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR: cir.store align(4) %[[INT_RESUMED]], %[[INT_VALUE]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>

// LLVM-LABEL: define dso_local void @_Z9int_yieldv()
// LLVM: %[[INT_COERCE_ADDR:.*]] = alloca i64, align 8
// LLVM: %[[INT_VALUE:.*]] = alloca { i32, i32 }, align 4
// LLVM: %[[INT_RESUME_ADDR:.*]] = alloca { i32, i32 }, align 4
// LLVM: %[[INT_ABI_RESULT:.*]] = call noundef i64 @_ZN26IntegerComplexYieldAwaiter12await_resumeEv
// LLVM: store i64 %[[INT_ABI_RESULT]], ptr %[[INT_COERCE_ADDR]], align 8
// LLVM: %[[INT_RESULT:.*]] = load { i32, i32 }, ptr %[[INT_COERCE_ADDR]], align 4
// LLVM: store { i32, i32 } %[[INT_RESULT]], ptr %[[INT_RESUME_ADDR]], align 4
// LLVM: %[[INT_RESUMED:.*]] = load { i32, i32 }, ptr %[[INT_RESUME_ADDR]], align 4
// LLVM: store { i32, i32 } %[[INT_RESUMED]], ptr %[[INT_VALUE]], align 4

// OGCG-LABEL: define dso_local void @_Z9int_yieldv()
// OGCG: %[[OG_INT_VALUE:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[OG_INT_RESULT:.*]] = call noundef i64 @_ZN26IntegerComplexYieldAwaiter12await_resumeEv
// OGCG: store i64 %[[OG_INT_RESULT]], ptr %[[OG_INT_COERCE:.*]], align 4
// OGCG: %[[OG_INT_COERCE_REAL:.*]] = getelementptr {{.*}}, ptr %[[OG_INT_COERCE]], i32 0, i32 0
// OGCG: %[[OG_INT_REAL:.*]] = load i32, ptr %[[OG_INT_COERCE_REAL]], align 4
// OGCG: store i32 %[[OG_INT_REAL]], ptr %[[OG_INT_REAL_TMP:.*]], align 4
// OGCG: %[[OG_INT_COERCE_IMAG:.*]] = getelementptr {{.*}}, ptr %[[OG_INT_COERCE]], i32 0, i32 1
// OGCG: %[[OG_INT_IMAG:.*]] = load i32, ptr %[[OG_INT_COERCE_IMAG]], align 4
// OGCG: store i32 %[[OG_INT_IMAG]], ptr %[[OG_INT_IMAG_TMP:.*]], align 4
// OGCG: %[[OG_INT_RESUMED_REAL:.*]] = load i32, ptr %[[OG_INT_REAL_TMP]], align 4
// OGCG: %[[OG_INT_RESUMED_IMAG:.*]] = load i32, ptr %[[OG_INT_IMAG_TMP]], align 4
// OGCG: %[[OG_INT_VALUE_REAL:.*]] = getelementptr {{.*}}, ptr %[[OG_INT_VALUE]], i32 0, i32 0
// OGCG: %[[OG_INT_VALUE_IMAG:.*]] = getelementptr {{.*}}, ptr %[[OG_INT_VALUE]], i32 0, i32 1
// OGCG: store i32 %[[OG_INT_RESUMED_REAL]], ptr %[[OG_INT_VALUE_REAL]], align 4
// OGCG: store i32 %[[OG_INT_RESUMED_IMAG]], ptr %[[OG_INT_VALUE_IMAG]], align 4
struct IntegerComplexYieldAwaiter {
  bool await_ready() { return false; }
  void await_suspend(std::coroutine_handle<>) {}
  _Complex int await_resume() { return {}; }
};

struct IntegerTask {
  struct promise_type {
    IntegerTask get_return_object() { return {}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    IntegerComplexYieldAwaiter yield_value(int) { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};

IntegerTask int_yield() {
  _Complex int value = co_yield 0;
}

// CIR-LABEL: cir.func coroutine {{.*}} @_Z9ref_yieldv
// CIR: %[[REF_VALUE:.*]] = cir.alloca "value" align(8) init const : !cir.ptr<!cir.ptr<!cir.complex<!cir.double>>>
// CIR: %[[REF_RESUME_ADDR:.*]] = cir.alloca "__coyield_resume_lval" align(8) : !cir.ptr<!cir.ptr<!cir.complex<!cir.double>>>
// CIR: cir.await(yield, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR:   %[[REF_RESULT:.*]] = cir.call @_ZN28ComplexReferenceYieldAwaiter12await_resumeEv
// CIR:   cir.store %[[REF_RESULT]], %[[REF_RESUME_ADDR]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.ptr<!cir.ptr<!cir.complex<!cir.double>>>
// CIR:   cir.yield
// CIR: },)
// CIR: %[[REF_RESUMED:.*]] = cir.load %[[REF_RESUME_ADDR]] : !cir.ptr<!cir.ptr<!cir.complex<!cir.double>>>, !cir.ptr<!cir.complex<!cir.double>>
// CIR: cir.store align(8) %[[REF_RESUMED]], %[[REF_VALUE]] : !cir.ptr<!cir.complex<!cir.double>>, !cir.ptr<!cir.ptr<!cir.complex<!cir.double>>>

// LLVM-LABEL: define dso_local void @_Z9ref_yieldv()
// LLVM: %[[REF_RESULT:.*]] = call noundef nonnull align 8 dereferenceable(16) ptr @_ZN28ComplexReferenceYieldAwaiter12await_resumeEv
// LLVM: store ptr %[[REF_RESULT]], ptr %[[REF_RESUME_ADDR:.*]], align 8
// LLVM: %[[REF_RESUMED:.*]] = load ptr, ptr %[[REF_RESUME_ADDR]], align 8
// LLVM: store ptr %[[REF_RESUMED]], ptr {{.*}}, align 8

// OGCG-LABEL: define dso_local void @_Z9ref_yieldv()
// OGCG: %[[OG_REF_VALUE:.*]] = alloca ptr, align 8
// OGCG: %[[OG_REF_RESULT:.*]] = call noundef nonnull align 8 dereferenceable(16) ptr @_ZN28ComplexReferenceYieldAwaiter12await_resumeEv
// OGCG: store ptr %[[OG_REF_RESULT]], ptr %[[OG_REF_RESUME_ADDR:.*]], align 8
// OGCG: %[[OG_REF_RESUMED:.*]] = load ptr, ptr %[[OG_REF_RESUME_ADDR]], align 8
// OGCG: store ptr %[[OG_REF_RESUMED]], ptr %[[OG_REF_VALUE]], align 8
_Complex double complex_storage;

struct ComplexReferenceYieldAwaiter {
  bool await_ready() { return false; }
  void await_suspend(std::coroutine_handle<>) {}
  _Complex double &await_resume() { return complex_storage; }
};

struct ReferenceTask {
  struct promise_type {
    ReferenceTask get_return_object() { return {}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    ComplexReferenceYieldAwaiter yield_value(int) { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};

ReferenceTask ref_yield() {
  _Complex double &value = co_yield 0;
}
