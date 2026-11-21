// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -fcxx-exceptions -fexceptions -fclangir \
// RUN:   -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR
// RUN: cir-opt --cir-flatten-cfg %t.cir -o %t.flat.cir
// RUN: FileCheck --input-file=%t.flat.cir %s --check-prefix=CIR-FLAT \
// RUN:   --implicit-check-not=cir.coroutine --implicit-check-not=cir.await \
// RUN:   --implicit-check-not=cir.co_return --implicit-check-not=cir.coro.exit \
// RUN:   --implicit-check-not=cir.coro.suspend_point
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -fcxx-exceptions -fexceptions -fclangir -emit-llvm \
// RUN:   -disable-llvm-passes %s -o %t.ll
// RUN: FileCheck --input-file=%t.ll %s --check-prefix=LLVM

#include "Inputs/coroutine.h"

struct Task {
  struct promise_type {
    Task get_return_object() noexcept { return {}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() noexcept {}
  };
};

struct ThrowingVoidAwaiter {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<>);
  void await_resume() noexcept;
};

struct ThrowingBoolAwaiter {
  bool await_ready() noexcept;
  bool await_suspend(std::coroutine_handle<>);
  void await_resume() noexcept;
};

struct HandleAwaiter {
  bool await_ready() noexcept;
  std::coroutine_handle<>
  await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

struct Cleanup {
  ~Cleanup() noexcept;
};

struct DefaultArgument {
  DefaultArgument() noexcept;
  ~DefaultArgument() noexcept;
};

struct DefaultArgumentAwaiter {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<>, DefaultArgument = {});
  void await_resume() noexcept;
};

// CIR-LABEL: cir.func coroutine {{.*}} @_Z10await_voidv()
// CIR:         }, body : {
// CIR:           cir.try {
// CIR:             cir.await(user, ready : {
// CIR:             }, suspend : {
// CIR:               cir.call @_ZN19ThrowingVoidAwaiter13await_suspendESt16coroutine_handleIvE
// CIR-NEXT:          cir.coro.suspend_point
// CIR:             } catch all
// CIR:               cir.call @_ZN4Task12promise_type19unhandled_exceptionEv

// The throwing await-suspend intrinsic must keep the coroutine body's
// exception handler reachable after flattening.
// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z10await_voidv()
// CIR-FLAT:         %[[FLAT_VOID_WRAPPER:.*]] = cir.get_global @_Z10await_voidv.__await_suspend_wrapper__await
// CIR-FLAT-NEXT:    cir.try_call_llvm_intrinsic "coro.await.suspend.void" {{.*}}%[[FLAT_VOID_WRAPPER]] ^[[VOID_CONT:bb[0-9]+]], ^[[VOID_UNWIND:bb[0-9]+]]
// CIR-FLAT:       ^[[VOID_CONT]]:
// CIR-FLAT:       ^[[VOID_UNWIND]]:
// CIR-FLAT-NEXT:    %{{.*}} = cir.eh.initiate
// CIR-FLAT:         cir.eh.dispatch
// CIR-FLAT:         cir.begin_catch
// CIR-FLAT:         cir.call @_ZN4Task12promise_type19unhandled_exceptionEv

// LLVM-LABEL: define {{.*}} @_Z10await_voidv()
// LLVM:         invoke void @llvm.coro.await.suspend.void({{.*}}@_Z10await_voidv.__await_suspend_wrapper__await)
// LLVM-NEXT:            to label %[[VOID_CONT:[a-zA-Z0-9._]+]] unwind label %[[VOID_LPAD:[a-zA-Z0-9._]+]]
// LLVM:       [[VOID_LPAD]]:
// LLVM-NEXT:    landingpad { ptr, i32 }
// LLVM:         call void @_ZN4Task12promise_type19unhandled_exceptionEv

Task await_void() {
  co_await ThrowingVoidAwaiter{};
}

// CIR-LABEL: cir.func coroutine {{.*}} @_Z10await_boolv()
// CIR:         }, body : {
// CIR:           cir.try {
// CIR:             cir.await(user, ready : {
// CIR:             }, suspend : {
// CIR:               %[[SUSPEND_RET:.*]] = cir.call @_ZN19ThrowingBoolAwaiter13await_suspendESt16coroutine_handleIvE
// CIR-NEXT:          cir.coro.suspend_point(%[[SUSPEND_RET]] : !cir.bool)
// CIR:             } catch all
// CIR:               cir.call @_ZN4Task12promise_type19unhandled_exceptionEv

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z10await_boolv()
// CIR-FLAT:         %[[FLAT_BOOL_WRAPPER:.*]] = cir.get_global @_Z10await_boolv.__await_suspend_wrapper__await
// CIR-FLAT-NEXT:    %[[FLAT_SUSPEND_RET:.*]] = cir.try_call_llvm_intrinsic "coro.await.suspend.bool" {{.*}}%[[FLAT_BOOL_WRAPPER]] ^[[BOOL_CONT:bb[0-9]+]], ^[[BOOL_UNWIND:bb[0-9]+]]
// CIR-FLAT:       ^[[BOOL_CONT]]:
// CIR-FLAT:         cir.brcond %[[FLAT_SUSPEND_RET]]
// CIR-FLAT:       ^[[BOOL_UNWIND]]:
// CIR-FLAT-NEXT:    %{{.*}} = cir.eh.initiate
// CIR-FLAT:         cir.eh.dispatch
// CIR-FLAT:         cir.begin_catch
// CIR-FLAT:         cir.call @_ZN4Task12promise_type19unhandled_exceptionEv

// LLVM-LABEL: define {{.*}} @_Z10await_boolv()
// LLVM:         %[[LLVM_SUSPEND_RET:.*]] = invoke i1 @llvm.coro.await.suspend.bool({{.*}}@_Z10await_boolv.__await_suspend_wrapper__await)
// LLVM-NEXT:            to label %[[BOOL_CONT:[a-zA-Z0-9._]+]] unwind label %[[BOOL_LPAD:[a-zA-Z0-9._]+]]
// LLVM:       [[BOOL_CONT]]:
// LLVM:         br i1 %[[LLVM_SUSPEND_RET]]
// LLVM:       [[BOOL_LPAD]]:
// LLVM-NEXT:    landingpad { ptr, i32 }
// LLVM:         call void @_ZN4Task12promise_type19unhandled_exceptionEv

Task await_bool() {
  co_await ThrowingBoolAwaiter{};
}

// A handle-returning await_suspend can unwind while resuming the returned
// coroutine, even when the await_suspend function itself is noexcept.
// CIR-LABEL: cir.func coroutine {{.*}} @_Z12await_handlev()
// CIR:         }, body : {
// CIR:           cir.try {
// CIR:             cir.await(user, ready : {
// CIR:             }, suspend : {
// CIR:               cir.call @_ZN13HandleAwaiter13await_suspendESt16coroutine_handleIvE
// CIR:               %[[HANDLE_RET:.*]] = cir.call @_ZNKSt16coroutine_handleIvE7addressEv
// CIR-NEXT:          cir.coro.suspend_point(%[[HANDLE_RET]] : !cir.ptr<!void>)
// CIR:             } catch all
// CIR:               cir.call @_ZN4Task12promise_type19unhandled_exceptionEv

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z12await_handlev()
// CIR-FLAT:         %[[FLAT_HANDLE_WRAPPER:.*]] = cir.get_global @_Z12await_handlev.__await_suspend_wrapper__await
// CIR-FLAT-NEXT:    cir.try_call_llvm_intrinsic "coro.await.suspend.handle" {{.*}}%[[FLAT_HANDLE_WRAPPER]] ^[[HANDLE_CONT:bb[0-9]+]], ^[[HANDLE_UNWIND:bb[0-9]+]]
// CIR-FLAT:       ^[[HANDLE_CONT]]:
// CIR-FLAT:       ^[[HANDLE_UNWIND]]:
// CIR-FLAT-NEXT:    %{{.*}} = cir.eh.initiate
// CIR-FLAT:         cir.eh.dispatch
// CIR-FLAT:         cir.begin_catch
// CIR-FLAT:         cir.call @_ZN4Task12promise_type19unhandled_exceptionEv

// LLVM-LABEL: define {{.*}} @_Z12await_handlev()
// LLVM:         invoke void @llvm.coro.await.suspend.handle({{.*}}@_Z12await_handlev.__await_suspend_wrapper__await)
// LLVM-NEXT:            to label %[[HANDLE_CONT:[a-zA-Z0-9._]+]] unwind label %[[HANDLE_LPAD:[a-zA-Z0-9._]+]]
// LLVM:       [[HANDLE_LPAD]]:
// LLVM-NEXT:    landingpad { ptr, i32 }
// LLVM:         call void @_ZN4Task12promise_type19unhandled_exceptionEv

Task await_handle() {
  co_await HandleAwaiter{};
}

// LLVM-LABEL: define {{.*}} @_Z23await_void_with_cleanupv()
// LLVM:         invoke void @llvm.coro.await.suspend.void({{.*}}@_Z23await_void_with_cleanupv.__await_suspend_wrapper__await)
// LLVM-NEXT:            to label %{{.*}} unwind label %[[CLEANUP_LPAD:[a-zA-Z0-9._]+]]
// LLVM:       [[CLEANUP_LPAD]]:
// LLVM-NEXT:    landingpad { ptr, i32 }
// LLVM:         call void @_ZN7CleanupD1Ev
// LLVM:         call void @_ZN4Task12promise_type19unhandled_exceptionEv

Task await_void_with_cleanup() {
  Cleanup cleanup;
  co_await ThrowingVoidAwaiter{};
}

Task await_default_argument() {
  co_await DefaultArgumentAwaiter{};
}

// The default argument is created in the wrapper and destroyed on both paths.
// CIR-FLAT-LABEL: cir.func always_inline internal private @_Z22await_default_argumentv.__await_suspend_wrapper__await(
// CIR-FLAT:         %[[DEFAULT_ARG:.*]] = cir.alloca "agg.tmp{{.*}}" align(1) : !cir.ptr<!rec_DefaultArgument>
// CIR-FLAT:         cir.call @_ZN15DefaultArgumentC1Ev(%[[DEFAULT_ARG]])
// CIR-FLAT:         cir.try_call @_ZN22DefaultArgumentAwaiter13await_suspendESt16coroutine_handleIvE15DefaultArgument({{.*}}, %[[DEFAULT_ARG]]) ^[[NORMAL:bb[0-9]+]], ^[[UNWIND:bb[0-9]+]]
// CIR-FLAT:       ^[[NORMAL]]:
// CIR-FLAT:         cir.call @_ZN15DefaultArgumentD1Ev(%[[DEFAULT_ARG]])
// CIR-FLAT:       ^[[UNWIND]]:
// CIR-FLAT-NEXT:    %[[INIT_TOKEN:.*]] = cir.eh.initiate cleanup
// CIR-FLAT-NEXT:    cir.br ^[[CLEANUP:bb[0-9]+]](%[[INIT_TOKEN]] : !cir.eh_token)
// CIR-FLAT:       ^[[CLEANUP]](%[[EH_TOKEN:.*]]: !cir.eh_token):
// CIR-FLAT-NEXT:    %[[CLEANUP_TOKEN:.*]] = cir.begin_cleanup %[[EH_TOKEN]]
// CIR-FLAT-NEXT:    cir.call @_ZN15DefaultArgumentD1Ev(%[[DEFAULT_ARG]])
// CIR-FLAT-NEXT:    cir.end_cleanup %[[CLEANUP_TOKEN]]
// CIR-FLAT-NEXT:    cir.resume %[[EH_TOKEN]]

// LLVM-LABEL: define internal void @_Z22await_default_argumentv.__await_suspend_wrapper__await(
// LLVM:         %[[DEFAULT_ARG:[0-9]+]] = alloca %struct.DefaultArgument
// LLVM:         call void @_ZN15DefaultArgumentC1Ev({{.*}}%[[DEFAULT_ARG]])
// LLVM:         invoke void @_ZN22DefaultArgumentAwaiter13await_suspendESt16coroutine_handleIvE15DefaultArgument({{.*}}%[[DEFAULT_ARG]])
// LLVM-NEXT:            to label %[[NORMAL:[0-9]+]] unwind label %[[UNWIND:[0-9]+]]
// LLVM:       [[NORMAL]]:
// LLVM:         call void @_ZN15DefaultArgumentD1Ev({{.*}}%[[DEFAULT_ARG]])
// LLVM:       [[UNWIND]]:
// LLVM-NEXT:    %{{.*}} = landingpad
// LLVM-NEXT:            cleanup
// LLVM:         call void @_ZN15DefaultArgumentD1Ev({{.*}}%[[DEFAULT_ARG]])
// LLVM:         resume
