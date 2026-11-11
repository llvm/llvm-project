// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu \
// RUN:   -fclangir -Wno-coroutine-missing-unhandled-exception \
// RUN:   -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s --check-prefix=CIR

#include "Inputs/coroutine.h"

struct Task {
  struct promise_type {
    Task get_return_object() { return {}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() {}
  };
};

struct HandleAwaiter {
  bool await_ready() noexcept;
  std::coroutine_handle<>
  await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

struct VoidAwaiter {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

struct BoolAwaiter {
  bool await_ready() noexcept;
  bool await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

// CIR-LABEL: cir.func coroutine {{.*}} @_Z13await_suspendv

// CIR: %[[HANDLE_WRAPPER:[0-9]+]] = cir.get_global @_Z13await_suspendv.__await_suspend_wrapper__await :
// CIR-NEXT: cir.coro.intrinsic.await.suspend.handle(%[[HANDLE_AWAITER:[0-9]+]], %[[FRAME:[0-9]+]], %[[HANDLE_WRAPPER]])
// CIR-NEXT: cir.coro.suspend_point

// CIR: %[[VOID_WRAPPER:[0-9]+]] = cir.get_global @_Z13await_suspendv.__await_suspend_wrapper__await2 :
// CIR-NEXT: cir.coro.intrinsic.await.suspend.void(%[[VOID_AWAITER:[0-9]+]], %[[FRAME]], %[[VOID_WRAPPER]])
// CIR-NEXT: cir.coro.suspend_point

// CIR: %[[BOOL_WRAPPER:[0-9]+]] = cir.get_global @_Z13await_suspendv.__await_suspend_wrapper__await3 :
// CIR-NEXT: %[[SUSPEND_RET:[0-9]+]] = cir.coro.intrinsic.await.suspend.bool(%[[BOOL_AWAITER:[0-9]+]], %[[FRAME]], %[[BOOL_WRAPPER]])
// CIR-NEXT: cir.coro.suspend_point(%[[SUSPEND_RET]])

Task await_suspend() {
  co_await HandleAwaiter{};
  co_await VoidAwaiter{};
  co_await BoolAwaiter{};
}

// CIR-LABEL: cir.func {{.*}} @_Z13await_suspendv.__await_suspend_wrapper__await(
// CIR: cir.call @_ZN13HandleAwaiter13await_suspendESt16coroutine_handleIvE
// CIR: cir.call @_ZNKSt16coroutine_handleIvE7addressEv
// CIR: cir.return

// CIR-LABEL: cir.func {{.*}} @_Z13await_suspendv.__await_suspend_wrapper__await2(
// CIR: cir.call @_ZN11VoidAwaiter13await_suspendESt16coroutine_handleIvE
// CIR: cir.return

// CIR-LABEL: cir.func {{.*}} @_Z13await_suspendv.__await_suspend_wrapper__await3(
// CIR: cir.call @_ZN11BoolAwaiter13await_suspendESt16coroutine_handleIvE
// CIR: cir.return
