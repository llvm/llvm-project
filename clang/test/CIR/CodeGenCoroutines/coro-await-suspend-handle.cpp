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

// CIR-LABEL: cir.func coroutine {{.*}} @_Z12await_handlev
// CIR: cir.coro.intrinsic.await.suspend.handle
// CIR: cir.coro.suspend_point
Task await_handle() {
  co_await HandleAwaiter{};
}

// CIR-LABEL: cir.func {{.*}} @_Z12await_handlev.__await_suspend_wrapper__await
// CIR: cir.call {{.*}}await_suspend
// CIR: cir.call {{.*}}address
// CIR: cir.return
