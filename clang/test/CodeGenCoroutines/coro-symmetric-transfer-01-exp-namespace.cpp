// RUN: %clang_cc1 -no-enable-noundef-analysis -triple x86_64-unknown-linux-gnu -fcoroutines-ts -std=c++14 -O0 -emit-llvm %s -o - -disable-llvm-passes | FileCheck %s
// RUN: %clang -fcoroutines-ts -std=c++14 -O0 -emit-llvm -c  %s -o %t -Xclang -disable-llvm-passes && %clang -c %t

#include "Inputs/coroutine-exp-namespace.h"

namespace coro = std::experimental::coroutines_v1;

struct detached_task {
  struct promise_type {
    detached_task get_return_object() noexcept {
      return detached_task{coro::coroutine_handle<promise_type>::from_promise(*this)};
    }

    void return_void() noexcept {}

    struct final_awaiter {
      bool await_ready() noexcept { return false; }
      coro::coroutine_handle<> await_suspend(coro::coroutine_handle<promise_type> h) noexcept {
        h.destroy();
        return {};
      }
      void await_resume() noexcept {}
    };

    void unhandled_exception() noexcept {}

    final_awaiter final_suspend() noexcept { return {}; }

    coro::suspend_always initial_suspend() noexcept { return {}; }
  };

  ~detached_task() {
    if (coro_) {
      coro_.destroy();
      coro_ = {};
    }
  }

  void start() && {
    auto tmp = coro_;
    coro_ = {};
    tmp.resume();
  }

  coro::coroutine_handle<promise_type> coro_;
};

detached_task foo() {
  co_return;
}

// check that the lifetime of the coroutine handle used to obtain the address is contained within single basic block, and hence does not live across suspension points.
// CHECK-LABEL: final.suspend:
// CHECK:         %{{.+}} = call token @llvm.coro.save(ptr null)
// CHECK:         call void @llvm.lifetime.start.p0(i64 8, ptr %[[HDL:.+]])
// CHECK:         %[[CALL:.+]] = call ptr @_ZN13detached_task12promise_type13final_awaiter13await_suspendENSt12experimental13coroutines_v116coroutine_handleIS0_EE(
// CHECK:         %[[HDL_CAST2:.+]] = getelementptr inbounds %"struct.std::experimental::coroutines_v1::coroutine_handle.0", ptr %[[HDL]], i32 0, i32 0
// CHECK:         store ptr %[[CALL]], ptr %[[HDL_CAST2]], align 8
// CHECK:         %[[HDL_TRANSFER:.+]] = call ptr @_ZNKSt12experimental13coroutines_v116coroutine_handleIvE7addressEv(ptr nonnull align 8 dereferenceable(8) %[[HDL]])
// CHECK:         call void @llvm.lifetime.end.p0(i64 8, ptr %[[HDL]])
// CHECK:         call void @llvm.coro.resume(ptr %[[HDL_TRANSFER]])
