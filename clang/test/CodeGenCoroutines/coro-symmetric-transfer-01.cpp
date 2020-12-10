// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcoroutines-ts -std=c++14 -O1 -emit-llvm %s -o - -disable-llvm-passes | FileCheck %s

#include "Inputs/coroutine.h"

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
// CHECK:         %[[PTR1:.+]] = bitcast %"struct.std::experimental::coroutines_v1::coroutine_handle.0"* %[[ADDR_TMP:.+]] to i8*
// CHECK-NEXT:    call void @llvm.lifetime.start.p0i8(i64 8, i8* %[[PTR1]])
// CHECK:         call i8* @{{.*address.*}}(%"struct.std::experimental::coroutines_v1::coroutine_handle.0"* {{[^,]*}} %[[ADDR_TMP]])
// CHECK-NEXT:    %[[PTR2:.+]] = bitcast %"struct.std::experimental::coroutines_v1::coroutine_handle.0"* %[[ADDR_TMP]] to i8*
// CHECK-NEXT:    call void @llvm.lifetime.end.p0i8(i64 8, i8* %[[PTR2]])
