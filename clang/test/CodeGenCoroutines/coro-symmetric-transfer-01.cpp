// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -O0 -emit-llvm %s -o - -disable-llvm-passes | FileCheck %s
// RUN: %clang -std=c++20 -O0 -emit-llvm -c  %s -o %t -Xclang -disable-llvm-passes && %clang -c %t

#include "Inputs/coroutine.h"

struct detached_task {
  struct promise_type {
    detached_task get_return_object() noexcept {
      return detached_task{std::coroutine_handle<promise_type>::from_promise(*this)};
    }

    void return_void() noexcept {}

    struct final_awaiter {
      bool await_ready() noexcept { return false; }
      std::coroutine_handle<> await_suspend(std::coroutine_handle<promise_type> h) noexcept {
        h.destroy();
        return {};
      }
      void await_resume() noexcept {}
    };

    void unhandled_exception() noexcept {}

    final_awaiter final_suspend() noexcept { return {}; }

    std::suspend_always initial_suspend() noexcept { return {}; }
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

  std::coroutine_handle<promise_type> coro_;
};

detached_task foo() {
  co_return;
}

// check that the lifetime of the coroutine handle used to obtain the address is contained within single basic block, and hence does not live across suspension points.
// CHECK-LABEL: final.suspend:
// CHECK:         %{{.+}} = call token @llvm.coro.save(i8* null)
// CHECK:         %[[HDL_CAST1:.+]] = bitcast %"struct.std::coroutine_handle.0"* %[[HDL:.+]] to i8*
// CHECK:         call void @llvm.lifetime.start.p0i8(i64 8, i8* %[[HDL_CAST1]])
// CHECK:         %[[CALL:.+]] = call i8* @_ZN13detached_task12promise_type13final_awaiter13await_suspendESt16coroutine_handleIS0_E(
// CHECK:         %[[HDL_CAST2:.+]] = getelementptr inbounds %"struct.std::coroutine_handle.0", %"struct.std::coroutine_handle.0"* %[[HDL]], i32 0, i32 0
// CHECK:         store i8* %[[CALL]], i8** %[[HDL_CAST2]], align 8
// CHECK:         %[[HDL_TRANSFER:.+]] = call noundef i8* @_ZNKSt16coroutine_handleIvE7addressEv(%"struct.std::coroutine_handle.0"* noundef {{.*}}%[[HDL]])
// CHECK:         %[[HDL_CAST3:.+]] = bitcast %"struct.std::coroutine_handle.0"* %[[HDL]] to i8*
// CHECK:         call void @llvm.lifetime.end.p0i8(i64 8, i8* %[[HDL_CAST3]])
// CHECK:         call void @llvm.coro.resume(i8* %[[HDL_TRANSFER]])
