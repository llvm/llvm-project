// This addresses https://github.com/llvm/llvm-project/issues/57339
// RUN: %clang_cc1 -triple %itanium_abi_triple -std=c++20 -fcxx-exceptions \
// RUN:     -fexceptions -S -emit-llvm -o - %s -O1 | FileCheck %s

#include "Inputs/coroutine.h"

struct gen {
  struct promise_type {
    gen get_return_object() noexcept {
      return gen{std::coroutine_handle<promise_type>::from_promise(*this)};
    }
    std::suspend_always initial_suspend() noexcept { return {}; }

    struct final_awaiter {
      ~final_awaiter() noexcept;
      bool await_ready() noexcept {
        return false;
      }
      void await_suspend(std::coroutine_handle<>) noexcept {}
      void await_resume() noexcept {}
    };

    final_awaiter final_suspend() noexcept {
      return {};
    }

    void unhandled_exception() {
      throw;
    }
    void return_void() {}
  };

  gen(std::coroutine_handle<promise_type> coro) noexcept
  : coro(coro)
  {
  }

  ~gen() noexcept {
    if (coro) {
      coro.destroy();
    }
  }

  gen(gen&& g) noexcept
  : coro(g.coro)
  {
    g.coro = {};
  }

  std::coroutine_handle<promise_type> coro;
};

struct X {};

gen maybe_throwing(bool x) {
  if (x) {
    throw X{};
  }
  co_return;
}

// CHECK: define{{.*}}@_Z14maybe_throwingb.destroy
// CHECK: %[[INDEX:.+]] = load i1, ptr %index.addr, align 1
// CHECK: br i1 %[[INDEX]], label %[[AFTERSUSPEND:.+]], label %[[CORO_FREE:.+]]
// CHECK: [[AFTERSUSPEND]]:
// CHECK: call{{.*}}_ZN3gen12promise_type13final_awaiterD1Ev(
// CHECK: [[CORO_FREE]]:
// CHECK: call{{.*}}_ZdlPv

void noexcept_call() noexcept;

gen no_throwing() {
  noexcept_call();
  co_return;
}

// CHECK: define{{.*}}@_Z11no_throwingv.resume({{.*}}%[[ARG:.+]])
// CHECK: resume:
// CHECK:   call{{.*}}@_Z13noexcept_callv()
// CHECK:   store ptr null, ptr %[[ARG]]
// CHECK:   ret void

// CHECK: define{{.*}}@_Z11no_throwingv.destroy({{.*}}%[[ARG:.+]])
// CHECK: %[[RESUME_FN_ADDR:.+]] = load ptr, ptr %[[ARG]]
// CHECK: %[[IF_NULL:.+]] = icmp eq ptr %[[RESUME_FN_ADDR]], null
// CHECK: br i1 %[[IF_NULL]], label %[[AFTERSUSPEND:.+]], label %[[CORO_FREE:.+]]
// CHECK: [[AFTERSUSPEND]]:
// CHECK: call{{.*}}_ZN3gen12promise_type13final_awaiterD1Ev(
// CHECK: [[CORO_FREE]]:
// CHECK: call{{.*}}_ZdlPv
