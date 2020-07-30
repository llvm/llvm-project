// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcoroutines-ts -std=c++14 -emit-llvm %s -o - | FileCheck %s
#include "Inputs/coroutine.h"

using namespace std::experimental;

struct coro {
  struct promise_type {
    coro get_return_object();
    suspend_never initial_suspend();
    suspend_never final_suspend() noexcept;
    void return_void();
    static void unhandled_exception();
  };
};

struct A {
  ~A();
  bool await_ready();
  int await_resume() { return 8; }
  template <typename F> void await_suspend(F);
};

extern "C" void consume(int);

// Verifies that domination is properly built during cleanup.
// Without CGCleanup.cpp fix verifier was reporting:
// Instruction does not dominate all uses!
//  %tmp.exprcleanup = alloca i32*, align 8
//  store i32* %x, i32** %tmp.exprcleanup, align 8


// CHECK-LABEL: f(
extern "C" coro f(int) {
  int x = 42;
  x = co_await A{};
  consume(x);
}
