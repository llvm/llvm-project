// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -std=c++20 -fsyntax-only -verify -Wunreachable-code

#include "Inputs/std-coroutine.h"

extern void abort(void) __attribute__((__noreturn__));

struct task {
  struct promise_type {
    std::suspend_always initial_suspend();
    std::suspend_always final_suspend() noexcept;
    void return_void();
    std::suspend_always yield_value(int) { return {}; }
    task get_return_object();
    void unhandled_exception();

    struct Awaiter {
      bool await_ready();
      void await_suspend(auto);
      int await_resume();
    };
    auto await_transform(const int& x) { return Awaiter{}; }
  };
};

task test1() {
  abort();
  co_yield 1;
}

task test2() {
  abort();
  1;  // expected-warning {{code will never be executed}}
  co_yield 1;
}

task test3() {
  abort();
  co_return;
}

task test4() {
  abort();
  1;  // expected-warning {{code will never be executed}}
  co_return;
}

task test5() {
  abort();
  co_await 1;
}

task test6() {
  abort();
  1;  // expected-warning {{code will never be executed}}
  co_await 3;
}

task test7() {
  // coroutine statements are not considered unreachable.
  co_await 1;
  abort();
  co_await 2;
}

task test8() {
  // coroutine statements are not considered unreachable.
  abort();
  co_return;
  1 + 1;  // expected-warning {{code will never be executed}}
}

task test9() {
  abort();
  // This warning is emitted on the declaration itself, rather the coroutine substmt.
  int x = co_await 1; // expected-warning {{code will never be executed}}
}
