// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -std=c++20 -fsyntax-only -verify -Wunreachable-code

#include "Inputs/std-coroutine.h"

extern void abort (void) __attribute__ ((__noreturn__));

struct task {
  struct promise_type {
    std::suspend_always initial_suspend();
    std::suspend_always final_suspend() noexcept;
    void return_void();
    std::suspend_always yield_value(int) { return {}; }
      task get_return_object();
    void unhandled_exception();
  };
};

task test1() {
  abort();
  co_yield 1;
}

task test2() {
  abort();
  1; // expected-warning {{code will never be executed}}
  co_yield 1;
}

task test3() {
  abort();
  co_return;
}

task test4() {
  abort();
  1; // expected-warning {{code will never be executed}}
  co_return;
}


task test5() {
  abort();
  co_await std::suspend_never{};
}

task test6() {
  abort();
  1; // expected-warning {{code will never be executed}}
  co_await std::suspend_never{};
}
