// Test that if the compiler will emit error message if the promise_type contain
// operator delete but none of them are available. This is required by the standard.
// RUN: %clang_cc1 %s -std=c++20 %s -fsyntax-only -verify

#include "Inputs/std-coroutine.h"

namespace std {
    typedef __SIZE_TYPE__ size_t;
    enum class align_val_t : size_t {};
}

struct task {
  struct promise_type {
    auto initial_suspend() { return std::suspend_always{}; }
    auto final_suspend() noexcept { return std::suspend_always{}; }
    auto get_return_object() { return task{}; }
    void unhandled_exception() {}
    void return_value(int) {}

    void operator delete(void *ptr, void *meaningless_placeholder); // expected-note {{member 'operator delete' declared here}}
  };
};

task f() { // expected-error 1+{{no suitable member 'operator delete' in 'promise_type'}}
  co_return 43;
}
