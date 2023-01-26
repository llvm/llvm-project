// Tests that the behavior will be good if there are multiple operator delete in the promise_type.
// RUN: %clang_cc1 %s -std=c++20 %s -fsyntax-only -verify
// expected-no-diagnostics

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

    void operator delete(void *ptr, void *meaningless_placeholder);
    void operator delete(void *ptr);
  };
};

task f() {
  co_return 43;
}
