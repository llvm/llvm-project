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

// From https://github.com/llvm/llvm-project/issues/60545
struct generator {
    struct promise_type {
        generator get_return_object();
        std::suspend_always initial_suspend();
        std::suspend_always final_suspend() noexcept;
        void return_void();
        [[noreturn]] void unhandled_exception();

        static void* operator new(std::size_t size);
        static void operator delete(void* ptr, const std::size_t size);
    };
};

generator goo() { co_return; }
