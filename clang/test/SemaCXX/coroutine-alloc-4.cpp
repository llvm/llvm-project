// Tests that we'll find aligned allocation function properly.
// RUN: %clang_cc1 %s -std=c++20 %s -fsyntax-only -verify -fcoro-aligned-allocation

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
    void *operator new(std::size_t); // expected-warning 1+{{under -fcoro-aligned-allocation, the non-aligned allocation function for the promise type 'f' has higher precedence than the global aligned allocation function}}
  };
};

task f() {
    co_return 43;
}

struct task2 {
  struct promise_type {
    auto initial_suspend() { return std::suspend_always{}; }
    auto final_suspend() noexcept { return std::suspend_always{}; }
    auto get_return_object() { return task2{}; }
    void unhandled_exception() {}
    void return_value(int) {}
    void *operator new(std::size_t, std::align_val_t);
  };
};

// no diagnostic expected
task2 f1() {
    co_return 43;
}

struct task3 {
  struct promise_type {
    auto initial_suspend() { return std::suspend_always{}; }
    auto final_suspend() noexcept { return std::suspend_always{}; }
    auto get_return_object() { return task3{}; }
    void unhandled_exception() {}
    void return_value(int) {}
    void *operator new(std::size_t, std::align_val_t) noexcept;
    void *operator new(std::size_t) noexcept;
    static auto get_return_object_on_allocation_failure() { return task3{}; }
  };
};

// no diagnostic expected
task3 f2() {
    co_return 43;
}

struct task4 {
  struct promise_type {
    auto initial_suspend() { return std::suspend_always{}; }
    auto final_suspend() noexcept { return std::suspend_always{}; }
    auto get_return_object() { return task4{}; }
    void unhandled_exception() {}
    void return_value(int) {}
    void *operator new(std::size_t, std::align_val_t, int, double, int) noexcept;
  };
};

// no diagnostic expected
task4 f3(int, double, int) {
    co_return 43;
}

struct task5 {
  struct promise_type {
    auto initial_suspend() { return std::suspend_always{}; }
    auto final_suspend() noexcept { return std::suspend_always{}; }
    auto get_return_object() { return task5{}; }
    void unhandled_exception() {}
    void return_value(int) {}
  };
};

// no diagnostic expected.
// The aligned allocation will be declared by the compiler.
task5 f4() {
    co_return 43;
}

namespace std {
  struct nothrow_t {};
  constexpr nothrow_t nothrow = {};
}

struct task6 {
  struct promise_type {
    auto initial_suspend() { return std::suspend_always{}; }
    auto final_suspend() noexcept { return std::suspend_always{}; }
    auto get_return_object() { return task6{}; }
    void unhandled_exception() {}
    void return_value(int) {}
    static task6 get_return_object_on_allocation_failure() { return task6{}; }
  };
};

task6 f5() { // expected-error 1+{{unable to find '::operator new(size_t, align_val_t, nothrow_t)' for 'f5'}}
    co_return 43;
}

void *operator new(std::size_t, std::align_val_t, std::nothrow_t) noexcept; 

task6 f6() {
    co_return 43;
}
