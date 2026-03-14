// RUN: %clang_cc1 -std=c++20 %s -fcxx-exceptions -fsyntax-only -Wexceptions -verify -fdeclspec

#include "Inputs/std-coroutine.h"

// expected-no-diagnostics

template <typename T>
struct promise;

template <typename T>
struct task {
    using promise_type = promise<T>;

    explicit task(promise_type& p) { throw 1; p.return_val = this; }

    T value;
};

template <typename T>
struct promise {
    task<T> get_return_object() { return task{*this}; }

    std::suspend_never initial_suspend() const noexcept { return {}; }

    std::suspend_never final_suspend() const noexcept { return {}; }

    template <typename U>
    void return_value(U&& val) { return_val->value = static_cast<U&&>(val); }

    void unhandled_exception() { throw 1; }

    task<T>* return_val;
};

task<int> a_ShouldNotDiag(const int a, const int b) {
  if (b == 0)
    throw b;

  co_return a / b;
}

task<int> b_ShouldNotDiag(const int a, const int b) noexcept {
  if (b == 0)
    throw b;

  co_return a / b;
}

const auto c_ShouldNotDiag = [](const int a, const int b) -> task<int> {
  if (b == 0)
    throw b;

  co_return a / b;
};

const auto d_ShouldNotDiag = [](const int a, const int b) noexcept -> task<int> {
  if (b == 0)
    throw b;

  co_return a / b;
};
