// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++23 -fexperimental-cxx-type-aware-allocators -fcoroutines -fexceptions -Wall -Wpedantic
// expected-no-diagnostics

#include "Inputs/std-coroutine.h"

namespace std {
   template <typename T> struct type_identity {
   typedef T type;
   };
   typedef __SIZE_TYPE__ size_t;
}

struct Allocator {};

struct resumable {
  struct promise_type {
    void *operator new(std::type_identity<promise_type>, std::size_t sz, int);
    void *operator new(std::type_identity<promise_type>, std::size_t sz, float);
    void operator delete(std::type_identity<promise_type>, void *);
    template <typename T> void operator delete(std::type_identity<T>, void *) = delete;

    resumable get_return_object() { return {}; }
    auto initial_suspend() { return std::suspend_always(); }
    auto final_suspend() noexcept { return std::suspend_always(); }
    void unhandled_exception() {}
    void return_void(){};
    std::suspend_always yield_value(int i);
  };
};

struct resumable2 {
  struct promise_type {
    template <typename... Args> void *operator new(std::type_identity<promise_type>, std::size_t sz, Args...);
    void operator delete(std::type_identity<promise_type>, void *);

    resumable2 get_return_object() { return {}; }
    auto initial_suspend() { return std::suspend_always(); }
    auto final_suspend() noexcept { return std::suspend_always(); }
    void unhandled_exception() {}
    void return_void(){};
    std::suspend_always yield_value(int i);
  };
};

resumable f1(int) {
  co_return;
}

resumable f2(float) {
  co_return;
}

resumable2 f3(int, float, const char*, Allocator) {
   co_yield 1;
   co_return;
}

resumable f4(int n = 10) {
   for (int i = 0; i < n; i++) co_yield i;
}
