// RUN: %clang_cc1 -triple arm64-apple-macosx -fsyntax-only -verify %s -std=c++26 -Wno-ext-cxx-type-aware-allocators -fcoroutines -fexceptions -Wall -Wpedantic


#include "Inputs/std-coroutine.h"

namespace std {
   template <typename T> struct type_identity {
   typedef T type;
   };
   typedef __SIZE_TYPE__ size_t;
   enum class align_val_t : size_t {};
}

struct Allocator {};

struct resumable {
  struct promise_type {
    void *operator new(std::type_identity<promise_type>, std::size_t sz, std::align_val_t, int); // #resumable_tan1
    void *operator new(std::type_identity<promise_type>, std::size_t sz, std::align_val_t, float); // #resumable_tan2
    void operator delete(std::type_identity<promise_type>, void *, std::size_t sz, std::align_val_t); // #resumable_tad1
    template <typename T> void operator delete(std::type_identity<T>, void *, std::size_t sz, std::align_val_t) = delete; // #resumable_tad2

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
    template <typename... Args> void *operator new(std::type_identity<promise_type>, std::size_t sz, std::align_val_t, Args...); // #resumable2_tan1
    void operator delete(std::type_identity<promise_type>, void *, std::size_t sz, std::align_val_t); // #resumable2_tad2

    resumable2 get_return_object() { return {}; }
    auto initial_suspend() { return std::suspend_always(); }
    auto final_suspend() noexcept { return std::suspend_always(); }
    void unhandled_exception() {}
    void return_void(){};
    std::suspend_always yield_value(int i);
  };
};


struct resumable3 {
  struct promise_type {
  // expected-error@-1 {{declaration of type aware 'operator new' in 'resumable3::promise_type' must have matching type aware 'operator delete'}}
  // expected-note@#resumable3_tan {{unmatched type aware 'operator new' declared here}}
    void *operator new(std::size_t sz, float);
    void *operator new(std::type_identity<promise_type>, std::size_t sz, std::align_val_t, float); // #resumable3_tan
    void operator delete(void *);

    resumable3 get_return_object() { return {}; }
    auto initial_suspend() { return std::suspend_always(); }
    auto final_suspend() noexcept { return std::suspend_always(); }
    void unhandled_exception() {}
    void return_void(){};
    std::suspend_always yield_value(int i);
  };
};
struct resumable4 {
  struct promise_type {
    // expected-error@-1 {{declaration of type aware 'operator delete' in 'resumable4::promise_type' must have matching type aware 'operator new'}}
    // expected-note@#resumable4_tad {{unmatched type aware 'operator delete' declared here}}
    void *operator new(std::size_t sz, float);
    template <typename T> void operator delete(std::type_identity<T>, void *, std::size_t, std::align_val_t); // #resumable4_tad

    resumable4 get_return_object() { return {}; }
    auto initial_suspend() { return std::suspend_always(); }
    auto final_suspend() noexcept { return std::suspend_always(); }
    void unhandled_exception() {}
    void return_void(){};
    std::suspend_always yield_value(int i);
  };
};
struct resumable5 {
  struct promise_type {
    // expected-error@-1 {{declaration of type aware 'operator delete' in 'resumable5::promise_type' must have matching type aware 'operator new'}}
    // expected-note@#resumable5_tad {{unmatched type aware 'operator delete' declared here}}
    void *operator new(std::size_t sz, float);
    void operator delete(void *);
    template <typename T> void operator delete(std::type_identity<T>, void *, std::size_t, std::align_val_t); // #resumable5_tad

    resumable5 get_return_object() { return {}; }
    auto initial_suspend() { return std::suspend_always(); }
    auto final_suspend() noexcept { return std::suspend_always(); }
    void unhandled_exception() {}
    void return_void(){};
    std::suspend_always yield_value(int i);
  };
};

resumable f1(int) {
  // expected-error@-1 {{'operator new' provided by 'std::coroutine_traits<resumable, int>::promise_type' (aka 'typename resumable::promise_type') is not usable with the function signature of 'f1'}}
  // expected-note@-2 {{type aware 'operator new' will not be used for coroutine allocation}}
  // expected-note@#resumable_tan1 {{type aware 'operator new' declared here}}
  // expected-note@#resumable_tan2 {{type aware 'operator new' declared here}}
  co_return;
}

resumable f2(float) {
  // expected-error@-1 {{'operator new' provided by 'std::coroutine_traits<resumable, float>::promise_type' (aka 'typename resumable::promise_type') is not usable with the function signature of 'f2'}}
  // expected-note@-2 {{type aware 'operator new' will not be used for coroutine allocation}}
  // expected-note@#resumable_tan1 {{type aware 'operator new' declared here}}
  // expected-note@#resumable_tan2 {{type aware 'operator new' declared here}}
  co_return;
}

resumable2 f3(int, float, const char*, Allocator) {
  // expected-error@-1 {{'operator new' provided by 'std::coroutine_traits<resumable2, int, float, const char *, Allocator>::promise_type' (aka 'typename resumable2::promise_type') is not usable with the function signature of 'f3'}}
  // expected-note@-2 {{type aware 'operator new' will not be used for coroutine allocation}}
  // expected-note@#resumable2_tan1 {{type aware 'operator new' declared here}}
  co_yield 1;
  co_return;
}

resumable f4(int n = 10) {
  // expected-error@-1 {{'operator new' provided by 'std::coroutine_traits<resumable, int>::promise_type' (aka 'typename resumable::promise_type') is not usable with the function signature of 'f4'}}
  // expected-note@-2 {{type aware 'operator new' will not be used for coroutine allocation}}
  // expected-note@#resumable_tan1 {{type aware 'operator new' declared here}}
  // expected-note@#resumable_tan2 {{type aware 'operator new' declared here}}
  for (int i = 0; i < n; i++)
    co_yield i;
}
resumable3 f5(float) {
  // expected-warning@-1 {{type aware 'operator new' will not be used for coroutine allocation}}
  // expected-note@#resumable3_tan {{type aware 'operator new' declared here}}
  co_return;
}

resumable4 f6(float) {
  // expected-error@-1 {{no suitable member 'operator delete' in 'promise_type'}}
  // expected-warning@-2 {{type aware 'operator delete' will not be used for coroutine allocation}}
  // expected-note@#resumable4_tad {{type aware 'operator delete' declared here}}
  // expected-note@#resumable4_tad {{member 'operator delete' declared here}}
  co_return;
}

resumable5 f7(float) {
  // expected-warning@-1 {{type aware 'operator delete' will not be used for coroutine allocation}}
  // expected-note@#resumable5_tad {{type aware 'operator delete' declared here}}
  co_return;
}
