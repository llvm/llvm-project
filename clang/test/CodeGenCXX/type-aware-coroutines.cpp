// RUN: %clang_cc1 -triple arm64-apple-macosx  %s -std=c++23 -fcoroutines -fexceptions -emit-llvm  -Wno-coro-type-aware-allocation-function -o - | FileCheck %s
// RUN: %clang_cc1 -triple arm64-apple-macosx  %s -std=c++26 -fcoroutines -fexceptions -emit-llvm  -Wno-coro-type-aware-allocation-function -o - | FileCheck %s

#include "Inputs/std-coroutine.h"

namespace std {
   template <typename T> struct type_identity {
   typedef T type;
   };
   typedef __SIZE_TYPE__ size_t;
   enum class align_val_t {};
}

struct Allocator {};

struct resumable {
  struct promise_type {
    promise_type();
    void *operator new(std::size_t sz, int);
    void *operator new(std::size_t sz, float);
    void *operator new(std::type_identity<promise_type>, std::size_t sz, std::align_val_t, int);
    void *operator new(std::type_identity<promise_type>, std::size_t sz, std::align_val_t, float);
    void operator delete(std::type_identity<promise_type>, void *, std::size_t sz, std::align_val_t);
    template <typename T> void operator delete(std::type_identity<T>, void *, std::size_t sz, std::align_val_t) = delete;
    void operator delete(void *);

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
    promise_type();
    template <typename... Args> void *operator new(std::size_t sz, Args...);
    template <typename... Args> void *operator new(std::type_identity<promise_type>, std::size_t sz, std::align_val_t, Args...);
    void operator delete(std::type_identity<promise_type>, void *, std::size_t sz, std::align_val_t);
    void operator delete(void *);

    resumable2 get_return_object() { return {}; }
    auto initial_suspend() { return std::suspend_always(); }
    auto final_suspend() noexcept { return std::suspend_always(); }
    void unhandled_exception() {}
    void return_void(){};
    std::suspend_always yield_value(int i);
  };
};

// CHECK-LABEL: void @f1
extern "C" resumable f1(int) {
  co_return;
// CHECK: coro.alloc:
// CHECK: _ZN9resumable12promise_typenwEmi
// CHECK: coro.free:
// CHECK: _ZN9resumable12promise_typedlEPv
}

// CHECK-LABEL: void @f2
extern "C" resumable f2(float) {
  co_return;
// CHECK: coro.alloc:
// CHECK: _ZN9resumable12promise_typenwEmf
// CHECK: coro.free:
// CHECK: _ZN9resumable12promise_typedlEPv
}

// CHECK-LABEL: void @f3
extern "C" resumable2 f3(int, float, const char*, Allocator) {
   co_yield 1;
   co_return;
// CHECK: coro.alloc:
// CHECK: _ZN10resumable212promise_typenwIJifPKc9AllocatorEEEPvmDpT_
// CHECK: coro.free:
// CHECK: _ZN10resumable212promise_typedlEPv
// CHECK: _ZN10resumable212promise_typedlEPv
}

// CHECK-LABEL: void @f4
extern "C" resumable f4(int n = 10) {
   for (int i = 0; i < n; i++) co_yield i;
// CHECK: coro.alloc:
// CHECK: call {{.*}}@_ZN9resumable12promise_typenwEmi(
// CHECK: coro.free:
// CHECK: call void @_ZN9resumable12promise_typedlEPv(
// CHECK: call void @_ZN9resumable12promise_typedlEPv(
}

struct resumable3 {
  struct promise_type {
    promise_type();
    resumable3 get_return_object() { return {}; }
    auto initial_suspend() { return std::suspend_always(); }
    auto final_suspend() noexcept { return std::suspend_always(); }
    void unhandled_exception() {}
    void return_void(){};
    std::suspend_always yield_value(int i);
  };
};
template <typename T> void *operator new(std::type_identity<T>, std::size_t sz, std::align_val_t);
template <typename T, typename... Args> void *operator new(std::type_identity<T>, std::size_t sz, std::align_val_t, Args...);
template <typename T> void operator delete(std::type_identity<T>, void *, std::size_t sz, std::align_val_t);
template <typename T, typename... Args> void operator delete(std::type_identity<T>, void *, std::size_t sz, std::align_val_t, Args...);

// CHECK-LABEL: void @f5
extern "C" resumable3 f5(float) {
  co_return;
// CHECK: coro.alloc:
// CHECK: call {{.*}}@_Znwm(
// CHECK: coro.free:
// CHECK: call void @_ZdlPvm(
// CHECK: call void @_ZdlPvm(
}

// CHECK-LABEL: void @f4.resume
// CHECK: coro.free:
// CHECK: _ZN9resumable12promise_typedlEPv

// CHECK-LABEL: void @f4.destroy
// CHECK: coro.free:
// CHECK: _ZN9resumable12promise_typedlEPv

// CHECK-LABEL: void @f4.cleanup
// CHECK: coro.free:
// CHECK: _ZN9resumable12promise_typedlEPv
