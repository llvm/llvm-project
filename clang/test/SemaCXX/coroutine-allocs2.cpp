// Tests that we wouldn't generate an allocation call in global scope with (std::size_t, p0, ..., pn)
// Although this test generates codes, it aims to test the semantics. So it is put here.
// RUN: %clang_cc1 %s -std=c++20 -S -triple %itanium_abi_triple -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s
#include "Inputs/std-coroutine.h"

namespace std {
typedef decltype(sizeof(int)) size_t;
}

struct Allocator {};

struct resumable {
  struct promise_type {

    resumable get_return_object() { return {}; }
    auto initial_suspend() { return std::suspend_always(); }
    auto final_suspend() noexcept { return std::suspend_always(); }
    void unhandled_exception() {}
    void return_void(){};
  };
};

void *operator new(std::size_t, void*);

resumable f1(void *) {
  co_return;
}

// CHECK: coro.alloc:
// CHECK-NEXT: [[SIZE:%.+]] = call i64 @llvm.coro.size.i64()
// CHECK-NEXT: call {{.*}} ptr @_Znwm(i64 noundef [[SIZE]])
