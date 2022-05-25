// Tests that we wouldn't generate an allocation call in global scope with (std::size_t, p0, ..., pn)
// RUN: %clang_cc1 %s -std=c++20 -S -triple x86_64-unknown-linux-gnu -emit-llvm -disable-llvm-passes %s -o - | FileCheck %s
#include "Inputs/coroutine.h"

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

void *operator new(std::size_t, void *);

resumable f1(void *) {
  co_return;
}

// CHECK: coro.alloc:
// CHECK-NEXT: [[SIZE:%.+]] = call [[BITWIDTH:.+]] @llvm.coro.size.[[BITWIDTH]]()
// CHECK-NEXT: call {{.*}} ptr @_Znwm([[BITWIDTH]] noundef [[SIZE]])
