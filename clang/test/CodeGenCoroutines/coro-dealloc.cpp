// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 \
// RUN:   -S -emit-llvm %s -o - -disable-llvm-passes \
// RUN:   -fsized-deallocation \
// RUN:   | FileCheck %s

#include "Inputs/coroutine.h"

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
  };
};

// Test the compiler will chose sized deallocation correctly.
// This is only enabled with `-fsized-deallocation` which is off by default.
void operator delete(void *ptr, std::size_t size) noexcept;

// CHECK: define{{.*}}@_Z1fv
// CHECK: %[[coro_free:.+]] = call{{.*}}@llvm.coro.free
// CHECK: coro.free:
// CHECK: %[[coro_size:.+]] = call{{.*}}@llvm.coro.size
// CHECK: call{{.*}}void @_ZdlPvm(ptr{{.*}}%[[coro_free]], i64{{.*}}%[[coro_size]])

task f() {
  co_return 43;
}
