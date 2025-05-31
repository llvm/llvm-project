// Tests that the combination of -fcoro-aligned-allocation and -fsized-deallocation works well.
// Test the compiler will chose sized deallocation correctly.
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 \
// RUN:   -fcoro-aligned-allocation -emit-llvm %s -o - -disable-llvm-passes \
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

// CHECK: define{{.*}}@_Z1fv
// CHECK: coro.free:
// CHECK: %[[coro_size:.+]] = call{{.*}}@llvm.coro.size
// CHECK: %[[coro_align:.+]] = call{{.*}}@llvm.coro.align
// CHECK: call{{.*}}void @_ZdlPvmSt11align_val_t(ptr{{.*}}, i64{{.*}}%[[coro_size]], i64{{.*}}%[[coro_align]])

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
    void operator delete(void *ptr);
  };
};

// CHECK: define{{.*}}@_Z2f2v
// CHECK: %[[FREE_HANDLE:.+]] = call{{.*}}ptr @llvm.coro.free(
// CHECK: coro.free:
// CHECK: call{{.*}}void @_ZN5task212promise_typedlEPv(ptr{{.*}} %[[FREE_HANDLE]])

task2 f2() {
  co_return 43;
}

struct task3 {
  struct promise_type {
    auto initial_suspend() { return std::suspend_always{}; }
    auto final_suspend() noexcept { return std::suspend_always{}; }
    auto get_return_object() { return task3{}; }
    void unhandled_exception() {}
    void return_value(int) {}
    void operator delete(void *ptr, std::size_t);
    void operator delete(void *ptr);
  };
};

// CHECK: define{{.*}}@_Z2f3v
// CHECK: %[[FREE_HANDLE:.+]] = call{{.*}}ptr @llvm.coro.free(
// CHECK: coro.free:
// CHECK: %[[coro_size:.+]] = call{{.*}}@llvm.coro.size
// CHECK: call{{.*}}void @_ZN5task312promise_typedlEPvm(ptr{{.*}} %[[FREE_HANDLE]], i64{{.*}}%[[coro_size]]

task3 f3() {
  co_return 43;
}

struct task4 {
  struct promise_type {
    auto initial_suspend() { return std::suspend_always{}; }
    auto final_suspend() noexcept { return std::suspend_always{}; }
    auto get_return_object() { return task4{}; }
    void unhandled_exception() {}
    void return_value(int) {}
    void operator delete(void *ptr, std::size_t);
    void operator delete(void *ptr, std::align_val_t);
    void operator delete(void *ptr);
  };
};

// CHECK: define{{.*}}@_Z2f4v
// CHECK: %[[FREE_HANDLE:.+]] = call{{.*}}ptr @llvm.coro.free(
// CHECK: coro.free:
// CHECK: %[[coro_align:.+]] = call{{.*}}@llvm.coro.align
// CHECK: call{{.*}}void @_ZN5task412promise_typedlEPvSt11align_val_t(ptr{{.*}} %[[FREE_HANDLE]], i64{{.*}}%[[coro_align]])

task4 f4() {
  co_return 43;
}

struct task5 {
  struct promise_type {
    auto initial_suspend() { return std::suspend_always{}; }
    auto final_suspend() noexcept { return std::suspend_always{}; }
    auto get_return_object() { return task5{}; }
    void unhandled_exception() {}
    void return_value(int) {}
    void operator delete(void *ptr, std::size_t);
    void operator delete(void *ptr, std::size_t, std::align_val_t);
    void operator delete(void *ptr);
  };
};

// CHECK: define{{.*}}@_Z2f5v
// CHECK: %[[FREE_HANDLE:.+]] = call{{.*}}ptr @llvm.coro.free(
// CHECK: coro.free:
// CHECK: %[[coro_size:.+]] = call{{.*}}@llvm.coro.size
// CHECK: %[[coro_align:.+]] = call{{.*}}@llvm.coro.align
// CHECK: call{{.*}}void @_ZN5task512promise_typedlEPvmSt11align_val_t(ptr{{.*}} %[[FREE_HANDLE]], i64{{.*}}%[[coro_size]], i64{{.*}}%[[coro_align]])

task5 f5() {
  co_return 43;
}
