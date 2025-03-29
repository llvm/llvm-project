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

// CHECK: define{{.*}}@_Z1fv(
// CHECK: coro.alloc:
// CHECK: %[[coro_size:.+]] = call{{.*}}@llvm.coro.size
// CHECK: %[[coro_align:.+]] = call{{.*}}@llvm.coro.align
// CHECK: %[[aligned_new:.+]] = call{{.*}}@_ZnwmSt11align_val_t({{.*}}%[[coro_size]],{{.*}}%[[coro_align]])

// CHECK: coro.free:
// CHECK: %[[coro_size_for_free:.+]] = call{{.*}}@llvm.coro.size
// CHECK: %[[coro_align_for_free:.+]] = call{{.*}}@llvm.coro.align
// CHECK: call void @_ZdlPvmSt11align_val_t({{.*}}%[[coro_size_for_free]],{{.*}}%[[coro_align_for_free]])

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
    static task2 get_return_object_on_allocation_failure() { return task2{}; }
  };
};

namespace std {
  struct nothrow_t {};
  constexpr nothrow_t nothrow = {};
}

void *operator new(std::size_t, std::align_val_t, std::nothrow_t) noexcept;

// CHECK: define{{.*}}@_Z2f2v(
// CHECK: coro.alloc:
// CHECK: %[[coro_size:.+]] = call{{.*}}@llvm.coro.size
// CHECK: %[[coro_align:.+]] = call{{.*}}@llvm.coro.align
// CHECK: %[[aligned_new:.+]] = call{{.*}}@_ZnwmSt11align_val_tSt9nothrow_t({{.*}}%[[coro_size]],{{.*}}%[[coro_align]])

// CHECK: coro.free:
// CHECK: %[[coro_size_for_free:.+]] = call{{.*}}@llvm.coro.size
// CHECK: %[[coro_align_for_free:.+]] = call{{.*}}@llvm.coro.align
// CHECK: call void @_ZdlPvmSt11align_val_t({{.*}}%[[coro_size_for_free]],{{.*}}%[[coro_align_for_free]])

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
    void operator delete(void *ptr);
  };
};

// CHECK: define{{.*}}@_Z2f3v
// CHECK: coro.free:
// CHECK: call{{.*}}void @_ZN5task312promise_typedlEPv(

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
    void operator delete(void *ptr, std::align_val_t);
    void operator delete(void *ptr);
  };
};

// CHECK: define{{.*}}@_Z2f4v
// CHECK: coro.free:
// CHECK: %[[coro_align_for_free:.+]] = call{{.*}}@llvm.coro.align
// CHECK: call{{.*}}void @_ZN5task412promise_typedlEPvSt11align_val_t({{.*}}, i64{{.*}}[[coro_align_for_free]]

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
    void *operator new(std::size_t);
  };
};

// CHECK: define{{.*}}@_Z2f5v
// CHECK: coro.alloc:
// CHECK: %[[coro_size:.+]] = call{{.*}}@llvm.coro.size
// CHECK: call{{.*}}ptr @_ZN5task512promise_typenwEm(i64{{.*}}%[[coro_size]])
task5 f5() {
  co_return 43;
}

struct task6 {
  struct promise_type {
    auto initial_suspend() { return std::suspend_always{}; }
    auto final_suspend() noexcept { return std::suspend_always{}; }
    auto get_return_object() { return task6{}; }
    void unhandled_exception() {}
    void return_value(int) {}
    void *operator new(std::size_t);
    void *operator new(std::size_t, int i);
  };
};

// CHECK: define{{.*}}@_Z2f6i
// CHECK: coro.alloc:
// CHECK: %[[coro_size:.+]] = call{{.*}}@llvm.coro.size
// CHECK: call{{.*}}ptr @_ZN5task612promise_typenwEmi(i64{{.*}}%[[coro_size]],
task6 f6(int i) {
  co_return i;
}

struct task7 {
  struct promise_type {
    auto initial_suspend() { return std::suspend_always{}; }
    auto final_suspend() noexcept { return std::suspend_always{}; }
    auto get_return_object() { return task7{}; }
    void unhandled_exception() {}
    void return_value(int) {}
    void *operator new(std::size_t);
    void *operator new(std::size_t, int i);
    void *operator new(std::size_t, std::align_val_t);
  };
};

// CHECK: define{{.*}}@_Z2f7i
// CHECK: coro.alloc:
// CHECK: %[[coro_size:.+]] = call{{.*}}@llvm.coro.size
// CHECK: %[[coro_align:.+]] = call{{.*}}@llvm.coro.align
// CHECK: call{{.*}}ptr @_ZN5task712promise_typenwEmSt11align_val_t(i64{{.*}}%[[coro_size]], i64{{.*}}[[coro_align]])
task7 f7(int i) {
  co_return i;
}

struct task8 {
  struct promise_type {
    auto initial_suspend() { return std::suspend_always{}; }
    auto final_suspend() noexcept { return std::suspend_always{}; }
    auto get_return_object() { return task8{}; }
    void unhandled_exception() {}
    void return_value(int) {}
    void *operator new(std::size_t);
    void *operator new(std::size_t, int i);
    void *operator new(std::size_t, std::align_val_t);
    void *operator new(std::size_t, std::align_val_t, int i);
  };
};

// CHECK: define{{.*}}@_Z2f8i
// CHECK: coro.alloc:
// CHECK: %[[coro_size:.+]] = call{{.*}}@llvm.coro.size
// CHECK: %[[coro_align:.+]] = call{{.*}}@llvm.coro.align
// CHECK: call{{.*}}ptr @_ZN5task812promise_typenwEmSt11align_val_ti(i64{{.*}}%[[coro_size]], i64{{.*}}[[coro_align]],
task8 f8(int i) {
  co_return i;
}
