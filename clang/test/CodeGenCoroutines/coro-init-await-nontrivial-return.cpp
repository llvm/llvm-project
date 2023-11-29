// RUN: %clang_cc1 -std=c++20 -triple=x86_64-- -emit-llvm -fcxx-exceptions \
// RUN:            -disable-llvm-passes %s -o - | FileCheck %s

#include "Inputs/coroutine.h"

struct NontrivialType {
  ~NontrivialType() {}
};

struct Task {
    struct promise_type;
    using handle_type = std::coroutine_handle<promise_type>;

    struct initial_suspend_awaiter {
        bool await_ready() {
            return false;
        }

        void await_suspend(handle_type h) {}

        NontrivialType await_resume() { return {}; }
    };

    struct promise_type {
        void return_void() {}
        void unhandled_exception() {}
        initial_suspend_awaiter initial_suspend() { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        Task get_return_object() {
            return Task{handle_type::from_promise(*this)};
        }
    };

    handle_type handler;
};

Task coro_create() {
    co_return;
}

// CHECK-LABEL: define{{.*}} ptr @_Z11coro_createv(
// CHECK: init.ready:
// CHECK-NEXT: store i1 true, ptr {{.*}}
// CHECK-NEXT: call void @_ZN4Task23initial_suspend_awaiter12await_resumeEv(
// CHECK-NEXT: call void @_ZN14NontrivialTypeD1Ev(
// CHECK-NEXT: store i1 false, ptr {{.*}}
