// RUN: %clang_cc1 -std=c++20 -triple=x86_64-- -emit-llvm -fcxx-exceptions \
// RUN:            -disable-llvm-passes %s -o - | FileCheck %s

#include "Inputs/coroutine.h"

struct NontrivialType {
  ~NontrivialType() {}
};

struct NontrivialTypeWithThrowingDtor {
  ~NontrivialTypeWithThrowingDtor() noexcept(false) {}
};

namespace can_throw {
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

// CHECK-LABEL: define{{.*}} ptr @_ZN9can_throw11coro_createEv(
// CHECK: init.ready:
// CHECK-NEXT: store i1 true, ptr {{.*}}
// CHECK-NEXT: call void @_ZN9can_throw4Task23initial_suspend_awaiter12await_resumeEv(
// CHECK-NEXT: call void @_ZN14NontrivialTypeD1Ev(
// CHECK-NEXT: store i1 false, ptr {{.*}}
}

template <typename R>
struct NoexceptResumeTask {
    struct promise_type;
    using handle_type = std::coroutine_handle<promise_type>;

    struct initial_suspend_awaiter {
        bool await_ready() {
            return false;
        }

        void await_suspend(handle_type h) {}

        R await_resume() noexcept { return {}; }
    };

    struct promise_type {
        void return_void() {}
        void unhandled_exception() {}
        initial_suspend_awaiter initial_suspend() { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        NoexceptResumeTask get_return_object() {
            return NoexceptResumeTask{handle_type::from_promise(*this)};
        }
    };

    handle_type handler;
};

namespace no_throw {
using InitNoThrowTask = NoexceptResumeTask<NontrivialType>;

InitNoThrowTask coro_create() {
    co_return;
}

// CHECK-LABEL: define{{.*}} ptr @_ZN8no_throw11coro_createEv(
// CHECK: init.ready:
// CHECK-NEXT: call void @_ZN18NoexceptResumeTaskI14NontrivialTypeE23initial_suspend_awaiter12await_resumeEv(
// CHECK-NEXT: call void @_ZN14NontrivialTypeD1Ev(
}

namespace throwing_dtor {
using InitTaskWithThrowingDtor = NoexceptResumeTask<NontrivialTypeWithThrowingDtor>;

InitTaskWithThrowingDtor coro_create() {
    co_return;
}

// CHECK-LABEL: define{{.*}} ptr @_ZN13throwing_dtor11coro_createEv(
// CHECK: init.ready:
// CHECK-NEXT: store i1 true, ptr {{.*}}
// CHECK-NEXT: call void @_ZN18NoexceptResumeTaskI30NontrivialTypeWithThrowingDtorE23initial_suspend_awaiter12await_resumeEv(
// CHECK-NEXT: call void @_ZN30NontrivialTypeWithThrowingDtorD1Ev(
// CHECK-NEXT: store i1 false, ptr {{.*}}
}
