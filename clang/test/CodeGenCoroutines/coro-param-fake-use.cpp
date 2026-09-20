// RUN: %clang_cc1 -std=c++20 -triple=x86_64-unknown-linux-gnu -emit-llvm -fextend-variable-liveness -o - %s -disable-llvm-passes -fexceptions | FileCheck %s

// See issue #192351
// Tests that parameters to a coroutine do not have fake uses inserted for them
// when we enable -fextend-variable-liveness, except for `this`, which is not
// stored in the coroutine frame.

#include "Inputs/coroutine.h"

struct task {
    struct promise_type {
        task get_return_object() noexcept { return {}; }
        std::suspend_never initial_suspend() noexcept { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
        void return_void() noexcept {}
        void unhandled_exception() noexcept {}
    };
};

class C {
public:
    C() {}

    // CHECK-LABEL: void @_ZN1C1fEb(ptr noundef{{.*}} %this, i1 noundef{{.*}} %b)
    task f(bool b) {
        // CHECK: store ptr %this, ptr %[[THIS_ADDR:.+]]
        // CHECK-NOT: llvm.fake.use

        // CHECK:      coro.ret:
        // CHECK-NEXT: call void @llvm.coro.end(
        // CHECK-NEXT: %[[THIS_FAKE_USE:.+]] = load ptr, ptr %[[THIS_ADDR]]
        // CHECK-NEXT: notail call void (...) @llvm.fake.use(ptr %[[THIS_FAKE_USE]])
        // CHECK-NEXT: ret void
        if (b) {
            co_await std::suspend_always{};
        }
    }
};

void foo() {
    C().f(false);
}
