// RUN: %clang_cc1 -std=c++20 -triple=x86_64-unknown-linux-gnu -emit-llvm -fextend-variable-liveness -o - %s -disable-llvm-passes -fexceptions | FileCheck %s

// See issue #192351
// Tests that when parameters to a coroutine have fake uses inserted for them by
// -fextend-variable-liveness, all such parameters (except `this`, which is not
// stored in the coroutine frame) have their fake uses inserted before the
// coroutine frame that they are contained in is freed.

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
        // CHECK: %[[B_EXT:.+]] = zext i1 %b to i8
        // CHECK: store i8 %[[B_EXT]], ptr %[[B_ADDR:.+]]

        // CHECK: coro.cleanup:
        // CHECK: %[[B_FAKEUSE:.+]] = load i8, ptr %[[B_ADDR]]
        // CHECK: call void (...) @llvm.fake.use(i8 %[[B_FAKEUSE]])
        // CHECK: call ptr @llvm.coro.free(

        // CHECK: coro.ret:
        // CHECK: call void @llvm.coro.end(
        // CHECK: %[[THIS_FAKE_USE:.+]] = load ptr, ptr %[[THIS_ADDR]]
        // CHECK: notail call void (...) @llvm.fake.use(ptr %[[THIS_FAKE_USE]])
        // CHECK: ret void
        if (b) {
            co_await std::suspend_always{};
        }
    }
};

void foo() {
    C().f(false);
}
