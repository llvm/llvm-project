// An end-to-end test to make sure coroutine passes are added for thinlto.
// REQUIRES: x86-registered-target
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++23 -ffat-lto-objects -flto=thin -emit-llvm %s -O3 -o - \
// RUN:  | FileCheck %s

#include "Inputs/coroutine.h"

class BasicCoroutine {
public:
    struct Promise {
        BasicCoroutine get_return_object() { return BasicCoroutine {}; }

        void unhandled_exception() noexcept { }

        void return_void() noexcept { }

        std::suspend_never initial_suspend() noexcept { return {}; }
        std::suspend_never final_suspend() noexcept { return {}; }
    };
    using promise_type = Promise;
};

// COM: match the embedded module, so we don't match something in it by accident.
// CHECK: @llvm.embedded.object = {{.*}}
// CHECK: @llvm.compiler.used = {{.*}}

BasicCoroutine coro() {
// CHECK: define {{.*}} void @_Z4corov() {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT: ret void
// CHECK-NEXT: }
    co_return;
}

int main() {
// CHECK: define {{.*}} i32 @main() {{.*}} {
// CHECK-NEXT: entry:
// CHECK-NEXT: tail call void @_Z4corov()
// CHECK-NEXT: ret i32 0
// CHECK-NEXT: }
    coro();
}

