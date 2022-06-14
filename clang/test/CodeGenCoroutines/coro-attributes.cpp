// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s
#include "Inputs/coroutine.h"

using namespace std;

struct coro {
  struct promise_type {
    coro get_return_object();
    suspend_never initial_suspend();
    suspend_never final_suspend() noexcept;
    void return_void();
    static void unhandled_exception();
  };
};

// CHECK: void @_Z3foov() #[[FOO_ATTR_NUM:[0-9]+]]
// CHECK: attributes #[[FOO_ATTR_NUM]] = { {{.*}} presplitcoroutine
coro foo() {
  co_await suspend_always{};
}
