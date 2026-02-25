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
// CHECK: declare token @llvm.coro.save(ptr) #[[SAVE_ATTR_NUM:[0-9]+]]
// CHECK: void @_Z3foov.__await_suspend_wrapper__init({{.*}}) #[[WRAPPER_ATTR_NUM:[0-9]+]]
// CHECK: void @_Z3foov.__await_suspend_wrapper__await({{.*}}) #[[WRAPPER_ATTR_NUM:[0-9]+]]
// CHECK: void @_Z3foov.__await_suspend_wrapper__final({{.*}}) #[[WRAPPER_ATTR_NUM:[0-9]+]]
// CHECK: attributes #[[FOO_ATTR_NUM]] = { {{.*}} presplitcoroutine
// CHECK: attributes #[[SAVE_ATTR_NUM]] = { {{.*}}nomerge
// CHECK: attributes #[[WRAPPER_ATTR_NUM]] = { {{.*}}"sample-profile-suffix-elision-policy"="selected"
coro foo() {
  co_await suspend_always{};
}
