// RUN: %clang_cc1 -std=c++14 -fcoroutines-ts -triple=x86_64-unknown-linux-gnu -emit-llvm %s -o - -disable-llvm-passes | FileCheck %s

#include "Inputs/coroutine.h"

namespace coro = std::experimental::coroutines_v1;

struct coro1 {
  struct promise_type {
    coro1 get_return_object();
    coro::suspend_never initial_suspend();
    coro::suspend_never final_suspend();
    void return_void();
  };
};

coro1 f() {
  co_await coro::suspend_never{};
}

// CHECK-LABEL: define void @_Z1fv(
// CHECK: call void @_ZNSt12experimental13coroutines_v113suspend_never12await_resumeEv(%"struct.std::experimental::coroutines_v1::suspend_never"*
// CHECK: call void @_ZN5coro112promise_type11return_voidEv(%"struct.coro1::promise_type"* %__promise)

struct coro2 {
  struct promise_type {
    coro2 get_return_object();
    coro::suspend_never initial_suspend();
    coro::suspend_never final_suspend();
    void return_value(int);
  };
};

coro2 g() {
  co_return 42;
}

// CHECK-LABEL: define void @_Z1gv(
// CHECK: call void @_ZNSt12experimental13coroutines_v113suspend_never12await_resumeEv(%"struct.std::experimental::coroutines_v1::suspend_never"*
// CHECK: call void @_ZN5coro212promise_type12return_valueEi(%"struct.coro2::promise_type"* %__promise, i32 42)
