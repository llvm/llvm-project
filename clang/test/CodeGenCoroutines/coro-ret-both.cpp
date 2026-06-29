// RUN: %clang_cc1 -std=c++20 -triple=x86_64-unknown-linux-gnu -emit-llvm %s -o - -disable-llvm-passes | FileCheck %s

#include "Inputs/coroutine.h"

struct coro1 {
  struct promise_type {
    coro1 get_return_object();
    std::suspend_never initial_suspend();
    std::suspend_never final_suspend() noexcept;
    void return_void();
    void return_value(int);
    void unhandled_exception() noexcept;
  };
};

// CHECK-LABEL: define {{.*}} void @_Z2f1v
coro1 f1() {
  // CHECK: call void @_ZN5coro112promise_type11return_voidEv
  co_return;
}

// CHECK-LABEL: define {{.*}} void @_Z2f2v
coro1 f2() {
  // CHECK: call void @_ZN5coro112promise_type12return_valueEi({{.*}}, i32 {{.*}} 1)
  co_return 1;
}

// CHECK-LABEL: define {{.*}} void @_Z2f3b
coro1 f3(bool b) {
  // CHECK: call void @_ZN5coro112promise_type11return_voidEv
  // CHECK: call void @_ZN5coro112promise_type12return_valueEi({{.*}}, i32 {{.*}} 2)
  if (b) co_return;
  co_return 2;
}

// CHECK-LABEL: define {{.*}} void @_Z2f4b
coro1 f4(bool b) {
  // CHECK: call void @_ZN5coro112promise_type12return_valueEi({{.*}}, i32 {{.*}} 3)
  // CHECK: call void @_ZN5coro112promise_type11return_voidEv
  if (b) co_return 3;
}

void returns_void();

// CHECK-LABEL: define {{.*}} void @_Z2f5v
coro1 f5() {
  // CHECK: call void @_Z12returns_voidv
  // CHECK: call void @_ZN5coro112promise_type11return_voidEv
  // CHECK-NOT: call void @_ZN5coro112promise_type12return_valueEi
  co_return returns_void();
}
