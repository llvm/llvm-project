// Test that return value alloca does not enter the coro frame
// Regression test for GH49843
// RUN: %clang_cc1 -std=c++20 -triple=x86_64-unknown-linux-gnu -fexceptions -fcxx-exceptions -emit-llvm -o - %s | FileCheck %s

#include "Inputs/coroutine.h"

struct tag { char data[8]; }; // `tag` can be any type. It could be empty, or an int, or anything. 

struct expected {
  char data; // No issues if this member isn't here.

  expected(tag) : data() {}

  struct promise_type {
    tag get_return_object() { return {}; } // No issues if we return an `expected` instead. 
    std::suspend_never initial_suspend() { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    tag return_value(tag) { return tag{}; }
    void unhandled_exception() {}
  };
};

// CHECK-LABEL: define {{.*}} i8 @_Z2f1v()
expected f1() {
  // CHECK: %[[Retval:.+]] = alloca %struct.expected, align 1, !coro.outside.frame

  // %Retval are captured
  // CHECK: gro.conv:
  // CHECK: invoke void @_ZN8expectedC1E3tag(ptr {{.*}} %[[Retval]], i64 {{.*}})

  // CHECK: %[[GEP:.+]] = getelementptr {{.*}} %struct.expected, ptr %[[Retval]], i32 0, i32 0
  // CHECK-NEXT: %[[Val:.+]] = load i8, ptr %[[GEP]], align 1
  // CHECK-NEXT: ret i8 %[[Val]]
  co_return {};
}
