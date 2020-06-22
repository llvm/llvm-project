// RUN: %clang_cc1 -std=c++14 -fcoroutines-ts -triple=x86_64-pc-windows-msvc18.0.0 -emit-llvm -o - %s -fexceptions -fcxx-exceptions -disable-llvm-passes | FileCheck %s
// -triple=x86_64-unknown-linux-gnu

#include "Inputs/coroutine.h"

namespace coro = std::experimental::coroutines_v1;

struct coro_t {
  void* p;
  ~coro_t();
  struct promise_type {
    coro_t get_return_object();
    coro::suspend_never initial_suspend();
    coro::suspend_never final_suspend() noexcept;
    void return_void();
    promise_type();
    ~promise_type();
    void unhandled_exception();
  };
};

struct Cleanup { ~Cleanup(); };
void may_throw();

coro_t f() {
  Cleanup cleanup;
  may_throw();
  co_return;
}

// CHECK-LABEL: define dso_local void @"?f@@YA?AUcoro_t@@XZ"(
// CHECK:  %gro.active = alloca i1
// CHECK:  store i1 false, i1* %gro.active

// CHECK:  invoke %"struct.coro_t::promise_type"* @"??0promise_type@coro_t@@QEAA@XZ"(
// CHECK:  invoke void @"?get_return_object@promise_type@coro_t@@QEAA?AU2@XZ"(
// CHECK:  store i1 true, i1* %gro.active

// CHECK:  %[[IS_ACTIVE:.+]] = load i1, i1* %gro.active
// CHECK:  br i1 %[[IS_ACTIVE]], label %[[CLEANUP1:.+]], label

// CHECK: [[CLEANUP1]]:
// CHECK:  %[[NRVO:.+]] = load i1, i1* %nrvo
// CHECK:  br i1 %[[NRVO]], label %{{.+}}, label %[[DTOR:.+]]

// CHECK: [[DTOR]]:
// CHECK:  call void @"??1coro_t@@QEAA@XZ"(
