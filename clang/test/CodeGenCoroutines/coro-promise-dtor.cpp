// RUN: %clang_cc1 -std=c++20 -triple=x86_64-pc-windows-msvc18.0.0 -emit-llvm -o - %s -fexceptions -fcxx-exceptions -disable-llvm-passes | FileCheck %s
// -triple=x86_64-unknown-linux-gnu

#include "Inputs/coroutine.h"

struct coro_t {
  void* p;
  ~coro_t();
  struct promise_type {
    coro_t get_return_object();
    std::suspend_never initial_suspend();
    std::suspend_never final_suspend() noexcept;
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

// CHECK:  invoke noundef %"struct.coro_t::promise_type"* @"??0promise_type@coro_t@@QEAA@XZ"(
// CHECK:  invoke void @"?get_return_object@promise_type@coro_t@@QEAA?AU2@XZ"(

// CHECK:  call void @"??1promise_type@coro_t@@QEAA@XZ"
