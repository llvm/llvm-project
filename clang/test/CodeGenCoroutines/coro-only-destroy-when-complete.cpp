// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 \
// RUN:     -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 \
// RUN:     -O3 -emit-llvm %s -o - | FileCheck %s --check-prefix=CHECK-O

#include "Inputs/coroutine.h"

using namespace std;

struct A;
struct A_promise_type {
  A get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend() noexcept;
  void return_value(int);
  void unhandled_exception();

  std::coroutine_handle<> handle;
};

struct Awaitable{
  bool await_ready();
  int await_resume();
  template <typename F>
  void await_suspend(F);
};
Awaitable something();

struct dtor {
    dtor();
    ~dtor();
};

struct [[clang::coro_only_destroy_when_complete]] A {
  using promise_type = A_promise_type;
  A();
  A(std::coroutine_handle<>);
  ~A();

  std::coroutine_handle<promise_type> handle;
};

A foo() {
    dtor d;
    co_await something();
    dtor d1;
    co_await something();
    dtor d2;
    co_return 43;
}

// CHECK: define{{.*}}@_Z3foov({{.*}}) #[[ATTR_NUM:[0-9]+]]
// CHECK: attributes #[[ATTR_NUM]] = {{.*}}coro_only_destroy_when_complete

// CHECK-O: define{{.*}}@_Z3foov.destroy
// CHECK-O: {{^.*}}:
// CHECK-O-NOT: br
// CHECK-O: ret void
