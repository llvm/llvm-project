// RUN: %clang_cc1 --std=c++20 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s

// Context: GH63818

#include "Inputs/coroutine.h"

struct coroutine {
  struct promise_type;
  std::coroutine_handle<promise_type> handle;
};

struct coroutine::promise_type {
  coroutine get_return_object() {
    return {std::coroutine_handle<promise_type>::from_promise(*this)};
  }
  std::suspend_never initial_suspend() noexcept { return {}; }
  std::suspend_always final_suspend() noexcept { return {}; }
  void return_void() {}
  void unhandled_exception() {}
};

struct Printy { ~Printy(); };

struct Printies {
  const Printy &a;
  const Printy &b;
  const Printy &c;
};

struct Awaiter : std::suspend_always {
  Printy await_resume() { return {}; }
};

// CHECK: define dso_local ptr @_Z5test1v()
coroutine test1() {
  // CHECK-NOT: @_ZN6PrintyD1Ev
  Printies p1{
    Printy(),
    co_await Awaiter{},
    // CHECK:       await.cleanup:
    // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
    // CHECK-NEXT:    br label %cleanup{{.*}}.from.await.cleanup
    // CHECK-NOT: @_ZN6PrintyD1Ev

    co_await Awaiter{}
    // CHECK:       await2.cleanup:
    // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
    // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
    // CHECK-NEXT:    br label %cleanup{{.*}}.from.await2.cleanup
    // CHECK-NOT: @_ZN6PrintyD1Ev
  };

  // CHECK-COUNT-3:       call void @_ZN6PrintyD1Ev
  // CHECK-NEXT:          br label

  // CHECK-NOT: @_ZN6PrintyD1Ev

  // CHECK: unreachable:
}

void bar(const Printy& a, const Printy& b);

// CHECK: define dso_local ptr @_Z5test2v()
coroutine test2() {
  // CHECK-NOT: @_ZN6PrintyD1Ev
  bar(
    Printy(),
    co_await Awaiter{}
    // CHECK:       await.cleanup:
    // CHECK-NEXT:    br label %cleanup{{.*}}.from.await.cleanup
    // CHECK-NOT: @_ZN6PrintyD1Ev
  );
  // CHECK: await.ready:
  // CHECK:   call void @_ZN6PrintyD1Ev
  // CHECK-NOT: @_ZN6PrintyD1Ev

  // CHECK: cleanup{{.*}}:
  // CHECK:   call void @_ZN6PrintyD1Ev
  // CHECK-NOT: @_ZN6PrintyD1Ev

  // CHECK: unreachable:
}
