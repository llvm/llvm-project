// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++20 \
// RUN:    -fsyntax-only -ast-dump | FileCheck %s

#include "Inputs/std-coroutine.h"

using namespace std;

struct Chat {
  struct promise_type {
    std::suspend_always initial_suspend() { return {}; }
    Chat get_return_object() {
      return std::coroutine_handle<promise_type>::from_promise(*this);
    }
    std::suspend_always yield_value(int m) { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    std::suspend_always return_value(int) { return {}; }
    void unhandled_exception() {}

    auto await_transform(int s) {
      struct awaiter {
        promise_type *promise;
        bool await_ready() { return true; }
        int await_resume() { return promise->message; }
        void await_suspend(std::coroutine_handle<>) {}
      };

      return awaiter{this};
    }
    int message;
  };

  Chat(std::coroutine_handle<promise_type> promise);

  std::coroutine_handle<promise_type> handle;
};

Chat f(int s)  {
  // CHECK:      CoyieldExpr {{.*}} <col:3, col:12>
  // CHECK-NEXT:   CXXMemberCallExpr {{.*}} <col:3, col:12> {{.*}}
  // CHECK-NEXT:     MemberExpr {{.*}} <col:3> {{.*}}
  // CHECK-NEXT:       DeclRefExpr {{.*}} <col:3> {{.*}}
  // CHECK-NEXT:     ImplicitCastExpr {{.*}} <col:12> {{.*}}
  // CHECK-NEXT:       DeclRefExpr {{.*}} <col:12> {{.*}}
  co_yield s;
  // CHECK:      CoreturnStmt {{.*}} <line:{{.*}}:3, col:13>
  co_return s;
  // CHECK:      CoawaitExpr {{.*}} <col:3, col:12> 'int'
  co_await s;
}
