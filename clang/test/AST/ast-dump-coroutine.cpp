// RUN: %clang_cc1 -triple x86_64-unknown-unknown %s -std=c++20 \
// RUN:    -fsyntax-only -ast-dump -ast-dump-filter test | FileCheck %s

#include "Inputs/std-coroutine.h"

using namespace std;

struct Task {
  struct promise_type {
    std::suspend_always initial_suspend() { return {}; }
    Task get_return_object() {
      return std::coroutine_handle<promise_type>::from_promise(*this);
    }
    std::suspend_always final_suspend() noexcept { return {}; }
    std::suspend_always return_void() { return {}; }
    void unhandled_exception() {}

    auto await_transform(int s) {
      struct awaiter {
        promise_type *promise;
        bool await_ready() { return true; }
        int await_resume() { return 1; }
        void await_suspend(std::coroutine_handle<>) {}
      };

      return awaiter{this};
    }
  };

  Task(std::coroutine_handle<promise_type> promise);

  std::coroutine_handle<promise_type> handle;
};

Task test()  {
  co_await 1;
// Writen souce code, verify no implicit bit for the co_await expr.
// CHECK:        CompoundStmt {{.*}}
// CHECK-NEXT:   | `-ExprWithCleanups {{.*}} 'int'
// CHECK-NEXT:   |   `-CoawaitExpr {{.*}} 'int'{{$}}
// CHECK-NEXT:   |     |-IntegerLiteral {{.*}} <col:12> 'int' 1
// CHECK-NEXT:   |     |-MaterializeTemporaryExpr {{.*}} 'awaiter'
// CHECK-NEXT:   |     | `-CXXMemberCallExpr {{.*}} 'awaiter'
// CHECK-NEXT:   |     |   |-MemberExpr {{.*}} .await_transform
}
// Verify the implicit AST nodes for coroutines.
// CHECK:        |-DeclStmt {{.*}}
// CHECK-NEXT:   | `-VarDecl {{.*}} implicit used __promise
// CHECK-NEXT:   |   `-CXXConstructExpr {{.*}}
// CHECK-NEXT:   |-ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT:   | `-CoawaitExpr {{.*}} 'void' implicit
// CHECK-NEXT:       |-CXXMemberCallExpr  {{.*}} 'std::suspend_always'
// CHECK-NEXT:       |   | `-MemberExpr {{.*}} .initial_suspend
//                   ...
// FIXME: the CoreturnStmt should be marked as implicit
// CHECK: CoreturnStmt {{.*}} <col:6>{{$}}

Task test2()  {
// Writen souce code, verify no implicit bit for the co_return expr.
// CHECK:        CompoundStmt {{.*}}
// CHECK-NEXT:   | `-CoreturnStmt {{.*}} <line:{{.*}}:{{.*}}>{{$}}
  co_return;
}
// Verify the implicit AST nodes for coroutines.
// CHECK:        |-DeclStmt {{.*}}
// CHECK-NEXT:   | `-VarDecl {{.*}} implicit used __promise
//               ...
// FIXME: the CoreturnStmt should be marked as implicit
// CHECK: CoreturnStmt {{.*}} <col:6>{{$}}
