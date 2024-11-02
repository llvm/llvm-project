// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -fcoroutines-ts -std=c++14 -fsyntax-only -ast-dump %s | FileCheck %s

#include "Inputs/std-coroutine-exp-namespace.h"

using namespace std::experimental;

struct Task {
  struct promise_type {
    Task get_return_object() noexcept {
      return Task{coroutine_handle<promise_type>::from_promise(*this)};
    }

    void return_void() noexcept {}

    struct final_awaiter {
      bool await_ready() noexcept { return false; }
      coroutine_handle<> await_suspend(coroutine_handle<promise_type> h) noexcept {
        h.destroy();
        return {};
      }
      void await_resume() noexcept {}
    };

    void unhandled_exception() noexcept {}

    final_awaiter final_suspend() noexcept { return {}; }

    suspend_always initial_suspend() noexcept { return {}; }

    template <typename Awaitable>
    auto await_transform(Awaitable &&awaitable) {
      return awaitable.co_viaIfAsync();
    }
  };

  using handle_t = coroutine_handle<promise_type>;

  class Awaiter {
  public:
    explicit Awaiter(handle_t coro) noexcept;
    Awaiter(Awaiter &&other) noexcept;
    Awaiter(const Awaiter &) = delete;
    ~Awaiter();

    bool await_ready() noexcept { return false; }
    handle_t await_suspend(coroutine_handle<> continuation) noexcept;
    void await_resume();

  private:
    handle_t coro_;
  };

  Task(handle_t coro) noexcept : coro_(coro) {}

  handle_t coro_;

  Task(const Task &t) = delete;
  Task(Task &&t) noexcept;
  ~Task();
  Task &operator=(Task t) noexcept;

  Awaiter co_viaIfAsync();
};

static Task foo() {
  co_return;
}

Task bar() {
  auto mode = 2;
  switch (mode) {
  case 1:
    co_await foo();
    break;
  case 2:
    co_await foo();
    break;
  default:
    break;
  }
}

// CHECK-LABEL: FunctionDecl {{.*}} bar 'Task ()'
// CHECK:         SwitchStmt
// CHECK:           CaseStmt
// CHECK:             ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT:          CoawaitExpr
// CHECK-NEXT:            CXXBindTemporaryExpr {{.*}} 'Task' (CXXTemporary {{.*}})
// CHECK:                 MaterializeTemporaryExpr {{.*}} 'Task::Awaiter':'Task::Awaiter'
// CHECK:                 ExprWithCleanups {{.*}} 'bool'
// CHECK-NEXT:              CXXMemberCallExpr {{.*}} 'bool'
// CHECK-NEXT:                MemberExpr {{.*}} .await_ready
// CHECK:                 CallExpr {{.*}} 'void'
// CHECK-NEXT:              ImplicitCastExpr {{.*}} 'void (*)(void *)'
// CHECK-NEXT:                DeclRefExpr {{.*}} '__builtin_coro_resume' 'void (void *)'
// CHECK-NEXT:              ExprWithCleanups {{.*}} 'void *'

// CHECK:           CaseStmt
// CHECK:             ExprWithCleanups {{.*}} 'void'
// CHECK-NEXT:          CoawaitExpr
// CHECK-NEXT:            CXXBindTemporaryExpr {{.*}} 'Task' (CXXTemporary {{.*}})
// CHECK:                 MaterializeTemporaryExpr {{.*}} 'Task::Awaiter':'Task::Awaiter'
// CHECK:                 ExprWithCleanups {{.*}} 'bool'
// CHECK-NEXT:              CXXMemberCallExpr {{.*}} 'bool'
// CHECK-NEXT:                MemberExpr {{.*}} .await_ready
// CHECK:                 CallExpr {{.*}} 'void'
// CHECK-NEXT:              ImplicitCastExpr {{.*}} 'void (*)(void *)'
// CHECK-NEXT:                DeclRefExpr {{.*}} '__builtin_coro_resume' 'void (void *)'
// CHECK-NEXT:              ExprWithCleanups {{.*}} 'void *'
