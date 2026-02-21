// RUN: c-index-test -test-load-source all -c %s -fsyntax-only -target x86_64-apple-darwin9 -std=c++20 -I%S/../SemaCXX/Inputs | FileCheck %s
#include "std-coroutine.h"

using std::suspend_always;
using std::suspend_never;

struct promise_void {
  void get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend() noexcept;
  void return_void();
  void unhandled_exception();
};

template <>
struct std::coroutine_traits<void> { using promise_type = promise_void; };

void CoroutineTestRet() {
  co_return;
}
// CHECK: [[@LINE-3]]:6: FunctionDecl=CoroutineTestRet:18:6 (Definition) (coroutine) Extent=[18:1 - 20:2]
// CHECK: [[@LINE-4]]:25: UnexposedStmt=
// CHECK-SAME: [[@LINE-5]]:25 - [[@LINE-3]]:2]
// CHECK: [[@LINE-5]]:3: UnexposedStmt=
// CHECK-SAME: [[@LINE-6]]:3 - [[@LINE-6]]:12]

class Coro {
public:
  struct promise_type {
    Coro get_return_object();
    suspend_always initial_suspend();
    suspend_always final_suspend() noexcept;
    void return_void();
    void unhandled_exception();
  };
  Coro(std::coroutine_handle<promise_void>);
};

class W{
public:
  Coro CoroutineMemberTest() {
    co_return;
  }
};

// CHECK: CXXMethod=CoroutineMemberTest:41:8 (Definition) (coroutine) Extent=[41:3 - 43:4] [access=public]

auto lambda = []() -> Coro {
  co_return;
};

// CHECK: LambdaExpr= (coroutine) Extent=[48:15 - 50:2]
