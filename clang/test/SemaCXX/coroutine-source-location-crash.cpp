// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++14 -fcoroutines-ts \
// RUN:    -fsyntax-only -ast-dump | FileCheck %s
#include "Inputs/std-coroutine.h"

using namespace std::experimental;

struct A {
  bool await_ready();
  void await_resume();
  template <typename F>
  void await_suspend(F);
};

struct coro_t {
  struct promise_type {
    coro_t get_return_object();
    suspend_never initial_suspend();
    suspend_never final_suspend();
    void return_void();
    static void unhandled_exception();
  };
};

// {{0x[0-9a-fA-F]+}} <line:[[@LINE+1]]:1, col:36>
// CHECK-LABEL: FunctionDecl {{.*}} f 'coro_t (int)'
coro_t f(int n) {
  A a{};
  // CHECK: CoawaitExpr {{0x[0-9a-fA-F]+}} <col:3, col:12>
  // CHECK-NEXT: DeclRefExpr {{0x[0-9a-fA-F]+}} <col:12>
  // CHECK-NEXT: CXXMemberCallExpr {{0x[0-9a-fA-F]+}} <col:12>
  // CHECK-NEXT: MemberExpr {{0x[0-9a-fA-F]+}} <col:12>
  co_await a;
}
