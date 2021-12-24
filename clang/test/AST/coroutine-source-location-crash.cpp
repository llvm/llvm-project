// Test without serialization:
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++20 \
// RUN:    -fsyntax-only -ast-dump | FileCheck %s
//
// Test with serialization:
// RUN: %clang_cc1 -triple x86_64-apple-darwin9 -std=c++20 -emit-pch -o %t %s
// RUN: %clang_cc1 -x c++ -triple x86_64-apple-darwin9 -std=c++20 -include-pch %t \
// RUN: -ast-dump-all /dev/null \
// RUN: | sed -e "s/ <undeserialized declarations>//" -e "s/ imported//" \
// RUN: | FileCheck %s

#include "Inputs/std-coroutine.h"

using namespace std;

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
    suspend_never final_suspend() noexcept;
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
