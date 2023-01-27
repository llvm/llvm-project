// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -I%S/Inputs -fclangir -fclangir-lifetime-check="history=all;remarks=all;history_limit=1" -clangir-verify-diagnostics -emit-cir %s -o %t.cir

#include "folly-coro.h"

folly::coro::Task<int> go(int const& val);
folly::coro::Task<int> go1() {
  auto task = go(1); // expected-note {{coroutine bound to resource with expired lifetime}}
                     // expected-note@-1 {{at the end of scope or full-expression}}
  co_return co_await task; // expected-remark {{pset => { task, invalid }}}
                           // expected-warning@-1 {{use of coroutine 'task' with dangling reference}}
}