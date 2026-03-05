// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 -I%S/Inputs -fclangir -fclangir-lifetime-check="history=all;remarks=all;history_limit=1" -clangir-verify-diagnostics -emit-cir %s -o %t.cir

#include "folly-coro.h"

folly::coro::Task<int> go(int const& val);
folly::coro::Task<int> go1() {
  auto task = go(1); // expected-note {{coroutine bound to resource with expired lifetime}}
                     // expected-note@-1 {{at the end of scope or full-expression}}
  co_return co_await task; // expected-remark {{pset => { task, invalid }}}
                           // expected-warning@-1 {{use of coroutine 'task' with dangling reference}}
}

folly::coro::Task<int> go1_lambda() {
  auto task = [i = 3]() -> folly::coro::Task<int> { // expected-note {{coroutine bound to lambda with expired lifetime}}
    co_return i;
  }(); // expected-note {{at the end of scope or full-expression}}
  co_return co_await task; // expected-remark {{pset => { task, invalid }}}
                           // expected-warning@-1 {{use of coroutine 'task' with dangling reference}}
}

folly::coro::Task<int> go2_lambda() {
  auto task = []() -> folly::coro::Task<int> { // expected-note {{coroutine bound to lambda with expired lifetime}}
    co_return 3;
  }(); // expected-note {{at the end of scope or full-expression}}
  co_return co_await task; // expected-remark {{pset => { task, invalid }}}
                           // expected-warning@-1 {{use of coroutine 'task' with dangling reference}}
}

folly::coro::Task<int> go3_lambda() {
  auto* fn = +[](int const& i) -> folly::coro::Task<int> { co_return i; };
  auto task = fn(3); // expected-note {{coroutine bound to resource with expired lifetime}}
                     // expected-note@-1 {{at the end of scope or full-expression}}
  co_return co_await task; // expected-remark {{pset => { task, invalid }}}
                           // expected-warning@-1 {{use of coroutine 'task' with dangling reference}}
}