// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++14 -fcoroutines-ts -fsyntax-only -Wignored-qualifiers -Wno-error=return-type -verify -fblocks -Wno-unreachable-code -Wno-unused-value

#if __has_feature(cxx_exceptions)
#error This test requires exceptions be disabled
#endif

#include "Inputs/std-coroutine.h"

using std::experimental::suspend_always;
using std::experimental::suspend_never;

struct promise_void {
  void get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
};

template <typename... T>
struct std::experimental::coroutine_traits<void, T...> { using promise_type = promise_void; };

void test0() { // expected-warning {{'promise_void' is required to declare the member 'unhandled_exception()' when exceptions are enabled}}
  co_return;
}
