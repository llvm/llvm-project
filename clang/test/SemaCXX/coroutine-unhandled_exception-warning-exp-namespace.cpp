// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++14 -fcoroutines-ts \
// RUN:    -fsyntax-only -Wignored-qualifiers -Wno-error=return-type -verify \
// RUN:    -fblocks -Wno-unreachable-code -Wno-unused-value

// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++14 -fcoroutines-ts \
// RUN:    -fsyntax-only -Wignored-qualifiers -Wno-error=return-type -verify \
// RUN:    -fblocks -Wno-unreachable-code -Wno-unused-value \
// RUN:    -DDISABLE_WARNING -Wno-deprecated-experimental-coroutine -Wno-coroutine-missing-unhandled-exception

#if __has_feature(cxx_exceptions)
#error This test requires exceptions be disabled
#endif

#include "Inputs/std-coroutine-exp-namespace.h"

using std::experimental::suspend_always;
using std::experimental::suspend_never;

#ifndef DISABLE_WARNING
struct promise_void { // expected-note {{defined here}}
#else
struct promise_void {
#endif
  void get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend() noexcept;
  void return_void();
};

template <typename... T>
struct std::experimental::coroutine_traits<void, T...> { using promise_type = promise_void; };

#ifndef DISABLE_WARNING
void test0() { // expected-warning {{'promise_void' is required to declare the member 'unhandled_exception()' when exceptions are enabled}}
  co_return;   // expected-warning {{support for 'std::experimental::coroutine_traits' will be removed}}
  // expected-note@Inputs/std-coroutine-exp-namespace.h:8 {{'coroutine_traits' declared here}}
}
#else
void test0() { // expected-no-diagnostics
  co_return;
}
#endif
