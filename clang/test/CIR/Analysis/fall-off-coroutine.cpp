// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -I %S/../../SemaCXX -fclangir -emit-cir -clangir-disable-passes -Wno-return-type %s -o %t.cir
// RUN: cir-opt %t.cir -cir-fall-off-analysis -o /dev/null 2>&1 | FileCheck %s

// With exceptions, the coroutine body is wrapped in an implicit try whose
// handler calls unhandled_exception(); leaving it is not flowing off.
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -I %S/../../SemaCXX -fcxx-exceptions -fexceptions -fclangir -emit-cir -clangir-disable-passes -Wno-return-type %s -o %t-eh.cir
// RUN: cir-opt %t-eh.cir -cir-fall-off-analysis -o /dev/null 2>&1 | FileCheck %s --check-prefixes=CHECK,EH

// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -I %S/../../SemaCXX -fsyntax-only -Wreturn-type -verify %s
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -I %S/../../SemaCXX -fcxx-exceptions -fexceptions -fsyntax-only -Wreturn-type -verify %s

#include "Inputs/std-coroutine.h"

using std::suspend_always;

struct awaitable {
  bool await_ready();
  void await_suspend(std::coroutine_handle<>);
  void await_resume();
} a;

// await_resume of the initial suspend may throw, so with exceptions the
// implicit try is also nested in an `if` that skips the body.
struct throwing_suspend {
  bool await_ready();
  void await_suspend(std::coroutine_handle<>);
  void await_resume();
};

struct promise_int {
  int get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend() noexcept;
  void return_value(int);
  void unhandled_exception();
};

struct promise_long {
  long get_return_object();
  throwing_suspend initial_suspend();
  suspend_always final_suspend() noexcept;
  void return_value(int);
  void unhandled_exception();
};

struct promise_float {
  float get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend() noexcept;
  void return_void();
  void unhandled_exception();
};

struct promise_void_return_value {
  void get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend() noexcept;
  void return_value(int);
  void unhandled_exception();
};

template <typename... T>
struct std::coroutine_traits<int, T...> { using promise_type = promise_int; };
template <typename... T>
struct std::coroutine_traits<long, T...> { using promise_type = promise_long; };
template <typename... T>
struct std::coroutine_traits<float, T...> { using promise_type = promise_float; };
template <typename... T>
struct std::coroutine_traits<void, T...> {
  using promise_type = promise_void_return_value;
};

[[noreturn]] void die();

// CHECK-DAG: fall-off-coroutine.cpp:[[@LINE+1]]:1: remark: fall-off: always
int always() {
  co_await a;
} // expected-warning {{non-void coroutine does not return a value}}

// CHECK-DAG: fall-off-coroutine.cpp:[[@LINE+1]]:1: remark: fall-off: maybe
int maybe(bool b) {
  if (b)
    co_return 1;
} // expected-warning {{non-void coroutine does not return a value in all control paths}}

// CHECK-DAG: fall-off-coroutine.cpp:[[@LINE+1]]:1: remark: fall-off: never
int never(bool b) {
  if (b)
    co_return 1;
  co_await a;
  co_return 2;
}

// CHECK-DAG: fall-off-coroutine.cpp:[[@LINE+1]]:1: remark: fall-off: never
int never_if_else(bool b) {
  if (b)
    co_return 1;
  else
    co_return 2;
}

// CHECK-DAG: fall-off-coroutine.cpp:[[@LINE+1]]:1: remark: fall-off: never-and-no-return
int ends_in_noreturn() {
  co_await a;
  die();
}

// CHECK-DAG: fall-off-coroutine.cpp:[[@LINE+1]]:1: remark: fall-off: never-and-no-return
int infinite_loop() {
  for (;;)
    co_await a;
}

// CHECK-DAG: fall-off-coroutine.cpp:[[@LINE+1]]:1: remark: fall-off: maybe
int loop_breaks(int n) {
  while (n--) {
    if (n == 3)
      break;
    co_return n;
  }
} // expected-warning {{non-void coroutine does not return a value in all control paths}}

// CHECK-DAG: fall-off-coroutine.cpp:[[@LINE+1]]:1: remark: fall-off: always
long throwing_initial_resume() {
  co_await a;
} // expected-warning {{non-void coroutine does not return a value}}

// CHECK-DAG: fall-off-coroutine.cpp:[[@LINE+1]]:1: remark: fall-off: never
long throwing_initial_resume_returns(bool b) {
  if (b)
    co_return 1;
  co_return 2;
}

// The fall-through handler, return_void(), is the implicit co_return.
// CHECK-DAG: fall-off-coroutine.cpp:[[@LINE+1]]:1: remark: fall-off: never
float return_void() {
  co_await a;
}

// Without return_void(), flowing off is undefined whatever the return type.
// CHECK-DAG: fall-off-coroutine.cpp:[[@LINE+1]]:1: remark: fall-off: always
void void_without_return_void(int) {
  co_await a;
} // expected-warning {{non-void coroutine does not return a value}}

// CHECK-DAG: fall-off-coroutine.cpp:[[@LINE+1]]:1: remark: fall-off: never
int use_lambda() {
  // CHECK-DAG: fall-off-coroutine.cpp:[[@LINE+1]]:{{[0-9]+}}: remark: fall-off: maybe
  auto l = [](bool b) -> int {
    if (b)
      co_return 1;
  }; // expected-warning {{non-void lambda does not return a value in all control paths}}
  (void)l(true);
  co_return 0;
}

#ifdef __EXCEPTIONS
// A try written in the body is not the implicit one.
// EH-DAG: fall-off-coroutine.cpp:[[@LINE+1]]:1: remark: fall-off: maybe
int user_try() {
  try {
    co_return 1;
  } catch (...) {
  }
} // expected-warning {{non-void coroutine does not return a value in all control paths}}

// EH-DAG: fall-off-coroutine.cpp:[[@LINE+1]]:1: remark: fall-off: never
int user_try_returns() {
  try {
    co_return 1;
  } catch (...) {
    co_return 2;
  }
}

// EH-DAG: fall-off-coroutine.cpp:[[@LINE+1]]:1: remark: fall-off: never-and-no-return
int ends_in_throw() {
  co_await a;
  throw 1;
}
#endif
