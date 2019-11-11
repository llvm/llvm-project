// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++14 -fcoroutines-ts -fsyntax-only -Wignored-qualifiers -Wno-error=return-type -verify -fblocks -Wall -Wextra -Wno-error=unreachable-code
#include "Inputs/std-coroutine.h"

using std::experimental::suspend_always;
using std::experimental::suspend_never;

struct awaitable {
  bool await_ready();
  void await_suspend(std::experimental::coroutine_handle<>); // FIXME: coroutine_handle
  void await_resume();
} a;

struct promise_void {
  void get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
  void unhandled_exception();
};

struct promise_void_return_value {
  void get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void unhandled_exception();
  void return_value(int);
};

struct VoidTagNoReturn {
  struct promise_type {
    VoidTagNoReturn get_return_object();
    suspend_always initial_suspend();
    suspend_always final_suspend();
    void unhandled_exception();
  };
};

struct VoidTagReturnValue {
  struct promise_type {
    VoidTagReturnValue get_return_object();
    suspend_always initial_suspend();
    suspend_always final_suspend();
    void unhandled_exception();
    void return_value(int);
  };
};

struct VoidTagReturnVoid {
  struct promise_type {
    VoidTagReturnVoid get_return_object();
    suspend_always initial_suspend();
    suspend_always final_suspend();
    void unhandled_exception();
    void return_void();
  };
};

struct promise_float {
  float get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
  void unhandled_exception();
};

struct promise_int {
  int get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_value(int);
  void unhandled_exception();
};

template <>
struct std::experimental::coroutine_traits<void> { using promise_type = promise_void; };

template <typename T1>
struct std::experimental::coroutine_traits<void, T1> { using promise_type = promise_void_return_value; };

template <typename... T>
struct std::experimental::coroutine_traits<float, T...> { using promise_type = promise_float; };

template <typename... T>
struct std::experimental::coroutine_traits<int, T...> { using promise_type = promise_int; };

void test0() { co_await a; }
float test1() { co_await a; }

int test2() {
  co_await a;
} // expected-warning {{non-void coroutine does not return a value}}

int test2a(bool b) {
  if (b)
    co_return 42;
} // expected-warning {{non-void coroutine does not return a value in all control paths}}

int test3() {
  co_await a;
b:
  goto b;
}

int test4() {
  co_return 42;
}

void test5(int) {
  co_await a;
} // expected-warning {{non-void coroutine does not return a value}}

void test6(int x) {
  if (x)
    co_return 42;
} // expected-warning {{non-void coroutine does not return a value in all control paths}}

void test7(int y) {
  if (y)
    co_return 42;
  else
    co_return 101;
}

VoidTagReturnVoid test8() {
  co_await a;
}

VoidTagReturnVoid test9(bool b) {
  if (b)
    co_return;
}

VoidTagReturnValue test10() {
  co_await a;
} // expected-warning {{non-void coroutine does not return a value}}

VoidTagReturnValue test11(bool b) {
  if (b)
    co_return 42;
} // expected-warning {{non-void coroutine does not return a value in all control paths}}
