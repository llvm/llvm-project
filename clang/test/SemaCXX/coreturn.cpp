// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++20 -fsyntax-only -Wignored-qualifiers -Wno-error=return-type -verify -fblocks -Wall -Wextra -Wno-error=unreachable-code
#include "Inputs/std-coroutine.h"

using std::suspend_always;
using std::suspend_never;

struct awaitable {
  bool await_ready();
  void await_suspend(std::coroutine_handle<>); // FIXME: coroutine_handle
  void await_resume();
} a;

struct promise_void {
  void get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend() noexcept;
  void return_void();
  void unhandled_exception();
};

struct promise_void_return_value {
  void get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend() noexcept;
  void unhandled_exception();
  void return_value(int);
};

struct VoidTagNoReturn {
  struct promise_type {
    VoidTagNoReturn get_return_object();
    suspend_always initial_suspend();
    suspend_always final_suspend() noexcept;
    void unhandled_exception();
  };
};

struct VoidTagReturnValue {
  struct promise_type {
    VoidTagReturnValue get_return_object();
    suspend_always initial_suspend();
    suspend_always final_suspend() noexcept;
    void unhandled_exception();
    void return_value(int);
  };
};

struct VoidTagReturnVoid {
  struct promise_type {
    VoidTagReturnVoid get_return_object();
    suspend_always initial_suspend();
    suspend_always final_suspend() noexcept;
    void unhandled_exception();
    void return_void();
  };
};

struct promise_float {
  float get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend() noexcept;
  void return_void();
  void unhandled_exception();
};

struct promise_int {
  int get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend() noexcept;
  void return_value(int);
  void unhandled_exception();
};

template <>
struct std::coroutine_traits<void> { using promise_type = promise_void; };

template <typename T1>
struct std::coroutine_traits<void, T1> { using promise_type = promise_void_return_value; };

template <typename... T>
struct std::coroutine_traits<float, T...> { using promise_type = promise_float; };

template <typename... T>
struct std::coroutine_traits<int, T...> { using promise_type = promise_int; };

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

namespace dependent_void_coreturn {
struct coro {
  struct promise_type {
    coro get_return_object();
    suspend_never initial_suspend();
    suspend_never final_suspend() noexcept;
    void unhandled_exception();
    void return_void();
  };
};

struct Ctx {
  template <typename T>
  T &get();
  void f(int);
};

template <typename T>
coro f(Ctx &ctx) {
  auto &v = ctx.get<T>();
  co_return ctx.f(v);
}

void use(Ctx &ctx) { f<int>(ctx); }
}

namespace dependent_coreturn_lambda_capture {
// A variable of the enclosing function referenced only from the type-dependent
// operand of a co_return must still be captured by the enclosing lambda.
struct coro_void {
  struct promise_type {
    coro_void get_return_object();
    suspend_never initial_suspend();
    suspend_never final_suspend() noexcept;
    void unhandled_exception();
    void return_void();
  };
};

struct coro_value {
  struct promise_type {
    coro_value get_return_object();
    suspend_never initial_suspend();
    suspend_never final_suspend() noexcept;
    void unhandled_exception();
    void return_value(int);
  };
};

template <typename T> void use_void(int &, T);
template <typename T> int use_value(int &, T);

// Instantiates the generic lambda from a context that no longer has the
// enclosing function's scopes on the stack.
template <typename F> void call(F f) {
  int t = 0;
  f(t);
}

void captured_by_dependent_coreturn(int p) {
  call([&](auto &t) -> coro_void {
    if (t)
      co_return;
    co_return use_void(p, t);
  });
  call([&](auto &t) -> coro_value {
    if (t)
      co_return 0;
    co_return use_value(p, t);
  });
  // Same, but the co_return is reached after a co_await rather than after
  // another co_return.
  call([&](auto &t) -> coro_void {
    co_await suspend_never{};
    co_return use_void(p, t);
  });
}
} // namespace dependent_coreturn_lambda_capture
