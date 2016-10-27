// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++14 -fcoroutines-ts -fsyntax-only -Wignored-qualifiers -Wno-error=return-type -verify -fblocks -Wno-unreachable-code -Wno-unused-value

struct awaitable {
  bool await_ready();
  void await_suspend(); // FIXME: coroutine_handle
  void await_resume();
} a;

struct suspend_always {
  bool await_ready() { return false; }
  void await_suspend() {}
  void await_resume() {}
};

struct suspend_never {
  bool await_ready() { return true; }
  void await_suspend() {}
  void await_resume() {}
};

namespace std {
namespace experimental {

template <class Ret, typename... T>
struct coroutine_traits { using promise_type = typename Ret::promise_type; };

template <class Promise = void>
struct coroutine_handle {};
}
}

struct promise_void {
  void get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
};

struct promise_float {
  float get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_void();
};

struct promise_int {
  int get_return_object();
  suspend_always initial_suspend();
  suspend_always final_suspend();
  void return_value(int);
};

template <typename... T>
struct std::experimental::coroutine_traits<void, T...> { using promise_type = promise_void; };

template <typename... T>
struct std::experimental::coroutine_traits<float, T...> { using promise_type = promise_float; };

template <typename... T>
struct std::experimental::coroutine_traits<int, T...> { using promise_type = promise_int; };

void test0() { co_await a; }
float test1() { co_await a; }

int test2() {
  co_await a;
} // expected-warning {{control reaches end of non-void coroutine}}

int test3() {
  co_await a;
b:
  goto b;
}
