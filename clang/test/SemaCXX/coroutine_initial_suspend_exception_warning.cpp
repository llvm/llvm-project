// RUN: %clang_cc1 -std=c++20 -verify %s -fcxx-exceptions -fexceptions -Wunused-result

namespace std {

template <class Ret, typename... T>
struct coroutine_traits { using promise_type = typename Ret::promise_type; };

template <class Promise = void>
struct coroutine_handle {
  static coroutine_handle from_address(void *);
  void *address() const noexcept;
};
template <>
struct coroutine_handle<void> {
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>);
  void *address() const noexcept;
};

struct suspend_always {
  bool await_ready() noexcept { return false; }
  void await_suspend(coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};

struct suspend_always_throws {
  bool await_ready() { return false; } // expected-note 1 {{must be declared with 'noexcept'}}
  void await_suspend(coroutine_handle<>) {} // expected-note 1 {{must be declared with 'noexcept'}}
  void await_resume() {} // no-warning
};

} // namespace std

using namespace std;

struct coro_t_1 {
  struct promise_type {
    coro_t_1 get_return_object();
    suspend_always initial_suspend();  // expected-note 1 {{must be declared with 'noexcept'}}
    suspend_always final_suspend() noexcept;
    void return_void();
    static void unhandled_exception();
  };
};

coro_t_1 f1() { // expected-warning {{a potentially throwing 'co_await __promise.initial_suspend()' may disable heap allocation elision; if it throws, the coroutine return value and state are destroyed in the reverse order of their construction}}
  co_return;
}

struct coro_t_2 {
  struct promise_type {
    coro_t_2 get_return_object();
    suspend_always_throws initial_suspend() noexcept;
    suspend_always final_suspend() noexcept;
    void return_void();
    static void unhandled_exception();
  };
};

coro_t_2 f2() { // expected-warning {{a potentially throwing 'co_await __promise.initial_suspend()' may disable heap allocation elision; if it throws, the coroutine return value and state are destroyed in the reverse order of their construction}}
  co_return;
}
