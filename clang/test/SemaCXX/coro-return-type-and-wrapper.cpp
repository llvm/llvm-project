// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++20 -fsyntax-only -verify -Wall -Wextra -Wno-c++23-lambda-attributes
#include "Inputs/std-coroutine.h"

using std::suspend_always;
using std::suspend_never;


template <typename T> struct [[clang::coro_return_type]] Gen {
  struct promise_type {
    Gen<T> get_return_object() {
      return {};
    }
    suspend_always initial_suspend();
    suspend_always final_suspend() noexcept;
    void unhandled_exception();
    void return_value(T t);

    template <typename U>
    auto await_transform(const Gen<U> &) {
      struct awaitable {
        bool await_ready() noexcept { return false; }
        void await_suspend(std::coroutine_handle<>) noexcept {}
        U await_resume() noexcept { return {}; }
      };
      return awaitable{};
    }
  };
};

Gen<int> foo_coro(int b);
Gen<int> foo_coro(int b) { co_return b; }

[[clang::coro_wrapper]] Gen<int> marked_wrapper1(int b) { return foo_coro(b); }

// expected-error@+1 {{neither a coroutine nor a coroutine wrapper}}
Gen<int> non_marked_wrapper(int b) { return foo_coro(b); }

namespace using_decl {
template <typename T> using Co = Gen<T>;

[[clang::coro_wrapper]] Co<int> marked_wrapper1(int b) { return foo_coro(b); }

// expected-error@+1 {{neither a coroutine nor a coroutine wrapper}}
Co<int> non_marked_wrapper(int b) { return foo_coro(b); }
} // namespace using_decl

namespace lambdas {

void foo() {
  auto coro_lambda = []() -> Gen<int> {
    co_return 1;
  };
  // expected-error@+1 {{neither a coroutine nor a coroutine wrapper}}
  auto not_allowed_wrapper = []() -> Gen<int> {
    return foo_coro(1);
  };
  auto allowed_wrapper = [] [[clang::coro_wrapper]] () -> Gen<int> {
    return foo_coro(1);
  };
}

Gen<int> coro_containing_lambda() {
  // expected-error@+1 {{neither a coroutine nor a coroutine wrapper}}
  auto wrapper_lambda = []() -> Gen<int> {
    return foo_coro(1);
  };
  co_return co_await wrapper_lambda();
}
} // namespace lambdas

namespace std_function {
namespace std {
template <typename> class function;

template <typename ReturnValue, typename... Args>
class function<ReturnValue(Args...)> {
public:
  template <typename T> function &operator=(T) {}
  template <typename T> function(T) {}
  // expected-error@+1 {{neither a coroutine nor a coroutine wrapper}}
  ReturnValue operator()(Args... args) const {
    return callable_->Invoke(args...);  // expected-note {{in instantiation of member}}
  }

private:
  class Callable {
  public:
    // expected-error@+1 {{neither a coroutine nor a coroutine wrapper}}
    ReturnValue Invoke(Args...) const { return {}; }
  };
  Callable* callable_;
};
} // namespace std

void use_std_function() {
  std::function<int(bool)> foo = [](bool b) { return b ? 1 : 2; };
  // expected-error@+1 {{neither a coroutine nor a coroutine wrapper}}
  std::function<Gen<int>(bool)> test1 = [](bool b) {
    return foo_coro(b);
  };
  std::function<Gen<int>(bool)> test2 = [](bool) -> Gen<int> {
    co_return 1;
  };
  std::function<Gen<int>(bool)> test3 = foo_coro;

  foo(true);   // Fine.
  test1(true); // expected-note 2 {{in instantiation of member}}
  test2(true);
  test3(true);
}
} // namespace std_function

// different_promise_type
class [[clang::coro_return_type]] Task{};
struct my_promise_type {
  Task get_return_object() {
    return {};
  }
  suspend_always initial_suspend();
  suspend_always final_suspend() noexcept;
  void unhandled_exception();
};
namespace std {
template<> class coroutine_traits<Task, int> {
    using promise_type = my_promise_type;
};
} // namespace std
// expected-error@+1 {{neither a coroutine nor a coroutine wrapper}}
Task foo(int) { return Task{}; }
