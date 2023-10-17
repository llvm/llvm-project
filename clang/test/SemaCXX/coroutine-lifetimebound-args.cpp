// RUN: %clang_cc1 -triple x86_64-apple-darwin9 %s -std=c++20 -fsyntax-only -verify -Wall -Wextra -Wno-error=unreachable-code -Wno-unused

#include "Inputs/std-coroutine.h"

using std::suspend_always;
using std::suspend_never;


#define CORO_TYPE [[clang::annotate("coro_type")]]
#define CORO_UNSAFE [[clang::annotate("coro_unsafe")]]

template <typename T> struct CORO_TYPE Gen {
  struct promise_type {
    Gen<T> get_return_object() {
      return {};
    }
    suspend_always initial_suspend();
    suspend_always final_suspend() noexcept;
    void unhandled_exception();
    void return_value(const T &t);

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

template <typename T> using Co = Gen<T>;

Gen<int> foo_coro(const int& b);

Gen<int> plain_return_foo_decl(int b) {
  return foo_coro(b); // expected-warning {{address of stack memory associated with parameter}}
}

Gen<int> foo_coro(const int& b) {
  if (b > 0)
    co_return 1;
  co_return 2;
}

int getInt() { return 0; }

Co<int> bar_coro(const int &b, int c) {
  int x = co_await foo_coro(b);
  int y = co_await foo_coro(1);
  int z = co_await foo_coro(getInt());
  auto unsafe1 = foo_coro(1); // expected-warning {{temporary whose address is used as value of local variable}}
  auto unsafe2 = foo_coro(getInt()); // expected-warning {{temporary whose address is used as value of local variable}}
  auto  safe1 = foo_coro(b);
  auto  safe2 = foo_coro(c);
  co_return co_await foo_coro(co_await foo_coro(1));
}

Gen<int> plain_return_co(int b) {
  return foo_coro(b); // expected-warning {{address of stack memory associated with parameter}}
}

Gen<int> safe_forwarding(const int& b) {
  return foo_coro(b);
}

Gen<int> unsafe_wrapper(int b) {
  return safe_forwarding(b); // expected-warning {{address of stack memory associated with parameter}}
}

Co<int> complex_plain_return(int b) {
  return b > 0 
      ? foo_coro(1)   // expected-warning {{returning address of local temporary object}}
      : bar_coro(0, 1); // expected-warning {{returning address of local temporary object}}
}

void lambdas() {
  auto unsafe_lambda = [](int b) {
    return foo_coro(b); // expected-warning {{address of stack memory associated with parameter}}
  };
  auto safe_lambda = [](int b) -> Co<int> {
    int x = co_await foo_coro(1);
    co_return x + co_await foo_coro(b);
  };
}
// =============================================================================
// Safe usage when parameters are value
// =============================================================================
namespace by_value {
Gen<int> value_coro(int b) { co_return co_await foo_coro(b); }

Gen<int> wrapper1(int b) { return value_coro(b); }
Gen<int> wrapper2(const int& b) { return value_coro(b); }
}

// =============================================================================
// std::function like wrappers. (Eg: https://godbolt.org/z/x3PfG3Gfb)
// =============================================================================
namespace std {

template <class T>
T&& forward(typename remove_reference<T>::type& t) noexcept;
template <class T>
T&& forward(typename remove_reference<T>::type&& t) noexcept;

template <bool, typename>
class function;

template <bool UseFp, typename ReturnValue, typename... Args>
class function<UseFp, ReturnValue(Args...)> {
   public:
    class Callable {
       public:
        ReturnValue operator()(Args&&...) const { return {}; }
    };
    Callable* callable_;
    ReturnValue operator()(Args... args) const
        requires (!UseFp)
    {
        return (*callable_)(std::forward<Args>(args)...); // expected-warning 3 {{address of stack memory}}
    }

    // Callable can also be a function pointer type.
    using FpCallableType = ReturnValue(Args&&...);
    FpCallableType* fp_callable_;
    ReturnValue operator()(Args... args) const
        requires(UseFp)
    {
        return fp_callable_(std::forward<Args>(args)...);
    }

    template <typename T>
    function& operator=(T) {}
    template <typename T>
    function(T) {}
    function() {}
};
}  // namespace std

namespace without_function_pointers {
template <typename T>
using fn = std::function<false, T>;

void use_std_function() {
    fn<Co<int>(const int&, const int&)> pass;
    pass(1, 1);
    // Lifetime issue with one parameter.
    fn<Co<int>(const int&, int)> fail;
    fail(1, 1);        // expected-note {{in instantiation of}}
    // Lifetime issue with both parameters.
    fn<Co<int>(int, int)> fail_twice;
    fail_twice(1, 1);  // expected-note {{in instantiation of}}
}
} // namespace without_function_pointers

// =============================================================================
// Future work: Function pointers needs to be fixed.
// =============================================================================
namespace with_function_pointers {
template <typename T>
using fn = std::function<true, T>;

void use_std_function() {
    fn<Co<int>(int, int)> fail;
    fail(1, 1); // FIXME: Must error.
}
} // namespace function_pointers

// =============================================================================
// Future work: Reference wrappers needs to be fixed.
// =============================================================================
namespace with_reference_wrappers {
struct RefWrapper {
  RefWrapper(const int &a): b(a) {}
  const int &b;
};
Co<int> RefWrapperCoro(RefWrapper a) { co_return a.b; }

Co<int> UnsafeWrapper(int a) {
  return RefWrapperCoro(a); // FIXME: Must error.
}
}