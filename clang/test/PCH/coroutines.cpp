// Test this without pch.
// RUN: %clang_cc1 -include %s -verify -std=c++20 %s

// Test with pch.
// RUN: %clang_cc1 -std=c++20  -emit-pch -o %t %s
// RUN: %clang_cc1 -include-pch %t -verify -std=c++20 %s

#ifndef HEADER
#define HEADER

namespace std {
template <typename... T> struct coroutine_traits;

template <class Promise = void> struct coroutine_handle {
  coroutine_handle() = default;
  static coroutine_handle from_address(void *) noexcept;
};
template <> struct coroutine_handle<void> {
  static coroutine_handle from_address(void *) noexcept;
  coroutine_handle() = default;
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) noexcept;
};
} // namespace std

struct suspend_always {
  bool await_ready() noexcept;
  void await_suspend(std::coroutine_handle<>) noexcept;
  void await_resume() noexcept;
};

template <typename... Args> struct std::coroutine_traits<void, Args...> {
  struct promise_type {
    void get_return_object() noexcept;
    suspend_always initial_suspend() noexcept;
    suspend_always final_suspend() noexcept;
    void return_void() noexcept;
    suspend_always yield_value(int) noexcept;
    promise_type();
    ~promise_type() noexcept;
    void unhandled_exception() noexcept;
  };
};

template <typename... Args> struct std::coroutine_traits<int, Args...> {
  struct promise_type {
    int get_return_object() noexcept;
    suspend_always initial_suspend() noexcept;
    suspend_always final_suspend() noexcept;
    void return_value(int) noexcept;
    promise_type();
    ~promise_type() noexcept;
    void unhandled_exception() noexcept;
  };
};

template <typename T>
void f(T x) {  // checks coawait_expr and coroutine_body_stmt
  co_yield 42; // checks coyield_expr
  co_await x;  // checks dependent_coawait
  co_return;   // checks coreturn_stmt
}

template <typename T>
int f2(T x) {  // checks coawait_expr and coroutine_body_stmt
  co_return x;   // checks coreturn_stmt with expr
}

struct S {};
S operator co_await(S) { return S(); }

template <typename T>
int f3(T x) {
  co_await x; // checks dependent_coawait with overloaded co_await operator
}

#else

// expected-no-diagnostics
void g() {
  f(suspend_always{});
  f2(42);
}

#endif
