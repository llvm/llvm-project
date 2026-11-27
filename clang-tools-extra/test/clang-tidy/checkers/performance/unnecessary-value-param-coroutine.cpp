// RUN: %check_clang_tidy -std=c++20-or-later %s performance-unnecessary-value-param %t -- -fix-errors
// RUN: %check_clang_tidy -std=c++20-or-later %s performance-unnecessary-value-param %t -- \
// RUN:   -config='{CheckOptions: {performance-unnecessary-value-param.IgnoreCoroutines: true}}' -fix-errors
// RUN: %check_clang_tidy -check-suffix=ALLOWED -std=c++20-or-later %s performance-unnecessary-value-param %t -- \
// RUN:   -config='{CheckOptions: {performance-unnecessary-value-param.IgnoreCoroutines: false}}' -fix-errors

namespace std {

template <class Ret, typename... T> struct coroutine_traits {
  using promise_type = typename Ret::promise_type;
};

template <class Promise = void> struct coroutine_handle {
  static coroutine_handle from_address(void *) noexcept;
  static coroutine_handle from_promise(Promise &promise);
  constexpr void *address() const noexcept;
};

template <> struct coroutine_handle<void> {
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) noexcept;
  static coroutine_handle from_address(void *);
  constexpr void *address() const noexcept;
};

struct suspend_always {
  bool await_ready() noexcept { return false; }
  void await_suspend(coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};

struct suspend_never {
  bool await_ready() noexcept { return true; }
  void await_suspend(coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};

} // namespace std

struct ReturnObject {
    struct promise_type {
        ReturnObject get_return_object() { return {}; }
        ReturnObject return_void() { return {}; }
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void unhandled_exception() {}
        std::suspend_always yield_value(int value) { return {}; }
    };
};

struct A {
  A(const A&);
};

ReturnObject foo_coroutine(const A a) {
// CHECK-MESSAGES-ALLOWED: [[@LINE-1]]:36: warning: the const qualified parameter 'a'
// CHECK-FIXES: ReturnObject foo_coroutine(const A a) {
  co_return;
}

ReturnObject foo_not_coroutine(const A a) {
// CHECK-MESSAGES: [[@LINE-1]]:40: warning: the const qualified parameter 'a'
// CHECK-MESSAGES-ALLOWED: [[@LINE-2]]:40: warning: the const qualified parameter 'a'
  return ReturnObject{};
}
