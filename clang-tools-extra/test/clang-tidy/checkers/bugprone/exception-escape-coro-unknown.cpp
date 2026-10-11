// RUN: %check_clang_tidy -std=c++20-or-later %s bugprone-exception-escape %t -- \
// RUN:     -config='{"CheckOptions": { \
// RUN:       "bugprone-exception-escape.TreatFunctionsWithoutSpecificationAsThrowing": "OnlyUndefined" \
// RUN:     }}' -- -fexceptions

namespace std {

template <class Ret, typename... T> struct coroutine_traits {
  using promise_type = typename Ret::promise_type;
};

template <class Promise = void> struct coroutine_handle {
  template <class OtherPromise>
  coroutine_handle(coroutine_handle<OtherPromise>) noexcept;
  static coroutine_handle from_address(void *) noexcept;
};

struct suspend_never {
  bool await_ready() noexcept { return true; }
  void await_suspend(coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};

} // namespace std

struct Task {
  struct promise_type {
    Task get_return_object() noexcept { return {}; }
    std::suspend_never initial_suspend() noexcept { return {}; }
    std::suspend_never final_suspend() noexcept { return {}; }
    void return_void() noexcept {}
    void unhandled_exception() noexcept {}
  };
};

void undefined();

Task calls_undefined() noexcept {
  // CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'calls_undefined' which should not throw exceptions
  undefined();
  co_return;
}
