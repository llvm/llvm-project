#pragma clang system_header

// Like the compiler-provided <coroutine>, but cut down to the bare minimum
// needed to write coroutines in tests.

namespace std {

template <typename R, typename...> struct coroutine_traits {
  using promise_type = typename R::promise_type;
};

template <typename Promise = void> struct coroutine_handle {
  static coroutine_handle from_address(void *addr) noexcept { return {}; }
  static coroutine_handle from_promise(Promise &promise) { return {}; }
  constexpr coroutine_handle() noexcept = default;
};

template <> struct coroutine_handle<void> {
  template <typename Promise>
  coroutine_handle(coroutine_handle<Promise>) noexcept {}
  static coroutine_handle from_address(void *addr) noexcept { return {}; }
  constexpr coroutine_handle() noexcept = default;
};

struct suspend_never {
  bool await_ready() noexcept { return true; }
  void await_suspend(coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};

} // namespace std
