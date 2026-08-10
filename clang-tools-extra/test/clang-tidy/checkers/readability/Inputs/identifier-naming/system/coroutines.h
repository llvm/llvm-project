#pragma once

namespace std {

template <typename ret_t, typename... args_t>
struct coroutine_traits {
  using promise_type = typename ret_t::promise_type;
};

template <class promise_t>
struct coroutine_handle {
  static constexpr coroutine_handle from_address(void *addr) noexcept { return {}; };
};

} // namespace std

struct never_suspend {
  bool await_ready() noexcept { return false; }
  template <typename coro_t>
  void await_suspend(coro_t handle) noexcept {}
  void await_resume() noexcept {}
};

struct task {
  struct promise_type {
    task get_return_object() noexcept { return {}; }
    never_suspend initial_suspend() noexcept { return {}; }
    never_suspend final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() {}
  };
};
