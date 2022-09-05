// Tests that we'll emit the proper diagnostic message if we failed to find `::operator new(size_­t, nothrow_­t)`.
// RUN: %clang_cc1 %s -std=c++20 %s -fsyntax-only -verify

namespace std {
template <typename... T>
struct coroutine_traits;

template <class Promise = void>
struct coroutine_handle {
  coroutine_handle() = default;
  static coroutine_handle from_address(void *) noexcept { return {}; }
};

template <>
struct coroutine_handle<void> {
  static coroutine_handle from_address(void *) { return {}; }
  coroutine_handle() = default;
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) noexcept {}
};

struct suspend_always {
  bool await_ready() noexcept { return false; }
  void await_suspend(std::coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};

struct nothrow_t {};
constexpr nothrow_t nothrow = {};

} // end namespace std

using SizeT = decltype(sizeof(int));

struct promise_on_alloc_failure_tag {};

template <>
struct std::coroutine_traits<int, promise_on_alloc_failure_tag> {
  struct promise_type {
    int get_return_object() { return 0; }
    std::suspend_always initial_suspend() { return {}; }
    std::suspend_always final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() {}
    static int get_return_object_on_allocation_failure() { return -1; }
  };
};

extern "C" int f(promise_on_alloc_failure_tag) { // expected-error 1+{{unable to find '::operator new(size_­t, nothrow_­t)' for 'f'}}
    co_return;
}
