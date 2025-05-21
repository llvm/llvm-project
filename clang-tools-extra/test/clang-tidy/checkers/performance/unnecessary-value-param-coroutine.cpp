// RUN: %check_clang_tidy -std=c++20-or-later %s performance-unnecessary-value-param %t -- -fix-errors

namespace std {

template <typename R, typename...> struct coroutine_traits {
  using promise_type = typename R::promise_type;
};

template <typename Promise = void> struct coroutine_handle;

template <> struct coroutine_handle<void> {
  static coroutine_handle from_address(void *addr) noexcept {
    coroutine_handle me;
    me.ptr = addr;
    return me;
  }
  void operator()() { resume(); }
  void *address() const noexcept { return ptr; }
  void resume() const {  }
  void destroy() const { }
  bool done() const { return true; }
  coroutine_handle &operator=(decltype(nullptr)) {
    ptr = nullptr;
    return *this;
  }
  coroutine_handle(decltype(nullptr)) : ptr(nullptr) {}
  coroutine_handle() : ptr(nullptr) {}
  //  void reset() { ptr = nullptr; } // add to P0057?
  explicit operator bool() const { return ptr; }

protected:
  void *ptr;
};

template <typename Promise> struct coroutine_handle : coroutine_handle<> {
  using coroutine_handle<>::operator=;

  static coroutine_handle from_address(void *addr) noexcept {
    coroutine_handle me;
    me.ptr = addr;
    return me;
  }

  Promise &promise() const {
    return *reinterpret_cast<Promise *>(
        __builtin_coro_promise(ptr, alignof(Promise), false));
  }
  static coroutine_handle from_promise(Promise &promise) {
    coroutine_handle p;
    p.ptr = __builtin_coro_promise(&promise, alignof(Promise), true);
    return p;
  }
};

struct suspend_always {
  bool await_ready() noexcept { return false; }
  void await_suspend(std::coroutine_handle<>) noexcept {}
  void await_resume() noexcept {}
};

} // namespace std

struct ReturnObject {
    struct promise_type {
        ReturnObject get_return_object() { return {}; }
        std::suspend_always initial_suspend() { return {}; }
        std::suspend_always final_suspend() noexcept { return {}; }
        void unhandled_exception() {}
        std::suspend_always yield_value(int value) { return {}; }
    };
};

struct A {
  A(const A&);
};

ReturnObject evaluateModels(const A timeMachineId) {
// No change for non-coroutine function expected because it is not safe.
// CHECK-FIXES: ReturnObject evaluateModels(const A timeMachineId) {
  co_return;
}
