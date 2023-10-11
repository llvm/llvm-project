// RUN: %check_clang_tidy -std=c++20 %s misc-coroutine-hostile-raii %t \
// RUN:   -config="{CheckOptions: \
// RUN:             {misc-coroutine-hostile-raii.RAIIDenyList: \
// RUN:               'my::Mutex; my::other::Mutex'}}"

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


#define SCOPED_LOCKABLE __attribute__ ((scoped_lockable))

namespace absl {
class SCOPED_LOCKABLE Mutex {};
using Mutex2 = Mutex;
} // namespace absl


ReturnObject scopedLockableTest() {
    absl::Mutex a;
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 'a' holds a lock across a suspension point of coroutine and could be unlocked by a different thread [misc-coroutine-hostile-raii]
    absl::Mutex2 b;
    // CHECK-MESSAGES: :[[@LINE-1]]:18: warning: 'b' holds a lock across a suspension point of coroutine and could be unlocked by a different thread [misc-coroutine-hostile-raii]
    {
        absl::Mutex no_warning_1;
        { absl::Mutex no_warning_2; }
    }

    co_yield 1;
    absl::Mutex c;
    // CHECK-MESSAGES: :[[@LINE-1]]:17: warning: 'c' holds a lock across a suspension point of coroutine and could be unlocked by a different thread [misc-coroutine-hostile-raii]
    co_await std::suspend_always{};
    for(int i=1;i<=10;++i ) {
      absl::Mutex d;
      // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: 'd' holds a lock across a suspension point of coroutine and could be unlocked by a different thread [misc-coroutine-hostile-raii]
      co_await std::suspend_always{};
      co_yield 1;
      absl::Mutex no_warning_3;
    }
    if (true) {
      absl::Mutex e;
      // CHECK-MESSAGES: :[[@LINE-1]]:19: warning: 'e' holds a lock across a suspension point of coroutine and could be unlocked by a different thread [misc-coroutine-hostile-raii]
      co_yield 1;
      absl::Mutex no_warning_4;
    }
    absl::Mutex no_warning_5;
}
namespace my {
class Mutex{};

namespace other {
class Mutex{};
} // namespace other

using Mutex2 = Mutex;
} // namespace my

ReturnObject denyListTest() {
    my::Mutex a;
    // CHECK-MESSAGES: :[[@LINE-1]]:15: warning: 'a' persists across a suspension point of coroutine [misc-coroutine-hostile-raii]
    my::other::Mutex b;
    // CHECK-MESSAGES: :[[@LINE-1]]:22: warning: 'b' persists across a suspension point of coroutine [misc-coroutine-hostile-raii]
    my::Mutex2 c;
    // CHECK-MESSAGES: :[[@LINE-1]]:16: warning: 'c' persists across a suspension point of coroutine [misc-coroutine-hostile-raii]
    co_yield 1;
}