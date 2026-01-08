// RUN: rm -rf %t
// RUN: split-file %s %t

// RUN: %clang_cc1 -std=c++20 %t/task.cppm -I%t -emit-reduced-module-interface -o %t/task.pcm
// RUN: %clang_cc1 -std=c++20 -fprebuilt-module-path=%t %t/user.cpp -verify -fsyntax-only

//--- coroutine.h

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
    void resume() const { __builtin_coro_resume(ptr); }
    void destroy() const { __builtin_coro_destroy(ptr); }
    bool done() const { return __builtin_coro_done(ptr); }
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

template <typename _PromiseT>
bool operator==(coroutine_handle<_PromiseT> const &_Left,
                coroutine_handle<_PromiseT> const &_Right) noexcept {
    return _Left.address() == _Right.address();
}

template <typename _PromiseT>
bool operator!=(coroutine_handle<_PromiseT> const &_Left,
                coroutine_handle<_PromiseT> const &_Right) noexcept {
    return !(_Left == _Right);
}

struct noop_coroutine_promise {};

template <>
struct coroutine_handle<noop_coroutine_promise> {
    operator coroutine_handle<>() const noexcept {
    return coroutine_handle<>::from_address(address());
    }

    constexpr explicit operator bool() const noexcept { return true; }
    constexpr bool done() const noexcept { return false; }

    constexpr void operator()() const noexcept {}
    constexpr void resume() const noexcept {}
    constexpr void destroy() const noexcept {}

    noop_coroutine_promise &promise() const noexcept {
    return *static_cast<noop_coroutine_promise *>(
        __builtin_coro_promise(this->__handle_, alignof(noop_coroutine_promise), false));
    }

    constexpr void *address() const noexcept { return __handle_; }

private:
    friend coroutine_handle<noop_coroutine_promise> noop_coroutine() noexcept;

    coroutine_handle() noexcept {
    this->__handle_ = __builtin_coro_noop();
    }

    void *__handle_ = nullptr;
};

using noop_coroutine_handle = coroutine_handle<noop_coroutine_promise>;

inline noop_coroutine_handle noop_coroutine() noexcept { return noop_coroutine_handle(); }

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

//--- task.cppm
module;
#include "coroutine.h"
export module task;
export template <typename T>
struct [[clang::coro_await_elidable]] Task {
  struct promise_type {
    struct FinalAwaiter {
      bool await_ready() const noexcept { return false; }

      template <typename P>
      std::coroutine_handle<> await_suspend(std::coroutine_handle<P> coro) noexcept {
        if (!coro)
          return std::noop_coroutine();
        return coro.promise().continuation;
      }
      void await_resume() noexcept {}
    };

    Task get_return_object() noexcept {
      return std::coroutine_handle<promise_type>::from_promise(*this);
    }

    std::suspend_always initial_suspend() noexcept { return {}; }
    FinalAwaiter final_suspend() noexcept { return {}; }
    void unhandled_exception() noexcept {}
    void return_value(T x) noexcept {
      value = x;
    }

    std::coroutine_handle<> continuation;
    T value;
  };

  Task(std::coroutine_handle<promise_type> handle) : handle(handle) {}
  ~Task() {
    if (handle)
      handle.destroy();
  }

  struct Awaiter {
    Awaiter(Task *t) : task(t) {}
    bool await_ready() const noexcept { return false; }
    void await_suspend(std::coroutine_handle<void> continuation) noexcept {}
    T await_resume() noexcept {
      return task->handle.promise().value;
    }

    Task *task;
  };

  auto operator co_await() {
    return Awaiter{this};
  }

private:
  std::coroutine_handle<promise_type> handle;
};

inline Task<int> callee() {
    co_return 1;
}

export inline Task<int> elidable() {
    co_return co_await callee();
}

namespace std {
    export using std::coroutine_traits;
    export using std::coroutine_handle;
    export using std::suspend_always;
}

//--- user.cpp
// expected-no-diagnostics
import task;
Task<int> test() {
    co_return co_await elidable();
}
