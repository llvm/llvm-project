// Test for PR56919. Tests the destroy function contains the call to delete function only.
//
// REQUIRES: x86-registered-target
//
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -std=c++20 %s -O3 -S -o - | FileCheck %s

#include "Inputs/coroutine.h"

namespace std {

template <typename T> struct remove_reference { using type = T; };
template <typename T> struct remove_reference<T &> { using type = T; };
template <typename T> struct remove_reference<T &&> { using type = T; };

template <typename T>
constexpr typename std::remove_reference<T>::type&& move(T &&t) noexcept {
  return static_cast<typename std::remove_reference<T>::type &&>(t);
}

}

template <typename T>
class Task final {
 public:
  using value_type = T;

  class promise_type final {
   public: 
    Task<void> get_return_object() { return Task<void>(std::coroutine_handle<promise_type>::from_promise(*this)); }

    void unhandled_exception() {}

    std::suspend_always initial_suspend() { return {}; }

    auto await_transform(Task<void> co) {
      return await_transform(std::move(co.handle_.promise()));
    }

    auto await_transform(promise_type&& awaited) {
      struct Awaitable {
        promise_type&& awaited;

        bool await_ready() { return false; }

        std::coroutine_handle<> await_suspend(
            const std::coroutine_handle<> handle) {
          // Register our handle to be resumed once the awaited promise's coroutine
          // finishes, and then resume that coroutine.
          awaited.registered_handle_ = handle;
          return std::coroutine_handle<promise_type>::from_promise(awaited);
        }

        void await_resume() {}

       private:
      };

      return Awaitable{std::move(awaited)};
    }

    void return_void() {}

    // At final suspend resume our registered handle.
    auto final_suspend() noexcept {
      struct FinalSuspendAwaitable final {
        bool await_ready() noexcept { return false; }

        std::coroutine_handle<> await_suspend(
            std::coroutine_handle<> h) noexcept {
          return to_resume;
        }

        void await_resume() noexcept {}

        std::coroutine_handle<> to_resume;
      };

      return FinalSuspendAwaitable{registered_handle_};
    }

   private:
    std::coroutine_handle<promise_type> my_handle() {
      return std::coroutine_handle<promise_type>::from_promise(*this);
    }

    std::coroutine_handle<> registered_handle_;
  };

  ~Task() {
    // Teach llvm that we are only ever destroyed when the coroutine body is done,
    // so there is no need for the jump table in the destroy function. Our coroutine
    // library doesn't expose handles to the user, so we know this constraint isn't
    // violated.
    if (!handle_.done()) {
      __builtin_unreachable();
    }

    handle_.destroy();
  }

 private:
  explicit Task(const std::coroutine_handle<promise_type> handle)
      : handle_(handle) {}

  const std::coroutine_handle<promise_type> handle_;
};

Task<void> Qux() { co_return; }
Task<void> Baz() { co_await Qux(); }
Task<void> Bar() { co_await Baz(); }

// CHECK: _Z3Quxv.destroy:{{.*}}
// CHECK-NEXT: #
// CHECK-NEXT: jmp	_ZdlPv

// CHECK: _Z3Bazv.destroy:{{.*}}
// CHECK-NEXT: #
// CHECK-NEXT: jmp	_ZdlPv

// CHECK: _Z3Barv.destroy:{{.*}}
// CHECK-NEXT: #
// CHECK-NEXT: jmp	_ZdlPv
