// Tests that we can mark await-suspend as noinline correctly.
//
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:     -O1 -disable-llvm-passes | FileCheck %s

#include "Inputs/coroutine.h"

struct Task {
  struct promise_type {
    struct FinalAwaiter {
      bool await_ready() const noexcept { return false; }
      template <typename PromiseType>
      std::coroutine_handle<> await_suspend(std::coroutine_handle<PromiseType> h) noexcept {
        return h.promise().continuation;
      }
      void await_resume() noexcept {}
    };

    Task get_return_object() noexcept {
      return std::coroutine_handle<promise_type>::from_promise(*this);
    }

    std::suspend_always initial_suspend() noexcept { return {}; }
    FinalAwaiter final_suspend() noexcept { return {}; }
    void unhandled_exception() noexcept {}
    void return_void() noexcept {}

    std::coroutine_handle<> continuation;
  };

  Task(std::coroutine_handle<promise_type> handle);
  ~Task();

private:
  std::coroutine_handle<promise_type> handle;
};

struct StatefulAwaiter {
    int value;
    bool await_ready() const noexcept { return false; }
    template <typename PromiseType>
    void await_suspend(std::coroutine_handle<PromiseType> h) noexcept {}
    void await_resume() noexcept {}
};

typedef std::suspend_always NoStateAwaiter;
using AnotherStatefulAwaiter = StatefulAwaiter;

template <class T>
struct TemplatedAwaiter {
    T value;
    bool await_ready() const noexcept { return false; }
    template <typename PromiseType>
    void await_suspend(std::coroutine_handle<PromiseType> h) noexcept {}
    void await_resume() noexcept {}
};


class Awaitable {};
StatefulAwaiter operator co_await(Awaitable) {
  return StatefulAwaiter{};
}

StatefulAwaiter GlobalAwaiter;
class Awaitable2 {};
StatefulAwaiter& operator co_await(Awaitable2) {
  return GlobalAwaiter;
}

struct AlwaysInlineStatefulAwaiter {
    void* value;
    bool await_ready() const noexcept { return false; }

    template <typename PromiseType>
    __attribute__((always_inline))
    void await_suspend(std::coroutine_handle<PromiseType> h) noexcept {}

    void await_resume() noexcept {}
};

Task testing() {
    co_await std::suspend_always{};
    co_await StatefulAwaiter{};
    co_await AnotherStatefulAwaiter{};
    
    // Test lvalue case.
    StatefulAwaiter awaiter;
    co_await awaiter;

    // The explicit call to await_suspend is not considered suspended.
    awaiter.await_suspend(std::coroutine_handle<void>::from_address(nullptr));

    co_await TemplatedAwaiter<int>{};
    TemplatedAwaiter<int> TemplatedAwaiterInstace;
    co_await TemplatedAwaiterInstace;

    co_await Awaitable{};
    co_await Awaitable2{};

    co_await AlwaysInlineStatefulAwaiter{};
}

struct AwaitTransformTask {
  struct promise_type {
    struct FinalAwaiter {
      bool await_ready() const noexcept { return false; }
      template <typename PromiseType>
      std::coroutine_handle<> await_suspend(std::coroutine_handle<PromiseType> h) noexcept {
        return h.promise().continuation;
      }
      void await_resume() noexcept {}
    };

    AwaitTransformTask get_return_object() noexcept {
      return std::coroutine_handle<promise_type>::from_promise(*this);
    }

    std::suspend_always initial_suspend() noexcept { return {}; }
    FinalAwaiter final_suspend() noexcept { return {}; }
    void unhandled_exception() noexcept {}
    void return_void() noexcept {}

    template <typename Awaitable>
    auto await_transform(Awaitable &&awaitable) {
      return awaitable;
    }

    std::coroutine_handle<> continuation;
  };

  AwaitTransformTask(std::coroutine_handle<promise_type> handle);
  ~AwaitTransformTask();

private:
  std::coroutine_handle<promise_type> handle;
};

struct awaitableWithGetAwaiter {
  bool await_ready() const noexcept { return false; }
  template <typename PromiseType>
  void await_suspend(std::coroutine_handle<PromiseType> h) noexcept {}
  void await_resume() noexcept {}
};

AwaitTransformTask testingWithAwaitTransform() {
  co_await awaitableWithGetAwaiter{};
}

// CHECK: define{{.*}}@_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE{{.*}}#[[NORMAL_ATTR:[0-9]+]]

// CHECK: define{{.*}}@_ZN15StatefulAwaiter13await_suspendIN4Task12promise_typeEEEvSt16coroutine_handleIT_E{{.*}}#[[NOINLINE_ATTR:[0-9]+]]

// CHECK: define{{.*}}@_ZN15StatefulAwaiter13await_suspendIvEEvSt16coroutine_handleIT_E{{.*}}#[[NORMAL_ATTR]]

// CHECK: define{{.*}}@_ZN16TemplatedAwaiterIiE13await_suspendIN4Task12promise_typeEEEvSt16coroutine_handleIT_E{{.*}}#[[NOINLINE_ATTR]]

// CHECK: define{{.*}}@_ZN27AlwaysInlineStatefulAwaiter13await_suspendIN4Task12promise_typeEEEvSt16coroutine_handleIT_E{{.*}}#[[ALWAYS_INLINE_ATTR:[0-9]+]]

// CHECK: define{{.*}}@_ZN4Task12promise_type12FinalAwaiter13await_suspendIS0_EESt16coroutine_handleIvES3_IT_E{{.*}}#[[NORMAL_ATTR]]

// CHECK: define{{.*}}@_ZN23awaitableWithGetAwaiter13await_suspendIN18AwaitTransformTask12promise_typeEEEvSt16coroutine_handleIT_E{{.*}}#[[NORMAL_ATTR]]

// CHECK: define{{.*}}@_ZN18AwaitTransformTask12promise_type12FinalAwaiter13await_suspendIS0_EESt16coroutine_handleIvES3_IT_E{{.*}}#[[NORMAL_ATTR]]

// CHECK-NOT: attributes #[[NORMAL_ATTR]] = noinline
// CHECK: attributes #[[NOINLINE_ATTR]] = {{.*}}noinline
// CHECK-NOT: attributes #[[ALWAYS_INLINE_ATTR]] = {{.*}}noinline
// CHECK: attributes #[[ALWAYS_INLINE_ATTR]] = {{.*}}alwaysinline
