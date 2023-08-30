// Tests that we can mark await-suspend as noinline correctly.
//
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:     -disable-llvm-passes | FileCheck %s

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

// CHECK-LABEL: @_Z7testingv

// Check `co_await __promise__.initial_suspend();` Since it returns std::suspend_always,
// which is an empty class, we shouldn't generate optimization blocker for it.
// CHECK: call token @llvm.coro.save
// CHECK: call void @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE{{.*}}#[[NORMAL_ATTR:[0-9]+]]

// Check the `co_await std::suspend_always{};` expression. We shouldn't emit the optimization
// blocker for it since it is an empty class.
// CHECK: call token @llvm.coro.save
// CHECK: call void @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE{{.*}}#[[NORMAL_ATTR]]

// Check `co_await StatefulAwaiter{};`. We need to emit the optimization blocker since
// the awaiter is not empty.
// CHECK: call token @llvm.coro.save
// CHECK: call void @_ZN15StatefulAwaiter13await_suspendIN4Task12promise_typeEEEvSt16coroutine_handleIT_E{{.*}}#[[NOINLINE_ATTR:[0-9]+]]

// Check `co_await AnotherStatefulAwaiter{};` to make sure that we can handle TypedefTypes.
// CHECK: call token @llvm.coro.save
// CHECK: call void @_ZN15StatefulAwaiter13await_suspendIN4Task12promise_typeEEEvSt16coroutine_handleIT_E{{.*}}#[[NOINLINE_ATTR]]

// Check `co_await awaiter;` to make sure we can handle lvalue cases.
// CHECK: call token @llvm.coro.save
// CHECK: call void @_ZN15StatefulAwaiter13await_suspendIN4Task12promise_typeEEEvSt16coroutine_handleIT_E{{.*}}#[[NOINLINE_ATTR]]

// Check `awaiter.await_suspend(...)` to make sure the explicit call the await_suspend won't be marked as noinline
// CHECK: call void @_ZN15StatefulAwaiter13await_suspendIvEEvSt16coroutine_handleIT_E{{.*}}#[[NORMAL_ATTR]]

// Check `co_await TemplatedAwaiter<int>{};` to make sure we can handle specialized template
// type.
// CHECK: call token @llvm.coro.save
// CHECK: call void @_ZN16TemplatedAwaiterIiE13await_suspendIN4Task12promise_typeEEEvSt16coroutine_handleIT_E{{.*}}#[[NOINLINE_ATTR]]

// Check `co_await TemplatedAwaiterInstace;` to make sure we can handle the lvalue from
// specialized template type.
// CHECK: call token @llvm.coro.save
// CHECK: call void @_ZN16TemplatedAwaiterIiE13await_suspendIN4Task12promise_typeEEEvSt16coroutine_handleIT_E{{.*}}#[[NOINLINE_ATTR]]

// Check `co_await Awaitable{};` to make sure we can handle awaiter returned by
// `operator co_await`;
// CHECK: call token @llvm.coro.save
// CHECK: call void @_ZN15StatefulAwaiter13await_suspendIN4Task12promise_typeEEEvSt16coroutine_handleIT_E{{.*}}#[[NOINLINE_ATTR]]

// Check `co_await Awaitable2{};` to make sure we can handle awaiter returned by
// `operator co_await` which returns a reference;
// CHECK: call token @llvm.coro.save
// CHECK: call void @_ZN15StatefulAwaiter13await_suspendIN4Task12promise_typeEEEvSt16coroutine_handleIT_E{{.*}}#[[NOINLINE_ATTR]]

// Check `co_await AlwaysInlineStatefulAwaiter{};` to make sure user can force the await_suspend function to get inlined.
// CHECK: call token @llvm.coro.save
// CHECK: call void @_ZN27AlwaysInlineStatefulAwaiter13await_suspendIN4Task12promise_typeEEEvSt16coroutine_handleIT_E{{.*}}#[[NORMAL_ATTR]]

// Check `co_await __promise__.final_suspend();`. We don't emit an blocker here since it is
// empty.
// CHECK: call token @llvm.coro.save
// CHECK: call ptr @_ZN4Task12promise_type12FinalAwaiter13await_suspendIS0_EESt16coroutine_handleIvES3_IT_E{{.*}}#[[NORMAL_ATTR]]

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

// CHECK-LABEL: @_Z25testingWithAwaitTransformv

// Init suspend
// CHECK: call token @llvm.coro.save
// CHECK-NOT: call void @llvm.coro.opt.blocker(
// CHECK: call void @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE{{.*}}#[[NORMAL_ATTR]]

// Check `co_await awaitableWithGetAwaiter{};`.
// CHECK: call token @llvm.coro.save
// CHECK-NOT: call void @llvm.coro.opt.blocker(
// Check call void @_ZN23awaitableWithGetAwaiter13await_suspendIN18AwaitTransformTask12promise_typeEEEvSt16coroutine_handleIT_E{{.*}}#[[NORMAL_ATTR]]

// Final suspend
// CHECK: call token @llvm.coro.save
// CHECK-NOT: call void @llvm.coro.opt.blocker(
// CHECK: call ptr @_ZN18AwaitTransformTask12promise_type12FinalAwaiter13await_suspendIS0_EESt16coroutine_handleIvES3_IT_E{{.*}}#[[NORMAL_ATTR]]

// CHECK-NOT: attributes #[[NORMAL_ATTR]] = noinline
// CHECK: attributes #[[NOINLINE_ATTR]] = {{.*}}noinline
