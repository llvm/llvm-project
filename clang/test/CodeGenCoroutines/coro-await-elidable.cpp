// This file tests the coro_await_elidable attribute semantics.
// RUN: %clang_cc1 -triple=x86_64-unknown-linux-gnu -std=c++20 -disable-llvm-passes -emit-llvm %s -o - | FileCheck %s

#include "Inputs/coroutine.h"
#include "Inputs/utility.h"

template <typename T>
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

// CHECK-LABEL: define{{.*}} @_Z6calleev{{.*}} {
Task<int> callee() {
  co_return 1;
}

// CHECK-LABEL: define{{.*}} @_Z8elidablev{{.*}} {
Task<int> elidable() {
  // CHECK: %[[TASK_OBJ:.+]] = alloca %struct.Task
  // CHECK: call void @_Z6calleev(ptr dead_on_unwind writable sret(%struct.Task) align 8 %[[TASK_OBJ]]) #[[ELIDE_SAFE:.+]]
  co_return co_await callee();
}

// CHECK-LABEL: define{{.*}} @_Z11nonelidablev{{.*}} {
Task<int> nonelidable() {
  // CHECK: %[[TASK_OBJ:.+]] = alloca %struct.Task
  auto t = callee();
  // Because we aren't co_awaiting a prvalue, we cannot elide here.
  // CHECK: call void @_Z6calleev(ptr dead_on_unwind writable sret(%struct.Task) align 8 %[[TASK_OBJ]])
  // CHECK-NOT: #[[ELIDE_SAFE]]
  co_await t;
  co_await std::move(t);

  co_return 1;
}

// CHECK-LABEL: define{{.*}} @_Z8addTasksO4TaskIiES1_{{.*}} {
Task<int> addTasks([[clang::coro_await_elidable_argument]] Task<int> &&t1, Task<int> &&t2) {
  int i1 = co_await t1;
  int i2 = co_await t2;
  co_return i1 + i2;
}

// CHECK-LABEL: define{{.*}} @_Z10returnSamei{{.*}} {
Task<int> returnSame(int i) {
  co_return i;
}

// CHECK-LABEL: define{{.*}} @_Z21elidableWithMustAwaitv{{.*}} {
Task<int> elidableWithMustAwait() {
  // CHECK: call void @_Z10returnSamei(ptr {{.*}}, i32 noundef 2) #[[ELIDE_SAFE]]
  // CHECK: call void @_Z10returnSamei(ptr {{.*}}, i32 noundef 3){{$}}
  co_return co_await addTasks(returnSame(2), returnSame(3));
}

template <typename... Args>
Task<int> sumAll([[clang::coro_await_elidable_argument]] Args && ... tasks);

// CHECK-LABEL: define{{.*}} @_Z16elidableWithPackv{{.*}} {
Task<int> elidableWithPack() {
  // CHECK: call void @_Z10returnSamei(ptr {{.*}}, i32 noundef 1){{$}}
  // CHECK: call void @_Z10returnSamei(ptr {{.*}}, i32 noundef 2) #[[ELIDE_SAFE]]
  // CHECK: call void @_Z10returnSamei(ptr {{.*}}, i32 noundef 3) #[[ELIDE_SAFE]]
  auto t = returnSame(1);
  co_return co_await sumAll(t, returnSame(2), returnSame(3));
}


// CHECK-LABEL: define{{.*}} @_Z25elidableWithPackRecursivev{{.*}} {
Task<int> elidableWithPackRecursive() {
  // CHECK: call void @_Z10returnSamei(ptr {{.*}}, i32 noundef 1) #[[ELIDE_SAFE]]
  // CHECK: call void @_Z10returnSamei(ptr {{.*}}, i32 noundef 2){{$}}
  // CHECK: call void @_Z10returnSamei(ptr {{.*}}, i32 noundef 3) #[[ELIDE_SAFE]]
  co_return co_await sumAll(addTasks(returnSame(1), returnSame(2)), returnSame(3));
}

// CHECK: attributes #[[ELIDE_SAFE]] = { coro_elide_safe }
