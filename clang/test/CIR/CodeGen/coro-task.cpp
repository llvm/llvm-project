// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -clangir-disable-emit-cxx-default -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s
// XFAIL: *

namespace std {

template <class Ret, typename... T>
struct coroutine_traits { using promise_type = typename Ret::promise_type; };

template <class Promise = void>
struct coroutine_handle {
  static coroutine_handle from_address(void *) noexcept;
};
template <>
struct coroutine_handle<void> {
  template <class PromiseType>
  coroutine_handle(coroutine_handle<PromiseType>) noexcept;
  static coroutine_handle from_address(void *);
};

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

namespace folly {
namespace coro {

using std::suspend_always;
using std::suspend_never;
using std::coroutine_handle;

using SemiFuture = int;

template<class T>
struct Task {
    struct promise_type {
        Task<T> get_return_object() noexcept;
        suspend_always initial_suspend() noexcept;
        suspend_always final_suspend() noexcept;
        void return_value(T);
        void unhandled_exception();
        auto yield_value(Task<T>) noexcept { return final_suspend(); }
    };
    bool await_ready() noexcept { return false; }
    void await_suspend(coroutine_handle<>) noexcept {}
    T await_resume();
};

template<>
struct Task<void> {
    struct promise_type {
        Task<void> get_return_object() noexcept;
        suspend_always initial_suspend() noexcept;
        suspend_always final_suspend() noexcept;
        void return_void() noexcept;
        void unhandled_exception() noexcept;
        auto yield_value(Task<void>) noexcept { return final_suspend(); }
    };
    bool await_ready() noexcept { return false; }
    void await_suspend(coroutine_handle<>) noexcept {}
    void await_resume() noexcept {}
    SemiFuture semi();
};

struct blocking_wait_fn {
  template <typename T>
  T operator()(Task<T>&& awaitable) const {
    return T();
  }
};

inline constexpr blocking_wait_fn blocking_wait{};
static constexpr blocking_wait_fn const& blockingWait = blocking_wait;

template <typename T>
Task<T> collectAllRange(Task<T>* awaitable);

template <typename... SemiAwaitables>
Task<void> collectAll(SemiAwaitables&&... awaitables);

struct co_invoke_fn {
  template <typename F, typename... A>
  Task<void> operator()(F&& f, A&&... a) const {
    return Task<void>();
  }
};

co_invoke_fn co_invoke;

}} // namespace folly::coro

// CHECK: ![[VoidTask:ty_.*]] = !cir.struct<"struct.folly::coro::Task", i8>
// CHECK: ![[VoidPromisse:ty_.*]] = !cir.struct<"struct.folly::coro::Task<void>::promise_type", i8>

// CHECK: module {{.*}} {
// CHECK-NEXT: cir.global external @_ZN5folly4coro9co_invokeE = #cir.zero : !cir.struct<"struct.folly::coro::co_invoke_fn", i8

// CHECK: cir.func builtin @__builtin_coro_id(i32, !cir.ptr<i8>, !cir.ptr<i8>, !cir.ptr<i8>) -> i32 attributes {builtin, sym_visibility = "private"}

using VoidTask = folly::coro::Task<void>;

VoidTask silly_task() {
  co_await std::suspend_always();
}

// CHECK: cir.func builtin @__builtin_coro_frame() -> !cir.ptr<i8> attributes {builtin, sym_visibility = "private"}

// CHECK: cir.func @_Z10silly_taskv() -> ![[VoidTask]] {

// Allocate promise.

// CHECK: %[[#VoidTaskAddr:]] = cir.alloca ![[VoidTask]], {{.*}}, ["__retval"]
// CHECK: %[[#VoidPromisseAddr:]] = cir.alloca ![[VoidPromisse]], {{.*}}, ["__promise"]

// Get coroutine id with __builtin_coro_id.

// CHECK: %[[#NullPtr:]] = cir.cst(#cir.null : !cir.ptr<i8>) : !cir.ptr<i8>
// CHECK: %[[#Align:]] = cir.cst(16 : i32) : i32
// CHECK: %[[#CoroId:]] = cir.call @__builtin_coro_id(%[[#Align]], %[[#NullPtr]], %[[#NullPtr]], %[[#NullPtr]])

// Maybe perform allocation calling operator new.

// CHECK: %[[#ShouldAlloc:]] = cir.call @__builtin_coro_alloc(%[[#CoroId]]) : (i32) -> !cir.bool
// CHECK: cir.if %[[#ShouldAlloc]] {
// CHECK:   %[[#CoroSize:]] = cir.call @__builtin_coro_size() : () -> i64
// CHECK:   %[[#CoroFrameAddr:]] = cir.call @_Znwm(%[[#CoroSize]]) : (i64) -> !cir.ptr<i8>
// CHECK: }

// Call promise.get_return_object() to retrieve the task object.

// CHECK: %[[#RetObj:]] = cir.call @_ZN5folly4coro4TaskIvE12promise_type17get_return_objectEv(%[[#VoidPromisseAddr]]) : {{.*}} -> ![[VoidTask]]
// CHECK: cir.store %[[#RetObj]], %[[#VoidTaskAddr]] : ![[VoidTask]]

// CHECK: }