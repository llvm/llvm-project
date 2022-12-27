// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -clangir-disable-emit-cxx-default -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s

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
// CHECK: ![[CoroHandleVoid:ty_.*]] = !cir.struct<"struct.std::coroutine_handle", i8>
// CHECK: ![[CoroHandlePromise:ty_.*]] = !cir.struct<"struct.std::coroutine_handle", i8>
// CHECK: ![[SuspendAlways:ty_.*]] = !cir.struct<"struct.std::suspend_always", i8>

// CHECK: module {{.*}} {
// CHECK-NEXT: cir.global external @_ZN5folly4coro9co_invokeE = #cir.zero : !ty_22struct2Efolly3A3Acoro3A3Aco_invoke_fn22

// CHECK: cir.func builtin @__builtin_coro_id(i32, !cir.ptr<i8>, !cir.ptr<i8>, !cir.ptr<i8>) -> i32 attributes {builtin, sym_visibility = "private"}
// CHECK: cir.func builtin @__builtin_coro_alloc(i32) -> !cir.bool attributes {builtin, sym_visibility = "private"}
// CHECK: cir.func builtin @__builtin_coro_size() -> i64 attributes {builtin, sym_visibility = "private"}
// CHECK: cir.func builtin @__builtin_coro_begin(i32, !cir.ptr<i8>) -> !cir.ptr<i8> attributes {builtin, sym_visibility = "private"}

using VoidTask = folly::coro::Task<void>;

VoidTask silly_task() {
  co_await std::suspend_always();
}

// CHECK: cir.func coroutine @_Z10silly_taskv() -> ![[VoidTask]] {{.*}} {

// Allocate promise.

// CHECK: %[[#VoidTaskAddr:]] = cir.alloca ![[VoidTask]], {{.*}}, ["__retval"]
// CHECK: %[[#SavedFrameAddr:]] = cir.alloca !cir.ptr<i8>, cir.ptr <!cir.ptr<i8>>, ["__coro_frame_addr"] {alignment = 8 : i64}
// CHECK: %[[#VoidPromisseAddr:]] = cir.alloca ![[VoidPromisse]], {{.*}}, ["__promise"]

// Get coroutine id with __builtin_coro_id.

// CHECK: %[[#NullPtr:]] = cir.cst(#cir.null : !cir.ptr<i8>) : !cir.ptr<i8>
// CHECK: %[[#Align:]] = cir.cst(16 : i32) : i32
// CHECK: %[[#CoroId:]] = cir.call @__builtin_coro_id(%[[#Align]], %[[#NullPtr]], %[[#NullPtr]], %[[#NullPtr]])

// Perform allocation calling operator 'new' depending on __builtin_coro_alloc and
// call __builtin_coro_begin for the final coroutine frame address.

// CHECK: %[[#ShouldAlloc:]] = cir.call @__builtin_coro_alloc(%[[#CoroId]]) : (i32) -> !cir.bool
// CHECK: cir.store %[[#NullPtr]], %[[#SavedFrameAddr]] : !cir.ptr<i8>, cir.ptr <!cir.ptr<i8>>
// CHECK: cir.if %[[#ShouldAlloc]] {
// CHECK:   %[[#CoroSize:]] = cir.call @__builtin_coro_size() : () -> i64
// CHECK:   %[[#AllocAddr:]] = cir.call @_Znwm(%[[#CoroSize]]) : (i64) -> !cir.ptr<i8>
// CHECK:   cir.store %[[#AllocAddr]], %[[#SavedFrameAddr]] : !cir.ptr<i8>, cir.ptr <!cir.ptr<i8>>
// CHECK: }
// CHECK: %[[#Load0:]] = cir.load %[[#SavedFrameAddr]] : cir.ptr <!cir.ptr<i8>>, !cir.ptr<i8>
// CHECK: %[[#CoroFrameAddr:]] = cir.call @__builtin_coro_begin(%[[#CoroId]], %[[#Load0]])

// Call promise.get_return_object() to retrieve the task object.

// CHECK: %[[#RetObj:]] = cir.call @_ZN5folly4coro4TaskIvE12promise_type17get_return_objectEv(%[[#VoidPromisseAddr]]) : {{.*}} -> ![[VoidTask]]
// CHECK: cir.store %[[#RetObj]], %[[#VoidTaskAddr]] : ![[VoidTask]]

// Start a new scope for the actual codegen for co_await, create temporary allocas for
// holding coroutine handle and the suspend_always struct.

// CHECK: cir.scope {
// CHECK:   %[[#SuspendAlwaysAddr:]] = cir.alloca ![[SuspendAlways]], {{.*}} ["ref.tmp0"] {alignment = 1 : i64}
// CHECK:   %[[#CoroHandleVoidAddr:]] = cir.alloca ![[CoroHandleVoid]], {{.*}} ["agg.tmp0"] {alignment = 1 : i64}
// CHECK:   %[[#CoroHandlePromiseAddr:]] = cir.alloca ![[CoroHandlePromise]], {{.*}} ["agg.tmp1"] {alignment = 1 : i64}

// Effectively execute `coawait promise_type::initial_suspend()` by calling initial_suspend() and getting
// the suspend_always struct to use for cir.await. Note that we return by-value since we defer ABI lowering
// to later passes, same is done elsewhere.

// CHECK:   %[[#Tmp0:]] = cir.call @_ZN5folly4coro4TaskIvE12promise_type15initial_suspendEv(%[[#VoidPromisseAddr]])
// CHECK:   cir.store %[[#Tmp0]], %[[#SuspendAlwaysAddr]]

//
// Here we start mapping co_await to cir.await.
//

// First regions `ready` has a special cir.yield code to veto suspension.

// CHECK:   cir.await(init, ready : {
// CHECK:     %[[#ReadyVeto:]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[#SuspendAlwaysAddr]])
// CHECK:     cir.if %[[#ReadyVeto]] {
// CHECK:       cir.yield nosuspend
// CHECK:     }
// CHECK:     cir.yield

// Second region `suspend` contains the actual suspend logic.
//
// - Start by getting the coroutine handle using from_address().
// - Implicit convert coroutine handle from task specific promisse
//   specialization to a void one.
// - Call suspend_always::await_suspend() passing the handle.
//
// FIXME: add veto support for non-void await_suspends.

// CHECK:   }, suspend : {
// CHECK:     %[[#FromAddrRes:]] = cir.call @_ZNSt16coroutine_handleIN5folly4coro4TaskIvE12promise_typeEE12from_addressEPv(%[[#CoroFrameAddr]])
// CHECK:     cir.store %[[#FromAddrRes]], %[[#CoroHandlePromiseAddr]] : ![[CoroHandlePromise]]
// CHECK:     %[[#CoroHandlePromiseReload:]] = cir.load %[[#CoroHandlePromiseAddr]]
// CHECK:     cir.call @_ZNSt16coroutine_handleIvEC1IN5folly4coro4TaskIvE12promise_typeEEES_IT_E(%[[#CoroHandleVoidAddr]], %[[#CoroHandlePromiseReload]])
// CHECK:     %[[#CoroHandleVoidReload:]] = cir.load %[[#CoroHandleVoidAddr]] : cir.ptr <![[CoroHandleVoid]]>, ![[CoroHandleVoid]]
// CHECK:     cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%[[#SuspendAlwaysAddr]], %[[#CoroHandleVoidReload]])
// CHECK:     cir.yield

// Third region `resume` handles coroutine resuming logic.

// CHECK:   }, resume : {
// CHECK:     cir.call @_ZNSt14suspend_always12await_resumeEv(%[[#SuspendAlwaysAddr]])
// CHECK:     cir.yield
// CHECK:   },)
// CHECK: }

// Since we already tested cir.await guts above, the remaining checks for:
// - The actual user written co_await
// - The promise call
// - The final suspend co_await
// - Return

// The actual user written co_await
// CHECK: cir.scope {
// CHECK:   cir.await(user, ready : {
// CHECK:   }, suspend : {
// CHECK:   }, resume : {
// CHECK:   },)
// CHECK: }

// The promise call
// CHECK: cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(%[[#VoidPromisseAddr]])

// The final suspend co_await
// CHECK: cir.scope {
// CHECK:   cir.await(final, ready : {
// CHECK:   }, suspend : {
// CHECK:   }, resume : {
// CHECK:   },)
// CHECK: }

// Call builtin coro end and return

// CHECK-NEXT: %[[#CoroEndArg0:]] = cir.cst(#cir.null : !cir.ptr<i8>)
// CHECK-NEXT: %[[#CoroEndArg1:]] = cir.cst(false) : !cir.bool
// CHECK-NEXT: = cir.call @__builtin_coro_end(%[[#CoroEndArg0]], %[[#CoroEndArg1]])

// CHECK: %[[#Tmp1:]] = cir.load %[[#VoidTaskAddr]]
// CHECK-NEXT: cir.return %[[#Tmp1]]
// CHECK-NEXT: }
