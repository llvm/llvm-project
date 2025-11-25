// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR

namespace std {

template<typename T> struct remove_reference       { typedef T type; };
template<typename T> struct remove_reference<T &>  { typedef T type; };
template<typename T> struct remove_reference<T &&> { typedef T type; };

template<typename T>
typename remove_reference<T>::type &&move(T &&t) noexcept;

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

struct string {
  int size() const;
  string();
  string(char const *s);
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

// FIXME: add CIRGen support here.
// struct blocking_wait_fn {
//   template <typename T>
//   T operator()(Task<T>&& awaitable) const {
//     return T();
//   }
// };

// inline constexpr blocking_wait_fn blocking_wait{};
// static constexpr blocking_wait_fn const& blockingWait = blocking_wait;

struct co_invoke_fn {
  template <typename F, typename... A>
  Task<void> operator()(F&& f, A&&... a) const {
    return Task<void>();
  }
};

co_invoke_fn co_invoke;

}} // namespace folly::coro

// CIR-DAG: ![[VoidTask:.*]] = !cir.record<struct "folly::coro::Task<void>" padded {!u8i}>
// CIR-DAG: ![[IntTask:.*]] = !cir.record<struct "folly::coro::Task<int>" padded {!u8i}>
// CIR-DAG: ![[VoidPromisse:.*]] = !cir.record<struct "folly::coro::Task<void>::promise_type" padded {!u8i}>
// CIR-DAG: ![[IntPromisse:.*]] = !cir.record<struct "folly::coro::Task<int>::promise_type" padded {!u8i}>
// CIR-DAG: ![[StdString:.*]] = !cir.record<struct "std::string" padded {!u8i}>
// CIR-DAG: ![[CoroHandleVoid:.*]] = !cir.record<struct "std::coroutine_handle<void>" padded {!u8i}>
// CIR-DAG: ![[CoroHandlePromiseVoid:rec_.*]]  = !cir.record<struct "std::coroutine_handle<folly::coro::Task<void>::promise_type>" padded {!u8i}>
// CIR-DAG: ![[CoroHandlePromiseInt:rec_.*]] = !cir.record<struct "std::coroutine_handle<folly::coro::Task<int>::promise_type>" padded {!u8i}>
// CIR-DAG: ![[SuspendAlways:.*]] = !cir.record<struct "std::suspend_always" padded {!u8i}>

// CIR: module {{.*}} {
// CIR-NEXT: cir.global external @_ZN5folly4coro9co_invokeE = #cir.zero : !rec_folly3A3Acoro3A3Aco_invoke_fn

// CIR: cir.func builtin private @__builtin_coro_id(!u32i, !cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>) -> !u32i
// CIR:  cir.func builtin private @__builtin_coro_alloc(!u32i) -> !cir.bool
// CIR:  cir.func builtin private @__builtin_coro_size() -> !u64i
// CIR:  cir.func builtin private @__builtin_coro_begin(!u32i, !cir.ptr<!void>) -> !cir.ptr<!void>

using VoidTask = folly::coro::Task<void>;

VoidTask silly_task() {
  co_await std::suspend_always();
}

// CIR: cir.func coroutine dso_local @_Z10silly_taskv() -> ![[VoidTask]]
// CIR: %[[VoidTaskAddr:.*]] = cir.alloca ![[VoidTask]], {{.*}}, ["__retval"]
// CIR: %[[SavedFrameAddr:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["__coro_frame_addr"]
// CIR: %[[VoidPromisseAddr:.*]] = cir.alloca ![[VoidPromisse]], {{.*}}, ["__promise"]

// Get coroutine id with __builtin_coro_id.

// CIR: %[[NullPtr:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR: %[[Align:.*]] = cir.const #cir.int<16> : !u32i
// CIR: %[[CoroId:.*]] = cir.call @__builtin_coro_id(%[[Align]], %[[NullPtr]], %[[NullPtr]], %[[NullPtr]])

// Perform allocation calling operator 'new' depending on __builtin_coro_alloc and
// call __builtin_coro_begin for the final coroutine frame address.

// CIR: %[[ShouldAlloc:.*]] = cir.call @__builtin_coro_alloc(%[[CoroId]]) : (!u32i) -> !cir.bool
// CIR: cir.store{{.*}} %[[NullPtr]], %[[SavedFrameAddr]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR: cir.if %[[ShouldAlloc]] {
// CIR:   %[[CoroSize:.*]] = cir.call @__builtin_coro_size() : () -> !u64i
// CIR:   %[[AllocAddr:.*]] = cir.call @_Znwm(%[[CoroSize]]) : (!u64i) -> !cir.ptr<!void>
// CIR:   cir.store{{.*}} %[[AllocAddr]], %[[SavedFrameAddr]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR: }
// CIR: %[[Load0:.*]] = cir.load{{.*}} %[[SavedFrameAddr]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR: %[[CoroFrameAddr:.*]] = cir.call @__builtin_coro_begin(%[[CoroId]], %[[Load0]])

// Call promise.get_return_object() to retrieve the task object.

// CIR: %[[RetObj:.*]] = cir.call @_ZN5folly4coro4TaskIvE12promise_type17get_return_objectEv(%[[VoidPromisseAddr]]) nothrow : {{.*}} -> ![[VoidTask]]
// CIR: cir.store{{.*}} %[[RetObj]], %[[VoidTaskAddr]] : ![[VoidTask]]
// Start a new scope for the actual codegen for co_await, create temporary allocas for
// holding coroutine handle and the suspend_always struct.

// CIR: cir.scope {
// CIR:   %[[SuspendAlwaysAddr:.*]] = cir.alloca ![[SuspendAlways]], {{.*}} ["ref.tmp0"] {alignment = 1 : i64}
// CIR:   %[[CoroHandleVoidAddr:.*]] = cir.alloca ![[CoroHandleVoid]], {{.*}} ["agg.tmp0"] {alignment = 1 : i64}
// CIR:   %[[CoroHandlePromiseAddr:.*]] = cir.alloca ![[CoroHandlePromiseVoid]], {{.*}} ["agg.tmp1"] {alignment = 1 : i64}

// Effectively execute `coawait promise_type::initial_suspend()` by calling initial_suspend() and getting
// the suspend_always struct to use for cir.await. Note that we return by-value since we defer ABI lowering
// to later passes, same is done elsewhere.

// CIR:   %[[Tmp0:.*]] = cir.call @_ZN5folly4coro4TaskIvE12promise_type15initial_suspendEv(%[[VoidPromisseAddr]])
// CIR:   cir.store{{.*}} %[[Tmp0:.*]], %[[SuspendAlwaysAddr]]

//
// Here we start mapping co_await to cir.await.
//

// First regions `ready` has a special cir.yield code to veto suspension.

// CIR:   cir.await(init, ready : {
// CIR:     %[[ReadyVeto:.*]] = cir.scope {
// CIR:       %[[TmpCallRes:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[SuspendAlwaysAddr]])
// CIR:       cir.yield %[[TmpCallRes:.*]] : !cir.bool
// CIR:     }
// CIR:     cir.condition(%[[ReadyVeto]])

// Second region `suspend` contains the actual suspend logic.
//
// - Start by getting the coroutine handle using from_address().
// - Implicit convert coroutine handle from task specific promisse
//   specialization to a void one.
// - Call suspend_always::await_suspend() passing the handle.
//
// FIXME: add veto support for non-void await_suspends.

// CIR:   }, suspend : {
// CIR:     %[[FromAddrRes:.*]] = cir.call @_ZNSt16coroutine_handleIN5folly4coro4TaskIvE12promise_typeEE12from_addressEPv(%[[CoroFrameAddr]])
// CIR:     cir.store{{.*}} %[[FromAddrRes]], %[[CoroHandlePromiseAddr]] : ![[CoroHandlePromiseVoid]]
// CIR:     %[[CoroHandlePromiseReload:.*]] = cir.load{{.*}} %[[CoroHandlePromiseAddr]]
// CIR:     cir.call @_ZNSt16coroutine_handleIvEC1IN5folly4coro4TaskIvE12promise_typeEEES_IT_E(%[[CoroHandleVoidAddr]], %[[CoroHandlePromiseReload]])
// CIR:     %[[CoroHandleVoidReload:.*]] = cir.load{{.*}} %[[CoroHandleVoidAddr]] : !cir.ptr<![[CoroHandleVoid]]>, ![[CoroHandleVoid]]
// CIR:     cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%[[SuspendAlwaysAddr]], %[[CoroHandleVoidReload]])
// CIR:     cir.yield
// CIR:   }, resume : {
// CIR:     cir.yield
// CIR:   },)
// CIR: }

folly::coro::Task<int> byRef(const std::string& s) {
  co_return s.size();
}

// CIR:  cir.func coroutine dso_local @_Z5byRefRKSt6string(%[[ARG:.*]]: !cir.ptr<![[StdString]]> {{.*}}) -> ![[IntTask]]
// CIR:    %[[AllocaParam:.*]] = cir.alloca !cir.ptr<![[StdString]]>, {{.*}}, ["s", init, const]
// CIR:    %[[IntTaskAddr:.*]] = cir.alloca ![[IntTask]], {{.*}}, ["__retval"]
// CIR:    %[[SavedFrameAddr:.*]]  = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["__coro_frame_addr"]
// CIR:    %[[AllocaFnUse:.*]] = cir.alloca !cir.ptr<![[StdString]]>, {{.*}}, ["s", init, const]
// CIR:    %[[IntPromisseAddr:.*]] = cir.alloca ![[IntPromisse]], {{.*}}, ["__promise"]
// CIR:    cir.store %[[ARG]], %[[AllocaParam]] : !cir.ptr<![[StdString]]>, {{.*}}

// Call promise.get_return_object() to retrieve the task object.

// CIR:    %[[LOAD:.*]] = cir.load %[[AllocaParam]] : !cir.ptr<!cir.ptr<![[StdString]]>>, !cir.ptr<![[StdString]]>
// CIR:    cir.store {{.*}} %[[LOAD]], %[[AllocaFnUse]] : !cir.ptr<![[StdString]]>, !cir.ptr<!cir.ptr<![[StdString]]>>
// CIR:    %[[RetObj:.*]] = cir.call @_ZN5folly4coro4TaskIiE12promise_type17get_return_objectEv(%4) nothrow : {{.*}} -> ![[IntTask]]
// CIR:    cir.store {{.*}} %[[RetObj]], %[[IntTaskAddr]] : ![[IntTask]]
// CIR:    cir.scope {
// CIR:      %[[SuspendAlwaysAddr:.*]] = cir.alloca ![[SuspendAlways]], {{.*}} ["ref.tmp0"] {alignment = 1 : i64}
// CIR:      %[[CoroHandleVoidAddr:.*]] = cir.alloca ![[CoroHandleVoid]], {{.*}} ["agg.tmp0"] {alignment = 1 : i64}
// CIR:      %[[CoroHandlePromiseAddr:.*]] = cir.alloca ![[CoroHandlePromiseInt]], {{.*}} ["agg.tmp1"] {alignment = 1 : i64}
// CIR:      %[[Tmp0:.*]] = cir.call @_ZN5folly4coro4TaskIiE12promise_type15initial_suspendEv(%[[IntPromisseAddr]])
// CIR:      cir.await(init, ready : {
// CIR:       %[[ReadyVeto:.*]] = cir.scope {
// CIR:         %[[TmpCallRes:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[SuspendAlwaysAddr]])
// CIR:         cir.yield %[[TmpCallRes:.*]] : !cir.bool
// CIR:       }
// CIR:       cir.condition(%[[ReadyVeto]])
// CIR:      }, suspend : {
// CIR:       %[[FromAddrRes:.*]] = cir.call @_ZNSt16coroutine_handleIN5folly4coro4TaskIiE12promise_typeEE12from_addressEPv(%[[CoroFrameAddr:.*]])
// CIR:       cir.store{{.*}} %[[FromAddrRes]], %[[CoroHandlePromiseAddr]] : ![[CoroHandlePromiseInt]]
// CIR:       %[[CoroHandlePromiseReload:.*]] = cir.load{{.*}} %[[CoroHandlePromiseAddr]]
// CIR:       cir.call @_ZNSt16coroutine_handleIvEC1IN5folly4coro4TaskIiE12promise_typeEEES_IT_E(%[[CoroHandleVoidAddr]], %[[CoroHandlePromiseReload]])
// CIR:       %[[CoroHandleVoidReload:.*]] = cir.load{{.*}} %[[CoroHandleVoidAddr]] : !cir.ptr<![[CoroHandleVoid]]>, ![[CoroHandleVoid]]
// CIR:       cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%[[SuspendAlwaysAddr]], %[[CoroHandleVoidReload]])
// CIR:       cir.yield
// CIR:      }, resume : {
// CIR:        cir.yield
// CIR:      },)
