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

template<typename T>
struct optional {
  optional();
  optional(const T&);
  T &operator*() &;
  T &&operator*() &&;
  T &value() &;
  T &&value() &&;
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
template <typename T>
T blockingWait(Task<T>&& awaitable) {
  return T();
}

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

// CIR: cir.func coroutine {{.*}} @_Z10silly_taskv() -> ![[VoidTask]]
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

// Third region `resume` handles coroutine resuming logic.

// CIR:   }, resume : {
// CIR:     cir.call @_ZNSt14suspend_always12await_resumeEv(%[[SuspendAlwaysAddr]])
// CIR:     cir.yield
// CIR:   },)
// CIR: }

// Since we already tested cir.await guts above, the remaining checks for:
// - The actual user written co_await
// - The promise call
// - The final suspend co_await
// - Return

// The actual user written co_await
// CIR: cir.scope {
// CIR:   cir.await(user, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR: }

// The promise call
// CHECK: cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(%[[VoidPromisseAddr]])

// The final suspend co_await
// CIR: cir.scope {
// CIR:   cir.await(final, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR: }

// Call builtin coro end and return

// CIR-NEXT: %[[CoroEndArg0:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR-NEXT: %[[CoroEndArg1:.*]] = cir.const #false
// CIR-NEXT: = cir.call @__builtin_coro_end(%[[CoroEndArg0]], %[[CoroEndArg1]])

// CIR: %[[Tmp1:.*]] = cir.load{{.*}} %[[VoidTaskAddr]]
// CIR-NEXT: cir.return %[[Tmp1]]
// CIR-NEXT: }

folly::coro::Task<int> byRef(const std::string& s) {
  co_return s.size();
}

// CIR:  cir.func coroutine {{.*}} @_Z5byRefRKSt6string(%[[ARG:.*]]: !cir.ptr<![[StdString]]> {{.*}}) -> ![[IntTask]]
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
// CIR:       }, resume : {
// CIR:         cir.call @_ZNSt14suspend_always12await_resumeEv(%[[SuspendAlwaysAddr]])
// CIR:         cir.yield
// CIR:       },)
// CIR:     }

// can't fallthrough
// CIR-NOT:   cir.await(user

// The final suspend co_await
// CIR: cir.scope {
// CIR:   cir.await(final, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR: }

folly::coro::Task<void> silly_coro() {
  std::optional<folly::coro::Task<int>> task;
  {
    std::string s = "yolo";
    task = byRef(s);
  }
  folly::coro::blockingWait(std::move(task.value()));
  co_return;
}

// Make sure we properly handle OnFallthrough coro body sub stmt and
// check there are not multiple co_returns emitted.

// CIR: cir.func coroutine {{.*}} @_Z10silly_corov() {{.*}} ![[VoidTask]]
// CIR: cir.await(init, ready : {
// CIR: cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv
// CIR-NOT: cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv
// CIR: cir.await(final, ready : {

folly::coro::Task<void> yield();
folly::coro::Task<void> yield1() {
  auto t = yield();
  co_yield t;
}

// CHECK: cir.func coroutine {{.*}} @_Z6yield1v() -> !rec_folly3A3Acoro3A3ATask3Cvoid3E

// CIR: cir.await(init, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: },)

//      CIR:  cir.scope {
// CIR-NEXT:   %[[SUSPEND_PTR:.*]] = cir.alloca ![[SuspendAlways]], !cir.ptr<![[SuspendAlways]]>
// CIR-NEXT:   %[[AWAITER_PTR:.*]] = cir.alloca ![[VoidTask]], !cir.ptr<![[VoidTask]]>
// CIR-NEXT:   %[[CORO_PTR:.*]] = cir.alloca ![[CoroHandleVoid]], !cir.ptr<![[CoroHandleVoid]]>
// CIR-NEXT:   %[[CORO2_PTR:.*]] = cir.alloca ![[CoroHandlePromiseVoid]], !cir.ptr<![[CoroHandlePromiseVoid]]>
// CIR-NEXT:   cir.copy {{.*}} to %[[AWAITER_PTR]] : !cir.ptr<![[VoidTask]]>
// CIR-NEXT:   %[[AWAITER:.*]] = cir.load{{.*}} %[[AWAITER_PTR]] : !cir.ptr<![[VoidTask]]>, ![[VoidTask]]
// CIR-NEXT:   %[[SUSPEND:.*]] = cir.call @_ZN5folly4coro4TaskIvE12promise_type11yield_valueES2_(%{{.+}}, %[[AWAITER]]) nothrow : (!cir.ptr<![[VoidPromisse]]>, ![[VoidTask]]) -> ![[SuspendAlways]]
// CIR-NEXT:   cir.store{{.*}} %[[SUSPEND]], %[[SUSPEND_PTR]] : ![[SuspendAlways]], !cir.ptr<![[SuspendAlways]]>
// CIR-NEXT:   cir.await(yield, ready : {
// CIR-NEXT:     %[[READY:.*]] = cir.scope {
// CIR-NEXT:       %[[A:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[SUSPEND_PTR]]) nothrow : (!cir.ptr<![[SuspendAlways]]>) -> !cir.bool
// CIR-NEXT:       cir.yield %[[A]] : !cir.bool
// CIR-NEXT:     } : !cir.bool
// CIR-NEXT:     cir.condition(%[[READY]])
// CIR-NEXT:   }, suspend : {
// CIR-NEXT:     %[[CORO2:.*]] = cir.call @_ZNSt16coroutine_handleIN5folly4coro4TaskIvE12promise_typeEE12from_addressEPv(%9) nothrow : (!cir.ptr<!void>) -> ![[CoroHandlePromiseVoid]]
// CIR-NEXT:     cir.store{{.*}} %[[CORO2]], %[[CORO2_PTR]] : ![[CoroHandlePromiseVoid]], !cir.ptr<![[CoroHandlePromiseVoid]]>
// CIR-NEXT:     %[[B:.*]] = cir.load{{.*}} %[[CORO2_PTR]] : !cir.ptr<![[CoroHandlePromiseVoid]]>, ![[CoroHandlePromiseVoid]]
// CIR-NEXT:     cir.call @_ZNSt16coroutine_handleIvEC1IN5folly4coro4TaskIvE12promise_typeEEES_IT_E(%[[CORO_PTR]], %[[B]]) nothrow : (!cir.ptr<![[CoroHandleVoid]]>, ![[CoroHandlePromiseVoid]]) -> ()
// CIR-NEXT:     %[[C:.*]] = cir.load{{.*}} %[[CORO_PTR]] : !cir.ptr<![[CoroHandleVoid]]>, ![[CoroHandleVoid]]
// CIR-NEXT:     cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%[[SUSPEND_PTR]], %[[C]]) nothrow : (!cir.ptr<![[SuspendAlways]]>, ![[CoroHandleVoid]]) -> ()
// CIR-NEXT:     cir.yield
// CIR-NEXT:   }, resume : {
// CIR-NEXT:     cir.call @_ZNSt14suspend_always12await_resumeEv(%[[SUSPEND_PTR]]) nothrow : (!cir.ptr<![[SuspendAlways]]>) -> ()
// CIR-NEXT:     cir.yield
// CIR-NEXT:   },)
// CIR-NEXT: }

// CIR: cir.await(final, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: },)

// CHECK: }

folly::coro::Task<int> go(int const& val);
folly::coro::Task<int> go1() {
  auto task = go(1);
  co_return co_await task;
}

// CIR: cir.func coroutine {{.*}} @_Z3go1v() {{.*}} ![[IntTask]]
// CIR: %[[IntTaskAddr:.*]] = cir.alloca ![[IntTask]], !cir.ptr<![[IntTask]]>, ["task", init]

// CIR:   cir.await(init, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR: }

// The call to go(1) has its own scope due to full-expression rules.
// CIR: cir.scope {
// CIR:   %[[OneAddr:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["ref.tmp1", init] {alignment = 4 : i64}
// CIR:   %[[One:.*]] = cir.const #cir.int<1> : !s32i
// CIR:   cir.store{{.*}} %[[One]], %[[OneAddr]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[IntTaskTmp:.*]] = cir.call @_Z2goRKi(%[[OneAddr]]) : (!cir.ptr<!s32i>) -> ![[IntTask]]
// CIR:   cir.store{{.*}} %[[IntTaskTmp]], %[[IntTaskAddr]] : ![[IntTask]], !cir.ptr<![[IntTask]]>
// CIR: }

// CIR: %[[CoReturnValAddr:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["__coawait_resume_rval"] {alignment = 1 : i64}
// CIR: cir.await(user, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR:   %[[ResumeVal:.*]] = cir.call @_ZN5folly4coro4TaskIiE12await_resumeEv(%3)
// CIR:   cir.store{{.*}} %[[ResumeVal]], %[[CoReturnValAddr]] : !s32i, !cir.ptr<!s32i>
// CIR: },)
// CIR: %[[V:.*]] = cir.load{{.*}} %[[CoReturnValAddr]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi({{.*}}, %[[V]])


folly::coro::Task<int> go1_lambda() {
  auto task = []() -> folly::coro::Task<int> {
    co_return 1;
  }();
  co_return co_await task;
}

// CIR: cir.func coroutine {{.*}} @_ZZ10go1_lambdavENK3$_0clEv{{.*}} ![[IntTask]]
// CIR: cir.func coroutine {{.*}} @_Z10go1_lambdav() {{.*}} ![[IntTask]]

folly::coro::Task<int> go4() {
  auto* fn = +[](int const& i) -> folly::coro::Task<int> { co_return i; };
  auto task = fn(3);
  co_return co_await std::move(task);
}

// CIR: cir.func coroutine {{.*}} @_Z3go4v() {{.*}} ![[IntTask]]

// CIR:   cir.await(init, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR: }

// CIR: %[[RES:.*]] = cir.scope {
// CIR:   %[[LAMBDA:.*]] = cir.alloca !rec_anon2E2, !cir.ptr<!rec_anon2E2>, ["ref.tmp1"] {alignment = 1 : i64}

// Get the lambda invoker ptr via `lambda operator folly::coro::Task<int> (*)(int const&)()`
// CIR:   %[[INVOKER:.*]] = cir.call @_ZZ3go4vENK3$_0cvPFN5folly4coro4TaskIiEERKiEEv(%[[LAMBDA]]) nothrow : (!cir.ptr<!rec_anon2E2>) -> !cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>
// CIR:   %[[PLUS:.*]] = cir.unary(plus, %[[INVOKER]]) : !cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>, !cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>
// CIR:   cir.yield %[[PLUS]] : !cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>
// CIR: }
// CIR: cir.store{{.*}} %[[RES]], %[[PTR_TASK:.*]] : !cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>, !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>>
// CIR: cir.scope {
// CIR:   %[[ARG:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["ref.tmp2", init] {alignment = 4 : i64}
// CIR:   %[[LAMBDA2:.*]] = cir.load{{.*}} %[[PTR_TASK]] : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>>, !cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>
// CIR:   %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
// CIR:   cir.store{{.*}} %[[THREE]], %[[ARG]] : !s32i, !cir.ptr<!s32i>

// Call invoker, which calls operator() indirectly.
// CIR:   %[[RES:.*]] = cir.call %[[LAMBDA2]](%[[ARG]]) : (!cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>, !cir.ptr<!s32i>) -> ![[IntTask]]
// CIR:   cir.store{{.*}} %[[RES]], %4 : ![[IntTask]], !cir.ptr<![[IntTask]]>
// CIR: }

// CIR:   cir.await(user, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR: }
