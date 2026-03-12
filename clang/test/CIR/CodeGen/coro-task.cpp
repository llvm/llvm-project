// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm  -disable-llvm-passes %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=OGCG

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

// OGCG-DAG: %[[VoidPromisse:"struct.folly::coro::Task<void>::promise_type"]] = type { i8 }
// OGCG-DAG: %[[VoidTask:"struct.folly::coro::Task"]] = type { i8 }
// OGCG-DAG: %[[SuspendAlways:"struct.std::suspend_always"]] = type { i8 }

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
// CIR: %[[SuspendAlwaysAddr:.*]] = cir.alloca ![[SuspendAlways]], {{.*}} ["ref.tmp0"]
// CIR: %[[CoroHandleVoidAddr:.*]] = cir.alloca ![[CoroHandleVoid]], {{.*}} ["agg.tmp0"]
// CIR: %[[CoroHandlePromiseAddr:.*]] = cir.alloca ![[CoroHandlePromiseVoid]], {{.*}} ["agg.tmp1"]

// OGCG: %[[VoidPromisseAddr:.*]] = alloca %[[VoidPromisse]], align 1
// OGCG: %[[VoidTaskAddr:.*]] = alloca %[[VoidTask]], align 1
// OGCG: %[[SuspendAlwaysAddr:.*]] = alloca %[[SuspendAlways]], align 1

// Get coroutine id with __builtin_coro_id.

// CIR: %[[NullPtr:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR: %[[Align:.*]] = cir.const #cir.int<16> : !u32i
// CIR: %[[CoroId:.*]] = cir.call @__builtin_coro_id(%[[Align]], %[[NullPtr]], %[[NullPtr]], %[[NullPtr]])

// OGCG: %[[CoroId:.*]] = call token @llvm.coro.id(i32 16, ptr %[[VoidPromisseAddr]], ptr null, ptr null)

// Perform allocation calling operator 'new' depending on __builtin_coro_alloc and
// call __builtin_coro_begin for the final coroutine frame address.

// CIR: %[[ShouldAlloc:.*]] = cir.call @__builtin_coro_alloc(%[[CoroId]]) : (!u32i) -> !cir.bool
// CIR: cir.store{{.*}} %[[NullPtr]], %[[SavedFrameAddr]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR: cir.if %[[ShouldAlloc]] {
// CIR:   %[[CoroSize:.*]] = cir.call @__builtin_coro_size() : () -> (!u64i {llvm.noundef})
// CIR:   %[[AllocAddr:.*]] = cir.call @_Znwm(%[[CoroSize]]) {allocsize = array<i32: 0>} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.noundef})
// CIR:   cir.store{{.*}} %[[AllocAddr]], %[[SavedFrameAddr]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR: }
// CIR: %[[Load0:.*]] = cir.load{{.*}} %[[SavedFrameAddr]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR: %[[CoroFrameAddr:.*]] = cir.call @__builtin_coro_begin(%[[CoroId]], %[[Load0]])

// OGCG: %[[ShouldAlloc:.*]]  = call i1 @llvm.coro.alloc(token %[[CoroId]])
// OGCG: br i1 %[[ShouldAlloc]], label %coro.alloc, label %coro.init

// OGCG: coro.alloc:
// OGCG:   %[[CoroSize:.*]] = call i64 @llvm.coro.size.i64()
// OGCG:   %[[Alloc_Frame:.*]] = call noalias noundef nonnull ptr @_Znwm(i64 noundef %[[CoroSize]])
// OGCG:   br label %coro.init

// OGCG: coro.init:
// OGCG:   %[[PtrToFramr:.*]] = phi ptr [ null, %entry ], [ %[[Alloc_Frame]], %coro.alloc ]
// OGCG:   %[[CoroFrameAddr:.*]] = call ptr @llvm.coro.begin(token %[[CoroId]], ptr %[[PtrToFramr]])

// Call promise.get_return_object() to retrieve the task object.

// CIR: %[[RetObj:.*]] = cir.call @_ZN5folly4coro4TaskIvE12promise_type17get_return_objectEv(%[[VoidPromisseAddr]]) nothrow : {{.*}} -> ![[VoidTask]]
// CIR: cir.store{{.*}} %[[RetObj]], %[[VoidTaskAddr]] : ![[VoidTask]]

// OGCG: call void @llvm.lifetime.start.p0(ptr %[[VoidPromisseAddr]])
// OGCG: call void @_ZN5folly4coro4TaskIvE12promise_type17get_return_objectEv(ptr noundef nonnull align 1 dereferenceable(1) %[[VoidPromisseAddr]])
// OGCG: call void @llvm.lifetime.start.p0(ptr %[[SuspendAlwaysAddr]])

// Start a new scope for the actual codegen for co_await, create temporary allocas for
// holding coroutine handle and the suspend_always struct.

// Effectively execute `coawait promise_type::initial_suspend()` by calling initial_suspend() and getting
// the suspend_always struct to use for cir.await. Note that we return by-value since we defer ABI lowering
// to later passes, same is done elsewhere.

// CIR: %[[Tmp0:.*]] = cir.call @_ZN5folly4coro4TaskIvE12promise_type15initial_suspendEv(%[[VoidPromisseAddr]])
// CIR: cir.store{{.*}} %[[Tmp0:.*]], %[[SuspendAlwaysAddr]]

// OGCG: call void @_ZN5folly4coro4TaskIvE12promise_type15initial_suspendEv(ptr noundef nonnull align 1 dereferenceable(1) %[[VoidPromisseAddr]])

//
// Here we start mapping co_await to cir.await.
//

// First regions `ready` has a special cir.yield code to veto suspension.

// CIR: cir.await(init, ready : {
// CIR:   %[[ReadyVeto:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[SuspendAlwaysAddr]])
// CIR:   cir.condition(%[[ReadyVeto]])

// OGCG: %[[Tmp0:.*]] = call noundef zeroext i1 @_ZNSt14suspend_always11await_readyEv(ptr noundef nonnull align 1 dereferenceable(1) %[[SuspendAlwaysAddr]])
// OGCG: br i1 %[[Tmp0]], label %init.ready, label %init.suspend

// Second region `suspend` contains the actual suspend logic.
//
// - Start by getting the coroutine handle using from_address().
// - Implicit convert coroutine handle from task specific promisse
//   specialization to a void one.
// - Call suspend_always::await_suspend() passing the handle.
//
// FIXME: add veto support for non-void await_suspends.

// CIR: }, suspend : {
// CIR:   %[[FromAddrRes:.*]] = cir.call @_ZNSt16coroutine_handleIN5folly4coro4TaskIvE12promise_typeEE12from_addressEPv(%[[CoroFrameAddr]])
// CIR:   cir.store{{.*}} %[[FromAddrRes]], %[[CoroHandlePromiseAddr]] : ![[CoroHandlePromiseVoid]]
// CIR:   %[[CoroHandlePromiseReload:.*]] = cir.load{{.*}} %[[CoroHandlePromiseAddr]]
// CIR:   cir.call @_ZNSt16coroutine_handleIvEC1IN5folly4coro4TaskIvE12promise_typeEEES_IT_E(%[[CoroHandleVoidAddr]], %[[CoroHandlePromiseReload]])
// CIR:   %[[CoroHandleVoidReload:.*]] = cir.load{{.*}} %[[CoroHandleVoidAddr]] : !cir.ptr<![[CoroHandleVoid]]>, ![[CoroHandleVoid]]
// CIR:   cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%[[SuspendAlwaysAddr]], %[[CoroHandleVoidReload]])
// CIR:   cir.yield

// OGCG: init.suspend:
// OGCG:   %[[Save:.*]] = call token @llvm.coro.save(ptr null)
// OGCG:   call void @llvm.coro.await.suspend.void(ptr %[[SuspendAlwaysAddr]], ptr %[[CoroFrameAddr]], ptr @_Z10silly_taskv.__await_suspend_wrapper__init)
// OGCG:   %[[TMP1:.*]] = call i8 @llvm.coro.suspend(token %[[Save]], i1 false)
// OGCG:   switch i8 %[[TMP1]], label %coro.ret [
// OGCG:     i8 0, label %init.ready
// OGCG:     i8 1, label %init.cleanup
// OGCG:   ]

// Third region `resume` handles coroutine resuming logic.

// CIR: }, resume : {
// CIR:   cir.call @_ZNSt14suspend_always12await_resumeEv(%[[SuspendAlwaysAddr]])
// CIR:   cir.yield
// CIR: },)

// OGCG: init.ready:
// OGCG:   call void @_ZNSt14suspend_always12await_resumeEv(ptr noundef nonnull align 1 dereferenceable(1) %[[SuspendAlwaysAddr]]
// OGCG:   br label %cleanup

// Since we already tested cir.await guts above, the remaining checks for:
// - The actual user written co_await
// - The promise call
// - The final suspend co_await
// - Return

// The actual user written co_await
// CIR: cir.await(user, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: },)

// OGCG: cleanup.cont
// OGCG: await.suspend:
// OGCG: await.ready:

// The promise call
// CHECK: cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(%[[VoidPromisseAddr]])

// OGCG: call void @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(ptr noundef nonnull align 1 dereferenceable(1) %[[VoidPromisseAddr]])

// The final suspend co_await
// CIR: cir.await(final, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: },)

// OGCG: coro.final:
// OGCG: final.suspend:
// OGCG: final.ready:

// Call builtin coro end and return

// CIR: %[[CoroEndArg0:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR: %[[CoroEndArg1:.*]] = cir.const #false
// CIR: = cir.call @__builtin_coro_end(%[[CoroEndArg0]], %[[CoroEndArg1]])

// CIR: %[[Tmp1:.*]] = cir.load{{.*}} %[[VoidTaskAddr]]
// CIR: cir.return %[[Tmp1]]
// CIR: }

// OGCG: coro.ret:
// OGCG:   call void @llvm.coro.end(ptr null, i1 false, token none)
// OGCG:   ret void

folly::coro::Task<int> byRef(const std::string& s) {
  co_return s.size();
}

// CIR:  cir.func coroutine {{.*}} @_Z5byRefRKSt6string(%[[ARG:.*]]: !cir.ptr<![[StdString]]> {{.*}}) -> ![[IntTask]]
// CIR:    %[[AllocaParam:.*]] = cir.alloca !cir.ptr<![[StdString]]>, {{.*}}, ["s", init, const]
// CIR:    %[[IntTaskAddr:.*]] = cir.alloca ![[IntTask]], {{.*}}, ["__retval"]
// CIR:    %[[SavedFrameAddr:.*]]  = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["__coro_frame_addr"]
// CIR:    %[[AllocaFnUse:.*]] = cir.alloca !cir.ptr<![[StdString]]>, {{.*}}, ["s", init, const]
// CIR:    %[[IntPromisseAddr:.*]] = cir.alloca ![[IntPromisse]], {{.*}}, ["__promise"]
// CIR:    %[[SuspendAlwaysAddr:.*]] = cir.alloca ![[SuspendAlways]], {{.*}} ["ref.tmp0"] {alignment = 1 : i64}
// CIR:    %[[CoroHandleVoidAddr:.*]] = cir.alloca ![[CoroHandleVoid]], {{.*}} ["agg.tmp0"] {alignment = 1 : i64}
// CIR:    %[[CoroHandlePromiseAddr:.*]] = cir.alloca ![[CoroHandlePromiseInt]], {{.*}} ["agg.tmp1"] {alignment = 1 : i64}
// CIR:    cir.store %[[ARG]], %[[AllocaParam]] : !cir.ptr<![[StdString]]>, {{.*}}

// Call promise.get_return_object() to retrieve the task object.

// CIR:    %[[LOAD:.*]] = cir.load %[[AllocaParam]] : !cir.ptr<!cir.ptr<![[StdString]]>>, !cir.ptr<![[StdString]]>
// CIR:    cir.store {{.*}} %[[LOAD]], %[[AllocaFnUse]] : !cir.ptr<![[StdString]]>, !cir.ptr<!cir.ptr<![[StdString]]>>
// CIR:    %[[RetObj:.*]] = cir.call @_ZN5folly4coro4TaskIiE12promise_type17get_return_objectEv(%[[IntPromisseAddr]]) nothrow : {{.*}} -> ![[IntTask]]
// CIR:    cir.store {{.*}} %[[RetObj]], %[[IntTaskAddr]] : ![[IntTask]]
// CIR:    %[[Tmp0:.*]] = cir.call @_ZN5folly4coro4TaskIiE12promise_type15initial_suspendEv(%[[IntPromisseAddr]])
// CIR:    cir.store{{.*}} %[[Tmp0]], %[[SuspendAlwaysAddr]]
// CIR:    cir.await(init, ready : {
// CIR:      %[[TmpCallRes:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[SuspendAlwaysAddr]])
// CIR:      cir.condition(%[[TmpCallRes]])
// CIR:    }, suspend : {
// CIR:      %[[FromAddrRes:.*]] = cir.call @_ZNSt16coroutine_handleIN5folly4coro4TaskIiE12promise_typeEE12from_addressEPv(%[[CoroFrameAddr:.*]])
// CIR:      cir.store{{.*}} %[[FromAddrRes]], %[[CoroHandlePromiseAddr]] : ![[CoroHandlePromiseInt]]
// CIR:      %[[CoroHandlePromiseReload:.*]] = cir.load{{.*}} %[[CoroHandlePromiseAddr]]
// CIR:      cir.call @_ZNSt16coroutine_handleIvEC1IN5folly4coro4TaskIiE12promise_typeEEES_IT_E(%[[CoroHandleVoidAddr]], %[[CoroHandlePromiseReload]])
// CIR:      %[[CoroHandleVoidReload:.*]] = cir.load{{.*}} %[[CoroHandleVoidAddr]] : !cir.ptr<![[CoroHandleVoid]]>, ![[CoroHandleVoid]]
// CIR:      cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%[[SuspendAlwaysAddr]], %[[CoroHandleVoidReload]])
// CIR:      cir.yield
// CIR:    }, resume : {
// CIR:      cir.call @_ZNSt14suspend_always12await_resumeEv(%[[SuspendAlwaysAddr]])
// CIR:      cir.yield
// CIR:    },)

// can't fallthrough
// CIR-NOT:   cir.await(user

// The final suspend co_await
// CIR: cir.await(final, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: },)

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
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: },)
// CIR: cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv
// CIR-NOT: cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv
// CIR: cir.await(final, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: },)

folly::coro::Task<void> yield();
folly::coro::Task<void> yield1() {
  auto t = yield();
  co_yield t;
}

// CIR: cir.func coroutine {{.*}} @_Z6yield1v() -> !rec_folly3A3Acoro3A3ATask3Cvoid3E

// Prologue allocas (still present in output)
// CIR-DAG: %[[RETVAL:.*]] = cir.alloca ![[VoidTask]], {{.*}} ["__retval"]
// CIR-DAG: %[[FRAME:.*]] = cir.alloca !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>, ["__coro_frame_addr"]
// CIR-DAG: %[[PROMISE:.*]] = cir.alloca ![[VoidPromisse]], {{.*}} ["__promise"]
// CIR-DAG: %[[SUSP0:.*]] = cir.alloca ![[SuspendAlways]], {{.*}} ["ref.tmp0"]
// CIR-DAG: %[[CH_VOID0:.*]] = cir.alloca ![[CoroHandleVoid]], {{.*}} ["agg.tmp0"]
// CIR-DAG: %[[CH_PROM0:.*]] = cir.alloca ![[CoroHandlePromiseVoid]], {{.*}} ["agg.tmp1"]
// CIR-DAG: %[[T_ADDR:.*]] = cir.alloca ![[VoidTask]], {{.*}} ["t", init]
// CIR-DAG: %[[SUSP1:.*]] = cir.alloca ![[SuspendAlways]], {{.*}} ["ref.tmp1"]
// CIR-DAG: %[[AWAITER_COPY_ADDR:.*]] = cir.alloca ![[VoidTask]], {{.*}} ["agg.tmp2"]
// CIR-DAG: %[[CH_VOID1:.*]] = cir.alloca ![[CoroHandleVoid]], {{.*}} ["agg.tmp3"]
// CIR-DAG: %[[CH_PROM1:.*]] = cir.alloca ![[CoroHandlePromiseVoid]], {{.*}} ["agg.tmp4"]
// CIR-DAG: %[[SUSP2:.*]] = cir.alloca ![[SuspendAlways]], {{.*}} ["ref.tmp2"]
// CIR-DAG: %[[CH_VOID2:.*]] = cir.alloca ![[CoroHandleVoid]], {{.*}} ["agg.tmp5"]
// CIR-DAG: %[[CH_PROM2:.*]] = cir.alloca ![[CoroHandlePromiseVoid]], {{.*}} ["agg.tmp6"]

// initial_suspend + await(init)
// CIR: %[[INIT_SUSP:.*]] = cir.call @_ZN5folly4coro4TaskIvE12promise_type15initial_suspendEv(%[[PROMISE]]){{.*}}
// CIR: cir.store{{.*}} %[[INIT_SUSP]], %[[SUSP0]]
// CIR: cir.await(init, ready : {
// CIR:   %[[READY0:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[SUSP0]]){{.*}}
// CIR:   cir.condition(%[[READY0]])
// CIR: }, suspend : {
// CIR:   %[[FROMADDR0:.*]] = cir.call @_ZNSt16coroutine_handleIN5folly4coro4TaskIvE12promise_typeEE12from_addressEPv(%{{.*}}){{.*}}
// CIR:   cir.store{{.*}} %[[FROMADDR0]], %[[CH_PROM0]]
// CIR:   %[[PROM_RELOAD0:.*]] = cir.load{{.*}} %[[CH_PROM0]]
// CIR:   cir.call @_ZNSt16coroutine_handleIvEC1IN5folly4coro4TaskIvE12promise_typeEEES_IT_E(%[[CH_VOID0]], %[[PROM_RELOAD0]]){{.*}}
// CIR:   %[[VOID_RELOAD0:.*]] = cir.load{{.*}} %[[CH_VOID0]]
// CIR:   cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%[[SUSP0]], %[[VOID_RELOAD0]]){{.*}}
// CIR:   cir.yield
// CIR: }, resume : {
// CIR:   cir.call @_ZNSt14suspend_always12await_resumeEv(%[[SUSP0]]){{.*}}
// CIR:   cir.yield
// CIR: },)

// yield_value + await(yield)
// CIR: %[[YIELD_TASK:.*]] = cir.call @_Z5yieldv(){{.*}}
// CIR: cir.store{{.*}} %[[YIELD_TASK]], %[[T_ADDR]]
// CIR: cir.copy %[[T_ADDR]] to %[[AWAITER_COPY_ADDR]]
// CIR: %[[AWAITER:.*]] = cir.load{{.*}} %[[AWAITER_COPY_ADDR]]
// CIR: %[[YIELD_SUSP:.*]] = cir.call @_ZN5folly4coro4TaskIvE12promise_type11yield_valueES2_(%[[PROMISE]], %[[AWAITER]]){{.*}}
// CIR: cir.store{{.*}} %[[YIELD_SUSP]], %[[SUSP1]]
// CIR: cir.await(yield, ready : {
// CIR:   %[[READY1:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[SUSP1]]){{.*}}
// CIR:   cir.condition(%[[READY1]])
// CIR: }, suspend : {
// CIR:   %[[FROMADDR1:.*]] = cir.call @_ZNSt16coroutine_handleIN5folly4coro4TaskIvE12promise_typeEE12from_addressEPv(%{{.*}}){{.*}}
// CIR:   cir.store{{.*}} %[[FROMADDR1]], %[[CH_PROM1]]
// CIR:   %[[PROM_RELOAD1:.*]] = cir.load{{.*}} %[[CH_PROM1]]
// CIR:   cir.call @_ZNSt16coroutine_handleIvEC1IN5folly4coro4TaskIvE12promise_typeEEES_IT_E(%[[CH_VOID1]], %[[PROM_RELOAD1]]){{.*}}
// CIR:   %[[VOID_RELOAD1:.*]] = cir.load{{.*}} %[[CH_VOID1]]
// CIR:   cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%[[SUSP1]], %[[VOID_RELOAD1]]){{.*}}
// CIR:   cir.yield
// CIR: }, resume : {
// CIR:   cir.call @_ZNSt14suspend_always12await_resumeEv(%[[SUSP1]]){{.*}}
// CIR:   cir.yield
// CIR: },)

// return_void + await(final)
// CIR: cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(%[[PROMISE]]){{.*}}
// CIR: %[[FINAL_SUSP:.*]] = cir.call @_ZN5folly4coro4TaskIvE12promise_type13final_suspendEv(%[[PROMISE]]){{.*}}
// CIR: cir.store{{.*}} %[[FINAL_SUSP]], %[[SUSP2]]
// CIR: cir.await(final, ready : {
// CIR:   %[[READY2:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[SUSP2]]){{.*}}
// CIR:   cir.condition(%[[READY2]])
// CIR: }, suspend : {
// CIR:   %[[FROMADDR2:.*]] = cir.call @_ZNSt16coroutine_handleIN5folly4coro4TaskIvE12promise_typeEE12from_addressEPv(%{{.*}}){{.*}}
// CIR:   cir.store{{.*}} %[[FROMADDR2]], %[[CH_PROM2]]
// CIR:   %[[PROM_RELOAD2:.*]] = cir.load{{.*}} %[[CH_PROM2]]
// CIR:   cir.call @_ZNSt16coroutine_handleIvEC1IN5folly4coro4TaskIvE12promise_typeEEES_IT_E(%[[CH_VOID2]], %[[PROM_RELOAD2]]){{.*}}
// CIR:   %[[VOID_RELOAD2:.*]] = cir.load{{.*}} %[[CH_VOID2]]
// CIR:   cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%[[SUSP2]], %[[VOID_RELOAD2]]){{.*}}
// CIR:   cir.yield
// CIR: }, resume : {
// CIR:   cir.call @_ZNSt14suspend_always12await_resumeEv(%[[SUSP2]]){{.*}}
// CIR:   cir.yield
// CIR: },)
// CIR: = cir.call @__builtin_coro_end(%{{.*}}, %{{.*}}){{.*}}
// CIR: %[[RETLOAD:.*]] = cir.load{{.*}} %[[RETVAL]]
// CIR: cir.return %[[RETLOAD]]
// CIR: }

// CHECK: }

folly::coro::Task<int> go(int const& val);
folly::coro::Task<int> go1() {
  auto task = go(1);
  co_return co_await task;
}

// CIR: cir.func coroutine {{.*}} @_Z3go1v() {{.*}} ![[IntTask]]
// CIR: %[[IntTaskAddr:.*]] = cir.alloca ![[IntTask]], !cir.ptr<![[IntTask]]>, ["task", init]

// CIR: cir.await(init, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: },)

// The call to go(1) has its own scope due to full-expression rules.
// CIR: cir.scope {
// CIR:   %[[OneAddr:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["ref.tmp1", init] {alignment = 4 : i64}
// CIR:   %[[One:.*]] = cir.const #cir.int<1> : !s32i
// CIR:   cir.store{{.*}} %[[One]], %[[OneAddr]] : !s32i, !cir.ptr<!s32i>
// CIR:   %[[IntTaskTmp:.*]] = cir.call @_Z2goRKi(%[[OneAddr]]) : (!cir.ptr<!s32i>{{.*}}) -> ![[IntTask]]
// CIR:   cir.store{{.*}} %[[IntTaskTmp]], %[[IntTaskAddr]] : ![[IntTask]], !cir.ptr<![[IntTask]]>
// CIR: }

// CIR: cir.await(user, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: %[[ResumeVal:.*]] = cir.call @_ZN5folly4coro4TaskIiE12await_resumeEv(%[[IntTaskAddr]])
// CIR: cir.store{{.*}} %[[ResumeVal]], %[[CoReturnValAddr:.*]] : !s32i, !cir.ptr<!s32i>
// CIR: },)
// CIR: %[[V:.*]] = cir.load{{.*}} %[[CoReturnValAddr]] : !cir.ptr<!s32i>, !s32i
// CIR: cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi({{.*}}, %[[V]])

// CIR: cir.await(final, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: },)


folly::coro::Task<int> go1_lambda() {
  auto task = []() -> folly::coro::Task<int> {
    co_return 1;
  }();
  co_return co_await task;
}

// CIR: cir.func coroutine {{.*}} @_ZZ10go1_lambdavENK3$_0clEv{{.*}} ![[IntTask]]
// CIR: cir.await(init, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: },)
// CIR: cir.await(final, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: },)
// CIR: cir.func coroutine {{.*}} @_Z10go1_lambdav() {{.*}} ![[IntTask]]
// CIR: cir.await(init, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: },)
// CIR: cir.await(user, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: },)
// CIR: cir.await(final, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: },)

folly::coro::Task<int> go4() {
  auto* fn = +[](int const& i) -> folly::coro::Task<int> { co_return i; };
  auto task = fn(3);
  co_return co_await std::move(task);
}

// CIR: cir.func coroutine{{.*}} @_ZZ3go4vENK3$_0clERKi(
// CIR: cir.await(init, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: },)
// CIR: cir.await(final, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: },)

// CIR: cir.func coroutine {{.*}} @_Z3go4v() {{.*}} ![[IntTask]]

// CIR: cir.await(init, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: },)

// Get the lambda invoker ptr via `lambda operator folly::coro::Task<int> (*)(int const&)()`
// CIR: %[[INVOKER:.*]] = cir.call @_ZZ3go4vENK3$_0cvPFN5folly4coro4TaskIiEERKiEEv(%{{.*}}) nothrow : {{.*}} -> (!cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>> {llvm.noundef})
// CIR: %[[PLUS:.*]] = cir.unary(plus, %[[INVOKER]]) : !cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>, !cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>
// CIR: cir.store{{.*}} %[[PLUS]], %[[FN_ADDR:.*]] : !cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>, !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>>
// CIR: cir.scope {
// CIR:   %[[ARG:.*]] = cir.alloca !s32i, !cir.ptr<!s32i>, ["ref.tmp2", init] {alignment = 4 : i64}
// CIR:   %[[FN:.*]] = cir.load{{.*}} %[[FN_ADDR]] : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>>, !cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>
// CIR:   %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
// CIR:   cir.store{{.*}} %[[THREE]], %[[ARG]] : !s32i, !cir.ptr<!s32i>

// Call invoker, which calls operator() indirectly.
// CIR:   %[[CALLRES:.*]] = cir.call %[[FN]](%[[ARG]]) : (!cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>, !cir.ptr<!s32i> {{.*}}) -> ![[IntTask]]
// CIR:   cir.store{{.*}} %[[CALLRES]], %[[TASK_ADDR:.*]] : ![[IntTask]], !cir.ptr<![[IntTask]]>
// CIR: }

// CIR: cir.await(user, ready : {
// CIR:   = cir.call @_ZN5folly4coro4TaskIiE11await_readyEv(%[[TASK_ADDR]])
// CIR:   cir.condition(
// CIR: }, suspend : {
// CIR:   cir.yield
// CIR: }, resume : {
// CIR:   cir.yield
// CIR: },)

// CIR: cir.await(final, ready : {
// CIR: }, suspend : {
// CIR: }, resume : {
// CIR: },)

// OGCG: define {{.*}}__await_suspend_wrapper__init(ptr noundef nonnull %[[Awaiter:.*]], ptr noundef %[[Handle:.*]])
// OGCG: entry:
// OGCG:   %[[AwaiterAddr:.*]] = alloca ptr
// OGCG:   %[[HandleAddr:.*]] = alloca ptr
// OGCG:   %[[CoroHandleVoidAddr:.*]] = alloca %"struct.std::coroutine_handle"
// OGCG:   store ptr %[[Awaiter:.*]], ptr %[[AwaiterAddr]]
// OGCG:   store ptr %[[Handle:.*]], ptr %[[HandleAddr]]
// OGCG:   %[[AwaiterReload:.*]] = load ptr, ptr %[[AwaiterAddr]]
// OGCG:   %[[CoroFrameAddr:.*]] = load ptr, ptr %[[HandleAddr]]
// OGCG:   call void @_ZNSt16coroutine_handleIN5folly4coro4TaskIvE12promise_typeEE12from_addressEPv(ptr noundef %[[CoroFrameAddr]])
// OGCG:   call void @_ZNSt16coroutine_handleIvEC1IN5folly4coro4TaskIvE12promise_typeEEES_IT_E(ptr noundef nonnull align 1 dereferenceable(1) %[[CoroHandleVoidAddr]])
// OGCG:   call void @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(ptr noundef nonnull align 1 dereferenceable(1) %[[AwaiterReload]])
// OGCG:   ret void
// OGCG: }
