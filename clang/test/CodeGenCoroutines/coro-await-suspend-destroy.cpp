// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   -disable-llvm-passes | FileCheck %s --check-prefix=CHECK
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm -o - %s \
// RUN:   -O2 | FileCheck %s --check-prefix=CHECK-OPT

// See `SemaCXX/coro-await-suspend-destroy-errors.cpp` for error checks.

#include "Inputs/coroutine.h"

// This is used to implement a few `await_suspend()`s annotated with the
// [[clang::coro_await_suspend_destroy]] attribute. As a consequence, it is only
// test-called, never emitted.
//
// The `operator new()` is meant to fail subsequent "no allocation" checks if
// this does get emitted.
//
// It is followed by the recommended `await_suspend` stub, to check it compiles.
#define STUB_AWAIT_SUSPEND(handle) \
    operator new(1); \
    await_suspend_destroy(handle.promise()); \
    handle.destroy()

// Use a dynamic `await_ready()` to ensure the suspend branch cannot be
// optimized away. Implements everything but `await_suspend()`.
struct BaseAwaiter {
  bool ready_;
  bool await_ready() { return ready_; }
  void await_resume() {}
  BaseAwaiter(bool ready) : ready_{ready} {}
};

// For a coroutine function to be a short-circuiting function, it needs a
// coroutine type with `std::suspend_never` for initial/final suspend
template <typename TaskT>
struct BasePromiseType {
  TaskT get_return_object() { return {}; }
  std::suspend_never initial_suspend() { return {}; }
  std::suspend_never final_suspend() noexcept { return {}; }
  void return_void() {}
  void unhandled_exception() {}
};

// The coros look the same, but `MaybeSuspendingAwaiter` handles them differently.
struct NonSuspendingTask {
  struct promise_type : BasePromiseType<NonSuspendingTask> {};
};
struct MaybeSuspendingTask {
  struct promise_type : BasePromiseType<MaybeSuspendingTask> {};
};

// When a coro only uses short-circuiting awaiters, it should elide allocations.
//   - `DestroyingAwaiter` is always short-circuiting
//   - `MaybeSuspendingAwaiter` short-circuits only in `NonSuspendingTask`

struct DestroyingAwaiter : BaseAwaiter {
  void await_suspend_destroy(auto& promise) {}
  [[clang::coro_await_suspend_destroy]]
  void await_suspend(auto handle) { STUB_AWAIT_SUSPEND(handle); }
};

struct MaybeSuspendingAwaiter : BaseAwaiter {
  // Without the attribute, the coro will use `await.suspend` intrinsics, which
  // currently trigger heap allocations for coro frames. Since the body isn't
  // visible, escape analysis should prevent heap elision.
  void await_suspend(std::coroutine_handle<MaybeSuspendingTask::promise_type>);

  void await_suspend_destroy(NonSuspendingTask::promise_type&) {}
  [[clang::coro_await_suspend_destroy]]
  void await_suspend(std::coroutine_handle<NonSuspendingTask::promise_type> h) {
    STUB_AWAIT_SUSPEND(h);
  }
};

// Should result in no allocation after optimization.
NonSuspendingTask test_single_destroying_await(bool ready) {
  co_await DestroyingAwaiter{ready};
}

// The reason this first `CHECK` test is so long is that it shows most of the
// unoptimized IR before coroutine lowering. The granular detail is provided per
// PR152623 code review, with the aim of helping future authors understand the
// intended control flow.
//
// This mostly shows the standard coroutine flow. Find **ATTRIBUTE-SPECIFIC** in
// the comments below to understand where the behavior diverges.

// Basic coro setup

// CHECK-LABEL: define{{.*}} void @_Z28test_single_destroying_awaitb
// CHECK: entry:
// CHECK: %__promise = alloca %"struct.NonSuspendingTask::promise_type", align 1
// CHECK: %[[PROMISE:.+]] = bitcast ptr %__promise to ptr
// CHECK-NEXT: %[[CORO_ID:.+]] = call token @llvm.coro.id(i32 {{[0-9]+}}, ptr %[[PROMISE]],
// CHECK-NEXT: %[[USE_DYNAMIC_ALLOC:.+]] = call i1 @llvm.coro.alloc(token %[[CORO_ID]])
// CHECK-NEXT: br i1 %[[USE_DYNAMIC_ALLOC]], label %coro.alloc, label %coro.init

// Conditional heap alloc -- must be elided after lowering

// CHECK: coro.alloc: ; preds = %entry
// CHECK: call{{.*}} @_Znwm

// Init coro frame & handle initial suspend

// CHECK: coro.init: ; preds = %coro.alloc, %entry
// CHECK: %[[FRAME:.+]] = call ptr @llvm.coro.begin(token %[[CORO_ID]]
//
// CHECK: call{{.*}} @_ZN15BasePromiseTypeI17NonSuspendingTaskE15initial_suspendEv
// CHECK-NEXT: %[[INIT_SUSPEND_READY:.+]] = call{{.*}} i1 @_ZNSt13suspend_never11await_readyEv
// CHECK-NEXT: br i1 %[[INIT_SUSPEND_READY]], label %init.ready, label %init.suspend
//
// CHECK: init.suspend: ; preds = %coro.init
// ... implementation omitted, not reached ...
//
// CHECK: init.ready: ; preds = %init.suspend, %coro.init

// Handle the user-visible `co_await` suspend point:

// CHECK: %[[CO_AWAIT_READY:.+]] = call{{.*}} i1 @_ZN11BaseAwaiter11await_readyEv(
// CHECK-NEXT: br i1 %[[CO_AWAIT_READY]], label %await.ready, label %await.suspend

// **ATTRIBUTE-SPECIFIC**
//
// This `co_await`'s suspend is trivial & lacks suspend intrinsics. For cleanup
// we branch to the same location as `await_resume`, but diverge later.

// CHECK: await.suspend:
// CHECK-NEXT: call void @_Z28test_single_destroying_awaitb.__await_suspend_wrapper__await(ptr %{{.+}}, ptr %[[FRAME]])
// CHECK-NEXT: br label %[[CO_AWAIT_CLEANUP:.+]]

// When ready, call `await_resume` :

// CHECK: await.ready:
// CHECK-NEXT: call{{.*}} @_ZN11BaseAwaiter12await_resumeEv(ptr{{.*}} %{{.+}})
// CHECK-NEXT: br label %[[CO_AWAIT_CLEANUP]]

// Further cleanup is conditional on whether we did "ready" or "suspend":

// CHECK: [[CO_AWAIT_CLEANUP]]: ; preds = %await.ready, %await.suspend
// CHECK-NEXT: %[[CLEANUP_PHI:.+]] = phi i32 [ 0, %await.ready ], [ 2, %await.suspend ]
// CHECK: switch i32 %[[CLEANUP_PHI]], label %[[ON_AWAIT_SUSPEND:.+]] [
// CHECK: i32 0, label %[[ON_AWAIT_READY:.+]]
// CHECK: ]

// On "ready", we `co_return` and do final suspend (not shown).

// CHECK: [[ON_AWAIT_READY]]: ; preds = %[[CO_AWAIT_CLEANUP]]
// CHECK-NEXT: call void @_ZN15BasePromiseTypeI17NonSuspendingTaskE11return_voidEv(
// CHECK-NEXT: br label %coro.final
//
// CHECK: coro.final: ; preds = %[[ON_AWAIT_READY]]
//
// ... here, we handle final suspend, and eventually ...
//
// CHECK: br label %[[ON_AWAIT_SUSPEND]]

// This [[ON_AWAIT_SUSPEND]] is actually the "destroy scope" code path,
// including conditional `operator delete`, which will be elided.

// CHECK: [[ON_AWAIT_SUSPEND]]:
// CHECK: %[[HEAP_OR_NULL:.+]] = call ptr @llvm.coro.free(token %[[CORO_ID]], ptr %[[FRAME]])
// CHECK-NEXT: %[[NON_NULL:.+]] = icmp ne ptr %[[HEAP_OR_NULL]], null
// CHECK-NEXT: br i1 %[[NON_NULL]], label %coro.free, label %after.coro.free

// The `operator delete()` call will be removed by optimizations.

// CHECK: coro.free:
// CHECK-NEXT: %[[CORO_SIZE:.+]] = call i64 @llvm.coro.size.i64()
// CHECK-NEXT: call void @_ZdlPvm(ptr noundef %[[HEAP_OR_NULL]], i64 noundef %[[CORO_SIZE]])
// CHECK-NEXT: br label %after.coro.free

// CHECK: after.coro.free:
//
// ... Not shown: Coro teardown finishes, and if we handle normal return vs
// exception.

// Don't let the matchers skip past the end of `test_single_destroying_await()`

// CHECK: }

// The optimized IR is thankfully brief.

// CHECK-OPT: define{{.*}} void @_Z28test_single_destroying_awaitb({{.*}} {
// CHECK-OPT-NEXT: entry:
// CHECK-OPT-NEXT: ret void
// CHECK-OPT-NEXT: }

///////////////////////////////////////////////////////////////////////////////
// The subsequent tests variations on the above theme. For brevity, they do not
// repeat the above coroutine skeleton, but merely check for heap allocations.
///////////////////////////////////////////////////////////////////////////////

// Multiple `co_await`s, all with `coro_await_suspend_destroy`.
NonSuspendingTask test_multiple_destroying_awaits(bool ready, bool condition) {
  co_await DestroyingAwaiter{ready};
  co_await MaybeSuspendingAwaiter{ready}; // Destroys `NonSuspendingTask`
  if (condition) {
    co_await DestroyingAwaiter{ready};
  }
}

// The unlowered IR has heaps allocs, but the optimized IR does not.

// CHECK-LABEL: define{{.*}} void @_Z31test_multiple_destroying_awaitsb
// CHECK: call{{.*}} @_Znwm
// CHECK: call{{.*}} @_ZdlPvm
// CHECK: }

// CHECK-OPT-LABEL: define{{.*}} void @_Z31test_multiple_destroying_awaitsb
// CHECK-OPT-NOT: call{{.*}} @llvm.coro.alloc
// CHECK-OPT-NOT: call{{.*}} malloc
// CHECK-OPT-NOT: call{{.*}} @_Znwm
// CHECK-OPT: }

// Same behavior as `test_multiple_destroying_awaits`, but with a
// `MaybeSuspendingTask`, and without a `MaybeSuspendingAwaiter`.
NonSuspendingTask test_multiple_destroying_awaits_too(bool ready, bool condition) {
  co_await DestroyingAwaiter{ready};
  co_await MaybeSuspendingAwaiter{ready}; // Destroys `NonSuspendingTask`
  if (condition) {
    co_await DestroyingAwaiter{ready};
  }
}

// The unlowered IR has heaps allocs, but the optimized IR does not.

// CHECK-LABEL: define{{.*}} void @_Z35test_multiple_destroying_awaits_toob
// CHECK: call{{.*}} @_Znwm
// CHECK: call{{.*}} @_ZdlPvm
// CHECK: }

// CHECK-OPT-LABEL: define{{.*}} void @_Z35test_multiple_destroying_awaits_toob
// CHECK-OPT-NOT: call{{.*}} @llvm.coro.alloc
// CHECK-OPT-NOT: call{{.*}} malloc
// CHECK-OPT-NOT: call{{.*}} @_Znwm
// CHECK-OPT: }

// Mixed awaits - some with `coro_await_suspend_destroy`, some without.
MaybeSuspendingTask test_mixed_awaits(bool ready) {
  co_await MaybeSuspendingAwaiter{ready}; // Suspends `MaybeSuspendingTask`
  co_await DestroyingAwaiter{ready};
}

// Both the unlowered & optimized IR have a heap allocation because not all
// awaits destroy the coroutine.

// CHECK-INITIAL-LABEL: define{{.*}} void @_Z17test_mixed_awaitsb
// CHECK: call{{.*}} @_Znwm
// CHECK: call{{.*}} @_ZdlPvm
// CHECK: }

// CHECK-OPT-LABEL: define{{.*}} void @_Z17test_mixed_awaitsb
// CHECK-OPT: call{{.*}} @_Znwm
// CHECK-OPT: call{{.*}} @_ZdlPvm
// CHECK-OPT: }

MaybeSuspendingTask test_unreachable_normal_suspend(bool ready) {
  co_await DestroyingAwaiter{false};
  // Unreachable in OPTIMIZED, so those builds don't see an allocation.
  co_await MaybeSuspendingAwaiter{ready}; // Would suspend `MaybeSuspendingTask`
}

// The unlowered IR has heaps allocs, but the optimized IR does not, since
// `co_await DestroyingAwaiter{false}` is effectively a `co_return`.

// CHECK-LABEL: define{{.*}} void @_Z31test_unreachable_normal_suspendb
// CHECK: call{{.*}} @_Znwm
// CHECK: call{{.*}} @_ZdlPvm
// CHECK: }

// CHECK-OPT-LABEL: define{{.*}} void @_Z31test_unreachable_normal_suspendb
// CHECK-OPT-NOT: call{{.*}} @llvm.coro.alloc
// CHECK-OPT-NOT: call{{.*}} malloc
// CHECK-OPT-NOT: call{{.*}} @_Znwm
// CHECK-OPT: }

// Template awaitable with `coro_await_suspend_destroy` attribute. Checks for
// bugs where we don't handle dependent types appropriately.
template<typename T>
struct TemplateDestroyingAwaiter : BaseAwaiter {
  void await_suspend_destroy(auto& promise) {}
  [[clang::coro_await_suspend_destroy]]
  void await_suspend(auto handle) { STUB_AWAIT_SUSPEND(handle); }
};

template <typename T>
NonSuspendingTask test_template_destroying_await(bool ready) {
  co_await TemplateDestroyingAwaiter<T>{ready};
}

template NonSuspendingTask test_template_destroying_await<int>(bool ready);

// CHECK-LABEL: define{{.*}} void @_Z30test_template_destroying_awaitIiE17NonSuspendingTaskb
// CHECK: call{{.*}} @_Znwm
// CHECK: call{{.*}} @_ZdlPvm
// CHECK: }

// CHECK-OPT-LABEL: define{{.*}} void @_Z30test_template_destroying_awaitIiE17NonSuspendingTaskb
// CHECK-OPT-NOT: call{{.*}} @llvm.coro.alloc
// CHECK-OPT-NOT: call{{.*}} malloc
// CHECK-OPT-NOT: call{{.*}} @_Znwm
// CHECK-OPT: }
