// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: cir-opt --cir-flatten-cfg %t.cir -o %t.flat.cir
// RUN: FileCheck --input-file=%t.flat.cir %s -check-prefix=CIR-FLAT --implicit-check-not=cir.coroutine --implicit-check-not=cir.await --implicit-check-not=cir.co_return --implicit-check-not=cir.coro.exit --implicit-check-not=cir.coro.suspend_point --implicit-check-not=cir.coro.cleanup
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm  -disable-llvm-passes %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=OGCG

#include "Inputs/coroutine.h"

// CIR-DAG: ![[VoidTask:.*]] = !cir.struct<"folly::coro::Task<void>" {pad !u8i}>
// CIR-DAG: ![[IntTask:.*]] = !cir.struct<"folly::coro::Task<int>" {pad !u8i}>
// CIR-DAG: ![[VoidPromisse:.*]] = !cir.struct<"folly::coro::Task<void>::promise_type" {pad !u8i}>
// CIR-DAG: ![[IntPromisse:.*]] = !cir.struct<"folly::coro::Task<int>::promise_type" {pad !u8i}>
// CIR-DAG: ![[StdString:.*]] = !cir.struct<"std::string" {pad !u8i}>
// CIR-DAG: ![[CoroHandleVoid:.*]] = !cir.struct<"std::coroutine_handle<void>" {pad !u8i}>
// CIR-DAG: ![[CoroHandlePromiseVoid:rec_.*]]  = !cir.struct<"std::coroutine_handle<folly::coro::Task<void>::promise_type>" {pad !u8i}>
// CIR-DAG: ![[CoroHandlePromiseInt:rec_.*]] = !cir.struct<"std::coroutine_handle<folly::coro::Task<int>::promise_type>" {pad !u8i}>
// CIR-DAG: ![[SuspendAlways:.*]] = !cir.struct<"std::suspend_always" {pad !u8i}>

// OGCG-DAG: %[[VoidPromisse:"struct.folly::coro::Task<void>::promise_type"]] = type { i8 }
// OGCG-DAG: %[[VoidTask:"struct.folly::coro::Task"]] = type { i8 }
// OGCG-DAG: %[[SuspendAlways:"struct.std::suspend_always"]] = type { i8 }

// CIR: module {{.*}} {
// CIR-NEXT: cir.global external @_ZN5folly4coro9co_invokeE = #cir.zero : !rec_folly3A3Acoro3A3Aco_invoke_fn

using VoidTask = folly::coro::Task<void>;

VoidTask silly_task() {
  co_await std::suspend_always();
}

// CIR: cir.func coroutine {{.*}} @_Z10silly_taskv() call_conv_lowered attributes
// CIR-NEXT: %[[VoidTaskAddr:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<![[VoidTask]]>
// CIR-NEXT: %[[SavedFrameAddr:.*]] = cir.alloca "__coro_frame_addr" {{.*}} : !cir.ptr<!cir.ptr<!void>>
// CIR-NEXT: %[[VoidPromisseAddr:.*]] = cir.alloca "__promise" {{.*}} : !cir.ptr<![[VoidPromisse]]>
// CIR-NEXT: %[[SuspendAlwaysAddr:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<![[SuspendAlways]]>
// CIR-NEXT: %[[CoroHandleVoidAddr:.*]] = cir.alloca "agg.tmp0" {{.*}} : !cir.ptr<![[CoroHandleVoid]]>
// CIR-NEXT: %[[CoroHandlePromiseAddr:.*]] = cir.alloca "agg.tmp1" {{.*}} : !cir.ptr<![[CoroHandlePromiseVoid]]>

// OGCG: %[[VoidPromisseAddr:.*]] = alloca %[[VoidPromisse]], align 1
// OGCG: %[[VoidTaskAddr:.*]] = alloca %[[VoidTask]], align 1
// OGCG: %[[SuspendAlwaysAddr:.*]] = alloca %[[SuspendAlways]], align 1

// Get coroutine id with __builtin_coro_id.

// CIR: %[[NullPtr:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR-NEXT: %[[Align:.*]] = cir.const #cir.int<16> : !u32i
// CIR-NEXT: %[[CoroId:.*]] = cir.coro.intrinsic.id(%[[Align]], %[[NullPtr]], %[[NullPtr]], %[[NullPtr]]) : (!u32i, !cir.ptr<!void>, !cir.ptr<!void>, !cir.ptr<!void>) -> token
// OGCG: %[[CoroId:.*]] = call token @llvm.coro.id(i32 16, ptr %[[VoidPromisseAddr]], ptr null, ptr null)

// Perform allocation calling operator 'new' depending on __builtin_coro_alloc and
// call __builtin_coro_begin for the final coroutine frame address.

// CIR-NEXT: %[[ShouldAlloc:.*]] = cir.coro.intrinsic.alloc(%[[CoroId]]) : (token) -> !cir.bool
// CIR-NEXT: cir.store{{.*}} %[[NullPtr]], %[[SavedFrameAddr]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR-NEXT: cir.if %[[ShouldAlloc]] {
// CIR-NEXT:   %[[CoroSize:.*]] = cir.coro.intrinsic.size() : () -> !u64i
// CIR-NEXT:   %[[AllocAddr:.*]] = cir.call @_Znwm(%[[CoroSize]]) {allocsize = array<i32: 0>} : (!u64i {llvm.noundef}) -> (!cir.ptr<!void> {llvm.noalias, llvm.nonnull, llvm.noundef})
// CIR-NEXT:   cir.store{{.*}} %[[AllocAddr]], %[[SavedFrameAddr]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR-NEXT: }
// CIR-NEXT: %[[Load0:.*]] = cir.load{{.*}} %[[SavedFrameAddr]] : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
// CIR-NEXT: %[[CoroFrameAddr:.*]] = cir.coro.intrinsic.begin(%[[CoroId]], %[[Load0]]) : (token, !cir.ptr<!void>) -> !cir.ptr<!void>

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

// Empty records are ignored, so calls returning one return nothing and the
// slot takes a poison placeholder.

// CIR: cir.coroutine initialSuspend : {

// CIR:   cir.call @_ZN5folly4coro4TaskIvE12promise_type17get_return_objectEv(%[[VoidPromisseAddr]]) nothrow : {{.*}} -> ()
// CIR:   %[[RetObj:.*]] = cir.const #cir.poison : ![[VoidTask]]
// CIR:   cir.store{{.*}} %[[RetObj]], %[[VoidTaskAddr]] : ![[VoidTask]]

// OGCG: call void @llvm.lifetime.start.p0(ptr %[[VoidPromisseAddr]])
// OGCG: call void @_ZN5folly4coro4TaskIvE12promise_type17get_return_objectEv(ptr noundef nonnull align 1 dereferenceable(1) %[[VoidPromisseAddr]])
// OGCG: call void @llvm.lifetime.start.p0(ptr %[[SuspendAlwaysAddr]])

// Start a new scope for the actual codegen for co_await, create temporary allocas for
// holding coroutine handle and the suspend_always struct.

// Effectively execute `coawait promise_type::initial_suspend()` by calling initial_suspend() and getting
// the suspend_always struct to use for cir.await.

// CIR:   cir.call @_ZN5folly4coro4TaskIvE12promise_type15initial_suspendEv(%[[VoidPromisseAddr]])
// CIR:   %[[Tmp0:.*]] = cir.const #cir.poison : ![[SuspendAlways]]
// CIR:   cir.store{{.*}} %[[Tmp0]], %[[SuspendAlwaysAddr]]

// OGCG: call void @_ZN5folly4coro4TaskIvE12promise_type15initial_suspendEv(ptr noundef nonnull align 1 dereferenceable(1) %[[VoidPromisseAddr]])

//
// Here we start mapping co_await to cir.await.
//

// First regions `ready` has a special cir.yield code to veto suspension.

// CIR-NEXT:   cir.await(init, ready : {
// CIR-NEXT:     %[[ReadyVeto:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[SuspendAlwaysAddr]])
// CIR-NEXT:     cir.condition(%[[ReadyVeto]])

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

// The handle arguments are ignored too, so only the object argument is passed.

// CIR-NEXT:   }, suspend : {
// CIR-NEXT:     cir.call @_ZNSt16coroutine_handleIN5folly4coro4TaskIvE12promise_typeEE12from_addressEPv(%[[CoroFrameAddr]])
// CIR-NEXT:     %[[FromAddrRes:.*]] = cir.const #cir.poison : ![[CoroHandlePromiseVoid]]
// CIR-NEXT:     cir.store{{.*}} %[[FromAddrRes]], %[[CoroHandlePromiseAddr]] : ![[CoroHandlePromiseVoid]]
// CIR-NEXT:     %[[CoroHandlePromiseReload:.*]] = cir.load{{.*}} %[[CoroHandlePromiseAddr]]
// CIR-NEXT:     cir.call @_ZNSt16coroutine_handleIvEC1IN5folly4coro4TaskIvE12promise_typeEEES_IT_E(%[[CoroHandleVoidAddr]])
// CIR-NEXT:     %[[CoroHandleVoidReload:.*]] = cir.load{{.*}} %[[CoroHandleVoidAddr]] : !cir.ptr<![[CoroHandleVoid]]>, ![[CoroHandleVoid]]
// CIR-NEXT:     cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%[[SuspendAlwaysAddr]])
// CIR-NEXT:     cir.coro.suspend_point

// OGCG: init.suspend:
// OGCG:   %[[Save:.*]] = call token @llvm.coro.save(ptr null)
// OGCG:   call void @llvm.coro.await.suspend.void(ptr %[[SuspendAlwaysAddr]], ptr %[[CoroFrameAddr]], ptr @_Z10silly_taskv.__await_suspend_wrapper__init)
// OGCG:   %[[TMP1:.*]] = call i8 @llvm.coro.suspend(token %[[Save]], i1 false)
// OGCG:   switch i8 %[[TMP1]], label %coro.ret [
// OGCG:     i8 0, label %init.ready
// OGCG:     i8 1, label %init.cleanup
// OGCG:   ]

// Third region `resume` handles coroutine resuming logic.

// CIR-NEXT:   }, resume : {
// CIR-NEXT:     cir.call @_ZNSt14suspend_always12await_resumeEv(%[[SuspendAlwaysAddr]])
// CIR-NEXT:     cir.yield
// CIR-NEXT:   },)
// CIR-NEXTY   cir.yield loc(#loc34)

// OGCG: init.ready:
// OGCG:   call void @_ZNSt14suspend_always12await_resumeEv(ptr noundef nonnull align 1 dereferenceable(1) %[[SuspendAlwaysAddr]]
// OGCG:   br label %cleanup

// Since we already tested cir.await guts above, the remaining checks for:
// - The actual user written co_await
// - The promise call
// - The final suspend co_await
// - Return

// CIR:   }, body : {

// The actual user written co_await
// CIR:     cir.await(user, ready : {
// CIR:     }, suspend : {
// CIR:     }, resume : {
// CIR:     },)

// OGCG: cleanup.cont
// OGCG: await.suspend:
// OGCG: await.ready:

// The promise call
// CIR:     cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(%[[VoidPromisseAddr]])
// CIR:     cir.co_return
// CIR:   }, finalSuspend : {

// OGCG: call void @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(ptr noundef nonnull align 1 dereferenceable(1) %[[VoidPromisseAddr]])

// The final suspend co_await
// CIR:   cir.await(final, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR-NEXT:   cir.yield

// OGCG: coro.final:
// OGCG: final.suspend:
// OGCG: final.ready:

// Cleanup of coroutine frame.
//
// `__builtin_coro_free` returns the frame pointer or null.
// If null, no dynamic allocation happened, so nothing to free.
// The `if` ensures we only call delete on non-null.

// CIR-NEXT: }, destroy : {

// CIR-NEXT:   %[[FreeMem:.*]] = cir.coro.intrinsic.free(%[[CoroId]], %[[CoroFrameAddr]]) : (token, !cir.ptr<!void>) -> !cir.ptr<!void>
// CIR-NEXT:   %[[NullPtr2:.*]] = cir.const #cir.ptr<null>
// CIR-NEXT:   %[[Cond:.*]] = cir.cmp ne %[[FreeMem]], %[[NullPtr2]]
// CIR-NEXT:   cir.if %[[Cond]] {
// CIR-NEXT:     %[[Size:.*]] = cir.coro.intrinsic.size()
// CIR-NEXT:     cir.call @_ZdlPvm(%[[FreeMem]], %[[Size]]){{.*}}
// CIR-NEXT:   }
// CIR-NEXT:   cir.yield
// CIR-NEXT: }, exit : {

// OGCG: %[[FreeMem:.*]] = call ptr @llvm.coro.free(token %[[CoroId]], ptr %[[CoroFrameAddr]])
// OGCG: %[[Cond:.*]] = icmp ne ptr %[[FreeMem]], null
// OGCG: br i1 %[[Cond]], label %coro.free, label %after.coro.free

// OGCG: coro.free:
// OGCG:   %[[Size:.*]] = call i64 @llvm.coro.size.i64()
// OGCG:   call void @_ZdlPvm(ptr {{.*}} %[[FreeMem]], i64 {{.*}} %[[Size]])
// OGCG:   br label %after.coro.free
// OGCG: after.coro.free:

// Call builtin coro end and return

// CIR-NEXT: %[[CoroEndArg0:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR-NEXT: %[[CoroEndArg1:.*]] = cir.const #false
// CIR-NEXT: %[[TK_NONE:.*]] = cir.token.none
// CIR-NEXT: cir.coro.intrinsic.end(%[[CoroEndArg0]], %[[CoroEndArg1]], %[[TK_NONE]]) : (!cir.ptr<!void>, !cir.bool, token){{.*}}

// CIR: %[[Tmp1:.*]] = cir.load{{.*}} %[[VoidTaskAddr]]
// CIR: cir.return
// CIR: }

// OGCG: coro.ret:
// OGCG:   call void @llvm.coro.end(ptr null, i1 false, token none)
// OGCG:   ret void

// OGCG: define {{.*}} @_Z10silly_taskv.__await_suspend_wrapper__init(ptr noundef nonnull %[[Awaiter:.*]], ptr noundef %[[Handle:.*]])
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

// After FlattenCFG, cir.coroutine is inlined into the function: the
// initialSuspend, body, finalSuspend, destroy (coro.free) and exit regions are
// placed in that order. Each cir.await becomes ready/suspend/resume blocks
// where suspend dispatches on llvm.coro.suspend: 0 resumes, 1 destroys
// (branches to the destroy region), and the default suspends (jumps straight
// to coro.end).
// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z10silly_taskv()
// CIR-FLAT:   %[[FRAME_ADDR:.*]] = cir.alloca "__coro_frame_addr"
// CIR-FLAT:   %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-FLAT:   %[[INIT_AWAITER:.*]] = cir.alloca "ref.tmp0"
// CIR-FLAT:   %[[USER_AWAITER:.*]] = cir.alloca "ref.tmp1"
// CIR-FLAT:   %[[FINAL_AWAITER:.*]] = cir.alloca "ref.tmp2"

// The frame allocation cir.if is flattened into coro.alloc/coro.init blocks.

// CIR-FLAT:   %[[NULL:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR-FLAT:   %[[ALIGN:.*]] = cir.const #cir.int<16> : !u32i
// CIR-FLAT:   %[[CORO_ID:.*]] = cir.coro.intrinsic.id(%[[ALIGN]], %[[NULL]], %[[NULL]], %[[NULL]])
// CIR-FLAT:   %[[SHOULD_ALLOC:.*]] = cir.coro.intrinsic.alloc(%[[CORO_ID]])
// CIR-FLAT:   cir.store %[[NULL]], %[[FRAME_ADDR]]
// CIR-FLAT:   cir.brcond %[[SHOULD_ALLOC]] ^[[CORO_ALLOC:bb[0-9]+]], ^[[CORO_INIT:bb[0-9]+]]
// CIR-FLAT: ^[[CORO_ALLOC]]:
// CIR-FLAT:   %[[SIZE:.*]] = cir.coro.intrinsic.size()
// CIR-FLAT:   %[[MEM:.*]] = cir.call @_Znwm(%[[SIZE]])
// CIR-FLAT:   cir.store %[[MEM]], %[[FRAME_ADDR]]
// CIR-FLAT:   cir.br ^[[CORO_INIT]]
// CIR-FLAT: ^[[CORO_INIT]]:
// CIR-FLAT:   %[[MEM_RELOAD:.*]] = cir.load %[[FRAME_ADDR]]
// CIR-FLAT:   %[[FRAME:.*]] = cir.coro.intrinsic.begin(%[[CORO_ID]], %[[MEM_RELOAD]])

// Initial suspend.

// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type17get_return_objectEv(%[[PROMISE]])
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type15initial_suspendEv(%[[PROMISE]])
// CIR-FLAT:   cir.br ^[[INIT_READY:bb[0-9]+]]
// CIR-FLAT: ^[[INIT_READY]]:
// CIR-FLAT:   %[[INIT_IS_READY:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[INIT_AWAITER]])
// CIR-FLAT:   cir.brcond %[[INIT_IS_READY]] ^[[INIT_RESUME:bb[0-9]+]], ^[[INIT_SUSPEND:bb[0-9]+]]
// CIR-FLAT: ^[[INIT_SUSPEND]]:
// CIR-FLAT:   %[[INIT_SAVE_ARG:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR-FLAT:   %[[INIT_SAVE:.*]] = cir.coro.intrinsic.save(%[[INIT_SAVE_ARG]]) : (!cir.ptr<!void>) -> token
// CIR-FLAT:   cir.call @_ZNSt16coroutine_handleIN5folly4coro4TaskIvE12promise_typeEE12from_addressEPv(%[[FRAME]])
// CIR-FLAT:   cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%[[INIT_AWAITER]])
// CIR-FLAT:   %[[INIT_IS_FINAL:.*]] = cir.const #false
// CIR-FLAT:   %[[INIT_SUSPEND_RES:.*]] = cir.coro.intrinsic.suspend(%[[INIT_SAVE]], %[[INIT_IS_FINAL]]) : (token, !cir.bool) -> !s8i
// CIR-FLAT:   cir.switch.flat %[[INIT_SUSPEND_RES]] : !s8i, ^[[INIT_SUSPENDED:bb[0-9]+]] [
// CIR-FLAT-NEXT:     0: ^[[INIT_RESUME]],
// CIR-FLAT-NEXT:     1: ^[[INIT_DESTROY:bb[0-9]+]]
// CIR-FLAT-NEXT:   ]
// CIR-FLAT: ^[[INIT_DESTROY]]:
// CIR-FLAT-NEXT:   cir.br ^[[CORO_CLEANUP:bb[0-9]+]]
// CIR-FLAT: ^[[INIT_SUSPENDED]]:
// CIR-FLAT:   cir.br ^[[CORO_RET:bb[0-9]+]]
// CIR-FLAT: ^[[INIT_RESUME]]:
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv(%[[INIT_AWAITER]])

// User written co_await.

// CIR-FLAT:   %[[USER_IS_READY:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[USER_AWAITER]])
// CIR-FLAT:   cir.brcond %[[USER_IS_READY]] ^[[USER_RESUME:bb[0-9]+]], ^[[USER_SUSPEND:bb[0-9]+]]
// CIR-FLAT: ^[[USER_SUSPEND]]:
// CIR-FLAT:   %[[USER_SAVE:.*]] = cir.coro.intrinsic.save(
// CIR-FLAT:   cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%[[USER_AWAITER]])
// CIR-FLAT:   %[[USER_IS_FINAL:.*]] = cir.const #false
// CIR-FLAT:   %[[USER_SUSPEND_RES:.*]] = cir.coro.intrinsic.suspend(%[[USER_SAVE]], %[[USER_IS_FINAL]])
// CIR-FLAT:   cir.switch.flat %[[USER_SUSPEND_RES]] : !s8i, ^[[USER_SUSPENDED:bb[0-9]+]] [
// CIR-FLAT-NEXT:     0: ^[[USER_RESUME]],
// CIR-FLAT-NEXT:     1: ^[[USER_DESTROY:bb[0-9]+]]
// CIR-FLAT-NEXT:   ]
// CIR-FLAT: ^[[USER_DESTROY]]:
// CIR-FLAT-NEXT:   cir.br ^[[CORO_CLEANUP]]
// CIR-FLAT: ^[[USER_SUSPENDED]]:
// CIR-FLAT:   cir.br ^[[CORO_RET]]
// CIR-FLAT: ^[[USER_RESUME]]:
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv(%[[USER_AWAITER]])

// The cir.co_return is replaced by a branch to the final suspend.

// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(%[[PROMISE]])
// CIR-FLAT-NEXT:   cir.br ^[[FINAL:bb[0-9]+]]
// CIR-FLAT: ^[[FINAL]]:
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type13final_suspendEv(%[[PROMISE]])

// Final suspend: same as above, but llvm.coro.suspend is marked final.

// CIR-FLAT:   %[[FINAL_IS_READY:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[FINAL_AWAITER]])
// CIR-FLAT:   cir.brcond %[[FINAL_IS_READY]] ^[[FINAL_RESUME:bb[0-9]+]], ^[[FINAL_SUSPEND:bb[0-9]+]]
// CIR-FLAT: ^[[FINAL_SUSPEND]]:
// CIR-FLAT:   %[[FINAL_SAVE:.*]] = cir.coro.intrinsic.save(
// CIR-FLAT:   cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%[[FINAL_AWAITER]])
// CIR-FLAT:   %[[FINAL_IS_FINAL:.*]] = cir.const #true
// CIR-FLAT:   %[[FINAL_SUSPEND_RES:.*]] = cir.coro.intrinsic.suspend(%[[FINAL_SAVE]], %[[FINAL_IS_FINAL]])
// CIR-FLAT:   cir.switch.flat %[[FINAL_SUSPEND_RES]] : !s8i, ^[[FINAL_SUSPENDED:bb[0-9]+]] [
// CIR-FLAT-NEXT:     0: ^[[FINAL_RESUME]],
// CIR-FLAT-NEXT:     1: ^[[FINAL_DESTROY:bb[0-9]+]]
// CIR-FLAT-NEXT:   ]
// CIR-FLAT: ^[[FINAL_DESTROY]]:
// CIR-FLAT-NEXT:   cir.br ^[[CORO_CLEANUP]]
// CIR-FLAT: ^[[FINAL_SUSPENDED]]:
// CIR-FLAT:   cir.br ^[[CORO_RET]]
// CIR-FLAT: ^[[FINAL_RESUME]]:
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv(%[[FINAL_AWAITER]])

// Falling off the end of the final suspend also runs the coroutine cleanup.

// CIR-FLAT:   cir.br ^[[CORO_CLEANUP]]

// Coroutine cleanup (the destroy region): free the frame if
// llvm.coro.free returns non-null, then go straight to the exit region, as OG
// does. Nothing is read after the frame is freed.

// CIR-FLAT: ^[[CORO_CLEANUP]]:
// CIR-FLAT:   %[[MEM_TO_FREE:.*]] = cir.coro.intrinsic.free(%[[CORO_ID]], %[[FRAME]])
// CIR-FLAT:   %[[NULL2:.*]] = cir.const #cir.ptr<null>
// CIR-FLAT:   %[[NEEDS_FREE:.*]] = cir.cmp ne %[[MEM_TO_FREE]], %[[NULL2]]
// CIR-FLAT:   cir.brcond %[[NEEDS_FREE]] ^[[CORO_FREE:bb[0-9]+]], ^[[AFTER_CORO_FREE:bb[0-9]+]]
// CIR-FLAT: ^[[CORO_FREE]]:
// CIR-FLAT:   %[[SIZE2:.*]] = cir.coro.intrinsic.size()
// CIR-FLAT:   cir.call @_ZdlPvm(%[[MEM_TO_FREE]], %[[SIZE2]])
// CIR-FLAT:   cir.br ^[[AFTER_CORO_FREE]]
// CIR-FLAT: ^[[AFTER_CORO_FREE]]:
// CIR-FLAT-NEXT:   cir.br ^[[CORO_RET]]

// The exit region; every suspend's default destination lands here.

// CIR-FLAT: ^[[CORO_RET]]:
// CIR-FLAT:   %[[END_ARG0:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR-FLAT:   %[[END_ARG1:.*]] = cir.const #false
// CIR-FLAT:   %[[END_TK:.*]] = cir.token.none
// CIR-FLAT:   cir.coro.intrinsic.end(%[[END_ARG0]], %[[END_ARG1]], %[[END_TK]])
// CIR-FLAT:   cir.return
// CIR-FLAT: }

folly::coro::Task<int> byRef(const std::string& s) {
  co_return s.size();
}

// CIR:  cir.func coroutine {{.*}} @_Z5byRefRKSt6string(%[[ARG:.*]]: !cir.ptr<![[StdString]]> {{.*}}) call_conv_lowered attributes
// CIR:    %[[IntTaskAddr:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<![[IntTask]]>
// CIR:    %[[AllocaParam:.*]] = cir.alloca "s" {{.*}} init const : !cir.ptr<!cir.ptr<![[StdString]]>>
// CIR:    %[[SavedFrameAddr:.*]]  = cir.alloca "__coro_frame_addr" {{.*}} : !cir.ptr<!cir.ptr<!void>>
// CIR:    %[[AllocaFnUse:.*]] = cir.alloca "s" {{.*}} init const : !cir.ptr<!cir.ptr<![[StdString]]>>
// CIR:    %[[IntPromisseAddr:.*]] = cir.alloca "__promise" {{.*}} : !cir.ptr<![[IntPromisse]]>
// CIR:    %[[SuspendAlwaysAddr:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<![[SuspendAlways]]>
// CIR:    %[[CoroHandleVoidAddr:.*]] = cir.alloca "agg.tmp0" {{.*}} : !cir.ptr<![[CoroHandleVoid]]>
// CIR:    %[[CoroHandlePromiseAddr:.*]] = cir.alloca "agg.tmp1" {{.*}} : !cir.ptr<![[CoroHandlePromiseInt]]>
// CIR:    cir.store %[[ARG]], %[[AllocaParam]] : !cir.ptr<![[StdString]]>, {{.*}}

// Call promise.get_return_object() to retrieve the task object.

// CIR:     %[[LOAD:.*]] = cir.load %[[AllocaParam]] : !cir.ptr<!cir.ptr<![[StdString]]>>, !cir.ptr<![[StdString]]>
// CIR:     cir.store {{.*}} %[[LOAD]], %[[AllocaFnUse]] : !cir.ptr<![[StdString]]>, !cir.ptr<!cir.ptr<![[StdString]]>>

// CIR:    cir.coroutine initialSuspend : {
// CIR:     cir.call @_ZN5folly4coro4TaskIiE12promise_type17get_return_objectEv(%[[IntPromisseAddr]]) nothrow : {{.*}} -> ()
// CIR:     %[[RetObj:.*]] = cir.const #cir.poison : ![[IntTask]]
// CIR:     cir.store {{.*}} %[[RetObj]], %[[IntTaskAddr]] : ![[IntTask]]
// CIR:     cir.call @_ZN5folly4coro4TaskIiE12promise_type15initial_suspendEv(%[[IntPromisseAddr]])
// CIR:     %[[Tmp0:.*]] = cir.const #cir.poison : ![[SuspendAlways]]
// CIR:     cir.store{{.*}} %[[Tmp0]], %[[SuspendAlwaysAddr]]
// CIR:     cir.await(init, ready : {
// CIR:       %[[TmpCallRes:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[SuspendAlwaysAddr]])
// CIR:       cir.condition(%[[TmpCallRes]])
// CIR:     }, suspend : {
// CIR:       cir.call @_ZNSt16coroutine_handleIN5folly4coro4TaskIiE12promise_typeEE12from_addressEPv(%[[CoroFrameAddr:.*]])
// CIR:       %[[FromAddrRes:.*]] = cir.const #cir.poison : ![[CoroHandlePromiseInt]]
// CIR:       cir.store{{.*}} %[[FromAddrRes]], %[[CoroHandlePromiseAddr]] : ![[CoroHandlePromiseInt]]
// CIR:       %[[CoroHandlePromiseReload:.*]] = cir.load{{.*}} %[[CoroHandlePromiseAddr]]
// CIR:       cir.call @_ZNSt16coroutine_handleIvEC1IN5folly4coro4TaskIiE12promise_typeEEES_IT_E(%[[CoroHandleVoidAddr]])
// CIR:       %[[CoroHandleVoidReload:.*]] = cir.load{{.*}} %[[CoroHandleVoidAddr]] : !cir.ptr<![[CoroHandleVoid]]>, ![[CoroHandleVoid]]
// CIR:       cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%[[SuspendAlwaysAddr]])
// CIR:       cir.coro.suspend_point
// CIR:     }, resume : {
// CIR:       cir.call @_ZNSt14suspend_always12await_resumeEv(%[[SuspendAlwaysAddr]])
// CIR:       cir.yield
// CIR:     },)
// CIR:     cir.yield
// CIR:     }, body : {
              // can't fallthrough
// CIR-NOT:   cir.await(user
// CIR:       cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[IntPromisseAddr]], %[[STRING_SIZE:.*]])
// CIR:        cir.co_return
// CIR:     }, finalSuspend : {

// The final suspend co_await
// CIR:     cir.await(final, ready : {
// CIR:     }, suspend : {
// CIR:     }, resume : {
// CIR:     },)
// CIR:     cir.yield
// CIR:    }, destroy : {
// CIR:    }, exit : {

// The body has no user await; co_return branches straight to final suspend.

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z5byRefRKSt6string
// CIR-FLAT:   %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv
// CIR-FLAT:   %[[SIZE:.*]] = cir.call @_ZNKSt6string4sizeEv(
// CIR-FLAT-NEXT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], %[[SIZE]])
// CIR-FLAT-NEXT:   cir.br ^[[FINAL:bb[0-9]+]]
// CIR-FLAT: ^[[FINAL]]:
// CIR-FLAT-NEXT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type13final_suspendEv(%[[PROMISE]])
// CIR-FLAT:   cir.coro.intrinsic.free
// CIR-FLAT:   cir.call @_ZdlPvm
// CIR-FLAT:   cir.coro.intrinsic.end
// CIR-FLAT:   cir.return

// OGCG: define {{.*}} void @_Z5byRefRKSt6string(
// OGCG: cleanup.cont:
// OGCG:   %[[SIZE:.*]] = call noundef i32 @_ZNKSt6string4sizeEv(
// OGCG-NEXT:   call void @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(ptr {{.*}} %__promise, i32 noundef %[[SIZE]])
// OGCG-NEXT:   br label %coro.final
// OGCG: coro.final:
// OGCG:   call void @_ZN5folly4coro4TaskIiE12promise_type13final_suspendEv(
// OGCG: coro.cleanup:
// OGCG:   call ptr @llvm.coro.free(
// OGCG: coro.ret:
// OGCG-NEXT:   call void @llvm.coro.end(ptr null, i1 false, token none)
// OGCG-NEXT:   ret void

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

// CIR: cir.func coroutine {{.*}} @_Z10silly_corov() call_conv_lowered attributes
// CIR: cir.coroutine initialSuspend : {
// CIR:   cir.await(init, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR: }, body : {
// CIR:   cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv
// CIR:   cir.co_return
// CIR: }, finalSuspend : {
// CIR:   cir.await(final, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR:   cir.yield
// CIR: }, destroy : {
// CIR: }, exit : {

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z10silly_corov()
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv
// CIR-FLAT:   cir.call @_ZNSt6stringC1EPKc
// CIR-FLAT:   cir.call @_Z5byRefRKSt6string
// CIR-FLAT:   cir.call @_ZN5folly4coro12blockingWaitIiEET_ONS0_4TaskIS2_EE
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv
// CIR-FLAT-NEXT:   cir.br ^[[FINAL:bb[0-9]+]]
// CIR-FLAT: ^[[FINAL]]:
// CIR-FLAT-NEXT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type13final_suspendEv
// CIR-FLAT:   cir.coro.intrinsic.end
// CIR-FLAT:   cir.return

// OGCG: define {{.*}} void @_Z10silly_corov()
// OGCG: cleanup.cont:
// OGCG:   call void @_ZNSt6stringC1EPKc(
// OGCG:   call void @_Z5byRefRKSt6string(
// OGCG:   call noundef i32 @_ZN5folly4coro12blockingWaitIiEET_ONS0_4TaskIS2_EE(
// OGCG:   call void @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(
// OGCG:   br label %coro.final
// OGCG: coro.final:
// OGCG:   call void @_ZN5folly4coro4TaskIvE12promise_type13final_suspendEv(
// OGCG: coro.cleanup:
// OGCG:   call ptr @llvm.coro.free(
// OGCG: coro.ret:
// OGCG-NEXT:   call void @llvm.coro.end(ptr null, i1 false, token none)
// OGCG-NEXT:   ret void

folly::coro::Task<void> yield();
folly::coro::Task<void> yield1() {
  auto t = yield();
  co_yield t;
}

// CIR: cir.func coroutine {{.*}} @_Z6yield1v() call_conv_lowered attributes

// Prologue allocas (still present in output)
// CIR-DAG: %[[RETVAL:.*]] = cir.alloca "__retval" {{.*}} : !cir.ptr<![[VoidTask]]>
// CIR-DAG: %[[FRAME:.*]] = cir.alloca "__coro_frame_addr" {{.*}} : !cir.ptr<!cir.ptr<!void>>
// CIR-DAG: %[[PROMISE:.*]] = cir.alloca "__promise" {{.*}} : !cir.ptr<![[VoidPromisse]]>
// CIR-DAG: %[[SUSP0:.*]] = cir.alloca "ref.tmp0" {{.*}} : !cir.ptr<![[SuspendAlways]]>
// CIR-DAG: %[[CH_VOID0:.*]] = cir.alloca "agg.tmp0" {{.*}} : !cir.ptr<![[CoroHandleVoid]]>
// CIR-DAG: %[[CH_PROM0:.*]] = cir.alloca "agg.tmp1" {{.*}} : !cir.ptr<![[CoroHandlePromiseVoid]]>
// CIR-DAG: %[[T_ADDR:.*]] = cir.alloca "t" {{.*}} init : !cir.ptr<![[VoidTask]]>
// CIR-DAG: %[[SUSP1:.*]] = cir.alloca "ref.tmp1" {{.*}} : !cir.ptr<![[SuspendAlways]]>
// CIR-DAG: %[[AWAITER_COPY_ADDR:.*]] = cir.alloca "agg.tmp2" {{.*}} : !cir.ptr<![[VoidTask]]>
// CIR-DAG: %[[CH_VOID1:.*]] = cir.alloca "agg.tmp3" {{.*}} : !cir.ptr<![[CoroHandleVoid]]>
// CIR-DAG: %[[CH_PROM1:.*]] = cir.alloca "agg.tmp4" {{.*}} : !cir.ptr<![[CoroHandlePromiseVoid]]>
// CIR-DAG: %[[SUSP2:.*]] = cir.alloca "ref.tmp2" {{.*}} : !cir.ptr<![[SuspendAlways]]>
// CIR-DAG: %[[CH_VOID2:.*]] = cir.alloca "agg.tmp5" {{.*}} : !cir.ptr<![[CoroHandleVoid]]>
// CIR-DAG: %[[CH_PROM2:.*]] = cir.alloca "agg.tmp6" {{.*}} : !cir.ptr<![[CoroHandlePromiseVoid]]>

// CIR: cir.coroutine initialSuspend : {
// initial_suspend + await(init)
// CIR:   cir.call @_ZN5folly4coro4TaskIvE12promise_type15initial_suspendEv(%[[PROMISE]]){{.*}}
// CIR:   %[[INIT_SUSP:.*]] = cir.const #cir.poison : ![[SuspendAlways]]
// CIR:   cir.store{{.*}} %[[INIT_SUSP]], %[[SUSP0]]
// CIR:   cir.await(init, ready : {
// CIR:     %[[READY0:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[SUSP0]]){{.*}}
// CIR:     cir.condition(%[[READY0]])
// CIR:   }, suspend : {
// CIR:     cir.call @_ZNSt16coroutine_handleIN5folly4coro4TaskIvE12promise_typeEE12from_addressEPv(%{{.*}}){{.*}}
// CIR:     %[[FROMADDR0:.*]] = cir.const #cir.poison : ![[CoroHandlePromiseVoid]]
// CIR:     cir.store{{.*}} %[[FROMADDR0]], %[[CH_PROM0]]
// CIR:     %[[PROM_RELOAD0:.*]] = cir.load{{.*}} %[[CH_PROM0]]
// CIR:     cir.call @_ZNSt16coroutine_handleIvEC1IN5folly4coro4TaskIvE12promise_typeEEES_IT_E(%[[CH_VOID0]]){{.*}}
// CIR:     %[[VOID_RELOAD0:.*]] = cir.load{{.*}} %[[CH_VOID0]]
// CIR:     cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%[[SUSP0]]){{.*}}
// CIR:     cir.coro.suspend_point
// CIR:   }, resume : {
// CIR:     cir.call @_ZNSt14suspend_always12await_resumeEv(%[[SUSP0]]){{.*}}
// CIR:     cir.yield
// CIR:   },)
// CIR:   cir.yield

// yield_value + await(yield)
// CIR: }, body : {
// CIR:    cir.call @_Z5yieldv(){{.*}}
// CIR:    %[[YIELD_TASK:.*]] = cir.const #cir.poison : ![[VoidTask]]
// CIR:    cir.store{{.*}} %[[YIELD_TASK]], %[[T_ADDR]]
// CIR:    %[[AWAITER:.*]] = cir.load{{.*}} %[[AWAITER_COPY_ADDR]]
// CIR:    cir.call @_ZN5folly4coro4TaskIvE12promise_type11yield_valueES2_(%[[PROMISE]]){{.*}}
// CIR:    %[[YIELD_SUSP:.*]] = cir.const #cir.poison : ![[SuspendAlways]]
// CIR:    cir.store{{.*}} %[[YIELD_SUSP]], %[[SUSP1]]
// CIR:    cir.await(yield, ready : {
// CIR:      %[[READY1:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[SUSP1]]){{.*}}
// CIR:      cir.condition(%[[READY1]])
// CIR:    }, suspend : {
// CIR:      cir.call @_ZNSt16coroutine_handleIN5folly4coro4TaskIvE12promise_typeEE12from_addressEPv(%{{.*}}){{.*}}
// CIR:      %[[FROMADDR1:.*]] = cir.const #cir.poison : ![[CoroHandlePromiseVoid]]
// CIR:      cir.store{{.*}} %[[FROMADDR1]], %[[CH_PROM1]]
// CIR:      %[[PROM_RELOAD1:.*]] = cir.load{{.*}} %[[CH_PROM1]]
// CIR:      cir.call @_ZNSt16coroutine_handleIvEC1IN5folly4coro4TaskIvE12promise_typeEEES_IT_E(%[[CH_VOID1]]){{.*}}
// CIR:      %[[VOID_RELOAD1:.*]] = cir.load{{.*}} %[[CH_VOID1]]
// CIR:      cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%[[SUSP1]]){{.*}}
// CIR:      cir.coro.suspend_point
// CIR:    }, resume : {
// CIR:      cir.call @_ZNSt14suspend_always12await_resumeEv(%[[SUSP1]]){{.*}}
// CIR:      cir.yield
// CIR:    },)
// CIR:    cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(%[[PROMISE]])
// CIR:    cir.co_return
// CIR: }, finalSuspend : {

// return_void + await(final)
// CIR:   cir.call @_ZN5folly4coro4TaskIvE12promise_type13final_suspendEv(%[[PROMISE]]){{.*}}
// CIR:   %[[FINAL_SUSP:.*]] = cir.const #cir.poison : ![[SuspendAlways]]
// CIR:   cir.store{{.*}} %[[FINAL_SUSP]], %[[SUSP2]]
// CIR:   cir.await(final, ready : {
// CIR:     %[[READY2:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[SUSP2]]){{.*}}
// CIR:     cir.condition(%[[READY2]])
// CIR:   }, suspend : {
// CIR:     cir.call @_ZNSt16coroutine_handleIN5folly4coro4TaskIvE12promise_typeEE12from_addressEPv(%{{.*}}){{.*}}
// CIR:     %[[FROMADDR2:.*]] = cir.const #cir.poison : ![[CoroHandlePromiseVoid]]
// CIR:     cir.store{{.*}} %[[FROMADDR2]], %[[CH_PROM2]]
// CIR:     %[[PROM_RELOAD2:.*]] = cir.load{{.*}} %[[CH_PROM2]]
// CIR:     cir.call @_ZNSt16coroutine_handleIvEC1IN5folly4coro4TaskIvE12promise_typeEEES_IT_E(%[[CH_VOID2]]){{.*}}
// CIR:     %[[VOID_RELOAD2:.*]] = cir.load{{.*}} %[[CH_VOID2]]
// CIR:     cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%[[SUSP2]]){{.*}}
// CIR:     cir.coro.suspend_point
// CIR:   }, resume : {
// CIR:     cir.call @_ZNSt14suspend_always12await_resumeEv(%[[SUSP2]]){{.*}}
// CIR:     cir.yield
// CIR:   },)
// CIR:   cir.yield
// CIR: }, destroy : {
// CIR: }, exit : {
// CIR: cir.coro.intrinsic.end(%{{.*}}, %{{.*}}, %{{.*}}){{.*}}
// CIR: %[[RETLOAD:.*]] = cir.load{{.*}} %[[RETVAL]]
// CIR: cir.return
// CIR: }

// co_yield is flattened like a user co_await (a non-final suspend).

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z6yield1v()
// CIR-FLAT:   %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-FLAT:   %[[YIELD_AWAITER:.*]] = cir.alloca "ref.tmp1"
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type11yield_valueES2_(%[[PROMISE]])
// CIR-FLAT:   %[[READY:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[YIELD_AWAITER]])
// CIR-FLAT:   cir.brcond %[[READY]] ^[[YIELD_RESUME:bb[0-9]+]], ^[[YIELD_SUSPEND:bb[0-9]+]]
// CIR-FLAT: ^[[YIELD_SUSPEND]]:
// CIR-FLAT:   %[[SAVE:.*]] = cir.coro.intrinsic.save(
// CIR-FLAT:   cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%[[YIELD_AWAITER]])
// CIR-FLAT:   %[[IS_FINAL:.*]] = cir.const #false
// CIR-FLAT:   %[[RES:.*]] = cir.coro.intrinsic.suspend(%[[SAVE]], %[[IS_FINAL]])
// CIR-FLAT:   cir.switch.flat %[[RES]] : !s8i, ^[[YIELD_SUSPENDED:bb[0-9]+]] [
// CIR-FLAT-NEXT:     0: ^[[YIELD_RESUME]],
// CIR-FLAT-NEXT:     1: ^[[YIELD_DESTROY:bb[0-9]+]]
// CIR-FLAT-NEXT:   ]
// CIR-FLAT: ^[[YIELD_DESTROY]]:
// CIR-FLAT-NEXT:   cir.br ^[[CORO_CLEANUP:bb[0-9]+]]
// CIR-FLAT: ^[[YIELD_SUSPENDED]]:
// CIR-FLAT-NEXT:   cir.br ^[[CORO_RET:bb[0-9]+]]
// CIR-FLAT: ^[[YIELD_RESUME]]:
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv(%[[YIELD_AWAITER]])
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(%[[PROMISE]])
// CIR-FLAT-NEXT:   cir.br ^[[FINAL:bb[0-9]+]]
// CIR-FLAT: ^[[FINAL]]:
// CIR-FLAT-NEXT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type13final_suspendEv(%[[PROMISE]])
// CIR-FLAT: ^[[CORO_CLEANUP]]:
// CIR-FLAT-NEXT:   cir.coro.intrinsic.free
// CIR-FLAT: ^[[CORO_RET]]:
// CIR-FLAT:   cir.coro.intrinsic.end
// CIR-FLAT:   cir.return

// OGCG: define {{.*}} void @_Z6yield1v()
// OGCG: cleanup.cont:
// OGCG:   call void @_Z5yieldv()
// OGCG:   call void @_ZN5folly4coro4TaskIvE12promise_type11yield_valueES2_(
// OGCG:   %[[READY:.*]] = call noundef zeroext i1 @_ZNSt14suspend_always11await_readyEv(
// OGCG:   br i1 %[[READY]], label %yield.ready, label %yield.suspend
// OGCG: yield.suspend:
// OGCG:   %[[SAVE:.*]] = call token @llvm.coro.save(ptr null)
// OGCG:   call void @llvm.coro.await.suspend.void({{.*}}, ptr @_Z6yield1v.__await_suspend_wrapper__yield)
// OGCG:   %[[SUSP:.*]] = call i8 @llvm.coro.suspend(token %[[SAVE]], i1 false)
// OGCG:   switch i8 %[[SUSP]], label %coro.ret [
// OGCG-NEXT:     i8 0, label %yield.ready
// OGCG-NEXT:     i8 1, label %yield.cleanup
// OGCG-NEXT:   ]
// OGCG: yield.ready:
// OGCG:   call void @_ZNSt14suspend_always12await_resumeEv(
// OGCG:   %[[SLOT:.*]] = phi i32 [ 0, %yield.ready ], [ 2, %yield.cleanup ]
// OGCG:   call void @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(
// OGCG-NEXT:   br label %coro.final
// OGCG: coro.final:
// OGCG:   call void @_ZN5folly4coro4TaskIvE12promise_type13final_suspendEv(
// OGCG: coro.cleanup:
// OGCG:   call ptr @llvm.coro.free(
// OGCG: coro.ret:
// OGCG-NEXT:   call void @llvm.coro.end(ptr null, i1 false, token none)
// OGCG-NEXT:   ret void

folly::coro::Task<int> go(int const& val);
folly::coro::Task<int> go1() {
  auto task = go(1);
  co_return co_await task;
}

// CIR: cir.func coroutine {{.*}} @_Z3go1v() call_conv_lowered attributes
// CIR: %[[IntTaskAddr:.*]] = cir.alloca "task" {{.*}} init : !cir.ptr<![[IntTask]]>
// CIR: %[[OneAddr:.*]] = cir.alloca "ref.tmp1" align(4) init : !cir.ptr<!s32i>

// CIR: cir.coroutine initialSuspend : {
// CIR:   cir.await(init, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR:   cir.yield

// CIR: }, body : {
// CIR:   %[[One:.*]] = cir.const #cir.int<1> : !s32i
// CIR:   cir.store{{.*}} %[[One]], %[[OneAddr]] : !s32i, !cir.ptr<!s32i>
// CIR:   cir.call @_Z2goRKi(%[[OneAddr]]) : (!cir.ptr<!s32i>{{.*}}) -> ()
// CIR:   %[[IntTaskTmp:.*]] = cir.const #cir.poison : ![[IntTask]]
// CIR:   cir.store{{.*}} %[[IntTaskTmp]], %[[IntTaskAddr]] : ![[IntTask]], !cir.ptr<![[IntTask]]>

// CIR:   cir.await(user, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   %[[ResumeVal:.*]] = cir.call @_ZN5folly4coro4TaskIiE12await_resumeEv(%[[IntTaskAddr]])
// CIR:   cir.store{{.*}} %[[ResumeVal]], %[[CoReturnValAddr:.*]] : !s32i, !cir.ptr<!s32i>
// CIR:   },)
// CIR:   %[[V:.*]] = cir.load{{.*}} %[[CoReturnValAddr]] : !cir.ptr<!s32i>, !s32i
// CIR:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi({{.*}}, %[[V]])
// CIR:   cir.co_return

// CIR: }, finalSuspend : {
// CIR:  cir.await(final, ready : {
// CIR:  }, suspend : {
// CIR:  }, resume : {
// CIR:  },)
// CIR:  cir.yield
// CIR: }, destroy : {
// CIR: }, exit : {

// The value produced in the resume region is spilled and reloaded after the
// await, so it is still available in the continuation block.

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z3go1v()
// CIR-FLAT:   %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-FLAT:   %[[TASK:.*]] = cir.alloca "task"
// CIR-FLAT:   %[[RESUME_RVAL:.*]] = cir.alloca "__coawait_resume_rval"
// CIR-FLAT:   cir.call @_Z2goRKi
// CIR-FLAT:   %[[READY:.*]] = cir.call @_ZN5folly4coro4TaskIiE11await_readyEv(%[[TASK]])
// CIR-FLAT:   cir.brcond %[[READY]] ^[[RESUME:bb[0-9]+]], ^[[SUSPEND:bb[0-9]+]]
// CIR-FLAT: ^[[SUSPEND]]:
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE13await_suspendESt16coroutine_handleIvE(%[[TASK]])
// CIR-FLAT:   cir.const #false
// CIR-FLAT:   cir.coro.intrinsic.suspend
// CIR-FLAT: ^[[RESUME]]:
// CIR-FLAT:   %[[RESUME_VAL:.*]] = cir.call @_ZN5folly4coro4TaskIiE12await_resumeEv(%[[TASK]])
// CIR-FLAT:   cir.store %[[RESUME_VAL]], %[[RESUME_RVAL]]
// CIR-FLAT:   %[[V:.*]] = cir.load %[[RESUME_RVAL]]
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], %[[V]])
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type13final_suspendEv(%[[PROMISE]])
// CIR-FLAT:   cir.coro.intrinsic.end
// CIR-FLAT:   cir.return

// Classic codegen passes the await_resume result straight to return_value
// (no __coawait_resume_rval spill). After the co_return it jumps to
// coro.final through the cleanup dispatch (slot 3).

// OGCG: define {{.*}} void @_Z3go1v()
// OGCG: cleanup.cont:
// OGCG:   store i32 1, ptr %[[REF:.*]], align 4
// OGCG:   call void @_Z2goRKi(ptr {{.*}} %[[REF]])
// OGCG:   br i1 %{{.*}}, label %await.ready, label %await.suspend
// OGCG: await.suspend:
// OGCG:   %[[SAVE:.*]] = call token @llvm.coro.save(ptr null)
// OGCG:   %[[SUSP:.*]] = call i8 @llvm.coro.suspend(token %[[SAVE]], i1 false)
// OGCG:   switch i8 %[[SUSP]], label %coro.ret [
// OGCG-NEXT:     i8 0, label %await.ready
// OGCG-NEXT:     i8 1, label %await.cleanup
// OGCG-NEXT:   ]
// OGCG: await.ready:
// OGCG-NEXT:   %[[RVAL:.*]] = call noundef i32 @_ZN5folly4coro4TaskIiE12await_resumeEv(
// OGCG-NEXT:   call void @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(ptr {{.*}} %__promise, i32 noundef %[[RVAL]])
// OGCG:   %[[SLOT:.*]] = phi i32 [ 3, %await.ready ], [ 2, %await.cleanup ]
// OGCG:   switch i32 %[[SLOT]], label %unreachable [
// OGCG-NEXT:     i32 2, label %coro.cleanup
// OGCG-NEXT:     i32 3, label %coro.final
// OGCG-NEXT:   ]
// OGCG: coro.final:
// OGCG:   call void @_ZN5folly4coro4TaskIiE12promise_type13final_suspendEv(
// OGCG: coro.cleanup:
// OGCG:   call ptr @llvm.coro.free(
// OGCG: coro.ret:
// OGCG-NEXT:   call void @llvm.coro.end(ptr null, i1 false, token none)
// OGCG-NEXT:   ret void

folly::coro::Task<int> go1_lambda() {
  auto task = []() -> folly::coro::Task<int> {
    co_return 1;
  }();
  co_return co_await task;
}

// CIR: cir.func coroutine {{.*}} @_Z10go1_lambdav() call_conv_lowered attributes
// CIR: cir.coroutine initialSuspend : {
// CIR:   cir.await(init, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR:   cir.yield
// CIR: }, body : {
// CIR:    cir.call @_ZZ10go1_lambdavENK3$_0clEv
// CIR:    cir.await(user, ready : {
// CIR:    }, suspend : {
// CIR:    }, resume : {
// CIR:      %[[RESUME_RES:.*]] = cir.call @_ZN5folly4coro4TaskIiE12await_resumeEv(%[[TASK:.*]])
// CIR:      cir.store %[[RESUME_RES]], %[[resume_rval:.*]] : !s32i, !cir.ptr<!s32i>
// CIR:    },)
// CIR:    %[[TMP1:.*]] = cir.load %[[resume_rval:.*]] : !cir.ptr<!s32i>
// CIR:    cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE:.*]], %[[TMP1]])
// CIR:    cir.co_return
// CIR: }, finalSuspend : {
// CIR:   cir.await(final, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR: }, destroy : {

// CIR: cir.func coroutine {{.*}} @_ZZ10go1_lambdavENK3$_0clEv{{.*}}) call_conv_lowered attributes
// CIR: cir.coroutine initialSuspend : {
// CIR:   cir.await(init, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR:   cir.yield
// CIR: }, body : {
// CIR:    %[[ONE:.*]] = cir.const #cir.int<1>
// CIR:    cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE:.*]], %[[ONE]])
// CIR:    cir.co_return
// CIR: }, finalSuspend : {
// CIR:   cir.await(final, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR:   cir.yield
// CIR: }, destroy : {

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z10go1_lambdav()
// CIR-FLAT:   %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-FLAT:   %[[TASK:.*]] = cir.alloca "task"
// CIR-FLAT:   %[[RESUME_RVAL:.*]] = cir.alloca "__coawait_resume_rval"
// CIR-FLAT:   cir.call @_ZZ10go1_lambdavENK3$_0clEv
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE11await_readyEv(%[[TASK]])
// CIR-FLAT:   cir.coro.intrinsic.suspend
// CIR-FLAT:   %[[RESUME_VAL:.*]] = cir.call @_ZN5folly4coro4TaskIiE12await_resumeEv(%[[TASK]])
// CIR-FLAT:   cir.store %[[RESUME_VAL]], %[[RESUME_RVAL]]
// CIR-FLAT:   %[[V:.*]] = cir.load %[[RESUME_RVAL]]
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], %[[V]])
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type13final_suspendEv(%[[PROMISE]])
// CIR-FLAT:   cir.return

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_ZZ10go1_lambdavENK3$_0clEv
// CIR-FLAT:   %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv
// CIR-FLAT:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], %[[ONE]])
// CIR-FLAT-NEXT:   cir.br ^[[FINAL:bb[0-9]+]]
// CIR-FLAT: ^[[FINAL]]:
// CIR-FLAT-NEXT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type13final_suspendEv(%[[PROMISE]])
// CIR-FLAT:   cir.return

// OGCG: define {{.*}} void @_Z10go1_lambdav()
// OGCG: cleanup.cont:
// OGCG:   call void @"_ZZ10go1_lambdavENK3$_0clEv"(
// OGCG:   br i1 %{{.*}}, label %await.ready, label %await.suspend
// OGCG: await.suspend:
// OGCG:   %[[SAVE:.*]] = call token @llvm.coro.save(ptr null)
// OGCG:   %[[SUSP:.*]] = call i8 @llvm.coro.suspend(token %[[SAVE]], i1 false)
// OGCG:   switch i8 %[[SUSP]], label %coro.ret [
// OGCG-NEXT:     i8 0, label %await.ready
// OGCG-NEXT:     i8 1, label %await.cleanup
// OGCG-NEXT:   ]
// OGCG: await.ready:
// OGCG-NEXT:   %[[RVAL:.*]] = call noundef i32 @_ZN5folly4coro4TaskIiE12await_resumeEv(
// OGCG-NEXT:   call void @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(ptr {{.*}} %__promise, i32 noundef %[[RVAL]])
// OGCG:   %[[SLOT:.*]] = phi i32 [ 3, %await.ready ], [ 2, %await.cleanup ]
// OGCG:   switch i32 %[[SLOT]], label %unreachable [
// OGCG-NEXT:     i32 2, label %coro.cleanup
// OGCG-NEXT:     i32 3, label %coro.final
// OGCG-NEXT:   ]
// OGCG: coro.final:
// OGCG:   call void @_ZN5folly4coro4TaskIiE12promise_type13final_suspendEv(
// OGCG: coro.cleanup:
// OGCG:   call ptr @llvm.coro.free(
// OGCG: coro.ret:
// OGCG-NEXT:   call void @llvm.coro.end(ptr null, i1 false, token none)
// OGCG-NEXT:   ret void

// OGCG: define internal void @"_ZZ10go1_lambdavENK3$_0clEv"(
// OGCG: cleanup.cont:
// OGCG-NEXT:   call void @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(ptr {{.*}} %__promise, i32 noundef 1)
// OGCG-NEXT:   br label %coro.final
// OGCG: coro.final:
// OGCG:   call void @_ZN5folly4coro4TaskIiE12promise_type13final_suspendEv(
// OGCG: coro.cleanup:
// OGCG:   call ptr @llvm.coro.free(
// OGCG: coro.ret:
// OGCG-NEXT:   call void @llvm.coro.end(ptr null, i1 false, token none)
// OGCG-NEXT:   ret void

folly::coro::Task<int> go4() {
  auto* fn = +[](int const& i) -> folly::coro::Task<int> { co_return i; };
  auto task = fn(3);
  co_return co_await std::move(task);
}

// CIR: cir.func coroutine {{.*}} @_Z3go4v() call_conv_lowered attributes
// CIR: cir.coroutine initialSuspend : {
// CIR:   cir.await(init, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR:   cir.yield
// CIR: }, body : {

// Get the lambda invoker ptr via `lambda operator folly::coro::Task<int> (*)(int const&)()`
// CIR:  %[[INVOKER:.*]] = cir.call @_ZZ3go4vENK3$_0cvPFN5folly4coro4TaskIiEERKiEEv(%{{.*}}) nothrow : {{.*}} -> (!cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>> {llvm.noundef})
// CIR:  cir.store{{.*}} %[[INVOKER]], %[[FN_ADDR:.*]] : !cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>, !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>>
// CIR:  %[[FN:.*]] = cir.load{{.*}} %[[FN_ADDR]] : !cir.ptr<!cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>>, !cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>>
// CIR:  %[[THREE:.*]] = cir.const #cir.int<3> : !s32i
// CIR:  cir.store{{.*}} %[[THREE]], %[[ARG:.*]] : !s32i, !cir.ptr<!s32i>

// Call invoker, which calls operator() indirectly. The callee is bitcast to
// the lowered signature.

// CIR: %[[FN_LOWERED:.*]] = cir.cast bitcast %[[FN]] : !cir.ptr<!cir.func<(!cir.ptr<!s32i>) -> ![[IntTask]]>> -> !cir.ptr<!cir.func<(!cir.ptr<!s32i>)>>
// CIR: cir.call %[[FN_LOWERED]](%[[ARG]]) : (!cir.ptr<!cir.func<(!cir.ptr<!s32i>)>>, !cir.ptr<!s32i> {{.*}}) -> ()
// CIR: %[[CALLRES:.*]] = cir.const #cir.poison : ![[IntTask]]
// CIR: cir.store{{.*}} %[[CALLRES]], %[[TASK_ADDR:.*]] : ![[IntTask]], !cir.ptr<![[IntTask]]>

// CIR:  cir.await(user, ready : {
// CIR:    = cir.call @_ZN5folly4coro4TaskIiE11await_readyEv(%[[TASK_ADDR]])
// CIR:    cir.condition(
// CIR:  }, suspend : {
// CIR:    cir.coro.suspend_point
// CIR:  }, resume : {
// CIR:    cir.yield
// CIR:  },)
// CIR:  cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi
// CIR:  cir.co_return
// CIR: }, finalSuspend : {

// CIR:  cir.await(final, ready : {
// CIR:  }, suspend : {
// CIR:  }, resume : {
// CIR:  },)
// CIR: }, destroy : {

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z3go4v()
// CIR-FLAT:   %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-FLAT:   %[[TASK:.*]] = cir.alloca "task"
// CIR-FLAT:   cir.call @_ZZ3go4vENK3$_0cvPFN5folly4coro4TaskIiEERKiEEv
// CIR-FLAT:   %[[FN:.*]] = cir.cast bitcast
// CIR-FLAT:   cir.call %[[FN]](
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE11await_readyEv(%[[TASK]])
// CIR-FLAT:   cir.coro.intrinsic.suspend
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12await_resumeEv(%[[TASK]])
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], {{.*}})
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type13final_suspendEv(%[[PROMISE]])
// CIR-FLAT:   cir.coro.intrinsic.free
// CIR-FLAT:   cir.return

// OGCG: define {{.*}} void @_Z3go4v()
// OGCG: cleanup.cont:
// OGCG:   %[[CONV:.*]] = call noundef ptr @"_ZZ3go4vENK3$_0cvPFN5folly4coro4TaskIiEERKiEEv"(
// OGCG:   store ptr %[[CONV]], ptr %[[FN_ADDR:.*]], align 8
// OGCG:   %[[FN:.*]] = load ptr, ptr %[[FN_ADDR]], align 8
// OGCG:   store i32 3, ptr %[[REF:.*]], align 4
// OGCG:   call void %[[FN]](ptr {{.*}} %[[REF]])
// OGCG:   br i1 %{{.*}}, label %await.ready, label %await.suspend
// OGCG: await.suspend:
// OGCG:   %[[SAVE:.*]] = call token @llvm.coro.save(ptr null)
// OGCG:   %[[SUSP:.*]] = call i8 @llvm.coro.suspend(token %[[SAVE]], i1 false)
// OGCG:   switch i8 %[[SUSP]], label %coro.ret [
// OGCG-NEXT:     i8 0, label %await.ready
// OGCG-NEXT:     i8 1, label %await.cleanup
// OGCG-NEXT:   ]
// OGCG: await.ready:
// OGCG-NEXT:   %[[RVAL:.*]] = call noundef i32 @_ZN5folly4coro4TaskIiE12await_resumeEv(
// OGCG-NEXT:   call void @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(ptr {{.*}} %__promise, i32 noundef %[[RVAL]])
// OGCG:   %[[SLOT:.*]] = phi i32 [ 3, %await.ready ], [ 2, %await.cleanup ]
// OGCG:   switch i32 %[[SLOT]], label %unreachable [
// OGCG-NEXT:     i32 2, label %coro.cleanup
// OGCG-NEXT:     i32 3, label %coro.final
// OGCG-NEXT:   ]
// OGCG: coro.final:
// OGCG:   call void @_ZN5folly4coro4TaskIiE12promise_type13final_suspendEv(
// OGCG: coro.cleanup:
// OGCG:   call ptr @llvm.coro.free(
// OGCG: coro.ret:
// OGCG-NEXT:   call void @llvm.coro.end(ptr null, i1 false, token none)
// OGCG-NEXT:   ret void

folly::coro::Task<int> co_returns(int x) {
  if (x < 0)
    co_return -1;
  else if (x == 1)
   co_return -2;

  co_await std::suspend_always();

  co_return x * 2;
}

// CIR: cir.func coroutine {{.*}} @_Z10co_returnsi
// CIR: cir.coroutine initialSuspend : {
// CIR:   cir.await(init, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR:   cir.yield
// CIR: }, body : {
// CIR:    cir.scope {
// CIR:      cir.if {{.*}} {
// CIR:        %[[MINUS_ONE:.*]] = cir.const #cir.int<-1>
// CIR:        cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE:.*]], %[[MINUS_ONE]])
// CIR:        cir.co_return
// CIR:      } else {
// CIR:       cir.if {{.*}} {
// CIR:         %[[MINUS_TWO:.*]] = cir.const #cir.int<-2>
// CIR          cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], %[[MINUS_TWO]])
// CIR          cir.co_return
// CIR:         }
// CIR:      }
// CIR:    }
// CIR:   cir.await(user, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR:   %[[X_LOAD:.*]] = cir.load {{.*}} %[[X:.*]]
// CIR:   %[[TWO:.*]] = cir.const #cir.int<2>
// CIR:   %[[RES:.*]] = cir.mul nsw %[[X_LOAD]], %[[TWO]]
// CIR:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], %[[RES]])
// CIR:   cir.co_return
// CIR: }, finalSuspend : {
// CIR:  cir.await(final, ready : {
// CIR:  }, suspend : {
// CIR:  }, resume : {
// CIR:  },)
// CIR:  cir.yield
// CIR: }, destroy : {


// OGCG: define {{.*}} @_Z10co_returnsi
// OGCG: coro.init:
// OGCG: init.suspend:
// OGCG: init.ready:
// OGCG: if.then:
// OGCG:   call void @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi({{.*}} %[[promise:.*]], {{.*}} -1)
// OGCG:   br label %coro.final
// OGCG: if.else:
// OGCG: if.then5:
// OGCG:   call void @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi({{.*}} %[[promise:.*]], {{.*}} -2)
// OGCG:   br label %coro.final
// OGCG: await.suspend:
// OGCG: await.ready:
// OGCG: coro.final:
// OGCG: final.suspend:
// OGCG: final.ready:

// Every cir.co_return, including the ones nested in cir.if, branches to the
// same final suspend block.

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z10co_returnsi
// CIR-FLAT:   cir.alloca "__coro_frame_addr"
// CIR-FLAT:   %[[X:.*]] = cir.alloca "x"
// CIR-FLAT:   %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv
// CIR-FLAT:   %[[LT:.*]] = cir.cmp lt
// CIR-FLAT:   cir.brcond %[[LT]] ^[[IF_LT:bb[0-9]+]], ^[[ELSE:bb[0-9]+]]
// CIR-FLAT: ^[[IF_LT]]:
// CIR-FLAT:   %[[MINUS_ONE:.*]] = cir.const #cir.int<-1> : !s32i
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], %[[MINUS_ONE]])
// CIR-FLAT-NEXT:   cir.br ^[[FINAL:bb[0-9]+]]
// CIR-FLAT: ^[[ELSE]]:
// CIR-FLAT:   %[[EQ:.*]] = cir.cmp eq
// CIR-FLAT:   cir.brcond %[[EQ]] ^[[IF_EQ:bb[0-9]+]], ^[[AFTER_IF:bb[0-9]+]]
// CIR-FLAT: ^[[IF_EQ]]:
// CIR-FLAT:   %[[MINUS_TWO:.*]] = cir.const #cir.int<-2> : !s32i
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], %[[MINUS_TWO]])
// CIR-FLAT-NEXT:   cir.br ^[[FINAL]]
// CIR-FLAT: ^[[AFTER_IF]]:

// The user co_await between the early returns and the last co_return.

// CIR-FLAT:   cir.call @_ZNSt14suspend_always11await_readyEv
// CIR-FLAT:   cir.coro.intrinsic.suspend
// CIR-FLAT:   cir.switch.flat %{{.*}} : !s8i, ^{{.*}} [
// CIR-FLAT-NEXT:     0: ^[[USER_RESUME:bb[0-9]+]],
// CIR-FLAT-NEXT:     1: ^[[USER_DESTROY:bb[0-9]+]]
// CIR-FLAT-NEXT:   ]
// CIR-FLAT: ^[[USER_DESTROY]]:
// CIR-FLAT-NEXT:   cir.br ^[[CORO_CLEANUP:bb[0-9]+]]
// CIR-FLAT: ^[[USER_RESUME]]:
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv
// CIR-FLAT:   %[[X_LOAD:.*]] = cir.load {{.*}} %[[X]]
// CIR-FLAT:   %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR-FLAT:   %[[MUL:.*]] = cir.mul nsw %[[X_LOAD]], %[[TWO]]
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], %[[MUL]])
// CIR-FLAT-NEXT:   cir.br ^[[FINAL]]
// CIR-FLAT: ^[[FINAL]]:
// CIR-FLAT-NEXT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type13final_suspendEv(%[[PROMISE]])
// CIR-FLAT: ^[[CORO_CLEANUP]]:
// CIR-FLAT-NEXT:   cir.coro.intrinsic.free
// CIR-FLAT:   cir.return

folly::coro::Task<int> implicit_yield_terminator(int x) {
  if (x < 0)
    co_return -1;
  else
    co_return 2;
}
// CIR: cir.func coroutine {{.*}} @_Z25implicit_yield_terminatori
// CIR:   }, body : {
// CIR:     cir.if %{{.*}} {
// CIR:       cir.co_return
// CIR:     } else {
// CIR:       cir.co_return
// CIR:     }
// CIR:     cir.yield
// CIR:   }, finalSuspend : {

// Both branches end in a cir.co_return; the implicit cir.yield after the
// cir.if is unreachable but still branches to the final suspend.

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z25implicit_yield_terminatori
// CIR-FLAT:   %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv
// CIR-FLAT:   cir.brcond %{{.*}} ^[[THEN:bb[0-9]+]], ^[[ELSE:bb[0-9]+]]
// CIR-FLAT: ^[[THEN]]:
// CIR-FLAT:   %[[MINUS_ONE:.*]] = cir.const #cir.int<-1> : !s32i
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], %[[MINUS_ONE]])
// CIR-FLAT-NEXT:   cir.br ^[[FINAL:bb[0-9]+]]
// CIR-FLAT: ^[[ELSE]]:
// CIR-FLAT:   %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], %[[TWO]])
// CIR-FLAT-NEXT:   cir.br ^[[FINAL]]
// CIR-FLAT: ^[[FINAL]]:
// CIR-FLAT-NEXT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type13final_suspendEv(%[[PROMISE]])
// CIR-FLAT:   cir.return

// OGCG: define {{.*}} void @_Z25implicit_yield_terminatori(
// OGCG: cleanup.cont:
// OGCG:   %[[CMP:.*]] = icmp slt i32 %{{.*}}, 0
// OGCG:   br i1 %[[CMP]], label %if.then, label %if.else
// OGCG: if.then:
// OGCG-NEXT:   call void @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(ptr {{.*}} %__promise, i32 noundef -1)
// OGCG-NEXT:   br label %coro.final
// OGCG: if.else:
// OGCG-NEXT:   call void @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(ptr {{.*}} %__promise, i32 noundef 2)
// OGCG-NEXT:   br label %coro.final
// OGCG: coro.final:
// OGCG:   call void @_ZN5folly4coro4TaskIiE12promise_type13final_suspendEv(
// OGCG: coro.cleanup:
// OGCG:   call ptr @llvm.coro.free(
// OGCG: coro.ret:
// OGCG-NEXT:   call void @llvm.coro.end(ptr null, i1 false, token none)
// OGCG-NEXT:   ret void

struct HasDtor {
  ~ HasDtor();
};

folly::coro::Task<int> co_return_with_dtor(int flag) {
  HasDtor local;
  if (flag)
    co_return 1;        // local dtor must run here
  co_return 2;
}

// CIR: cir.func coroutine {{.*}} @_Z19co_return_with_dtori

// CIR: %[[LOCAL:.*]] = cir.alloca "local" {{.*}} : !cir.ptr<!rec_HasDtor>
// CIR: cir.coroutine initialSuspend : {
// CIR:   cir.await(init, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR:   cir.yield

// CIR: }, body : {
// CIR:   cir.cleanup.scope {
// CIR:     cir.scope {
// CIR:       %[[CAST_FLAG:.*]] = cir.cast int_to_bool %[[FLAG:.*]]
// CIR:       cir.if %[[CAST_FLAG]] {
// CIR:         %[[ONE:.*]] = cir.const #cir.int<1>
// CIR:         cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[promise:.*]], %[[ONE]])
// CIR:         cir.co_return
// CIR:       }
// CIR:     }
// CIR:     %[[TWO:.*]] = cir.const #cir.int<2>
// CIR:     cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[promise]], %[[TWO]])
// CIR:     cir.co_return
// CIR:   } cleanup normal {
// CIR:     cir.call @_ZN7HasDtorD1Ev(%[[LOCAL]])
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.yield

// CIR: }, finalSuspend : {
// CIR:   cir.await(final, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)
// CIR:   cir.yield
// CIR: }, destroy : {
// CIR:   cir.yield
// CIR: }

// OGCG: define {{.*}} void @_Z19co_return_with_dtori
// OGCG:   %[[LOCAL:.*]] = alloca %struct.HasDtor
// OGCG: coro.init:
// OGCG: init.suspend:
// OGCG: init.ready:
// OGCG: if.then:
// OGCG:   call void @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi({{.*}} %[[promise:.*]], {{.*}} 1)
// OGCG:   br label %cleanup4
// OGCG: if.end:
// OGCG:   call void @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi({{.*}} %[[promise]], {{.*}} 2)
// OGCG:   br label %cleanup4
// OGCG: cleanup4:
// OGCG:   call void @_ZN7HasDtorD1Ev({{.*}} %[[LOCAL]])
// OGCG:   br label %coro.final

// A cir.co_return nested in a cleanup scope exits through the cleanup: it
// records its destination in the cleanup dest slot, runs ~HasDtor, and only
// then continues to the final suspend.

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z19co_return_with_dtori
// CIR-FLAT:   %[[CLEANUP_DEST_SLOT:.*]] = cir.alloca "__cleanup_dest_slot"
// CIR-FLAT:   %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-FLAT:   %[[LOCAL:.*]] = cir.alloca "local"
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv
// CIR-FLAT:   %[[FLAG:.*]] = cir.cast int_to_bool
// CIR-FLAT:   cir.brcond %[[FLAG]] ^[[IF_BODY:bb[0-9]+]], ^[[IF_END:bb[0-9]+]]
// CIR-FLAT: ^[[IF_BODY]]:
// CIR-FLAT:   %[[ONE:.*]] = cir.const #cir.int<1> : !s32i
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], %[[ONE]])
// CIR-FLAT-NEXT:   %[[DEST_RET1:.*]] = cir.const #cir.int<1> : !s32i
// CIR-FLAT-NEXT:   cir.store %[[DEST_RET1]], %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.br ^[[DTOR:bb[0-9]+]]
// CIR-FLAT: ^[[IF_END]]:
// CIR-FLAT-NEXT:   cir.br ^[[AFTER_IF:bb[0-9]+]]
// CIR-FLAT: ^[[AFTER_IF]]:
// CIR-FLAT:   %[[TWO:.*]] = cir.const #cir.int<2> : !s32i
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], %[[TWO]])
// CIR-FLAT-NEXT:   %[[DEST_RET2:.*]] = cir.const #cir.int<2> : !s32i
// CIR-FLAT-NEXT:   cir.store %[[DEST_RET2]], %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.br ^[[DTOR]]

// The normal fallthrough out of the cleanup scope gets slot 0. It is dead
// here because both paths end in a cir.co_return.
// CIR-FLAT: ^[[FALLTHROUGH:bb[0-9]+]]:  // no predecessors
// CIR-FLAT-NEXT:   %[[DEST_FALLTHROUGH:.*]] = cir.const #cir.int<0> : !s32i
// CIR-FLAT-NEXT:   cir.store %[[DEST_FALLTHROUGH]], %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.br ^[[DTOR]]

// CIR-FLAT: ^[[DTOR]]:
// CIR-FLAT-NEXT:   cir.call @_ZN7HasDtorD1Ev(%[[LOCAL]])
// CIR-FLAT-NEXT:   cir.br ^[[DTOR_EXIT:bb[0-9]+]]
// CIR-FLAT: ^[[DTOR_EXIT]]:
// CIR-FLAT-NEXT:   %[[DEST:.*]] = cir.load %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.switch.flat %[[DEST]] : !s32i, ^[[DEFAULT:bb[0-9]+]] [
// CIR-FLAT-NEXT:     0: ^[[FROM_FALLTHROUGH:bb[0-9]+]],
// CIR-FLAT-NEXT:     1: ^[[FROM_RET1:bb[0-9]+]],
// CIR-FLAT-NEXT:     2: ^[[FROM_RET2:bb[0-9]+]]
// CIR-FLAT-NEXT:   ]
// CIR-FLAT: ^[[FROM_FALLTHROUGH]]:
// CIR-FLAT-NEXT:   cir.br ^[[AFTER_CLEANUP:bb[0-9]+]]
// CIR-FLAT: ^[[FROM_RET1]]:
// CIR-FLAT-NEXT:   cir.br ^[[FINAL:bb[0-9]+]]
// CIR-FLAT: ^[[FROM_RET2]]:
// CIR-FLAT-NEXT:   cir.br ^[[FINAL]]
// CIR-FLAT: ^[[DEFAULT]]:
// CIR-FLAT-NEXT:   cir.unreachable
// CIR-FLAT: ^[[AFTER_CLEANUP]]:
// CIR-FLAT-NEXT:   cir.br ^[[FINAL]]
// CIR-FLAT: ^[[FINAL]]:
// CIR-FLAT-NEXT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type13final_suspendEv(%[[PROMISE]])
// CIR-FLAT:   cir.coro.intrinsic.free
// CIR-FLAT:   cir.return

VoidTask silly_task_with_dtor() {
  HasDtor local;
  co_await std::suspend_always();
}

// CIR: cir.func coroutine {{.*}} @_Z20silly_task_with_dtorv
// CIR:   %[[LOCAL:.*]] = cir.alloca "local"
// CIR: }, body : {
// CIR:   cir.cleanup.scope {
// CIR:     cir.await(user, ready : {
// CIR:     }, suspend : {
// CIR:     }, resume : {
// CIR:     },)
// CIR:     cir.yield
// CIR:   } cleanup normal {
// CIR:     cir.call @_ZN7HasDtorD1Ev(%[[LOCAL]])
// CIR:     cir.yield
// CIR:   }
// CIR:   cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv
// CIR:   cir.co_return
// CIR: }, finalSuspend : {

// The user await is nested in the lifetime of `local`, so both of its exits
// run ~HasDtor first and then dispatch on the cleanup dest slot: resume
// (slot 0) continues into the rest of the body (return_void + final suspend),
// and destroy (slot 1) goes on to the coroutine cleanup (coro.free) without
// running the rest of the body. This matches classic codegen below, where
// the destroy case (slot 2) jumps from the ~HasDtor cleanup to %coro.cleanup.

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z20silly_task_with_dtorv()
// CIR-FLAT:   %[[CLEANUP_DEST_SLOT:.*]] = cir.alloca "__cleanup_dest_slot"
// CIR-FLAT:   %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-FLAT:   %[[LOCAL:.*]] = cir.alloca "local"
// CIR-FLAT:   %[[USER_AWAITER:.*]] = cir.alloca "ref.tmp1"
// CIR-FLAT:   %[[READY:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[USER_AWAITER]])
// CIR-FLAT-NEXT:   cir.brcond %[[READY]] ^[[USER_RESUME:bb[0-9]+]], ^[[USER_SUSPEND:bb[0-9]+]]
// CIR-FLAT: ^[[USER_SUSPEND]]:
// CIR-FLAT:   %[[RES:.*]] = cir.coro.intrinsic.suspend
// CIR-FLAT:   cir.switch.flat %[[RES]] : !s8i, ^{{.*}} [
// CIR-FLAT-NEXT:     0: ^[[USER_RESUME]],
// CIR-FLAT-NEXT:     1: ^[[USER_DESTROY:bb[0-9]+]]
// CIR-FLAT-NEXT:   ]
// CIR-FLAT: ^[[USER_DESTROY]]:
// CIR-FLAT-NEXT:   %[[DEST_DESTROY:.*]] = cir.const #cir.int<1> : !s32i
// CIR-FLAT-NEXT:   cir.store %[[DEST_DESTROY]], %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.br ^[[DTOR:bb[0-9]+]]
// CIR-FLAT: ^[[USER_RESUME]]:
// CIR-FLAT-NEXT:   cir.call @_ZNSt14suspend_always12await_resumeEv(%[[USER_AWAITER]])
// CIR-FLAT:   %[[DEST_NORMAL:.*]] = cir.const #cir.int<0> : !s32i
// CIR-FLAT-NEXT:   cir.store %[[DEST_NORMAL]], %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.br ^[[DTOR]]
// CIR-FLAT: ^[[DTOR]]:
// CIR-FLAT-NEXT:   cir.call @_ZN7HasDtorD1Ev(%[[LOCAL]])
// CIR-FLAT-NEXT:   cir.br ^[[DTOR_EXIT:bb[0-9]+]]
// CIR-FLAT: ^[[DTOR_EXIT]]:
// CIR-FLAT-NEXT:   %[[DEST:.*]] = cir.load %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.switch.flat %[[DEST]] : !s32i, ^{{.*}} [
// CIR-FLAT-NEXT:     0: ^[[DTOR_TO_BODY:bb[0-9]+]],
// CIR-FLAT-NEXT:     1: ^[[DTOR_TO_DESTROY:bb[0-9]+]]
// CIR-FLAT-NEXT:   ]
// CIR-FLAT: ^[[DTOR_TO_BODY]]:
// CIR-FLAT-NEXT:   cir.br ^[[AFTER_DTOR:bb[0-9]+]]

// Destroy: after ~HasDtor, go on to the coroutine cleanup.
// CIR-FLAT: ^[[DTOR_TO_DESTROY]]:
// CIR-FLAT-NEXT:   cir.br ^[[CORO_CLEANUP:bb[0-9]+]]

// Resume: the rest of the body.
// CIR-FLAT: ^[[AFTER_DTOR]]:
// CIR-FLAT-NEXT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(%[[PROMISE]])
// CIR-FLAT-NEXT:   cir.br ^[[FINAL:bb[0-9]+]]
// CIR-FLAT: ^[[FINAL]]:
// CIR-FLAT-NEXT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type13final_suspendEv(%[[PROMISE]])
// CIR-FLAT: ^[[CORO_CLEANUP]]:
// CIR-FLAT-NEXT:   cir.coro.intrinsic.free
// CIR-FLAT:   cir.coro.intrinsic.end
// CIR-FLAT:   cir.return

// OGCG: define {{.*}} void @_Z20silly_task_with_dtorv
// OGCG:   %[[LOCAL:.*]] = alloca %struct.HasDtor
// OGCG: await.suspend:
// OGCG:   switch i8 %{{.*}}, label %coro.ret [
// OGCG-NEXT:     i8 0, label %await.ready
// OGCG-NEXT:     i8 1, label %await.cleanup
// OGCG-NEXT:   ]
// OGCG: await.cleanup:
// OGCG-NEXT:   br label %[[AWAIT_CLEANUP:cleanup[0-9]+]]
// OGCG: await.ready:
// OGCG-NEXT:   call void @_ZNSt14suspend_always12await_resumeEv(
// OGCG-NEXT:   br label %[[AWAIT_CLEANUP]]

// Slot 0 = normal, slot 2 = destroy. Both reach the ~HasDtor cleanup.
// OGCG: [[AWAIT_CLEANUP]]:
// OGCG-NEXT:   %[[SLOT1:.*]] = phi i32 [ 0, %await.ready ], [ 2, %await.cleanup ]
// OGCG:   switch i32 %[[SLOT1]], label %[[DTOR:cleanup[0-9]+]] [
// OGCG-NEXT:     i32 0, label %[[AWAIT_CONT:cleanup.cont[0-9]+]]
// OGCG-NEXT:   ]
// OGCG: [[AWAIT_CONT]]:
// OGCG-NEXT:   br label %[[DTOR]]
// OGCG: [[DTOR]]:
// OGCG-NEXT:   %[[SLOT2:.*]] = phi i32 [ %[[SLOT1]], %[[AWAIT_CLEANUP]] ], [ 0, %[[AWAIT_CONT]] ]
// OGCG-NEXT:   call void @_ZN7HasDtorD1Ev({{.*}} %[[LOCAL]])
// OGCG:   switch i32 %[[SLOT2]], label %unreachable [
// OGCG-NEXT:     i32 0, label %[[DTOR_CONT:cleanup.cont[0-9]+]]
// OGCG-NEXT:     i32 2, label %coro.cleanup
// OGCG-NEXT:   ]

// Normal: return_void + final suspend. Destroy: %coro.cleanup (coro.free).
// OGCG: [[DTOR_CONT]]:
// OGCG-NEXT:   call void @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(
// OGCG-NEXT:   br label %coro.final
// OGCG: coro.final:
// OGCG:   call void @_ZN5folly4coro4TaskIvE12promise_type13final_suspendEv(
// OGCG: coro.cleanup:
// OGCG:   call ptr @llvm.coro.free(
// OGCG: coro.ret:
// OGCG-NEXT:   call void @llvm.coro.end(ptr null, i1 false, token none)
// OGCG-NEXT:   ret void

VoidTask await_in_loop(int n) {
  for (int i = 0; i < n; ++i)
    co_await std::suspend_always();
}

// CIR: cir.func coroutine {{.*}} @_Z13await_in_loopi
// CIR: }, body : {
// CIR:   cir.scope {
// CIR:     cir.for : cond {
// CIR:     } body {
// CIR:       cir.await(user, ready : {
// CIR:       }, suspend : {
// CIR:         cir.coro.suspend_point
// CIR:       }, resume : {
// CIR:       },)
// CIR:       cir.yield
// CIR:     } step {
// CIR:   cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv
// CIR:   cir.co_return
// CIR: }, finalSuspend : {

// The destroy case of the user await leaves the loop and goes to the
// coroutine cleanup, like classic codegen below. Resuming continues to the
// step (++i).

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z13await_in_loopi(
// CIR-FLAT:   %[[I:.*]] = cir.alloca "i"
// CIR-FLAT:   %[[USER_AWAITER:.*]] = cir.alloca "ref.tmp1"
// CIR-FLAT:   cir.store {{.*}} %{{.*}}, %[[I]]
// CIR-FLAT-NEXT:   cir.br ^[[COND:bb[0-9]+]]
// CIR-FLAT: ^[[COND]]:
// CIR-FLAT:   %[[CMP:.*]] = cir.cmp lt
// CIR-FLAT-NEXT:   cir.brcond %[[CMP]] ^{{.*}}, ^[[LOOP_EXIT:bb[0-9]+]]
// CIR-FLAT:   %[[READY:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[USER_AWAITER]])
// CIR-FLAT-NEXT:   cir.brcond %[[READY]] ^[[USER_RESUME:bb[0-9]+]], ^[[USER_SUSPEND:bb[0-9]+]]
// CIR-FLAT: ^[[USER_SUSPEND]]:
// CIR-FLAT:   %[[RES:.*]] = cir.coro.intrinsic.suspend
// CIR-FLAT:   cir.switch.flat %[[RES]] : !s8i, ^{{.*}} [
// CIR-FLAT-NEXT:     0: ^[[USER_RESUME]],
// CIR-FLAT-NEXT:     1: ^[[USER_DESTROY:bb[0-9]+]]
// CIR-FLAT-NEXT:   ]
// CIR-FLAT: ^[[USER_DESTROY]]:
// CIR-FLAT-NEXT:   cir.br ^[[CORO_CLEANUP:bb[0-9]+]]
// CIR-FLAT: ^[[USER_RESUME]]:
// CIR-FLAT-NEXT:   cir.call @_ZNSt14suspend_always12await_resumeEv(%[[USER_AWAITER]])
// CIR-FLAT:   %[[I_VAL:.*]] = cir.load {{.*}}%[[I]]
// CIR-FLAT-NEXT:   %[[INC:.*]] = cir.inc nsw %[[I_VAL]]
// CIR-FLAT-NEXT:   cir.store {{.*}}%[[INC]], %[[I]]
// CIR-FLAT-NEXT:   cir.br ^[[COND]]
// CIR-FLAT: ^[[LOOP_EXIT]]:
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type13final_suspendEv
// CIR-FLAT: ^[[CORO_CLEANUP]]:
// CIR-FLAT-NEXT:   cir.coro.intrinsic.free
// CIR-FLAT:   cir.coro.intrinsic.end
// CIR-FLAT:   cir.return

// In classic codegen the destroy case (slot 2) leaves the loop through the
// loop cleanup and reaches %coro.cleanup; only the normal case (slot 0)
// continues to for.inc.

// OGCG: define {{.*}} void @_Z13await_in_loopi(
// OGCG: await.suspend:
// OGCG:   switch i8 %{{.*}}, label %coro.ret [
// OGCG-NEXT:     i8 0, label %await.ready
// OGCG-NEXT:     i8 1, label %await.cleanup
// OGCG-NEXT:   ]
// OGCG: await.cleanup:
// OGCG-NEXT:   br label %[[AWAIT_CLEANUP:cleanup[0-9]+]]
// OGCG: [[AWAIT_CLEANUP]]:
// OGCG-NEXT:   %[[SLOT1:.*]] = phi i32 [ 0, %await.ready ], [ 2, %await.cleanup ]
// OGCG:   switch i32 %[[SLOT1]], label %[[LOOP_CLEANUP:cleanup[0-9]+]] [
// OGCG-NEXT:     i32 0, label %[[AWAIT_CONT:cleanup.cont[0-9]+]]
// OGCG-NEXT:   ]
// OGCG: [[AWAIT_CONT]]:
// OGCG-NEXT:   br label %for.inc
// OGCG: [[LOOP_CLEANUP]]:
// OGCG-NEXT:   %[[SLOT2:.*]] = phi i32 [ %[[SLOT1]], %[[AWAIT_CLEANUP]] ], [ 4, %for.cond.cleanup ]
// OGCG:   switch i32 %[[SLOT2]], label %unreachable [
// OGCG-NEXT:     i32 4, label %for.end
// OGCG-NEXT:     i32 2, label %coro.cleanup
// OGCG-NEXT:   ]
// OGCG: for.end:
// OGCG-NEXT:   call void @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(
// OGCG-NEXT:   br label %coro.final
// OGCG: coro.cleanup:
// OGCG:   call ptr @llvm.coro.free(

VoidTask await_in_if(bool b) {
  if (b)
    co_await std::suspend_always();
}

// CIR: cir.func coroutine {{.*}} @_Z11await_in_ifb
// CIR: }, body : {
// CIR:   cir.scope {
// CIR:     cir.if %{{.*}} {
// CIR:       cir.await(user, ready : {
// CIR:       }, suspend : {
// CIR:         cir.coro.suspend_point
// CIR:       }, resume : {
// CIR:       },)
// CIR:     }
// CIR:   }
// CIR:   cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv
// CIR:   cir.co_return
// CIR: }, finalSuspend : {

// The destroy case of the user await inside the cir.if goes straight to the
// coroutine cleanup.

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z11await_in_ifb(
// CIR-FLAT:   cir.alloca "__coro_frame_addr"
// CIR-FLAT:   %[[B:.*]] = cir.alloca "b"
// CIR-FLAT:   %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-FLAT:   %[[COND:.*]] = cir.load {{.*}}%[[B]]
// CIR-FLAT-NEXT:   cir.brcond %[[COND]] ^[[THEN:bb[0-9]+]], ^[[IF_END:bb[0-9]+]]
// CIR-FLAT: ^[[THEN]]:
// CIR-FLAT-NEXT:   %[[USER_AWAITER:.*]] = cir.alloca "ref.tmp1"
// CIR-FLAT:   %[[READY:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[USER_AWAITER]])
// CIR-FLAT-NEXT:   cir.brcond %[[READY]] ^[[USER_RESUME:bb[0-9]+]], ^[[USER_SUSPEND:bb[0-9]+]]
// CIR-FLAT: ^[[USER_SUSPEND]]:
// CIR-FLAT:   %[[RES:.*]] = cir.coro.intrinsic.suspend
// CIR-FLAT:   cir.switch.flat %[[RES]] : !s8i, ^{{.*}} [
// CIR-FLAT-NEXT:     0: ^[[USER_RESUME]],
// CIR-FLAT-NEXT:     1: ^[[USER_DESTROY:bb[0-9]+]]
// CIR-FLAT-NEXT:   ]
// CIR-FLAT: ^[[USER_DESTROY]]:
// CIR-FLAT-NEXT:   cir.br ^[[CORO_CLEANUP:bb[0-9]+]]
// CIR-FLAT: ^[[USER_RESUME]]:
// CIR-FLAT-NEXT:   cir.call @_ZNSt14suspend_always12await_resumeEv(%[[USER_AWAITER]])
// CIR-FLAT: ^[[IF_END]]:
// CIR-FLAT-NEXT:   cir.br ^[[AFTER_IF:bb[0-9]+]]
// CIR-FLAT: ^[[AFTER_IF]]:
// CIR-FLAT-NEXT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(%[[PROMISE]])
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type13final_suspendEv(%[[PROMISE]])
// CIR-FLAT: ^[[CORO_CLEANUP]]:
// CIR-FLAT-NEXT:   cir.coro.intrinsic.free
// CIR-FLAT:   cir.coro.intrinsic.end
// CIR-FLAT:   cir.return

// OGCG: define {{.*}} void @_Z11await_in_ifb(
// OGCG: cleanup.cont:
// OGCG:   br i1 %{{.*}}, label %if.then, label %if.end
// OGCG: await.suspend:
// OGCG:   switch i8 %{{.*}}, label %coro.ret [
// OGCG-NEXT:     i8 0, label %await.ready
// OGCG-NEXT:     i8 1, label %await.cleanup
// OGCG-NEXT:   ]
// OGCG: await.cleanup:
// OGCG-NEXT:   br label %[[AWAIT_CLEANUP:cleanup[0-9]+]]
// OGCG: [[AWAIT_CLEANUP]]:
// OGCG-NEXT:   %[[SLOT:.*]] = phi i32 [ 0, %await.ready ], [ 2, %await.cleanup ]
// OGCG:   switch i32 %[[SLOT]], label %unreachable [
// OGCG-NEXT:     i32 0, label %[[AWAIT_CONT:cleanup.cont[0-9]+]]
// OGCG-NEXT:     i32 2, label %coro.cleanup
// OGCG-NEXT:   ]
// OGCG: [[AWAIT_CONT]]:
// OGCG-NEXT:   br label %if.end
// OGCG: if.end:
// OGCG-NEXT:   call void @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(
// OGCG-NEXT:   br label %coro.final
// OGCG: coro.cleanup:
// OGCG:   call ptr @llvm.coro.free(

VoidTask await_in_loop_with_dtor(int n) {
  for (int i = 0; i < n; ++i) {
    HasDtor local;
    co_await std::suspend_always();
  }
}

// CIR: cir.func coroutine {{.*}} @_Z23await_in_loop_with_dtori
// CIR: }, body : {
// CIR:   cir.for : cond {
// CIR:   } body {
// CIR:     cir.cleanup.scope {
// CIR:       cir.await(user, ready : {
// CIR:       },)
// CIR:       cir.yield
// CIR:     } cleanup normal {
// CIR:       cir.call @_ZN7HasDtorD1Ev
// CIR:       cir.yield
// CIR:     }
// CIR:   } step {
// CIR:   cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv
// CIR:   cir.co_return
// CIR: }, finalSuspend : {

// Same as silly_task_with_dtor inside a loop: after ~HasDtor, resume (slot 0)
// continues to the step (++i), and destroy (slot 1) leaves the loop and goes
// to the coroutine cleanup, like classic codegen below.

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z23await_in_loop_with_dtori(
// CIR-FLAT:   %[[CLEANUP_DEST_SLOT:.*]] = cir.alloca "__cleanup_dest_slot"
// CIR-FLAT:   %[[I:.*]] = cir.alloca "i"
// CIR-FLAT:   cir.store {{.*}} %{{.*}}, %[[I]]
// CIR-FLAT-NEXT:   cir.br ^[[COND:bb[0-9]+]]
// CIR-FLAT: ^[[COND]]:
// CIR-FLAT:   %[[CMP:.*]] = cir.cmp lt
// CIR-FLAT-NEXT:   cir.brcond %[[CMP]] ^{{.*}}, ^[[LOOP_EXIT:bb[0-9]+]]
// CIR-FLAT:   %[[LOCAL:.*]] = cir.alloca "local"
// CIR-FLAT-NEXT:   %[[USER_AWAITER:.*]] = cir.alloca "ref.tmp1"
// CIR-FLAT:   %[[READY:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[USER_AWAITER]])
// CIR-FLAT-NEXT:   cir.brcond %[[READY]] ^[[USER_RESUME:bb[0-9]+]], ^[[USER_SUSPEND:bb[0-9]+]]
// CIR-FLAT: ^[[USER_SUSPEND]]:
// CIR-FLAT:   %[[RES:.*]] = cir.coro.intrinsic.suspend
// CIR-FLAT:   cir.switch.flat %[[RES]] : !s8i, ^{{.*}} [
// CIR-FLAT-NEXT:     0: ^[[USER_RESUME]],
// CIR-FLAT-NEXT:     1: ^[[USER_DESTROY:bb[0-9]+]]
// CIR-FLAT-NEXT:   ]
// CIR-FLAT: ^[[USER_DESTROY]]:
// CIR-FLAT-NEXT:   %[[DEST_DESTROY:.*]] = cir.const #cir.int<1> : !s32i
// CIR-FLAT-NEXT:   cir.store %[[DEST_DESTROY]], %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.br ^[[DTOR:bb[0-9]+]]
// CIR-FLAT: ^[[USER_RESUME]]:
// CIR-FLAT-NEXT:   cir.call @_ZNSt14suspend_always12await_resumeEv(%[[USER_AWAITER]])
// CIR-FLAT:   %[[DEST_NORMAL:.*]] = cir.const #cir.int<0> : !s32i
// CIR-FLAT-NEXT:   cir.store %[[DEST_NORMAL]], %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.br ^[[DTOR]]
// CIR-FLAT: ^[[DTOR]]:
// CIR-FLAT-NEXT:   cir.call @_ZN7HasDtorD1Ev(%[[LOCAL]])
// CIR-FLAT-NEXT:   cir.br ^[[DTOR_EXIT:bb[0-9]+]]
// CIR-FLAT: ^[[DTOR_EXIT]]:
// CIR-FLAT-NEXT:   %[[DEST:.*]] = cir.load %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.switch.flat %[[DEST]] : !s32i, ^{{.*}} [
// CIR-FLAT-NEXT:     0: ^[[DTOR_TO_BODY:bb[0-9]+]],
// CIR-FLAT-NEXT:     1: ^[[DTOR_TO_DESTROY:bb[0-9]+]]
// CIR-FLAT-NEXT:   ]
// CIR-FLAT: ^[[DTOR_TO_BODY]]:
// CIR-FLAT-NEXT:   cir.br ^[[AFTER_DTOR:bb[0-9]+]]

// Destroy: after ~HasDtor, go on to the coroutine cleanup.
// CIR-FLAT: ^[[DTOR_TO_DESTROY]]:
// CIR-FLAT-NEXT:   cir.br ^[[CORO_CLEANUP:bb[0-9]+]]

// Resume: the end of the loop body, then the step.
// CIR-FLAT: ^[[AFTER_DTOR]]:
// CIR-FLAT-NEXT:   cir.br ^[[BODY_END:bb[0-9]+]]
// CIR-FLAT: ^[[BODY_END]]:
// CIR-FLAT-NEXT:   cir.br ^[[STEP:bb[0-9]+]]
// CIR-FLAT: ^[[STEP]]:
// CIR-FLAT-NEXT:   %[[I_VAL:.*]] = cir.load {{.*}}%[[I]]
// CIR-FLAT-NEXT:   %[[INC:.*]] = cir.inc nsw %[[I_VAL]]
// CIR-FLAT-NEXT:   cir.store {{.*}}%[[INC]], %[[I]]
// CIR-FLAT-NEXT:   cir.br ^[[COND]]
// CIR-FLAT: ^[[LOOP_EXIT]]:
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type13final_suspendEv
// CIR-FLAT: ^[[CORO_CLEANUP]]:
// CIR-FLAT-NEXT:   cir.coro.intrinsic.free
// CIR-FLAT:   cir.coro.intrinsic.end
// CIR-FLAT:   cir.return

// Classic codegen: after ~HasDtor, only slot 0 continues to for.inc; the
// destroy case (slot 2) goes through the loop cleanup to %coro.cleanup.

// OGCG: define {{.*}} void @_Z23await_in_loop_with_dtori(
// OGCG:   %[[LOCAL:.*]] = alloca %struct.HasDtor
// OGCG: await.cleanup:
// OGCG-NEXT:   br label %[[AWAIT_CLEANUP:cleanup[0-9]+]]
// OGCG: [[AWAIT_CLEANUP]]:
// OGCG-NEXT:   %[[SLOT1:.*]] = phi i32 [ 0, %await.ready ], [ 2, %await.cleanup ]
// OGCG:   switch i32 %[[SLOT1]], label %[[DTOR:cleanup[0-9]+]] [
// OGCG: [[DTOR]]:
// OGCG-NEXT:   %[[SLOT2:.*]] = phi i32 [ %[[SLOT1]], %[[AWAIT_CLEANUP]] ], [ 0, %{{.*}} ]
// OGCG-NEXT:   call void @_ZN7HasDtorD1Ev({{.*}} %[[LOCAL]])
// OGCG:   switch i32 %[[SLOT2]], label %[[LOOP_CLEANUP:cleanup[0-9]+]] [
// OGCG-NEXT:     i32 0, label %[[DTOR_CONT:cleanup.cont[0-9]+]]
// OGCG-NEXT:   ]
// OGCG: [[DTOR_CONT]]:
// OGCG-NEXT:   br label %for.inc
// OGCG: [[LOOP_CLEANUP]]:
// OGCG-NEXT:   %[[SLOT3:.*]] = phi i32 [ %[[SLOT2]], %[[DTOR]] ], [ 4, %for.cond.cleanup ]
// OGCG:   switch i32 %[[SLOT3]], label %unreachable [
// OGCG-NEXT:     i32 4, label %for.end
// OGCG-NEXT:     i32 2, label %coro.cleanup
// OGCG-NEXT:   ]
// OGCG: for.end:
// OGCG-NEXT:   call void @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(
// OGCG-NEXT:   br label %coro.final
// OGCG: coro.cleanup:
// OGCG:   call ptr @llvm.coro.free(

VoidTask await_in_if_with_dtor(bool b) {
  if (b) {
    HasDtor local;
    co_await std::suspend_always();
  }
}

// CIR: cir.func coroutine {{.*}} @_Z21await_in_if_with_dtorb
// CIR: }, body : {
// CIR:   cir.if %{{.*}} {
// CIR:     cir.cleanup.scope {
// CIR:       cir.await(user, ready : {
// CIR:       },)
// CIR:       cir.yield
// CIR:     } cleanup normal {
// CIR:       cir.call @_ZN7HasDtorD1Ev
// CIR:       cir.yield
// CIR:     }
// CIR:   }
// CIR:   cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv
// CIR:   cir.co_return
// CIR: }, finalSuspend : {

// Same as silly_task_with_dtor inside a cir.if: after ~HasDtor, resume
// (slot 0) continues after the if (return_void + final suspend), and destroy
// (slot 1) goes to the coroutine cleanup, like classic codegen below.

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z21await_in_if_with_dtorb(
// CIR-FLAT:   %[[CLEANUP_DEST_SLOT:.*]] = cir.alloca "__cleanup_dest_slot"
// CIR-FLAT:   cir.alloca "__coro_frame_addr"
// CIR-FLAT:   %[[B:.*]] = cir.alloca "b"
// CIR-FLAT:   %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-FLAT:   %[[COND:.*]] = cir.load {{.*}}%[[B]]
// CIR-FLAT-NEXT:   cir.brcond %[[COND]] ^[[THEN:bb[0-9]+]], ^[[IF_END:bb[0-9]+]]
// CIR-FLAT: ^[[THEN]]:
// CIR-FLAT-NEXT:   %[[LOCAL:.*]] = cir.alloca "local"
// CIR-FLAT-NEXT:   %[[USER_AWAITER:.*]] = cir.alloca "ref.tmp1"
// CIR-FLAT:   %[[READY:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[USER_AWAITER]])
// CIR-FLAT-NEXT:   cir.brcond %[[READY]] ^[[USER_RESUME:bb[0-9]+]], ^[[USER_SUSPEND:bb[0-9]+]]
// CIR-FLAT: ^[[USER_SUSPEND]]:
// CIR-FLAT:   %[[RES:.*]] = cir.coro.intrinsic.suspend
// CIR-FLAT:   cir.switch.flat %[[RES]] : !s8i, ^{{.*}} [
// CIR-FLAT-NEXT:     0: ^[[USER_RESUME]],
// CIR-FLAT-NEXT:     1: ^[[USER_DESTROY:bb[0-9]+]]
// CIR-FLAT-NEXT:   ]
// CIR-FLAT: ^[[USER_DESTROY]]:
// CIR-FLAT-NEXT:   %[[DEST_DESTROY:.*]] = cir.const #cir.int<1> : !s32i
// CIR-FLAT-NEXT:   cir.store %[[DEST_DESTROY]], %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.br ^[[DTOR:bb[0-9]+]]
// CIR-FLAT: ^[[USER_RESUME]]:
// CIR-FLAT-NEXT:   cir.call @_ZNSt14suspend_always12await_resumeEv(%[[USER_AWAITER]])
// CIR-FLAT:   %[[DEST_NORMAL:.*]] = cir.const #cir.int<0> : !s32i
// CIR-FLAT-NEXT:   cir.store %[[DEST_NORMAL]], %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.br ^[[DTOR]]
// CIR-FLAT: ^[[DTOR]]:
// CIR-FLAT-NEXT:   cir.call @_ZN7HasDtorD1Ev(%[[LOCAL]])
// CIR-FLAT-NEXT:   cir.br ^[[DTOR_EXIT:bb[0-9]+]]
// CIR-FLAT: ^[[DTOR_EXIT]]:
// CIR-FLAT-NEXT:   %[[DEST:.*]] = cir.load %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.switch.flat %[[DEST]] : !s32i, ^{{.*}} [
// CIR-FLAT-NEXT:     0: ^[[DTOR_TO_BODY:bb[0-9]+]],
// CIR-FLAT-NEXT:     1: ^[[DTOR_TO_DESTROY:bb[0-9]+]]
// CIR-FLAT-NEXT:   ]
// CIR-FLAT: ^[[DTOR_TO_BODY]]:
// CIR-FLAT-NEXT:   cir.br ^[[AFTER_DTOR:bb[0-9]+]]

// Destroy: after ~HasDtor, go on to the coroutine cleanup.
// CIR-FLAT: ^[[DTOR_TO_DESTROY]]:
// CIR-FLAT-NEXT:   cir.br ^[[CORO_CLEANUP:bb[0-9]+]]

// Resume: the end of the if, then the rest of the body.
// CIR-FLAT: ^[[AFTER_DTOR]]:
// CIR-FLAT-NEXT:   cir.br ^[[IF_END]]
// CIR-FLAT: ^[[IF_END]]:
// CIR-FLAT-NEXT:   cir.br ^[[AFTER_IF:bb[0-9]+]]
// CIR-FLAT: ^[[AFTER_IF]]:
// CIR-FLAT-NEXT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(%[[PROMISE]])
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type13final_suspendEv(%[[PROMISE]])
// CIR-FLAT: ^[[CORO_CLEANUP]]:
// CIR-FLAT-NEXT:   cir.coro.intrinsic.free
// CIR-FLAT:   cir.coro.intrinsic.end
// CIR-FLAT:   cir.return

// OGCG: define {{.*}} void @_Z21await_in_if_with_dtorb(
// OGCG:   %[[LOCAL:.*]] = alloca %struct.HasDtor
// OGCG: await.cleanup:
// OGCG-NEXT:   br label %[[AWAIT_CLEANUP:cleanup[0-9]+]]
// OGCG: [[AWAIT_CLEANUP]]:
// OGCG-NEXT:   %[[SLOT1:.*]] = phi i32 [ 0, %await.ready ], [ 2, %await.cleanup ]
// OGCG:   switch i32 %[[SLOT1]], label %[[DTOR:cleanup[0-9]+]] [
// OGCG: [[DTOR]]:
// OGCG-NEXT:   %[[SLOT2:.*]] = phi i32 [ %[[SLOT1]], %[[AWAIT_CLEANUP]] ], [ 0, %{{.*}} ]
// OGCG-NEXT:   call void @_ZN7HasDtorD1Ev({{.*}} %[[LOCAL]])
// OGCG:   switch i32 %[[SLOT2]], label %unreachable [
// OGCG-NEXT:     i32 0, label %[[DTOR_CONT:cleanup.cont[0-9]+]]
// OGCG-NEXT:     i32 2, label %coro.cleanup
// OGCG-NEXT:   ]
// OGCG: [[DTOR_CONT]]:
// OGCG-NEXT:   br label %if.end
// OGCG: if.end:
// OGCG-NEXT:   call void @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(
// OGCG-NEXT:   br label %coro.final
// OGCG: coro.cleanup:
// OGCG:   call ptr @llvm.coro.free(

folly::coro::Task<int> co_return_in_loop(int n) {
  HasDtor outer;
  for (int i = 0; i < n; ++i) {
    HasDtor inner;
    if (i == 3)
      co_return i;
    co_await std::suspend_always();
  }
  co_return -1;
}

// CIR: cir.func coroutine {{.*}} @_Z17co_return_in_loopi
// CIR: }, body : {
// CIR:   cir.cleanup.scope {
// CIR:     cir.for : cond {
// CIR:     } body {
// CIR:       cir.cleanup.scope {
// CIR:         cir.if %{{.*}} {
// CIR:           cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi
// CIR-NEXT:      cir.co_return
// CIR:         cir.await(user, ready : {
// CIR:       } cleanup normal {
// CIR-NEXT:    cir.call @_ZN7HasDtorD1Ev
// CIR:     } step {
// CIR:     cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi
// CIR-NEXT:  cir.co_return
// CIR:   } cleanup normal {
// CIR-NEXT:  cir.call @_ZN7HasDtorD1Ev
// CIR: }, finalSuspend : {

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z17co_return_in_loopi(
// CIR-FLAT:   %[[CLEANUP_DEST_SLOT:.*]] = cir.alloca "__cleanup_dest_slot"
// CIR-FLAT:   %[[PROMISE:.*]] = cir.alloca "__promise"
// CIR-FLAT:   %[[OUTER:.*]] = cir.alloca "outer"
// CIR-FLAT:   %[[I:.*]] = cir.alloca "i"
// CIR-FLAT:   %[[LT:.*]] = cir.cmp lt
// CIR-FLAT-NEXT:   cir.brcond %[[LT]] ^{{.*}}, ^[[LOOP_EXIT:bb[0-9]+]]
// CIR-FLAT:   %[[INNER:.*]] = cir.alloca "inner"
// CIR-FLAT:   %[[EQ:.*]] = cir.cmp eq
// CIR-FLAT-NEXT:   cir.brcond %[[EQ]] ^[[IF_THEN:bb[0-9]+]], ^{{.*}}

// co_return i: slot 1 for the inner cleanup scope.
// CIR-FLAT: ^[[IF_THEN]]:
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], %{{.*}})
// CIR-FLAT-NEXT:   %[[RET_IN:.*]] = cir.const #cir.int<1> : !s32i
// CIR-FLAT-NEXT:   cir.store %[[RET_IN]], %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.br ^[[INNER_DTOR:bb[0-9]+]]

// The user await: destroy uses slot 2, resume slot 0.
// CIR-FLAT:   %[[RES:.*]] = cir.coro.intrinsic.suspend
// CIR-FLAT:   cir.switch.flat %[[RES]] : !s8i, ^{{.*}} [
// CIR-FLAT-NEXT:     0: ^[[USER_RESUME:bb[0-9]+]],
// CIR-FLAT-NEXT:     1: ^[[USER_DESTROY:bb[0-9]+]]
// CIR-FLAT-NEXT:   ]
// CIR-FLAT: ^[[USER_DESTROY]]:
// CIR-FLAT-NEXT:   %[[DESTROY_IN:.*]] = cir.const #cir.int<2> : !s32i
// CIR-FLAT-NEXT:   cir.store %[[DESTROY_IN]], %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.br ^[[INNER_DTOR]]
// CIR-FLAT: ^[[USER_RESUME]]:
// CIR-FLAT-NEXT:   cir.call @_ZNSt14suspend_always12await_resumeEv
// CIR-FLAT:   %[[NORMAL_IN:.*]] = cir.const #cir.int<0> : !s32i
// CIR-FLAT-NEXT:   cir.store %[[NORMAL_IN]], %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.br ^[[INNER_DTOR]]

// ~inner, then dispatch: 0 continues the loop, 1 (co_return) and
// 2 (destroy) leave it through ~outer.
// CIR-FLAT: ^[[INNER_DTOR]]:
// CIR-FLAT-NEXT:   cir.call @_ZN7HasDtorD1Ev(%[[INNER]])
// CIR-FLAT-NEXT:   cir.br ^[[INNER_EXIT:bb[0-9]+]]
// CIR-FLAT: ^[[INNER_EXIT]]:
// CIR-FLAT-NEXT:   %[[DEST_IN:.*]] = cir.load %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.switch.flat %[[DEST_IN]] : !s32i, ^{{.*}} [
// CIR-FLAT-NEXT:     0: ^{{.*}},
// CIR-FLAT-NEXT:     1: ^[[INNER_TO_RET:bb[0-9]+]],
// CIR-FLAT-NEXT:     2: ^[[INNER_TO_DESTROY:bb[0-9]+]]
// CIR-FLAT-NEXT:   ]
// CIR-FLAT: ^[[INNER_TO_RET]]:
// CIR-FLAT-NEXT:   %[[RET_OUT:.*]] = cir.const #cir.int<1> : !s32i
// CIR-FLAT-NEXT:   cir.store %[[RET_OUT]], %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.br ^[[OUTER_DTOR:bb[0-9]+]]
// CIR-FLAT: ^[[INNER_TO_DESTROY]]:
// CIR-FLAT-NEXT:   %[[DESTROY_OUT:.*]] = cir.const #cir.int<2> : !s32i
// CIR-FLAT-NEXT:   cir.store %[[DESTROY_OUT]], %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.br ^[[OUTER_DTOR]]

// CIR-FLAT:   %[[I_VAL:.*]] = cir.load {{.*}}%[[I]]
// CIR-FLAT-NEXT:   cir.inc nsw %[[I_VAL]]
// CIR-FLAT: ^[[LOOP_EXIT]]:
// CIR-FLAT:   %[[MINUS_ONE:.*]] = cir.const #cir.int<-1> : !s32i
// CIR-FLAT-NEXT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], %[[MINUS_ONE]])
// CIR-FLAT-NEXT:   %[[RET2_OUT:.*]] = cir.const #cir.int<3> : !s32i
// CIR-FLAT-NEXT:   cir.store %[[RET2_OUT]], %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.br ^[[OUTER_DTOR]]

// ~outer, then both co_returns go to the final suspend and the destroy to
// the coroutine cleanup.
// CIR-FLAT: ^[[OUTER_DTOR]]:
// CIR-FLAT-NEXT:   cir.call @_ZN7HasDtorD1Ev(%[[OUTER]])
// CIR-FLAT-NEXT:   cir.br ^[[OUTER_EXIT:bb[0-9]+]]
// CIR-FLAT: ^[[OUTER_EXIT]]:
// CIR-FLAT-NEXT:   %[[DEST_OUT:.*]] = cir.load %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT-NEXT:   cir.switch.flat %[[DEST_OUT]] : !s32i, ^{{.*}} [
// CIR-FLAT-NEXT:     0: ^{{.*}},
// CIR-FLAT-NEXT:     1: ^[[OUTER_TO_RET:bb[0-9]+]],
// CIR-FLAT-NEXT:     2: ^[[OUTER_TO_DESTROY:bb[0-9]+]],
// CIR-FLAT-NEXT:     3: ^[[OUTER_TO_RET2:bb[0-9]+]]
// CIR-FLAT-NEXT:   ]
// CIR-FLAT: ^[[OUTER_TO_RET]]:
// CIR-FLAT-NEXT:   cir.br ^[[FINAL:bb[0-9]+]]
// CIR-FLAT: ^[[OUTER_TO_DESTROY]]:
// CIR-FLAT-NEXT:   cir.br ^[[CORO_CLEANUP:bb[0-9]+]]
// CIR-FLAT: ^[[OUTER_TO_RET2]]:
// CIR-FLAT-NEXT:   cir.br ^[[FINAL]]
// CIR-FLAT: ^[[FINAL]]:
// CIR-FLAT-NEXT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type13final_suspendEv(%[[PROMISE]])
// CIR-FLAT: ^[[CORO_CLEANUP]]:
// CIR-FLAT-NEXT:   cir.coro.intrinsic.free
// CIR-FLAT:   cir.coro.intrinsic.end
// CIR-FLAT:   cir.return

// Classic codegen: the co_return (slot 3) and the destroy (slot 2) both run
// ~inner and ~outer, then go to %coro.final and %coro.cleanup.

// OGCG: define {{.*}} void @_Z17co_return_in_loopi(
// OGCG:   %[[OUTER:.*]] = alloca %struct.HasDtor
// OGCG:   %[[INNER:.*]] = alloca %struct.HasDtor
// OGCG: if.then:
// OGCG:   call void @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(
// OGCG-NEXT:   br label %[[INNER_DTOR:cleanup[0-9]+]]
// OGCG: await.cleanup:
// OGCG-NEXT:   br label %[[AWAIT_CLEANUP:cleanup[0-9]+]]
// OGCG: [[AWAIT_CLEANUP]]:
// OGCG-NEXT:   %[[SLOT1:.*]] = phi i32 [ 0, %await.ready ], [ 2, %await.cleanup ]
// OGCG: [[INNER_DTOR]]:
// OGCG-NEXT:   %[[SLOT2:.*]] = phi i32 [ 3, %if.then ], [ %[[SLOT1]], %[[AWAIT_CLEANUP]] ], [ 0, %{{.*}} ]
// OGCG-NEXT:   call void @_ZN7HasDtorD1Ev({{.*}} %[[INNER]])
// OGCG:   switch i32 %[[SLOT2]], label %{{.*}} [
// OGCG-NEXT:     i32 0, label %{{.*}}
// OGCG-NEXT:   ]
// OGCG: for.end:
// OGCG-NEXT:   call void @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi({{.*}}, i32 noundef -1)
// OGCG:   %[[SLOT4:.*]] = phi i32 [ %{{.*}}, %{{.*}} ], [ 3, %for.end ]
// OGCG-NEXT:   call void @_ZN7HasDtorD1Ev({{.*}} %[[OUTER]])
// OGCG:   switch i32 %[[SLOT4]], label %unreachable [
// OGCG-NEXT:     i32 3, label %coro.final
// OGCG-NEXT:     i32 2, label %coro.cleanup
// OGCG-NEXT:   ]

folly::coro::Task<int __complex__> fetchData() noexcept {
  int __complex__ a;
  co_return a;
}

folly::coro::Task<int __complex__> complex_co_await() noexcept { 
  co_await fetchData();
}

// CIR: cir.func coroutine {{.*}} @_Z16complex_co_awaitv

// CIR: %[[COERCE_ADDR:.*]] = cir.alloca "coerce" {{.*}} : !cir.ptr<!u64i>
// CIR: %[[COMPLEX_ADDR:.*]] = cir.alloca "ref.tmp1" {{.*}} : !cir.ptr<!rec_folly3A3Acoro3A3ATask3C_Complex_int3E>
// CIR: %[[RESUME_VAL_ADDR:.*]] = cir.alloca "__coawait_resume_rval" {{.*}} : !cir.ptr<!cir.complex<!s32i>>

// CIR: cir.coroutine initialSuspend : {
// CIR:   cir.await(init, ready : {
// CIR:   }, suspend : {
// CIR:   }, resume : {
// CIR:   },)

// CIR: }, body : {
// CIR:     cir.call @_Z9fetchDatav() nothrow : () -> ()
// CIR:     %[[CALL:.*]] = cir.const #cir.poison : !rec_folly3A3Acoro3A3ATask3C_Complex_int3E
// CIR:     cir.store {{.*}} %[[CALL]], %[[COMPLEX_ADDR]] : !rec_folly3A3Acoro3A3ATask3C_Complex_int3E, !cir.ptr<!rec_folly3A3Acoro3A3ATask3C_Complex_int3E>

// CIR:     cir.await(user, ready : {
// CIR:     }, suspend : {
// CIR:     }, resume : {
// _Complex int is one eightbyte in class INTEGER, so it returns as !u64i and
// is reloaded through the coerce slot.

// CIR:         %[[RESUME_VAL:.*]] = cir.call @_ZN5folly4coro4TaskICiE12await_resumeEv(%[[COMPLEX_ADDR]]) : (!cir.ptr<!rec_folly3A3Acoro3A3ATask3C_Complex_int3E> {llvm.align = 1 : i64, llvm.dereferenceable = 1 : i64, llvm.nonnull, llvm.noundef}) -> (!u64i {llvm.noundef})
// CIR:         cir.store %[[RESUME_VAL]], %[[COERCE_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR:         %[[COERCE_CAST:.*]] = cir.cast bitcast %[[COERCE_ADDR]] : !cir.ptr<!u64i> -> !cir.ptr<!cir.complex<!s32i>>
// CIR:         %[[RESUME_COMPLEX:.*]] = cir.load %[[COERCE_CAST]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR:         cir.store %[[RESUME_COMPLEX]], %[[RESUME_VAL_ADDR]] : !cir.complex<!s32i>, !cir.ptr<!cir.complex<!s32i>>
// CIR:       },)
// CIR:       %[[V:.*]] = cir.load %[[RESUME_VAL_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR:       cir.yield
// CIR: }, finalSuspend : {
// CIR: }, destroy : {
// CIR: }, exit : {

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z9fetchDatav()

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z16complex_co_awaitv()
// CIR-FLAT:   %[[COERCE_ADDR:.*]] = cir.alloca "coerce"
// CIR-FLAT:   %[[TASK_ADDR:.*]] = cir.alloca "ref.tmp1"
// CIR-FLAT:   %[[RESUME_VAL_ADDR:.*]] = cir.alloca "__coawait_resume_rval"
// CIR-FLAT:   cir.call @_Z9fetchDatav()
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskICiE11await_readyEv(%[[TASK_ADDR]])
// CIR-FLAT:   cir.coro.intrinsic.suspend
// CIR-FLAT:   %[[RESUME_VAL:.*]] = cir.call @_ZN5folly4coro4TaskICiE12await_resumeEv(%[[TASK_ADDR]])
// CIR-FLAT-NEXT:   cir.store %[[RESUME_VAL]], %[[COERCE_ADDR]] : !u64i, !cir.ptr<!u64i>
// CIR-FLAT-NEXT:   %[[COERCE_CAST:.*]] = cir.cast bitcast %[[COERCE_ADDR]]
// CIR-FLAT-NEXT:   %[[RESUME_COMPLEX:.*]] = cir.load %[[COERCE_CAST]]
// CIR-FLAT-NEXT:   cir.store %[[RESUME_COMPLEX]], %[[RESUME_VAL_ADDR]]
// CIR-FLAT:   cir.load %[[RESUME_VAL_ADDR]] : !cir.ptr<!cir.complex<!s32i>>, !cir.complex<!s32i>
// CIR-FLAT:   cir.coro.intrinsic.free
// CIR-FLAT:   cir.return

// OGCG: define dso_local void @_Z16complex_co_awaitv()

// OGCG: %[[RESUME_VAL_ADDR:.*]] = alloca { i32, i32 }, align 4
// OGCG: %[[RESUME_REAL_ADDR:.*]] = alloca i32, align 4
// OGCG: %[[RESUME_IMAG_ADDR:.*]] = alloca i32, align 4

// OGCG: coro.init:
// OGCG: await.ready:
// OGCG:   %[[RESUME_VAL:.*]] = call noundef i64 @_ZN5folly4coro4TaskICiE12await_resumeEv(ptr noundef nonnull align 1 dereferenceable(1) %{{.*}})
// OGCG:   store i64 %[[RESUME_VAL]], ptr %[[RESUME_VAL_ADDR]], align 4
// OGCG:   %[[RESUME_REAL_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[RESUME_VAL_ADDR]], i32 0, i32 0
// OGCG:   %[[RESUME_REAL:.*]] = load i32, ptr %[[RESUME_REAL_PTR]], align 4
// OGCG:   store i32 %[[RESUME_REAL]], ptr %[[RESUME_REAL_ADDR]], align 4
// OGCG:   %[[RESUME_IMAG_PTR:.*]] = getelementptr inbounds nuw { i32, i32 }, ptr %[[RESUME_VAL_ADDR]], i32 0, i32 1
// OGCG:   %[[RESUME_IMAG:.*]] = load i32, ptr %[[RESUME_IMAG_PTR]], align 4
// OGCG:   store i32 %[[RESUME_IMAG]], ptr %[[RESUME_IMAG_ADDR]], align 4
// OGCG:   br label %[[CLEANUP_FROM_AWAIT_READY:.*]]

// OGCG: [[CLEANUP_CONT:.*]]:
// OGCG:   %[[RESUME_REAL:.*]] = load i32, ptr %[[RESUME_REAL_ADDR]], align 4
// OGCG:   %[[RESUME_IMAG:.*]] = load i32, ptr %[[RESUME_IMAG_ADDR]], align 4

void void_coroutine() {
  co_await std::suspend_never();
}

// CIR: cir.func coroutine {{.*}} @_Z14void_coroutinev
// CIR: cir.coroutine initialSuspend : {
// CIR: }, exit : {
// CIR:   cir.coro.intrinsic.end
// CIR:   cir.return
// CIR: }

// CIR-FLAT-LABEL: cir.func flattened_coroutine {{.*}} @_Z14void_coroutinev()
// CIR-FLAT:   cir.call @_ZNSt13suspend_never11await_readyEv
// CIR-FLAT:   cir.call @_ZNSt16coroutine_traitsIvJEE12promise_type11return_voidEv
// CIR-FLAT:   cir.call @_ZNSt16coroutine_traitsIvJEE12promise_type13final_suspendEv
// CIR-FLAT:   %[[IS_FINAL:.*]] = cir.const #true
// CIR-FLAT:   cir.coro.intrinsic.suspend(%{{.*}}, %[[IS_FINAL]])
// CIR-FLAT:   cir.coro.intrinsic.free
// CIR-FLAT:   cir.coro.intrinsic.end
// CIR-FLAT-NEXT:   cir.return

// OGCG: define {{.*}} void @_Z14void_coroutinev()
// OGCG: coro.ret:
// OGCG-NEXT:   call void @llvm.coro.end
// OGCG-NEXT:   ret void
