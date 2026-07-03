// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -emit-cir %s -o %t.cir
// RUN: cir-opt --cir-flatten-cfg %t.cir -o %t.flat.cir
// RUN: FileCheck --input-file=%t.flat.cir %s -check-prefix=CIR-FLAT

#include "Inputs/coroutine.h"

using VoidTask = folly::coro::Task<void>;

VoidTask silly_task() {
  co_await std::suspend_always();
}

// CIR-FLAT: cir.func flattened_coroutine {{.*}} @_Z10silly_taskv

// CIR-FLAT: %[[CLEANUP_DEST_SLOT:.*]] = cir.alloca "__cleanup_dest_slot"
// CIR-FLAT: %[[NullPtr:.*]] = cir.const #cir.ptr<null>
// CIR-FLAT: %[[Align:.*]] = cir.const #cir.int<16>
// CIR-FLAT: %[[CoroId:.*]] = cir.call @__builtin_coro_id(%[[Align]], %[[NullPtr]], %[[NullPtr]], %[[NullPtr]])
// CIR-FLAT: %[[SUSPEND_POINT:.*]] = cir.alloca "__coroutine_suspend_point"
// CIR-FLAT: %[[SavedFrameAddr:.*]] = cir.alloca "__coro_frame_addr"
// CIR-FLAT: %[[SuspendAlwaysAddr:.*]] = cir.alloca "ref.tmp0"
// CIR-FLAT: %[[ShouldAlloc:.*]] = cir.call @__builtin_coro_alloc(%[[CoroId]]) : (!u32i) -> !cir.bool
// CIR-FLAT: cir.store %[[NullPtr]], %[[SavedFrameAddr]]
// CIR-FLAT: cir.brcond %[[ShouldAlloc]] ^[[CORO_ALLOC:.*]], ^[[CORO_INIT:.*]]
// CIR-FLAT: ^[[CORO_ALLOC]]:
// CIR-FLAT:   %[[CoroSize:.*]] = cir.call @__builtin_coro_size()
// CIR-FLAT:   %[[AllocAddr:.*]] = cir.call @_Znwm(%[[CoroSize]])
// CIR-FLAT:   cir.store %[[AllocAddr]], %[[SavedFrameAddr]] : !cir.ptr<!void>, !cir.ptr<!cir.ptr<!void>>
// CIR-FLAT:   cir.br ^[[CORO_INIT]]
// CIR-FLAT: ^[[CORO_INIT]]:
// CIR-FLAT:   %[[LOAD_CORO_FRAME:.*]] = cir.load %[[SavedFrameAddr]]
// CIR-FLAT:   %[[CoroFrameAddr:.*]] = cir.call @__builtin_coro_begin(%[[CoroId]], %[[LOAD_CORO_FRAME]])
// CIR-FLAT:   cir.br ^[[INIT_AWAIT_READY:.*]]
// CIR-FLAT: ^[[INIT_AWAIT_READY]]:
// CIR-FLAT:  %[[RetObj:.*]] = cir.call @_ZN5folly4coro4TaskIvE12promise_type17get_return_objectEv(%[[VoidPromisseAddr:.*]])
// CIR-FLAT:   cir.store {{.*}} %[[RetObj]], %[[VoidTaskAddr:.*]]
// CIR-FLAT:   %[[Tmp0:.*]] = cir.call @_ZN5folly4coro4TaskIvE12promise_type15initial_suspendEv(%[[VoidPromisseAddr]])
// CIR-FLAT:   cir.store {{.*}} %[[Tmp0]], %[[SuspendAlwaysAddr]]
// CIR-FLAT:   cir.br ^[[INIT_AWAIT_READY_CONT:.*]]
// CIR-FLAT: ^[[INIT_AWAIT_READY_CONT]]:
// CIR-FLAT:   %[[ShouldSuspend:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[SuspendAlwaysAddr]])
// CIR-FLAT:   cir.brcond %[[ShouldSuspend]] ^[[AWAIT_INIT_RESUME:.*]], ^[[AWAIT_SUSPEND:.*]]
// CIR-FLAT: ^[[AWAIT_SUSPEND]]:  // pred: ^bb4
// CIR-FLAT:   %[[NULLPTR2:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>

// TODO (cir): We should use a cir.token or mlir.token instead of returning !u32i.
// CIR-FLAT:   %[[SAVE_TOKEN:.*]] = cir.call_llvm_intrinsic "llvm.coro.save" %[[NULLPTR2]] : (!cir.ptr<!void>) -> !u32i
// CIR-FLAT:   %27 = cir.call @_ZNSt16coroutine_handleIN5folly4coro4TaskIvE12promise_typeEE12from_addressEPv(%21)
// CIR-FLAT:   cir.store align(1) %27, %10
// CIR-FLAT:   %28 = cir.load align(1) %10
// CIR-FLAT:   cir.call @_ZNSt16coroutine_handleIvEC1IN5folly4coro4TaskIvE12promise_typeEEES_IT_E(%9, %28)
// CIR-FLAT:   %29 = cir.load align(1) %9
// CIR-FLAT:   cir.call @_ZNSt14suspend_always13await_suspendESt16coroutine_handleIvE(%8, %29)
// CIR-FLAT:   %[[IS_FINAL_SUSPEND:.*]] = cir.const #false
// CIR-FLAT:   %[[SUSPEND_RESULT:.*]] = cir.call_llvm_intrinsic "llvm.coro.suspend" %[[SAVE_TOKEN]], %[[IS_FINAL_SUSPEND]]
// CIR-FLAT:   cir.switch.flat %[[SUSPEND_RESULT]] : !u32i, ^[[CORO_RET:.*]] [
// CIR-FLAT:     0: ^[[AWAIT_INIT_RESUME]],
// CIR-FLAT:     1: ^[[INIT_CLEANUP_DESTROY:.*]]
// CIR-FLAT:   ]
// CIR-FLAT: ^[[INIT_CLEANUP_DESTROY]]:
// CIR-FLAT:   cir.const #cir.int<0>
// CIR-FLAT:   cir.store {{.*}}, %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT:   cir.br ^[[CLEANUP_CORO_FREE:.*]]
// CIR-FLAT: ^[[AWAIT_INIT_RESUME]]:
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv(%[[SuspendAlwaysAddr]])
// CIR-FLAT:   cir.br ^[[CORO_BODY:.*]]

// The remaining await operations follow the same lowering pattern as above,
// so we only check a few key instructions here instead of matching the entire IR

// CIR-FLAT: ^[[CORO_BODY:.*]]:
// CIR-FLAT:   cir.br ^[[USER_AWAIT:.*]]
// CIR-FLAT: ^[[USER_AWAIT]]:
// CIR-FLAT:   cir.br ^[[USER_AWAIT_READY_CONT:.*]]
// CIR-FLAT: ^[[USER_AWAIT_READY_CONT]]:
// CIR-FLAT:   %[[ShouldSuspend2:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv
// CIR-FLAT:   cir.brcond %[[ShouldSuspend2]] ^[[AWAIT_USER_RESUME:.*]], ^[[AWAIT_USER_SUSPEND:.*]]
// CIR-FLAT: ^[[AWAIT_USER_SUSPEND]]
// CIR-FLAT:   %[[NULLPTR3:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR-FLAT:   %[[SAVE_TOKEN2:.*]] = cir.call_llvm_intrinsic "llvm.coro.save" %[[NULLPTR3]]
// CIR-FLAT:   %[[IS_FINAL_SUSPEND2:.*]] = cir.const #false
// CIR-FLAT:   %[[SUSPEND_RESULT2:.*]] = cir.call_llvm_intrinsic "llvm.coro.suspend" %[[SAVE_TOKEN2]], %[[IS_FINAL_SUSPEND2]]
// CIR-FLAT:   cir.switch.flat %[[SUSPEND_RESULT2]] : !u32i, ^[[CORO_RET]] [
// CIR-FLAT:     0: ^[[AWAIT_USER_RESUME]],
// CIR-FLAT:     1: ^[[user_cleanup_destroy:.*]]
// CIR-FLAT:   ]
// CIR-FLAT: ^[[user_cleanup_destroy]]:
// CIR-FLAT:   cir.const #cir.int<1>
// CIR-FLAT:   cir.store {{.*}}, %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT:   cir.br ^[[CLEANUP_CORO_FREE:.*]]
// CIR-FLAT: ^[[AWAIT_USER_RESUME]]:
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv
// CIR-FLAT:   cir.br ^[[CO_RETURN:.*]]
// CIR-FLAT: ^[[CO_RETURN]]:
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv(%[[VoidPromisseAddr]])

// This corresponds to the implicit cir.co_return path, which exits the
// cir.coro.body and branches to the final suspend block.
// CIR-FLAT:   cir.br ^[[FINAL_SUSPEND_BB:.*]]
// CIR-FLAT: ^[[FINAL_SUSPEND_BB]]:
// CIR-FLAT:   %[[final_suspend:.*]] = cir.call @_ZN5folly4coro4TaskIvE12promise_type13final_suspendEv(%[[VoidPromisseAddr]])
// CIR-FLAT:   cir.store {{.*}} %[[final_suspend]], %{{.*}}
// CIR-FLAT:   cir.br ^[[FINAL_AWAIT_READY_CONT:.*]]
// CIR-FLAT: ^[[FINAL_AWAIT_READY_CONT]]:
// CIR-FLAT:   %[[ShouldSuspend3:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv
// CIR-FLAT:   cir.brcond %[[ShouldSuspend3]] ^[[AWAIT_FINAL_RESUME:.*]], ^[[AWAIT_FINAL_SUSPEND:.*]]
// CIR-FLAT: ^[[AWAIT_FINAL_SUSPEND]]:
// CIR-FLAT:   %[[NULLPTR4:.*]] = cir.const #cir.ptr<null>
// CIR-FLAT:   %[[SAVE_TOKEN3:.*]] = cir.call_llvm_intrinsic "llvm.coro.save" %[[NULLPTR4]]
// CIR-FLAT:   %[[IS_FINAL_SUSPEND3:.*]] = cir.const #true
// CIR-FLAT:   %[[SUSPEND_RESULT3:.*]] = cir.call_llvm_intrinsic "llvm.coro.suspend" %[[SAVE_TOKEN3]], %[[IS_FINAL_SUSPEND3]]
// CIR-FLAT:   cir.switch.flat %[[SUSPEND_RESULT3]] : !u32i, ^[[CORO_RET]] [
// CIR-FLAT:     0: ^[[AWAIT_FINAL_RESUME]],
// CIR-FLAT:     1: ^[[final_cleanup_destroy:.*]]
// CIR-FLAT:   ]
// CIR-FLAT: ^[[final_cleanup_destroy]]:
// CIR-FLAT:   cir.const #cir.int<2>
// CIR-FLAT:   cir.store {{.*}}, %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT:   cir.br ^[[CLEANUP_CORO_FREE:.*]]
// CIR-FLAT: ^[[AWAIT_FINAL_RESUME]]:
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv
// CIR-FLAT:   cir.br ^[[FIANL_CLEANUP_EXIT:.*]]
// CIR-FLAT: ^[[FIANL_CLEANUP_EXIT]]:
// CIR-FLAT:   cir.const #cir.int<3> : !s32i
// CIR-FLAT:   cir.store %{{.*}}, %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT:   cir.br ^[[CLEANUP_CORO_FREE]]

// Check whether llvm.coro.free returned a non-null pointer, indicating
// that the coroutine frame must be deallocated.
// CIR-FLAT: ^[[CLEANUP_CORO_FREE]]:
// CIR-FLAT:   %[[SHOULD_FREE:.*]] = cir.call @__builtin_coro_free(%[[CoroId]], %[[CoroFrameAddr]])
// CIR-FLAT:   %[[NULLPTR5:.*]] = cir.const #cir.ptr<null>
// CIR-FLAT:   %[[NEEDS_FREE:.*]] = cir.cmp ne %[[SHOULD_FREE]], %[[NULLPTR5]]
// CIR-FLAT:   cir.brcond %[[NEEDS_FREE]] ^[[FREE_FRAME:.*]], ^[[EXIT_CLEANUP:.*]]
// CIR-FLAT: ^[[FREE_FRAME]]:
// CIR-FLAT:   %[[CoroSize2:.*]] = cir.call @__builtin_coro_size() : () -> (!u64i {llvm.noundef})
// CIR-FLAT:   cir.call @_ZdlPvm(%[[SHOULD_FREE]], %[[CoroSize2]])
// CIR-FLAT:   cir.br ^[[EXIT_CLEANUP]]
// CIR-FLAT: ^[[EXIT_CLEANUP]]:
// CIR-FLAT:   cir.br ^[[EXIT_CLEANUP_SWITCH:.*]]
// CIR-FLAT: ^[[EXIT_CLEANUP_SWITCH]]:
// CIR-FLAT:   %[[LOAD_DEST_SLOT:.*]] = cir.load %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT:   cir.switch.flat %[[LOAD_DEST_SLOT]] : !s32i, ^[[DEFAULT_EXIT:.*]] [
// CIR-FLAT:     0: ^[[EXIT1:.*]],
// CIR-FLAT:     1: ^[[EXIT2:.*]],
// CIR-FLAT:     2: ^[[EXIT3:.*]],
// CIR-FLAT:     3: ^[[EXIT4:.*]]
// CIR-FLAT:   ]
// CIR-FLAT: ^[[EXIT1]]:
// CIR-FLAT:   cir.br ^[[TO_RET:.*]]
// CIR-FLAT: ^[[EXIT2]]:
// CIR-FLAT:   cir.br ^[[TO_RET]]
// CIR-FLAT: ^[[EXIT3]]:
// CIR-FLAT:   cir.br ^[[TO_RET]]
// CIR-FLAT: ^[[EXIT4]]:
// CIR-FLAT:   cir.br ^[[TO_RET]]
// CIR-FLAT: ^[[DEFAULT_EXIT]]:
// CIR-FLAT:   cir.unreachable
// CIR-FLAT: ^[[TO_RET]]:
// CIR-FLAT:   cir.br ^[[CORO_RET]]
// CIR-FLAT: ^[[CORO_RET]]:
// CIR-FLAT:   %58 = cir.const #cir.int<1> : !s32i
// CIR-FLAT:   cir.store %58, %5 : !s32i, !cir.ptr<!s32i>
// CIR-FLAT:   %[[NULLPTR6:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR-FLAT:   %[[UNWIND:.*]] = cir.const #false
// CIR-FLAT:   cir.call @__builtin_coro_end(%[[NULLPTR6]], %[[UNWIND]])
// CIR-FLAT:   cir.return
// CIR-FLAT: }

struct HasDtor {
  ~HasDtor();
};

VoidTask silly_task_with_dtor() {
  HasDtor local;
  co_await std::suspend_always();
}

// CIR-FLAT: cir.func flattened_coroutine {{.*}} @_Z20silly_task_with_dtorv
// CIR-FLAT:   %[[CLEANUP_DEST_SLOT:.*]] = cir.alloca "__cleanup_dest_slot"
// CIR-FLAT:   %[[SuspendAlwaysAddr:.*]] = cir.alloca "ref.tmp0"

// CIR-FLAT:   %[[ShouldSuspend:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv(%[[SuspendAlwaysAddr]])
// CIR-FLAT:   cir.brcond %[[ShouldSuspend]]  ^[[AWAIT_INIT_RESUME:.*]], ^[[AWAIT_INIT_SUSPEND:.*]]
// CIR-FLAT: ^[[AWAIT_INIT_SUSPEND]]:
// CIR-FLAT:   %[[NULLPTR:.*]] = cir.const #cir.ptr<null>
// CIR-FLAT:   %[[SAVE_TOKEN:.*]] = cir.call_llvm_intrinsic "llvm.coro.save" %[[NULLPTR]]
// CIR-FLAT:   %[[IS_FINAL_SUSPEND:.*]] = cir.const #false
// CIR-FLAT:   %[[SUSPEND_RESULT:.*]] = cir.call_llvm_intrinsic "llvm.coro.suspend" %[[SAVE_TOKEN]], %[[IS_FINAL_SUSPEND]]
// CIR-FLAT:   cir.switch.flat %[[SUSPEND_RESULT]] : !u32i, ^[[CORO_RET:.*]] [
// CIR-FLAT:     0: ^[[AWAIT_INIT_RESUME]],
// CIR-FLAT:     1: ^[[INIT_CLEANUP_DESTROY:.*]]
// CIR-FLAT:   ]
// CIR-FLAT: ^[[INIT_CLEANUP_DESTROY]]:
// CIR-FLAT:   cir.const #cir.int<0>
// CIR-FLAT:   cir.store {{.*}}, %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT:   cir.br ^[[CLEANUP_CORO_FREE:.*]]
// CIR-FLAT: ^[[AWAIT_INIT_RESUME]]:
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv(%[[SuspendAlwaysAddr]])
// CIR-FLAT:   cir.br ^[[AWAIT_INIT_RESUME_CONT:.*]]
// CIR-FLAT: ^[[AWAIT_INIT_RESUME_CONT]]:
// CIR-FLAT:   cir.br ^[[CORO_BODY:.*]]
// CIR-FLAT: ^[[CORO_BODY]]:
// CIR-FLAT:   cir.br ^[[HAS_DTOR_CLEANUP_SCOPE:.*]]
// CIR-FLAT: ^[[HAS_DTOR_CLEANUP_SCOPE]]:
// CIR-FLAT:   cir.br ^[[USER_AWAIT_READY:.*]]
// CIR-FLAT: ^[[USER_AWAIT_READY]]:
// CIR-FLAT:   %[[ShouldSuspend2:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv
// CIR-FLAT:   cir.brcond %[[ShouldSuspend2]] ^[[AWAIT_USER_RESUME:.*]], ^[[AWAIT_USER_SUSPEND:.*]]
// CIR-FLAT: ^[[AWAIT_USER_SUSPEND]]:
// CIR-FLAT:   %[[NULLPTR2:.*]] = cir.const #cir.ptr<null>
// CIR-FLAT:   %[[SAVE_TOKEN2:.*]] = cir.call_llvm_intrinsic "llvm.coro.save" %[[NULLPTR2]]
// CIR-FLAT:   %[[IS_FINAL_SUSPEND2:.*]] = cir.const #false
// CIR-FLAT:   %[[SUSPEND_RESULT2:.*]] = cir.call_llvm_intrinsic "llvm.coro.suspend" %[[SAVE_TOKEN2]], %[[IS_FINAL_SUSPEND2]]

// The destroy branch cannot jump directly to the coroutine cleanup.
// Since this await is nested within the lifetime of a local HasDtor object,
// destruction must first run HasDtor::~HasDtor() before proceeding to the
// coroutine-frame cleanup.
//
// Destroy path:
// USER_CLEANUP_DESTROY -> HasDtor::~HasDtor() -> coro.free cleanup

// CIR-FLAT:   cir.switch.flat %[[SUSPEND_RESULT2]] : !u32i, ^[[CORO_RET]] [
// CIR-FLAT:     0: ^[[AWAIT_USER_RESUME]],
// CIR-FLAT:     1: ^[[USER_CLEANUP_DESTROY:.*]]
// CIR-FLAT:   ]
// CIR-FLAT: ^[[USER_CLEANUP_DESTROY]]:
// CIR-FLAT:   cir.const #cir.int<0>
// CIR-FLAT:   cir.store {{.*}}, %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT:   cir.br ^[[HAS_DTR:.*]]
// CIR-FLAT: ^[[AWAIT_USER_RESUME]]:
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv
// CIR-FLAT:   cir.br ^[[CO_RETURN:.*]]
// CIR-FLAT: ^[[CO_RETURN]]:
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type11return_voidEv
// CIR-FLAT:   cir.const #cir.int<1>
// CIR-FLAT:   cir.store {{.*}}, %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT:   cir.br ^[[HAS_DTR]]
// CIR-FLAT: ^[[HAS_DTR]]:
// CIR-FLAT:   cir.call @_ZN7HasDtorD1Ev
// CIR-FLAT:   cir.br ^[[HAS_DTR_CONT:.*]]
// CIR-FLAT: ^[[HAS_DTR_CONT]]:
// CIR-FLAT:   %[[LOAD_DEST_SLOT:.*]] = cir.load %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT:   cir.switch.flat %[[LOAD_DEST_SLOT]] : !s32i, ^[[DEFAULT:.*]] [
// CIR-FLAT:     0: ^[[FROM_USER_CLEANUP_DESTROY:.*]],
// CIR-FLAT:     1: ^[[FROM_CO_RETURN:.*]]
// CIR-FLAT:   ]
// CIR-FLAT: ^[[FROM_USER_CLEANUP_DESTROY]]:
// CIR-FLAT:   cir.br ^[[EXIT_TO_CORO_FREE:.*]]
// CIR-FLAT: ^[[FROM_CO_RETURN]]:
// CIR-FLAT:   cir.br ^[[FINAL_SUSPEND_BB:.*]]
// CIR-FLAT: ^[[DEFAULT]]:
// CIR-FLAT:   cir.unreachable
// CIR-FLAT: ^[[EXIT_TO_CORO_FREE]]:
// CIR-FLAT:   cir.const #cir.int<1>
// CIR-FLAT:   cir.store {{.*}}, %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT:   cir.br ^[[CLEANUP_CORO_FREE]]
// CIR-FLAT: ^[[FINAL_SUSPEND_BB:.*]]:
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIvE12promise_type13final_suspendEv
// CIR-FLAT:   cir.br ^[[FINAL_AWAIT_READY_CONT:.*]]
// CIR-FLAT:   cir.call @_ZNSt14suspend_always11await_readyEv
// CIR-FLAT:   cir.brcond {{.*}} ^[[AWAIT_FINAL_RESUME:.*]], ^[[AWAIT_FINAL_SUSPEND:.*]]
// CIR-FLAT: ^[[AWAIT_FINAL_SUSPEND]]:
// CIR-FLAT:   cir.call_llvm_intrinsic "llvm.coro.save"
// CIR-FLAT:   cir.const #true
// CIR-FLAT:   %[[SUSPEND_RESULT3:.*]] = cir.call_llvm_intrinsic "llvm.coro.suspend"
// CIR-FLAT:   cir.switch.flat %[[SUSPEND_RESULT3]] : !u32i, ^[[CORO_RET]] [
// CIR-FLAT:     0: ^[[AWAIT_FINAL_RESUME]],
// CIR-FLAT:     1: ^[[FINAL_CLEANUP_DESTROY:.*]]
// CIR-FLAT:   ]
// CIR-FLAT: ^[[FINAL_CLEANUP_DESTROY]]:
// CIR-FLAT:   cir.const #cir.int<2>
// CIR-FLAT:   cir.store {{.*}}, %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT:   cir.br ^bb28
// CIR-FLAT: ^[[AWAIT_FINAL_RESUME]]:
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv
// CIR-FLAT:   cir.br ^[[CORO_BODY_EXIT:.*]]
// CIR-FLAT: ^[[CORO_BODY_EXIT]]:
// CIR-FLAT:   cir.const #cir.int<3> : !s32i
// CIR-FLAT:   cir.store {{.*}}, %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT:   cir.br ^[[CLEANUP_CORO_FREE]]
// CIR-FLAT: ^[[CLEANUP_CORO_FREE]]:
// CIR-FLAT:   %[[SHOULD_FREE:.*]] = cir.call @__builtin_coro_free
// CIR-FLAT:   %[[NULLPTR5:.*]] = cir.const #cir.ptr<null> : !cir.ptr<!void>
// CIR-FLAT:   %[[NEEDS_FREE:.*]] = cir.cmp ne %[[SHOULD_FREE]], %[[NULLPTR5]]
// CIR-FLAT:   cir.brcond %[[NEEDS_FREE]] ^[[FREE_FRAME:.*]], ^[[EXIT_CLEANUP:.*]]
// CIR-FLAT: ^[[FREE_FRAME]]:
// CIR-FLAT:   %60 = cir.call @__builtin_coro_size() : () -> (!u64i {llvm.noundef})
// CIR-FLAT:   cir.call @_ZdlPvm(%57, %60) nothrow : (!cir.ptr<!void> {llvm.noundef}, !u64i {llvm.noundef}) -> ()
// CIR-FLAT:   cir.br ^[[EXIT_CLEANUP]]
// CIR-FLAT: ^[[EXIT_CLEANUP]]:
// CIR-FLAT:   cir.br ^[[EXIT_CLEANUP_SWITCH:.*]]
// CIR-FLAT: ^[[EXIT_CLEANUP_SWITCH]]:
// CIR-FLAT:   %[[LOAD_DEST_SLOT:.*]] = cir.load %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT:   cir.switch.flat %[[LOAD_DEST_SLOT:.*]]
// CIR-FLAT: ^[[CORO_RET]]:
// CIR-FLAT:   cir.return

folly::coro::Task<int> co_returns(int flag) {
  if (flag == 1) {
    co_return 1;
  } else if (flag == 2) {
    co_return 2;
  }
  co_return 3;
}

// CIR-FLAT: cir.func flattened_coroutine {{.*}} @_Z10co_returnsi
// CIR-FLAT:   %[[CLEANUP_DEST_SLOT:.*]] = cir.alloca "__cleanup_dest_slot"
// CIR-FLAT:   %[[SUSPEND_POINT:.*]] = cir.alloca "__coroutine_suspend_point"

// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv
// CIR-FLAT:   %[[LOAD_FLAG:.*]] = cir.load {{.*}} %[[FLAG:.*]]
// CIR-FLAT:   %[[ONE:.*]] = cir.const #cir.int<1>
// CIR-FLAT:   %[[EQ_ONE:.*]] = cir.cmp eq %[[LOAD_FLAG]], %[[ONE]]
// CIR-FLAT:   cir.brcond %[[EQ_ONE]] ^[[IF_EQ_ONE:.*]], ^[[ELSE_IF:.*]]
// CIR-FLAT: ^[[IF_EQ_ONE]]:
// CIR-FLAT:   %[[ONE:.*]] = cir.const #cir.int<1>
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE:.*]], %[[ONE]])
// CIR-FLAT:   cir.br ^[[FINAL_SUSPEND_BB:.*]]
// CIR-FLAT: ^[[ELSE_IF]]:
// CIR-FLAT:   cir.br ^[[ELSE_IF_CONT:.*]]
// CIR-FLAT: ^[[ELSE_IF_CONT:.*]]:
// CIR-FLAT:   %[[LOAD_FLAG2:.*]] = cir.load {{.*}} %[[FLAG]]
// CIR-FLAT:   %[[TWO:.*]] = cir.const #cir.int<2>
// CIR-FLAT:   %[[EQ_TWO:.*]] = cir.cmp eq %[[LOAD_FLAG2]], %[[TWO]]
// CIR-FLAT:   cir.brcond %[[EQ_TWO]] ^[[IF_EQ_TWO:.*]], ^[[ELSE:.*]]
// CIR-FLAT: ^[[IF_EQ_TWO]]:
// CIR-FLAT:   %[[TWO:.*]] = cir.const #cir.int<2>
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], %[[TWO]])
// CIR-FLAT:   cir.br ^[[FINAL_SUSPEND_BB]]
// CIR-FLAT: ^[[ELSE]]:
// CIR-FLAT:   cir.br ^[[ELSE_CONT:.*]]
// CIR-FLAT: ^[[ELSE_CONT]]:
// CIR-FLAT:   %[[THREE:.*]] = cir.const #cir.int<3>
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], %[[THREE]])
// CIR-FLAT:   cir.br ^[[FINAL_SUSPEND_BB]]
// CIR-FLAT: ^[[FINAL_SUSPEND_BB]]:
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type13final_suspendEv



folly::coro::Task<int> co_return_with_dtor(int flag) {
  HasDtor local;
  if (flag)
    co_return 1;        // local dtor must run here
  co_return 2;
}

// CIR-FLAT:  cir.func flattened_coroutine {{.*}} @_Z19co_return_with_dtori
// CIR-FLAT:  %[[CLEANUP_DEST_SLOT:.*]] = cir.alloca "__cleanup_dest_slot"
// CIR-FLAT:  %[[SUSPEND_POINT:.*]] = cir.alloca "__coroutine_suspend_point"
// CIR-FLAT:  %[[ShouldSuspend:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv
// CIR-FLAT:  cir.brcond %[[ShouldSuspend]] ^[[AWAIT_INIT_RESUME:.*]], ^[[AWAIT_INIT_SUSPEND:.*]]
// CIR-FLAT: ^[[AWAIT_INIT_SUSPEND]]:
// CIR-FLAT:   cir.switch.flat {{.*}} : !u32i, ^[[CORO_RET:.*]] [
// CIR-FLAT:     0: ^[[AWAIT_INIT_RESUME]],
// CIR-FLAT:     1: ^[[INIT_CLEANUP_DESTROY:.*]]
// CIR-FLAT:   ]
// CIR-FLAT: ^[[INIT_CLEANUP_DESTROY]]:
// CIR-FLAT:   cir.const #cir.int<0> : !s32i
// CIR-FLAT:   cir.store {{.*}}, %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT:   cir.br ^[[CLEANUP_CORO_FREE:.*]]
// CIR-FLAT: ^[[AWAIT_INIT_RESUME]]:
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv(%10)
// CIR-FLAT:   cir.br ^[[CORO_BODY:.*]]
// CIR-FLAT: ^[[CORO_BODY]]:
// CIR-FLAT:   cir.br ^[[HAS_DTR_CLEANUP_SCOPE:.*]]
// CIR-FLAT: ^[[HAS_DTR_CLEANUP_SCOPE]]:
// CIR-FLAT:   cir.br ^[[IF_SCOPE:.*]]
// CIR-FLAT: ^[[IF_SCOPE]]:
// CIR-FLAT:   cir.br ^[[IF_SCOPE_CONT:.*]]
// CIR-FLAT: ^[[IF_SCOPE_CONT]]:
// CIR-FLAT:   %[[LOAD_FLAG:.*]] = cir.load {{.*}} %[[FLAG_ARG:.*]]
// CIR-FLAT:   %[[CAST_TO_BOOL:.*]] = cir.cast int_to_bool %[[LOAD_FLAG]]
// CIR-FLAT:   cir.brcond %[[CAST_TO_BOOL]] ^[[IF_BODY:.*]], ^[[IF_CONT:.*]]
// CIR-FLAT: ^[[IF_BODY]]:
// CIR-FLAT:   %[[ONE:.*]] = cir.const #cir.int<1>
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE:.*]], %[[ONE]])
// CIR-FLAT:   cir.const #cir.int<0>
// CIR-FLAT:   cir.store {{.*}}, %[[CLEANUP_DEST_SLOT]]

// This cir.co_return exits through the cleanup path. Control is transferred
// to the destructor cleanup block before reaching the final suspend point.
// co_return -> cleanup -> final suspend
// CIR-FLAT:   cir.br ^[[HAS_DTOR_CLEANUP:.*]]
// CIR-FLAT: ^[[IF_CONT]]:
// CIR-FLAT:   %[[TWO:.*]] = cir.const #cir.int<2>
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type12return_valueEi(%[[PROMISE]], %[[TWO]])
// CIR-FLAT:   cir.const #cir.int<1>
// CIR-FLAT:   cir.store {{.*}}, %[[CLEANUP_DEST_SLOT]]

// This cir.co_return exits through the cleanup path. Control is transferred
// to the destructor cleanup block before reaching the final suspend point.
// co_return -> cleanup -> final suspend
// CIR-FLAT:   cir.br ^[[HAS_DTOR_CLEANUP]]
// CIR-FLAT: ^[[HAS_DTOR_CLEANUP]]:
// CIR-FLAT:   cir.call @_ZN7HasDtorD1Ev
// CIR-FLAT:   cir.br ^[[CLEANUP_EXIT:.*]]
// CIR-FLAT: ^[[CLEANUP_EXIT]]:
// CIR-FLAT:   %[[LOAD_CLEANUP:.*]] = cir.load %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT:   cir.switch.flat %[[LOAD_CLEANUP:.*]] : !s32i, ^[[DEFAULT:.*]] [
// CIR-FLAT:     0: ^[[EXIT1:.*]],
// CIR-FLAT:     1: ^[[EXIT2:.*]]
// CIR-FLAT:   ]
// CIR-FLAT: ^[[EXIT1]]:
// CIR-FLAT:   cir.br ^[[FINAL_SUSPEND_BB:.*]]
// CIR-FLAT: ^[[EXIT2]]:
// CIR-FLAT:   cir.br ^[[FINAL_SUSPEND_BB]]
// CIR-FLAT: ^[[DEFAULT]]:
// CIR-FLAT:   cir.unreachable
// CIR-FLAT: ^[[FINAL_SUSPEND_BB]]:
// CIR-FLAT:   cir.call @_ZN5folly4coro4TaskIiE12promise_type13final_suspendEv
// CIR-FLAT:   %[[ShouldSuspend2:.*]] = cir.call @_ZNSt14suspend_always11await_readyEv
// CIR-FLAT:   cir.brcond %[[ShouldSuspend2]] ^[[AWAIT_FINAL_RESUME:.*]], ^[[AWAIT_FINAL_SUSPEND:.*]]
// CIR-FLAT: ^[[AWAIT_FINAL_SUSPEND]]:
// CIR-FLAT:   %[[IS_FINAL_SUSPEND:.*]] = cir.const #true
// CIR-FLAT:   cir.call_llvm_intrinsic "llvm.coro.suspend" {{.*}}, %[[IS_FINAL_SUSPEND]]
// CIR-FLAT:   cir.switch.flat {{.*}} : !u32i, ^[[CORO_RET]] [
// CIR-FLAT:     0: ^[[AWAIT_FINAL_RESUME]],
// CIR-FLAT:     1: ^[[FINAL_CLEANUP_DESTROY:.*]]
// CIR-FLAT:   ]
// CIR-FLAT: ^[[FINAL_CLEANUP_DESTROY]]:
// CIR-FLAT:   cir.const #cir.int<2>
// CIR-FLAT:   cir.store {{.*}}, %[[CLEANUP_DEST_SLOT]]
// CIR-FLAT:   cir.br ^[[CLEANUP_CORO_FREE]]
// CIR-FLAT: ^[[AWAIT_FINAL_RESUME]]:
// CIR-FLAT:   cir.call @_ZNSt14suspend_always12await_resumeEv

// CIR-FLAT: ^[[CLEANUP_CORO_FREE]]:
// CIR-FLAT:   cir.call @__builtin_coro_free
// CIR-FLAT:   cir.const #cir.ptr<null>
// CIR-FLAT:   cir.brcond %{{.*}} ^[[FREE_FRAME:.*]], ^[[EXIT_CLEANUP:.*]]
// CIR-FLAT: ^[[FREE_FRAME]]:
// CIR-FLAT:   cir.call @__builtin_coro_size()
// CIR-FLAT:   cir.call @_ZdlPvm
