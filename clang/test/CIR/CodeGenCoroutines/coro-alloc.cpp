// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -fclangir -Wno-coroutine-missing-unhandled-exception -emit-cir %s -o %t.cir
// RUN: FileCheck --input-file=%t.cir %s -check-prefix=CIR
// RUN: %clang_cc1 -std=c++20 -triple x86_64-unknown-linux-gnu -emit-llvm  -disable-llvm-passes -Wno-coroutine-missing-unhandled-exception %s -o %t-cir.ll
// RUN: FileCheck --input-file=%t-cir.ll %s --check-prefix=LLVM
#include "Inputs/coroutine.h"
namespace std {

struct nothrow_t {};
constexpr nothrow_t nothrow = {};

} // end namespace std

// Required when get_return_object_on_allocation_failure() is defined by
// the promise. The nothrow overload prevents allocation failure from
// throwing std::bad_alloc.
using SizeT = decltype(sizeof(int));
void* operator new(SizeT __sz, const std::nothrow_t&) noexcept;
void  operator delete(void* __p, const std::nothrow_t&) noexcept;

struct promise_on_alloc_failure_tag {};

template <>
struct std::coroutine_traits<int, promise_on_alloc_failure_tag> {
  struct promise_type {
    int get_return_object() { return 0; }
    suspend_always initial_suspend() { return {}; }
    suspend_always final_suspend() noexcept { return {}; }
    void return_void() {}
    static int get_return_object_on_allocation_failure() { return -1; }
  };
};

int f4(promise_on_alloc_failure_tag) {

  co_return;
}

// CIR-LABEL: @_Z2f428promise_on_alloc_failure_tag(
// CIR: %[[RetVal:.*]] = cir.alloca "__retval"
// CIR: %[[FrameAddr:.*]] = cir.alloca "__coro_frame_addr"
// CIR: %[[NULL_INIT:.*]] = cir.const #cir.ptr<null>
// CIR: %[[CORO_ID:.*]] = cir.coro.intrinsic.id(
// CIR: %[[ShouldAlloc:.*]] = cir.coro.intrinsic.alloc(%[[CORO_ID]])
// CIR: cir.store %[[NULL_INIT]], %[[FrameAddr]]
// CIR: cir.if %[[ShouldAlloc]] {
// CIR:   %[[CORO_SIZE:.*]] = cir.coro.intrinsic.size()
// CIR:   %[[STD_NOTHROW:.*]] = cir.get_global @_ZStL7nothrow
// CIR:   %[[ALLOC_ADDR:.*]] = cir.call @_ZnwmRKSt9nothrow_t(%[[CORO_SIZE]], %[[STD_NOTHROW]])
// CIR:   cir.store %[[ALLOC_ADDR]], %[[FrameAddr]]
// CIR:   %[[NULL_PTR:.*]] = cir.const #cir.ptr<null>
// CIR:   %[[IS_NULL_PTR:.*]] = cir.cmp eq %[[ALLOC_ADDR]], %[[NULL_PTR]]
// CIR:   cir.if %[[IS_NULL_PTR]] {
// CIR:     %[[FailRet:.*]] = cir.call @_ZNSt16coroutine_traitsIiJ28promise_on_alloc_failure_tagEE12promise_type39get_return_object_on_allocation_failureEv
// CIR:     cir.store %[[FailRet]], %[[RetVal]] : !s32i
// CIR:     %[[RET:.*]] = cir.load %[[RetVal]] : !cir.ptr<!s32i>
// CIR:     cir.return %[[RET]] : !s32i
// CIR:   ^[[UNREACHABLE:.*]]:
// CIR:     cir.unreachable
// CIR:   }
// CIR: }

// LLVM-LABEL: @_Z2f428promise_on_alloc_failure_tag(
// LLVM: %[[RetVal:.*]] = alloca i32
// LLVM: %[[ID:.*]] = call token @llvm.coro.id(i32 16
// LLVM: %[[SIZE:.*]] = call i64 @llvm.coro.size.i64()
// LLVM: %[[MEM:.*]] = call noalias noundef ptr @_ZnwmRKSt9nothrow_t(i64 noundef %[[SIZE]], ptr noundef nonnull align 1 dereferenceable(1) @_ZStL7nothrow)
// LLVM: %[[OK:.*]] = icmp ne ptr %[[MEM]], null
// LLVM: br i1 %[[OK]], label %[[OKBB:.*]], label %[[ERRBB:.*]]

// LLVM: [[ERRBB]]:
// LLVM:   %[[FailRet:.*]] = call noundef i32 @_ZNSt16coroutine_traitsIiJ28promise_on_alloc_failure_tagEE12promise_type39get_return_object_on_allocation_failureEv()
// LLVM:   store i32 %[[FailRet]], ptr %[[RetVal]]
// LLVM:   br label %[[RetBB:.*]]

// LLVM: [[OKBB]]:
// LLVM:   %[[OkRet:.*]] = call noundef i32 @_ZNSt16coroutine_traitsIiJ28promise_on_alloc_failure_tagEE12promise_type17get_return_objectEv({{.*}}

// LLVM: [[RetBB]]:
// LLVM:   %[[LoadRet:.*]] = load i32, ptr %[[RetVal]], align 4
// LLVM:   ret i32 %[[LoadRet]]
