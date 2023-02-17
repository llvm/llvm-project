// RUN: %clang_cc1 -std=c++14 -fcoroutines-ts -triple=x86_64-pc-windows-msvc18.0.0 -emit-llvm %s -o - -fexceptions -fcxx-exceptions -disable-llvm-passes | FileCheck %s
// RUN: %clang_cc1 -std=c++14 -fcoroutines-ts -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s -fexceptions -fcxx-exceptions -disable-llvm-passes | FileCheck --check-prefix=CHECK-LPAD %s

#include "Inputs/coroutine-exp-namespace.h"

namespace coro = std::experimental::coroutines_v1;

namespace std {
using exception_ptr = int;
exception_ptr current_exception();
} // namespace std

struct coro_t {
  struct promise_type {
    coro_t get_return_object() {
      coro::coroutine_handle<promise_type>{};
      return {};
    }
    coro::suspend_never initial_suspend() { return {}; }
    coro::suspend_never final_suspend() noexcept { return {}; }
    void return_void() {}
    void unhandled_exception() noexcept;
  };
};

struct Cleanup {
  ~Cleanup();
};
void may_throw();

coro_t f() {
  Cleanup x;
  may_throw();
  co_return;
}

// CHECK: @"?f@@YA?AUcoro_t@@XZ"(
// CHECK:   invoke void @"?may_throw@@YAXXZ"()
// CHECK:       to label %{{.+}} unwind label %[[EHCLEANUP:.+]]
// CHECK: [[EHCLEANUP]]:
// CHECK:   %[[INNERPAD:.+]] = cleanuppad within none []
// CHECK:   call void @"??1Cleanup@@QEAA@XZ"(
// CHECK:   cleanupret from %[[INNERPAD]] unwind label %[[CATCHSW:.+]]
// CHECK: [[CATCHSW]]:
// CHECK:   %[[CATCHSWTOK:.+]] = catchswitch within none [label %[[CATCH:.+]]] unwind label
// CHECK: [[CATCH]]:
// CHECK:   %[[CATCHTOK:.+]] = catchpad within [[CATCHSWTOK:.+]]
// CHECK:   call void @"?unhandled_exception@promise_type@coro_t@@QEAAXXZ"
// CHECK:   catchret from %[[CATCHTOK]] to label %[[CATCHRETDEST:.+]]
// CHECK: [[CATCHRETDEST]]:
// CHECK-NEXT: br label %[[TRYCONT:.+]]
// CHECK: [[TRYCONT]]:
// CHECK-NEXT: br label %[[COROFIN:.+]]
// CHECK: [[COROFIN]]:
// CHECK-NEXT: call void @llvm.lifetime.start.p0(
// CHECK-NEXT: call void @"?final_suspend@promise_type@coro_t@@QEAA?AUsuspend_never@coroutines_v1@experimental@std@@XZ"(

// CHECK-LPAD: @_Z1fv(
// CHECK-LPAD:   invoke void @_Z9may_throwv()
// CHECK-LPAD:       to label %[[CONT:.+]] unwind label %[[CLEANUP:.+]]
// CHECK-LPAD: [[CLEANUP]]:
// CHECK-LPAD:   call void @_ZN7CleanupD1Ev(ptr {{[^,]*}} %x) #2
// CHECK-LPAD:   br label %[[CATCH:.+]]

// CHECK-LPAD: [[CATCH]]:
// CHECK-LPAD:    call ptr @__cxa_begin_catch
// CHECK-LPAD:    call void @_ZN6coro_t12promise_type19unhandled_exceptionEv(ptr {{[^,]*}} %__promise) #2
// CHECK-LPAD:    invoke void @__cxa_end_catch()
// CHECK-LPAD-NEXT:  to label %[[CATCHRETDEST:.+]] unwind label
// CHECK-LPAD: [[CATCHRETDEST]]:
// CHECK-LPAD-NEXT: br label %[[TRYCONT:.+]]
// CHECK-LPAD: [[TRYCONT]]:
// CHECK-LPAD: br label %[[COROFIN:.+]]
// CHECK-LPAD: [[COROFIN]]:
// CHECK-LPAD-NEXT: call void @llvm.lifetime.start.p0(
// CHECK-LPAD-NEXT: call void @_ZN6coro_t12promise_type13final_suspendEv(
