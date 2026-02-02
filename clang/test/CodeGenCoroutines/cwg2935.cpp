// Verifies lifetime of __gro local variable
// Verify that coroutine promise and allocated memory are freed up on exception.
// RUN: %clang_cc1 -std=c++20 -triple=x86_64-unknown-linux-gnu -emit-llvm -o - %s -disable-llvm-passes | FileCheck %s

#include "Inputs/coroutine.h"

using namespace std;

struct task {
  struct promise_type {
    task get_return_object();
    suspend_always initial_suspend();
    suspend_always final_suspend() noexcept;
    void return_void();
    void unhandled_exception();
  };
  ~task();
};

task f1() {
  co_return;
}
// CHECK:       define dso_local void @_Z2f1v
// CHECK:       entry:
// CHECK-NEXT:    %result.ptr = alloca ptr, align 8
// CHECK-NEXT:    %__promise = alloca %"struct.task::promise_type", align 1
// CHECK-NEXT:    %direct.gro.active = alloca i1, align 1
// CHECK:       coro.init:                                        ; preds = %coro.alloc, %entry
// CHECK-NEXT:    %3 = phi ptr [ null, %entry ], [ %call, %coro.alloc ]
// CHECK-NEXT:    %4 = call ptr @llvm.coro.begin(token %0, ptr %3)
// CHECK-NEXT:    call void @llvm.lifetime.start.p{{.*}}(ptr %__promise) #2
// CHECK-NEXT:    call void @_ZN4task12promise_type17get_return_objectEv(ptr dead_on_unwind writable sret(%struct.task) align 1 %agg.result, ptr noundef nonnull align 1 dereferenceable(1) %__promise)
// CHECK-NEXT:    store i1 true, ptr %direct.gro.active, align 1
// CHECK:       cleanup.cont:                                     ; preds = %cleanup
// CHECK-NEXT:    store i1 false, ptr %direct.gro.active, align 1
// CHECK-NEXT:    call void @_ZN4task12promise_type11return_voidEv(ptr noundef nonnull align 1 dereferenceable(1) %__promise)
// CHECK-NEXT:    br label %coro.final
// CHECK:       cleanup.cont{{.*}}:
// CHECK-NEXT:    store i1 false, ptr %direct.gro.active, align 1
// CHECK-NEXT:    br label %cleanup{{.*}}
// CHECK:       cleanup.action:                                   ; preds = %cleanup{{.*}}
// CHECK-NEXT:    call void @_ZN4taskD1Ev(ptr noundef nonnull align 1 dereferenceable(1) %agg.result) #2
// CHECK-NEXT:    br label %cleanup.done

struct intial_awaiter_resume_throw : std::suspend_always {
  void await_resume();
};

struct task_resume_throw {
  struct promise_type {
    task_resume_throw get_return_object();
    suspend_always initial_suspend() noexcept;
    suspend_always final_suspend() noexcept;
    void return_void();
    void unhandled_exception();
  };
  ~task_resume_throw();
};

task_resume_throw f2() {
  co_return;
}

// CHECK:       define dso_local void @_Z2f2v
// CHECK-NOT:     %direct.gro.active = alloca i1, align 1