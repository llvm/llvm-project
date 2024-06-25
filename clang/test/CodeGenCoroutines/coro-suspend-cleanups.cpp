// RUN: %clang_cc1 --std=c++20 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck %s

#include "Inputs/coroutine.h"

struct Printy {
  Printy(const char *name) : name(name) {}
  ~Printy() {}
  const char *name;
};

struct coroutine {
  struct promise_type;
  std::coroutine_handle<promise_type> handle;
  ~coroutine() {
    if (handle) handle.destroy();
  }
};

struct coroutine::promise_type {
  coroutine get_return_object() {
    return {std::coroutine_handle<promise_type>::from_promise(*this)};
  }
  std::suspend_never initial_suspend() noexcept { return {}; }
  std::suspend_always final_suspend() noexcept { return {}; }
  void return_void() {}
  void unhandled_exception() {}
};

struct Awaiter : std::suspend_always {
  Printy await_resume() { return {"awaited"}; }
};

int foo() { return 2; }

coroutine ArrayInitCoro() {
  // Verify that:
  //  - We do the necessary stores for array cleanups.
  //  - Array cleanups are called by await.cleanup.
  //  - We activate the cleanup after the first element and deactivate it in await.ready (see cleanup.isactive).

  // CHECK-LABEL: define dso_local void @_Z13ArrayInitCorov
  // CHECK: %arrayinit.endOfInit = alloca ptr, align 8
  // CHECK: %cleanup.isactive = alloca i1, align 1
  Printy arr[2] = {
    Printy("a"),
    // CHECK:       store i1 true, ptr %cleanup.isactive.reload.addr, align 1
    // CHECK-NEXT:  store ptr %arr.reload.addr, ptr %arrayinit.endOfInit.reload.addr, align 8
    // CHECK-NEXT:  call void @_ZN6PrintyC1EPKc(ptr noundef nonnull align 8 dereferenceable(8) %arr.reload.addr, ptr noundef @.str)
    // CHECK-NEXT:  %arrayinit.element = getelementptr inbounds %struct.Printy, ptr %arr.reload.addr, i64 1
    // CHECK-NEXT:  %arrayinit.element.spill.addr = getelementptr inbounds %_Z13ArrayInitCorov.Frame, ptr %0, i32 0, i32 10
    // CHECK-NEXT:  store ptr %arrayinit.element, ptr %arrayinit.element.spill.addr, align 8
    // CHECK-NEXT:  store ptr %arrayinit.element, ptr %arrayinit.endOfInit.reload.addr, align 8
    co_await Awaiter{}
    // CHECK-NEXT:  @_ZNSt14suspend_always11await_readyEv
    // CHECK-NEXT:  br i1 %{{.+}}, label %await.ready, label %CoroSave30
  };
  // CHECK:       await.cleanup:                                    ; preds = %AfterCoroSuspend{{.*}}
  // CHECK-NEXT:    br label %cleanup{{.*}}.from.await.cleanup

  // CHECK:       cleanup{{.*}}.from.await.cleanup:                      ; preds = %await.cleanup
  // CHECK:         br label %cleanup{{.*}}

  // CHECK:       await.ready:
  // CHECK-NEXT:    %arrayinit.element.reload.addr = getelementptr inbounds %_Z13ArrayInitCorov.Frame, ptr %0, i32 0, i32 10
  // CHECK-NEXT:    %arrayinit.element.reload = load ptr, ptr %arrayinit.element.reload.addr, align 8
  // CHECK-NEXT:    call void @_ZN7Awaiter12await_resumeEv
  // CHECK-NEXT:    store i1 false, ptr %cleanup.isactive.reload.addr, align 1
  // CHECK-NEXT:    br label %cleanup{{.*}}.from.await.ready

  // CHECK:       cleanup{{.*}}:                                         ; preds = %cleanup{{.*}}.from.await.ready, %cleanup{{.*}}.from.await.cleanup
  // CHECK:         %cleanup.is_active = load i1, ptr %cleanup.isactive.reload.addr, align 1
  // CHECK-NEXT:    br i1 %cleanup.is_active, label %cleanup.action, label %cleanup.done

  // CHECK:       cleanup.action:
  // CHECK:         %arraydestroy.isempty = icmp eq ptr %arr.reload.addr, %{{.*}}
  // CHECK-NEXT:    br i1 %arraydestroy.isempty, label %arraydestroy.done{{.*}}, label %arraydestroy.body.from.cleanup.action
  // Ignore rest of the array cleanup.
}

coroutine ArrayInitWithCoReturn() {
  // CHECK-LABEL: define dso_local void @_Z21ArrayInitWithCoReturnv
  // Verify that we start to emit the array destructor.
  // CHECK: %arrayinit.endOfInit = alloca ptr, align 8
  Printy arr[2] = {"a", ({
                      if (foo()) {
                        co_return;
                      }
                      "b";
                    })};
}
