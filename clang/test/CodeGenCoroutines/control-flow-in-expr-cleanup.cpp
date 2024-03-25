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

struct Printies {
  Printy a;
  Printy b;
  Printy c;
};

void ParenInit() {
  // CHECK-LABEL: define dso_local void @_Z9ParenInitv()
  // CHECK: [[CLEANUP_DEST:%.+]] = alloca i32, align 4
  Printies ps(Printy("a"), 
              // CHECK: call void @_ZN6PrintyC1EPKc
              ({
                if (foo()) return;
                // CHECK:     if.then:
                // CHECK-NEXT:   store i32 1, ptr [[CLEANUP_DEST]], align 4
                // CHECK-NEXT:   br label %cleanup
                Printy("b");
                // CHECK:     if.end:
                // CHECK-NEXT:  call void @_ZN6PrintyC1EPKc
              }),
              ({
                if (foo()) return;
                // CHECK:     if.then{{.*}}:
                // CHECK-NEXT:  store i32 1, ptr [[CLEANUP_DEST]], align 4
                // CHECK-NEXT:  call void @_ZN6PrintyD1Ev
                // CHECK-NEXT:  br label %cleanup
                Printy("c");
                // CHECK:     if.end{{.*}}:
                // CHECK-NEXT:  call void @_ZN6PrintyC1EPKc
                // CHECK-NEXT:  call void @_ZN8PrintiesD1Ev
                // CHECK-NEXT:  br label %return
              }));
  // CHECK:     cleanup:
  // CHECK-NEXT:  call void @_ZN6PrintyD1Ev
  // CHECK-NEXT:  br label %return
}

coroutine ParenInitCoro() {
  // CHECK-LABEL: define dso_local void @_Z13ParenInitCorov
  // CHECK: [[ACTIVE1:%.+]] = alloca i1, align 1
  // CHECK: [[ACTIVE2:%.+]] = alloca i1, align 1
  Printies ps(Printy("a"), Printy("b"),
    // CHECK:       call void @_ZN6PrintyC1EPKc
    // CHECK-NEXT:  store i1 true, ptr [[ACTIVE2]].reload.addr, align 1
    // CHECK:       call void @_ZN6PrintyC1EPKc
    // CHECK-NEXT:  store i1 true, ptr [[ACTIVE1]].reload.addr, align 1
    co_await Awaiter{}

    // CHECK:     await.cleanup:
    // CHECK-NEXT:  br label %[[CLEANUP:.+]].from.await.cleanup

    // CHECK:     [[CLEANUP]].from.await.cleanup:
    // CHECK:       br label %[[CLEANUP]]

    // CHECK:     await.ready:
    // CHECK:       store i1 false, ptr [[ACTIVE1]].reload.addr, align 1
    // CHECK-NEXT:  store i1 false, ptr [[ACTIVE2]].reload.addr, align 1 
    // CHECK-NEXT:  br label %[[CLEANUP]].from.await.ready

    // CHECK:     [[CLEANUP]].from.await.ready:
    // CHECK:       br label %[[CLEANUP]]

    // CHECK:     [[CLEANUP]]:
    // CHECK:       [[IS_ACTIVE1:%.+]] = load i1, ptr [[ACTIVE1]].reload.addr, align 1
    // CHECK-NEXT:  br i1 [[IS_ACTIVE1]], label %[[ACTION1:.+]], label %[[DONE1:.+]]

    // CHECK:     [[ACTION1]]:
    // CHECK:       call void @_ZN6PrintyD1Ev
    // CHECK:       br label %[[DONE1]]

    // CHECK:     [[DONE1]]:
    // CHECK:       [[IS_ACTIVE2:%.+]] = load i1, ptr [[ACTIVE2]].reload.addr, align 1
    // CHECK-NEXT:  br i1 [[IS_ACTIVE2]], label %[[ACTION2:.+]], label %[[DONE2:.+]]

    // CHECK:     [[ACTION2]]:
    // CHECK:       call void @_ZN6PrintyD1Ev
    // CHECK:       br label %[[DONE2]]
  );
}

void break_in_stmt_expr() {
  // Verify that the "break" in "if.then".calls dtor before jumping to "for.end".

  // CHECK-LABEL: define dso_local void @_Z18break_in_stmt_exprv()
  Printies p{Printy("a"), 
            // CHECK: call void @_ZN6PrintyC1EPKc
            ({
                for (;;) {
                    Printies ps{
                      Printy("b"), 
                      // CHECK: for.cond:
                      // CHECK:   call void @_ZN6PrintyC1EPKc
                      ({
                        if (foo()) {
                          break;
                          // CHECK:       if.then:
                          // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
                          // CHECK-NEXT:    br label %for.end
                        }
                        Printy("c");
                        // CHECK:       if.end:
                        // CHECK-NEXT:    call void @_ZN6PrintyC1EPKc
                      }),
                      Printy("d")};
                      // CHECK:           call void @_ZN6PrintyC1EPKc
                      // CHECK-NEXT:      call void @_ZN8PrintiesD1Ev
                      // CHECK-NEXT:      br label %for.cond
                }
                Printy("e");
  // CHECK:       for.end:
  // CHECK-NEXT:    call void @_ZN6PrintyC1EPKc
              }),
              Printy("f")};
  // CHECK:         call void @_ZN6PrintyC1EPKc
  // CHECK-NEXT:    call void @_ZN8PrintiesD1Ev
}

void goto_in_stmt_expr() {
  // Verify that:
  //  - correct branch fixups for deactivated normal cleanups are generated correctly.

  // CHECK-LABEL: define dso_local void @_Z17goto_in_stmt_exprv()
  // CHECK: [[CLEANUP_DEST_SLOT:%cleanup.dest.slot.*]] = alloca i32, align 4
  {
    Printies p1{Printy("a"), // CHECK: call void @_ZN6PrintyC1EPKc
                ({
                  {
                    Printies p2{Printy("b"),
                                // CHECK: call void @_ZN6PrintyC1EPKc
                                ({
                                  if (foo() == 1) {
                                    goto in;
                                    // CHECK:       if.then:
                                    // CHECK-NEXT:    store i32 2, ptr [[CLEANUP_DEST_SLOT]], align 4
                                    // CHECK-NEXT:    br label %[[CLEANUP1:.+]]
                                  }
                                  if (foo() == 2) {
                                    goto out;
                                    // CHECK:       if.then{{.*}}:
                                    // CHECK-NEXT:    store i32 3, ptr [[CLEANUP_DEST_SLOT]], align 4
                                    // CHECK-NEXT:    br label %[[CLEANUP1]]
                                  }
                                  Printy("c");
                                  // CHECK:       if.end{{.*}}:
                                  // CHECK-NEXT:    call void @_ZN6PrintyC1EPKc
                                }),
                                Printy("d")};
                                // CHECK:           call void @_ZN6PrintyC1EPKc
                                // CHECK-NEXT:      call void @_ZN8PrintiesD1Ev
                                // CHECK-NEXT:      br label %in

                  }
                in:
                  Printy("e");
                // CHECK:       in:                                               ; preds = %if.end{{.*}}, %[[CLEANUP1]]
                // CHECK-NEXT:    call void @_ZN6PrintyC1EPKc
                }),
                Printy("f")};
                // CHECK:         call void @_ZN6PrintyC1EPKc
                // CHECK-NEXT:    call void @_ZN8PrintiesD1Ev
                // CHECK-NEXT:    br label %out
  }
out:
  return;
  // CHECK:       out:
  // CHECK-NEXT:    ret void

  // CHECK:       [[CLEANUP1]]:                                          ; preds = %if.then{{.*}}, %if.then
  // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
  // CHECK-NEXT:    %cleanup.dest = load i32, ptr [[CLEANUP_DEST_SLOT]], align 4
  // CHECK-NEXT:    switch i32 %cleanup.dest, label %[[CLEANUP2:.+]] [
  // CHECK-NEXT:      i32 2, label %in
  // CHECK-NEXT:    ]

  // CHECK:       [[CLEANUP2]]:                                         ; preds = %[[CLEANUP1]]
  // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
  // CHECK-NEXT:    %cleanup.dest{{.*}} = load i32, ptr [[CLEANUP_DEST_SLOT]], align 4
  // CHECK-NEXT:    switch i32 %cleanup.dest{{.*}}, label %unreachable [
  // CHECK-NEXT:      i32 3, label %out
  // CHECK-NEXT:    ]
}

void ArrayInit() {
  // Printy arr[4] = {ctorA, ctorB, stmt-exprC, stmt-exprD};
  // Verify that:
  //  - We do the necessary stores for array cleanups (endOfInit and last constructed element).
  //  - We update the array init element correctly for ctorA, ctorB and stmt-exprC.
  //  - stmt-exprC and stmt-exprD share the array body dtor code (see %cleanup).

  // CHECK-LABEL: define dso_local void @_Z9ArrayInitv()
  // CHECK: %arrayinit.endOfInit = alloca ptr, align 8
  // CHECK: %cleanup.dest.slot = alloca i32, align 4
  // CHECK: %arrayinit.begin = getelementptr inbounds [4 x %struct.Printy], ptr %arr, i64 0, i64 0
  // CHECK: store ptr %arrayinit.begin, ptr %arrayinit.endOfInit, align 8
  Printy arr[4] = {
    Printy("a"),
    // CHECK: call void @_ZN6PrintyC1EPKc(ptr noundef nonnull align 8 dereferenceable(8) %arrayinit.begin, ptr noundef @.str)
    // CHECK: [[ARRAYINIT_ELEMENT1:%.+]] = getelementptr inbounds %struct.Printy, ptr %arrayinit.begin, i64 1
    // CHECK: store ptr [[ARRAYINIT_ELEMENT1]], ptr %arrayinit.endOfInit, align 8
    Printy("b"),
    // CHECK: call void @_ZN6PrintyC1EPKc(ptr noundef nonnull align 8 dereferenceable(8) [[ARRAYINIT_ELEMENT1]], ptr noundef @.str.1)
    // CHECK: [[ARRAYINIT_ELEMENT2:%.+]] = getelementptr inbounds %struct.Printy, ptr [[ARRAYINIT_ELEMENT1]], i64 1
    // CHECK: store ptr [[ARRAYINIT_ELEMENT2]], ptr %arrayinit.endOfInit, align 8
    ({
    // CHECK: br i1 {{.*}}, label %if.then, label %if.end
      if (foo()) {
        return;
      // CHECK:       if.then:
      // CHECK-NEXT:    store i32 1, ptr %cleanup.dest.slot, align 4
      // CHECK-NEXT:    br label %cleanup
      }
      // CHECK:       if.end:
      Printy("c");
      // CHECK-NEXT:    call void @_ZN6PrintyC1EPKc
      // CHECK-NEXT:    %arrayinit.element2 = getelementptr inbounds %struct.Printy, ptr %arrayinit.element1, i64 1
      // CHECK-NEXT:    store ptr %arrayinit.element2, ptr %arrayinit.endOfInit, align 8
    }),
    ({
    // CHECK: br i1 {{%.+}} label %[[IF_THEN2:.+]], label %[[IF_END2:.+]]
      if (foo()) {
        return;
      // CHECK:       [[IF_THEN2]]:
      // CHECK-NEXT:    store i32 1, ptr %cleanup.dest.slot, align 4
      // CHECK-NEXT:    br label %cleanup
      }
      // CHECK:       [[IF_END2]]:
      Printy("d");
      // CHECK-NEXT:    call void @_ZN6PrintyC1EPKc
      // CHECK-NEXT:    %array.begin = getelementptr inbounds [4 x %struct.Printy], ptr %arr, i32 0, i32 0
      // CHECK-NEXT:    %0 = getelementptr inbounds %struct.Printy, ptr %array.begin, i64 4
      // CHECK-NEXT:    br label %[[ARRAY_DESTROY_BODY1:.+]]
  }),
  };

  // CHECK:       [[ARRAY_DESTROY_BODY1]]:
  // CHECK-NEXT:    %arraydestroy.elementPast{{.*}} = phi ptr [ %0, %[[IF_END2]] ], [ %arraydestroy.element{{.*}}, %[[ARRAY_DESTROY_BODY1]] ]
  // CHECK-NEXT:    %arraydestroy.element{{.*}} = getelementptr inbounds %struct.Printy, ptr %arraydestroy.elementPast{{.*}}, i64 -1
  // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
  // CHECK-NEXT:    %arraydestroy.done{{.*}} = icmp eq ptr %arraydestroy.element{{.*}}, %array.begin
  // CHECK-NEXT:    br i1 %arraydestroy.done{{.*}}, label %[[ARRAY_DESTROY_DONE1:.+]], label %[[ARRAY_DESTROY_BODY1]]

  // CHECK:       [[ARRAY_DESTROY_DONE1]]:
  // CHECK-NEXT:    ret void

  // CHECK:       cleanup:
  // CHECK-NEXT:    %1 = load ptr, ptr %arrayinit.endOfInit, align 8
  // CHECK-NEXT:    %arraydestroy.isempty = icmp eq ptr %arrayinit.begin, %1
  // CHECK-NEXT:    br i1 %arraydestroy.isempty, label %[[ARRAY_DESTROY_DONE2:.+]], label %[[ARRAY_DESTROY_BODY2:.+]]

  // CHECK:       [[ARRAY_DESTROY_BODY2]]:
  // CHECK-NEXT:    %arraydestroy.elementPast = phi ptr [ %1, %cleanup ], [ %arraydestroy.element, %[[ARRAY_DESTROY_BODY2]] ]
  // CHECK-NEXT:    %arraydestroy.element = getelementptr inbounds %struct.Printy, ptr %arraydestroy.elementPast, i64 -1
  // CHECK-NEXT:    call void @_ZN6PrintyD1Ev(ptr noundef nonnull align 8 dereferenceable(8) %arraydestroy.element)
  // CHECK-NEXT:    %arraydestroy.done = icmp eq ptr %arraydestroy.element, %arrayinit.begin
  // CHECK-NEXT:    br i1 %arraydestroy.done, label %[[ARRAY_DESTROY_DONE2]], label %[[ARRAY_DESTROY_BODY2]]

  // CHECK:       [[ARRAY_DESTROY_DONE2]]:
  // CHECK-NEXT:    br label %[[ARRAY_DESTROY_DONE1]]
}

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
    // CHECK:       %arrayinit.begin = getelementptr inbounds [2 x %struct.Printy], ptr %arr.reload.addr, i64 0, i64 0
    // CHECK-NEXT:  %arrayinit.begin.spill.addr = getelementptr inbounds %_Z13ArrayInitCorov.Frame, ptr %0, i32 0, i32 10
    // CHECK-NEXT:  store ptr %arrayinit.begin, ptr %arrayinit.begin.spill.addr, align 8
    // CHECK-NEXT:  store i1 true, ptr %cleanup.isactive.reload.addr, align 1
    // CHECK-NEXT:  store ptr %arrayinit.begin, ptr %arrayinit.endOfInit.reload.addr, align 8
    // CHECK-NEXT:  call void @_ZN6PrintyC1EPKc(ptr noundef nonnull align 8 dereferenceable(8) %arrayinit.begin, ptr noundef @.str)
    // CHECK-NEXT:  %arrayinit.element = getelementptr inbounds %struct.Printy, ptr %arrayinit.begin, i64 1
    // CHECK-NEXT:  %arrayinit.element.spill.addr = getelementptr inbounds %_Z13ArrayInitCorov.Frame, ptr %0, i32 0, i32 11
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
  // CHECK-NEXT:    %arrayinit.element.reload.addr = getelementptr inbounds %_Z13ArrayInitCorov.Frame, ptr %0, i32 0, i32 11
  // CHECK-NEXT:    %arrayinit.element.reload = load ptr, ptr %arrayinit.element.reload.addr, align 8
  // CHECK-NEXT:    call void @_ZN7Awaiter12await_resumeEv
  // CHECK-NEXT:    store i1 false, ptr %cleanup.isactive.reload.addr, align 1
  // CHECK-NEXT:    br label %cleanup{{.*}}.from.await.ready

  // CHECK:       cleanup{{.*}}:                                         ; preds = %cleanup{{.*}}.from.await.ready, %cleanup{{.*}}.from.await.cleanup
  // CHECK:         %cleanup.is_active = load i1, ptr %cleanup.isactive.reload.addr, align 1
  // CHECK-NEXT:    br i1 %cleanup.is_active, label %cleanup.action, label %cleanup.done

  // CHECK:       cleanup.action:
  // CHECK:         %arraydestroy.isempty = icmp eq ptr %arrayinit.begin.reload{{.*}}, %{{.*}}
  // CHECK-NEXT:    br i1 %arraydestroy.isempty, label %arraydestroy.done{{.*}}, label %arraydestroy.body.from.cleanup.action
  // Ignore rest of the array cleanup.
}

void ArraySubobjects() {
  struct S {
    Printy arr1[2];
    Printy arr2[2];
    Printy p;
  };
  // CHECK-LABEL: define dso_local void @_Z15ArraySubobjectsv()
  // CHECK: %arrayinit.endOfInit = alloca ptr, align 8
  S s{{Printy("a"), Printy("b")},
      // CHECK: call void @_ZN6PrintyC1EPKc
      // CHECK: call void @_ZN6PrintyC1EPKc
      {Printy("a"),
      // CHECK: [[ARRAYINIT_BEGIN:%.+]] = getelementptr inbounds [2 x %struct.Printy]
      // CHECK: store ptr [[ARRAYINIT_BEGIN]], ptr %arrayinit.endOfInit, align 8
      // CHECK: call void @_ZN6PrintyC1EPKc
      // CHECK: [[ARRAYINIT_ELEMENT:%.+]] = getelementptr inbounds %struct.Printy
      // CHECK: store ptr [[ARRAYINIT_ELEMENT]], ptr %arrayinit.endOfInit, align 8
      ({
         if (foo()) {
           return;
           // CHECK:      if.then:
           // CHECK-NEXT:   [[V0:%.+]] = load ptr, ptr %arrayinit.endOfInit, align 8
           // CHECK-NEXT:   %arraydestroy.isempty = icmp eq ptr [[ARRAYINIT_BEGIN]], [[V0]]
           // CHECK-NEXT:   br i1 %arraydestroy.isempty, label %[[ARRAY_DESTROY_DONE:.+]], label %[[ARRAY_DESTROY_BODY:.+]]
         }
         Printy("b");
       })
      },
      Printy("c")
      // CHECK:       if.end:
      // CHECK-NEXT:    call void @_ZN6PrintyC1EPKc
      // CHECK:         call void @_ZN6PrintyC1EPKc
      // CHECK-NEXT:    call void @_ZZ15ArraySubobjectsvEN1SD1Ev
      // CHECK-NEXT:    br label %return
    };
    // CHECK:       return:
    // CHECK-NEXT:    ret void

    // CHECK:       [[ARRAY_DESTROY_BODY]]:
    // CHECK-NEXT:    %arraydestroy.elementPast = phi ptr [ %0, %if.then ], [ %arraydestroy.element, %[[ARRAY_DESTROY_BODY]] ]
    // CHECK-NEXT:    %arraydestroy.element = getelementptr inbounds %struct.Printy, ptr %arraydestroy.elementPast, i64 -1
    // CHECK-NEXT:    call void @_ZN6PrintyD1Ev(ptr noundef nonnull align 8 dereferenceable(8) %arraydestroy.element)
    // CHECK-NEXT:    %arraydestroy.done = icmp eq ptr %arraydestroy.element, [[ARRAYINIT_BEGIN]]
    // CHECK-NEXT:    br i1 %arraydestroy.done, label %[[ARRAY_DESTROY_DONE]], label %[[ARRAY_DESTROY_BODY]]

    // CHECK:       [[ARRAY_DESTROY_DONE]]
    // CHECK-NEXT:    [[ARRAY_BEGIN:%.+]] = getelementptr inbounds [2 x %struct.Printy], ptr %arr1, i32 0, i32 0
    // CHECK-NEXT:    [[V1:%.+]] = getelementptr inbounds %struct.Printy, ptr [[ARRAY_BEGIN]], i64 2
    // CHECK-NEXT:    br label %[[ARRAY_DESTROY_BODY2:.+]]

    // CHECK:       [[ARRAY_DESTROY_BODY2]]:
    // CHECK-NEXT:    %arraydestroy.elementPast5 = phi ptr [ %1, %[[ARRAY_DESTROY_DONE]] ], [ %arraydestroy.element6, %[[ARRAY_DESTROY_BODY2]] ]
    // CHECK-NEXT:    %arraydestroy.element6 = getelementptr inbounds %struct.Printy, ptr %arraydestroy.elementPast5, i64 -1
    // CHECK-NEXT:    call void @_ZN6PrintyD1Ev(ptr noundef nonnull align 8 dereferenceable(8) %arraydestroy.element6)
    // CHECK-NEXT:    %arraydestroy.done7 = icmp eq ptr %arraydestroy.element6, [[ARRAY_BEGIN]]
    // CHECK-NEXT:    br i1 %arraydestroy.done7, label %[[ARRAY_DESTROY_DONE2:.+]], label %[[ARRAY_DESTROY_BODY2]]


    // CHECK:     [[ARRAY_DESTROY_DONE2]]:
    // CHECK-NEXT:  br label %return
}

void LambdaInit() {
  // CHECK-LABEL: define dso_local void @_Z10LambdaInitv()
  auto S = [a = Printy("a"), b = ({
                               if (foo()) {
                                 return;
                                 // CHECK:       if.then:
                                 // CHECK-NEXT:    call void @_ZN6PrintyD1Ev
                                 // CHECK-NEXT:    br label %return
                               }
                               Printy("b");
                             })]() { return a; };
}

void LifetimeExtended() {
  // CHECK-LABEL: define dso_local void @_Z16LifetimeExtendedv
  struct PrintyRefBind {
    const Printy &a;
    const Printy &b;
  };
  PrintyRefBind ps = {Printy("a"), ({
                        if (foo()) {
                          return;
                          // CHECK: if.then:
                          // CHECK-NEXT: call void @_ZN6PrintyD1Ev
                          // CHECK-NEXT: br label %return
                        }
                        Printy("b");
                      })};
}

void NewArrayInit() {
  // CHECK-LABEL: define dso_local void @_Z12NewArrayInitv()
  // CHECK: %array.init.end = alloca ptr, align 8
  // CHECK: store ptr %0, ptr %array.init.end, align 8
  Printy *array = new Printy[3]{
    "a",
    // CHECK: call void @_ZN6PrintyC1EPKc
    // CHECK: store ptr %array.exp.next, ptr %array.init.end, align 8
    "b", 
    // CHECK: call void @_ZN6PrintyC1EPKc
    // CHECK: store ptr %array.exp.next1, ptr %array.init.end, align 8
    ({
        if (foo()) {
          return;
          // CHECK: if.then:
          // CHECK:   br i1 %arraydestroy.isempty, label %arraydestroy.done{{.*}}, label %arraydestroy.body
        }
        "b";
        // CHECK: if.end:
        // CHECK:   call void @_ZN6PrintyC1EPKc
    })};
  // CHECK:       arraydestroy.body:
  // CHECK-NEXT:    %arraydestroy.elementPast = phi ptr [ %{{.*}}, %if.then ], [ %arraydestroy.element, %arraydestroy.body ]
  // CHECK-NEXT:    %arraydestroy.element = getelementptr inbounds %struct.Printy, ptr %arraydestroy.elementPast, i64 -1
  // CHECK-NEXT:    call void @_ZN6PrintyD1Ev(ptr noundef nonnull align 8 dereferenceable(8) %arraydestroy.element)
  // CHECK-NEXT:    %arraydestroy.done = icmp eq ptr %arraydestroy.element, %0
  // CHECK-NEXT:    br i1 %arraydestroy.done, label %arraydestroy.done{{.*}}, label %arraydestroy.body

  // CHECK:       arraydestroy.done{{.*}}:                               ; preds = %arraydestroy.body, %if.then
  // CHECK-NEXT:    br label %return
}

void ArrayInitWithContinue() {
  // CHECK-LABEL: @_Z21ArrayInitWithContinuev
  // Verify that we start to emit the array destructor.
  // CHECK: %arrayinit.endOfInit = alloca ptr, align 8
  for (int i = 0; i < 1; ++i) {
    Printy arr[2] = {"a", ({
                       if (foo()) {
                         continue;
                       }
                       "b";
                     })};
  }
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
