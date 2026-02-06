// RUN: %clang_cc1 --std=c++20 -fexceptions -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck -check-prefixes=EH %s
// RUN: %clang_cc1 --std=c++20 -triple x86_64-linux-gnu -emit-llvm %s -o - | FileCheck -check-prefixes=NOEH,CHECK %s

struct Printy {
  Printy(const char *name) : name(name) {}
  ~Printy() {}
  const char *name;
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
  // CHECK: store ptr %arr, ptr %arrayinit.endOfInit, align 8
  Printy arr[4] = {
    Printy("a"),
    // CHECK: call void @_ZN6PrintyC1EPKc(ptr noundef nonnull align 8 dereferenceable(8) %arr, ptr noundef @.str)
    // CHECK: [[ARRAYINIT_ELEMENT1:%.+]] = getelementptr inbounds %struct.Printy, ptr %arr, i64 1
    // CHECK: store ptr [[ARRAYINIT_ELEMENT1]], ptr %arrayinit.endOfInit, align 8
    Printy("b"),
    // CHECK: call void @_ZN6PrintyC1EPKc(ptr noundef nonnull align 8 dereferenceable(8) [[ARRAYINIT_ELEMENT1]], ptr noundef @.str.1)
    // CHECK: [[ARRAYINIT_ELEMENT2:%.+]] = getelementptr inbounds %struct.Printy, ptr %arr, i64 2
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
      // CHECK-NEXT:    %arrayinit.element2 = getelementptr inbounds %struct.Printy, ptr %arr, i64 3
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
  // CHECK-NEXT:    %arraydestroy.isempty = icmp eq ptr %arr, %1
  // CHECK-NEXT:    br i1 %arraydestroy.isempty, label %[[ARRAY_DESTROY_DONE2:.+]], label %[[ARRAY_DESTROY_BODY2:.+]]

  // CHECK:       [[ARRAY_DESTROY_BODY2]]:
  // CHECK-NEXT:    %arraydestroy.elementPast = phi ptr [ %1, %cleanup ], [ %arraydestroy.element, %[[ARRAY_DESTROY_BODY2]] ]
  // CHECK-NEXT:    %arraydestroy.element = getelementptr inbounds %struct.Printy, ptr %arraydestroy.elementPast, i64 -1
  // CHECK-NEXT:    call void @_ZN6PrintyD1Ev(ptr noundef nonnull align 8 dereferenceable(8) %arraydestroy.element)
  // CHECK-NEXT:    %arraydestroy.done = icmp eq ptr %arraydestroy.element, %arr
  // CHECK-NEXT:    br i1 %arraydestroy.done, label %[[ARRAY_DESTROY_DONE2]], label %[[ARRAY_DESTROY_BODY2]]

  // CHECK:       [[ARRAY_DESTROY_DONE2]]:
  // CHECK-NEXT:    br label %[[ARRAY_DESTROY_DONE1]]
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
      // CHECK: store ptr %arr2, ptr %arrayinit.endOfInit, align 8
      // CHECK: call void @_ZN6PrintyC1EPKc
      // CHECK: [[ARRAYINIT_ELEMENT:%.+]] = getelementptr inbounds %struct.Printy
      // CHECK: store ptr [[ARRAYINIT_ELEMENT]], ptr %arrayinit.endOfInit, align 8
      ({
         if (foo()) {
           return;
           // CHECK:      if.then:
           // CHECK-NEXT:   [[V0:%.+]] = load ptr, ptr %arrayinit.endOfInit, align 8
           // CHECK-NEXT:   %arraydestroy.isempty = icmp eq ptr %arr2, [[V0]]
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
    // CHECK-NEXT:    %arraydestroy.done = icmp eq ptr %arraydestroy.element, %arr2
    // CHECK-NEXT:    br i1 %arraydestroy.done, label %[[ARRAY_DESTROY_DONE]], label %[[ARRAY_DESTROY_BODY]]

    // CHECK:       [[ARRAY_DESTROY_DONE]]
    // CHECK-NEXT:    [[ARRAY_BEGIN:%.+]] = getelementptr inbounds [2 x %struct.Printy], ptr %arr1, i32 0, i32 0
    // CHECK-NEXT:    [[V1:%.+]] = getelementptr inbounds %struct.Printy, ptr [[ARRAY_BEGIN]], i64 2
    // CHECK-NEXT:    br label %[[ARRAY_DESTROY_BODY2:.+]]

    // CHECK:       [[ARRAY_DESTROY_BODY2]]:
    // CHECK-NEXT:    %arraydestroy.elementPast4 = phi ptr [ %1, %[[ARRAY_DESTROY_DONE]] ], [ %arraydestroy.element5, %[[ARRAY_DESTROY_BODY2]] ]
    // CHECK-NEXT:    %arraydestroy.element5 = getelementptr inbounds %struct.Printy, ptr %arraydestroy.elementPast4, i64 -1
    // CHECK-NEXT:    call void @_ZN6PrintyD1Ev(ptr noundef nonnull align 8 dereferenceable(8) %arraydestroy.element5)
    // CHECK-NEXT:    %arraydestroy.done6 = icmp eq ptr %arraydestroy.element5, [[ARRAY_BEGIN]]
    // CHECK-NEXT:    br i1 %arraydestroy.done6, label %[[ARRAY_DESTROY_DONE2:.+]], label %[[ARRAY_DESTROY_BODY2]]


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

struct PrintyRefBind {
  const Printy &a;
  const Printy &b;
};

struct Temp {
  Temp();
  ~Temp();
};
Temp CreateTemp();
Printy CreatePrinty();
Printy CreatePrinty(const Temp&);

void LifetimeExtended() {
  // CHECK-LABEL: define dso_local void @_Z16LifetimeExtendedv
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

void ConditionalLifetimeExtended() {
  // CHECK-LABEL: @_Z27ConditionalLifetimeExtendedv()

  // Verify that we create two cleanup flags.
  //  1. First for the cleanup which is deactivated after full expression.
  //  2. Second for the life-ext cleanup which is activated if the branch is taken.

  // Note: We use `CreateTemp()` to ensure that life-ext destroy cleanup is not at
  // the top of EHStack on deactivation. This ensures using active flags.

  Printy* p1 = nullptr;
  // CHECK:       store i1 false, ptr [[BRANCH1_DEFERRED:%cleanup.cond]], align 1
  // CHECK-NEXT:  store i1 false, ptr [[BRANCH1_LIFEEXT:%cleanup.cond.*]], align 1
  PrintyRefBind ps = {
      p1 != nullptr ? static_cast<const Printy&>(CreatePrinty())
      // CHECK:       cond.true:
      // CHECK-NEXT:    call void @_Z12CreatePrintyv
      // CHECK-NEXT:    store i1 true, ptr [[BRANCH1_DEFERRED]], align 1
      // CHECK-NEXT:    store i1 true, ptr [[BRANCH1_LIFEEXT]], align 1
      // CHECK-NEXT:    br label %{{.*}}
      : foo() ? static_cast<const Printy&>(CreatePrinty(CreateTemp()))
              : *p1,
      ({
        if (foo()) return;
        Printy("c");
        // CHECK:       if.end:
        // CHECK-NEXT:    call void @_ZN6PrintyC1EPKc
        // CHECK-NEXT:    store ptr
      })};
      // CHECK-NEXT:      store i1 false, ptr [[BRANCH1_DEFERRED]], align 1
      // CHECK-NEXT:      store i32 0, ptr %cleanup.dest.slot, align 4
      // CHECK-NEXT:      br label %cleanup

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

void DestroyInConditionalCleanup() {
  // EH-LABEL: DestroyInConditionalCleanupv()
  // NOEH-LABEL: DestroyInConditionalCleanupv()
  struct A {
    A() {}
    ~A() {}
  };

  struct Value {
    Value(A) {}
    ~Value() {}
  };

  struct V2 {
    Value K;
    Value V;
  };
  // Verify we use conditional cleanups.
  (void)(foo() ? V2{A(), A()} : V2{A(), A()});
  // NOEH:   cond.true:
  // NOEH:      call void @_ZZ27DestroyInConditionalCleanupvEN1AC1Ev
  // NOEH:      store ptr %{{.*}}, ptr %cond-cleanup.save

  // EH:   cond.true:
  // EH:        invoke void @_ZZ27DestroyInConditionalCleanupvEN1AC1Ev
  // EH:        store ptr %{{.*}}, ptr %cond-cleanup.save
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

struct [[clang::trivial_abi]] HasTrivialABI {
  HasTrivialABI();
  ~HasTrivialABI();
};
void AcceptTrivialABI(HasTrivialABI, int);
void TrivialABI() {
  // CHECK-LABEL: define dso_local void @_Z10TrivialABIv()
  AcceptTrivialABI(HasTrivialABI(), ({
                     if (foo()) return;
                     // CHECK:      if.then:
                     // CHECK-NEXT:   call void @_ZN13HasTrivialABID1Ev
                     // CHECK-NEXT:   br label %return
                     0;
                   }));
}

namespace CleanupFlag {
struct A {
  A() {}
  ~A() {}
};

struct B {
  B(const A&) {}
  B() {}
  ~B() {}
};

struct S {
  A a;
  B b;
};

int AcceptS(S s);

void Accept2(int x, int y);

void InactiveNormalCleanup() {
  // CHECK-LABEL: define {{.*}}InactiveNormalCleanupEv()
  
  // The first A{} below is an inactive normal cleanup which
  // is not popped from EHStack on deactivation. This needs an
  // "active" cleanup flag.

  // CHECK: [[ACTIVE:%cleanup.isactive.*]] = alloca i1, align 1
  // CHECK: call void [[A_CTOR:@.*AC1Ev]]
  // CHECK: store i1 true, ptr [[ACTIVE]], align 1
  // CHECK: call void [[A_CTOR]]
  // CHECK: call void [[B_CTOR:@.*BC1ERKNS_1AE]]
  // CHECK: store i1 false, ptr [[ACTIVE]], align 1
  // CHECK: call noundef i32 [[ACCEPTS:@.*AcceptSENS_1SE]]
  Accept2(AcceptS({.a = A{}, .b = A{}}), ({
            if (foo()) return;
            // CHECK: if.then:
            // CHECK:   br label %cleanup
            0;
            // CHECK: if.end:
            // CHECK:   call void [[ACCEPT2:@.*Accept2Eii]]
            // CHECK:   br label %cleanup
          }));
  // CHECK: cleanup:
  // CHECK:   call void [[S_DTOR:@.*SD1Ev]]
  // CHECK:   call void [[A_DTOR:@.*AD1Ev]]
  // CHECK:   %cleanup.is_active = load i1, ptr [[ACTIVE]]
  // CHECK:   br i1 %cleanup.is_active, label %cleanup.action, label %cleanup.done

  // CHECK: cleanup.action:
  // CHECK:   call void [[A_DTOR]]

  // The "active" cleanup flag is not required for unused cleanups.
  Accept2(AcceptS({.a = A{}, .b = A{}}), 0);
  // CHECK: cleanup.cont:
  // CHECK:   call void [[A_CTOR]]
  // CHECK-NOT: store i1 true
  // CHECK:   call void [[A_CTOR]]
  // CHECK:   call void [[B_CTOR]]
  // CHECK-NOT: store i1 false
  // CHECK:   call noundef i32 [[ACCEPTS]]
  // CHECK:   call void [[ACCEPT2]]
  // CHECK:   call void [[S_DTOR]]
  // CHECK:   call void [[A_DTOR]]
  // CHECK:   br label %return
}
}  // namespace CleanupFlag
