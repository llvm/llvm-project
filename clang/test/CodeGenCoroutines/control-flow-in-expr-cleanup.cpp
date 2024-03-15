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

struct PrintiesCopy {
  Printy a;
  Printy b;
  Printy c;
};

void ParenInit() {
  // CHECK: define dso_local void @_Z9ParenInitv()
  // CHECK: [[CLEANUP_DEST:%.+]] = alloca i32, align 4
  PrintiesCopy ps(Printy("a"), 
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
                    // CHECK-NEXT:  call void @_ZN12PrintiesCopyD1Ev
                    // CHECK-NEXT:  br label %return
                  }));
  // CHECK:     cleanup:
  // CHECK-NEXT:  call void @_ZN6PrintyD1Ev
  // CHECK-NEXT:  br label %return
}

coroutine ParenInitCoro() {
  // CHECK: define dso_local void @_Z13ParenInitCorov
  // CHECK: [[ACTIVE1:%.+]] = alloca i1, align 1
  // CHECK: [[ACTIVE2:%.+]] = alloca i1, align 1
  PrintiesCopy ps(Printy("a"), Printy("b"),
    // CHECK:       call void @_ZN6PrintyC1EPKc
    // CHECK-NEXT:  store i1 true, ptr [[ACTIVE1]].reload.addr, align 1
    // CHECK-NEXT:  store i1 true, ptr [[ACTIVE2]].reload.addr, align 1
    // CHECK:       call void @_ZN6PrintyC1EPKc
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

// TODO: Add more assertions after preliminary review.
// struct S {
//   Printy arr1[2];
//   Printy arr2[2];
//   Printy p;
// };

// void ArraySubobjects() {
//   S s{{Printy("a"), Printy("b")},
//       {Printy("a"), ({
//          if (foo() == 1) {
//            return;
//          }
//          Printy("b");
//        })},
//       ({
//         if (foo() == 2) {
//           return;
//         }
//         Printy("b");
//       })};
// }

// coroutine ArraySubobjectsCoro() {
//   S s{{Printy("a"), Printy("b")},
//       {Printy("a"), co_await Awaiter{}},
//       co_await Awaiter{}};
// }

// struct A {
//   Printy a;
// };
// struct B : A {
//   Printy b;
//   Printy c;
// };

// void BaseClassCtors() {
//   auto S = B({Printy("a")}, Printy("b"), ({
//                return;
//                Printy("c");
//              }));
// }

// coroutine BaseClassCtorsCoro() {
//   auto S = B({Printy("a")}, Printy("b"), co_await Awaiter{});
// }

// void LambdaInit() {
//   auto S = [a = Printy("a"), b = ({
//                                return;
//                                Printy("b");
//                              })]() { return a; };
// }

// coroutine LambdaInitCoro() {
//   auto S = [a = Printy("a"), b = co_await Awaiter{}]() { return a; };
// }
