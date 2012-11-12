// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - -fcxx-exceptions -fexceptions | FileCheck %s

// Test IR generation for partial destruction of aggregates.

void opaque();

// Initializer lists.
namespace test0 {
  struct A { A(int); A(); ~A(); void *v; };
  void test() {
    A as[10] = { 5, 7 };
    opaque();
  }
  // CHECK:    define void @_ZN5test04testEv()
  // CHECK:      [[AS:%.*]] = alloca [10 x [[A:%.*]]], align
  // CHECK-NEXT: [[ENDVAR:%.*]] = alloca [[A]]*
  // CHECK-NEXT: [[EXN:%.*]] = alloca i8*
  // CHECK-NEXT: [[SEL:%.*]] = alloca i32

  // Initialize.
  // CHECK-NEXT: [[E_BEGIN:%.*]] = getelementptr inbounds [10 x [[A]]]* [[AS]], i64 0, i64 0
  // CHECK-NEXT: store [[A]]* [[E_BEGIN]], [[A]]** [[ENDVAR]]
  // CHECK-NEXT: invoke void @_ZN5test01AC1Ei([[A]]* [[E_BEGIN]], i32 5)
  // CHECK:      [[E1:%.*]] = getelementptr inbounds [[A]]* [[E_BEGIN]], i64 1
  // CHECK-NEXT: store [[A]]* [[E1]], [[A]]** [[ENDVAR]]
  // CHECK-NEXT: invoke void @_ZN5test01AC1Ei([[A]]* [[E1]], i32 7)
  // CHECK:      [[E2:%.*]] = getelementptr inbounds [[A]]* [[E1]], i64 1
  // CHECK-NEXT: store [[A]]* [[E2]], [[A]]** [[ENDVAR]]
  // CHECK-NEXT: [[E_END:%.*]] = getelementptr inbounds [[A]]* [[E_BEGIN]], i64 10
  // CHECK-NEXT: br label
  // CHECK:      [[E_CUR:%.*]] = phi [[A]]* [ [[E2]], {{%.*}} ], [ [[E_NEXT:%.*]], {{%.*}} ]
  // CHECK-NEXT: invoke void @_ZN5test01AC1Ev([[A]]* [[E_CUR]])
  // CHECK:      [[E_NEXT]] = getelementptr inbounds [[A]]* [[E_CUR]], i64 1
  // CHECK-NEXT: store [[A]]* [[E_NEXT]], [[A]]** [[ENDVAR]]
  // CHECK-NEXT: [[T0:%.*]] = icmp eq [[A]]* [[E_NEXT]], [[E_END]]
  // CHECK-NEXT: br i1 [[T0]],

  // Run.
  // CHECK:      invoke void @_Z6opaquev()

  // Normal destroy.
  // CHECK:      [[ED_BEGIN:%.*]] = getelementptr inbounds [10 x [[A]]]* [[AS]], i32 0, i32 0
  // CHECK-NEXT: [[ED_END:%.*]] = getelementptr inbounds [[A]]* [[ED_BEGIN]], i64 10
  // CHECK-NEXT: br label
  // CHECK:      [[ED_AFTER:%.*]] = phi [[A]]* [ [[ED_END]], {{%.*}} ], [ [[ED_CUR:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[ED_CUR]] = getelementptr inbounds [[A]]* [[ED_AFTER]], i64 -1
  // CHECK-NEXT: invoke void @_ZN5test01AD1Ev([[A]]* [[ED_CUR]])
  // CHECK:      [[T0:%.*]] = icmp eq [[A]]* [[ED_CUR]], [[ED_BEGIN]]
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      ret void

  // Partial destroy for initialization.
  // CHECK:      landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
  // CHECK-NEXT:   cleanup
  // CHECK:      [[PARTIAL_END:%.*]] = load [[A]]** [[ENDVAR]]
  // CHECK-NEXT: [[T0:%.*]] = icmp eq [[A]]* [[E_BEGIN]], [[PARTIAL_END]]
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[E_AFTER:%.*]] = phi [[A]]* [ [[PARTIAL_END]], {{%.*}} ], [ [[E_CUR:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[E_CUR]] = getelementptr inbounds [[A]]* [[E_AFTER]], i64 -1
  // CHECK-NEXT: invoke void @_ZN5test01AD1Ev([[A]]* [[E_CUR]])
  // CHECK:      [[T0:%.*]] = icmp eq [[A]]* [[E_CUR]], [[E_BEGIN]]
  // CHECK-NEXT: br i1 [[T0]],

  // Primary EH destructor.
  // CHECK:      landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
  // CHECK-NEXT:   cleanup
  // CHECK:      [[E0:%.*]] = getelementptr inbounds [10 x [[A]]]* [[AS]], i32 0, i32 0
  // CHECK-NEXT: [[E_END:%.*]] = getelementptr inbounds [[A]]* [[E0]], i64 10
  // CHECK-NEXT: br label

  // Partial destructor for primary normal destructor.
  // FIXME: There's some really bad block ordering here which causes
  // the partial destroy for the primary normal destructor to fall
  // within the primary EH destructor.
  // CHECK:      landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
  // CHECK-NEXT:   cleanup
  // CHECK:      [[T0:%.*]] = icmp eq [[A]]* [[ED_BEGIN]], [[ED_CUR]]
  // CHECK-NEXT: br i1 [[T0]]
  // CHECK:      [[EDD_AFTER:%.*]] = phi [[A]]* [ [[ED_CUR]], {{%.*}} ], [ [[EDD_CUR:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[EDD_CUR]] = getelementptr inbounds [[A]]* [[EDD_AFTER]], i64 -1
  // CHECK-NEXT: invoke void @_ZN5test01AD1Ev([[A]]* [[EDD_CUR]])
  // CHECK:      [[T0:%.*]] = icmp eq [[A]]* [[EDD_CUR]], [[ED_BEGIN]]
  // CHECK-NEXT: br i1 [[T0]]

  // Back to the primary EH destructor.
  // CHECK:      [[E_AFTER:%.*]] = phi [[A]]* [ [[E_END]], {{%.*}} ], [ [[E_CUR:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[E_CUR]] = getelementptr inbounds [[A]]* [[E_AFTER]], i64 -1
  // CHECK-NEXT: invoke void @_ZN5test01AD1Ev([[A]]* [[E_CUR]])
  // CHECK:      [[T0:%.*]] = icmp eq [[A]]* [[E_CUR]], [[E0]]
  // CHECK-NEXT: br i1 [[T0]],

}

namespace test1 {
  struct A { A(); A(int); ~A(); };
  struct B { A x, y, z; int w; };

  void test() {
    B v = { 5, 6, 7, 8 };
  }
  // CHECK:    define void @_ZN5test14testEv()
  // CHECK:      [[V:%.*]] = alloca [[B:%.*]], align 4
  // CHECK-NEXT: alloca i8*
  // CHECK-NEXT: alloca i32
  // CHECK-NEXT: [[X:%.*]] = getelementptr inbounds [[B]]* [[V]], i32 0, i32 0
  // CHECK-NEXT: call void @_ZN5test11AC1Ei([[A:%.*]]* [[X]], i32 5)
  // CHECK-NEXT: [[Y:%.*]] = getelementptr inbounds [[B]]* [[V]], i32 0, i32 1
  // CHECK-NEXT: invoke void @_ZN5test11AC1Ei([[A]]* [[Y]], i32 6)
  // CHECK:      [[Z:%.*]] = getelementptr inbounds [[B]]* [[V]], i32 0, i32 2
  // CHECK-NEXT: invoke void @_ZN5test11AC1Ei([[A]]* [[Z]], i32 7)
  // CHECK:      [[W:%.*]] = getelementptr inbounds [[B]]* [[V]], i32 0, i32 3
  // CHECK-NEXT: store i32 8, i32* [[W]], align 4
  // CHECK-NEXT: call void @_ZN5test11BD1Ev([[B]]* [[V]])
  // CHECK-NEXT: ret void

  // FIXME: again, the block ordering is pretty bad here
  // CHECK:      landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
  // CHECK-NEXT:   cleanup
  // CHECK:      landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
  // CHECK-NEXT:   cleanup
  // CHECK:      invoke void @_ZN5test11AD1Ev([[A]]* [[Y]])
  // CHECK:      invoke void @_ZN5test11AD1Ev([[A]]* [[X]])
}

namespace test2 {
  struct A { A(); ~A(); };

  void test() {
    A v[4][7];

    // CHECK:    define void @_ZN5test24testEv()
    // CHECK:      [[V:%.*]] = alloca [4 x [7 x [[A:%.*]]]], align 1
    // CHECK-NEXT: alloca i8*
    // CHECK-NEXT: alloca i32

    // Main initialization loop.
    // CHECK-NEXT: [[BEGIN:%.*]] = getelementptr inbounds [4 x [7 x [[A]]]]* [[V]], i32 0, i32 0, i32 0
    // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds [[A]]* [[BEGIN]], i64 28
    // CHECK-NEXT: br label
    // CHECK:      [[CUR:%.*]] = phi [[A]]* [ [[BEGIN]], {{%.*}} ], [ [[NEXT:%.*]], {{%.*}} ]
    // CHECK-NEXT: invoke void @_ZN5test21AC1Ev([[A]]* [[CUR]])
    // CHECK:      [[NEXT:%.*]] = getelementptr inbounds [[A]]* [[CUR]], i64 1
    // CHECK-NEXT: [[DONE:%.*]] = icmp eq [[A]]* [[NEXT]], [[END]]
    // CHECK-NEXT: br i1 [[DONE]],

    // Partial destruction landing pad.
    // CHECK:      landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
    // CHECK-NEXT:   cleanup
    // CHECK:      [[EMPTY:%.*]] = icmp eq [[A]]* [[BEGIN]], [[CUR]]
    // CHECK-NEXT: br i1 [[EMPTY]],
    // CHECK:      [[PAST:%.*]] = phi [[A]]* [ [[CUR]], {{%.*}} ], [ [[DEL:%.*]], {{%.*}} ]
    // CHECK-NEXT: [[DEL]] = getelementptr inbounds [[A]]* [[PAST]], i64 -1
    // CHECK-NEXT: invoke void @_ZN5test21AD1Ev([[A]]* [[DEL]])
    // CHECK:      [[T0:%.*]] = icmp eq [[A]]* [[DEL]], [[BEGIN]]
    // CHECK-NEXT: br i1 [[T0]],
  }

}

// PR10351
namespace test3 {
  struct A { A(); ~A(); void *p; };
  struct B {
    B() {}
    A a;
  };

  B *test() {
    return new B[10];
    // invoke void @_ZN5test31BD1Ev(
  }
}
