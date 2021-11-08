// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - -fcxx-exceptions -fexceptions -std=c++03 | FileCheck %s -check-prefixes=CHECK,CHECKv03
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - -fcxx-exceptions -fexceptions -std=c++11 | FileCheck %s -check-prefixes=CHECK,CHECKv11

// Test IR generation for partial destruction of aggregates.

void opaque();

// Initializer lists.
namespace test0 {
  struct A { A(int); A(); ~A(); void *v; };
  void test() {
    A as[10] = { 5, 7 };
    opaque();
  }
  // CHECK-LABEL:    define{{.*}} void @_ZN5test04testEv()
  // CHECK-SAME: personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
  // CHECK:      [[AS:%.*]] = alloca [10 x [[A:%.*]]], align
  // CHECK-NEXT: [[ENDVAR:%.*]] = alloca [[A]]*
  // CHECK-NEXT: [[EXN:%.*]] = alloca i8*
  // CHECK-NEXT: [[SEL:%.*]] = alloca i32

  // Initialize.
  // CHECK-NEXT: [[E_BEGIN:%.*]] = getelementptr inbounds [10 x [[A]]], [10 x [[A]]]* [[AS]], i64 0, i64 0
  // CHECK-NEXT: store [[A]]* [[E_BEGIN]], [[A]]** [[ENDVAR]]
  // CHECK-NEXT: invoke void @_ZN5test01AC1Ei([[A]]* {{[^,]*}} [[E_BEGIN]], i32 5)
  // CHECK:      [[E1:%.*]] = getelementptr inbounds [[A]], [[A]]* [[E_BEGIN]], i64 1
  // CHECK-NEXT: store [[A]]* [[E1]], [[A]]** [[ENDVAR]]
  // CHECK-NEXT: invoke void @_ZN5test01AC1Ei([[A]]* {{[^,]*}} [[E1]], i32 7)
  // CHECK:      [[E2:%.*]] = getelementptr inbounds [[A]], [[A]]* [[E1]], i64 1
  // CHECK-NEXT: store [[A]]* [[E2]], [[A]]** [[ENDVAR]]
  // CHECK-NEXT: [[E_END:%.*]] = getelementptr inbounds [[A]], [[A]]* [[E_BEGIN]], i64 10
  // CHECK-NEXT: br label
  // CHECK:      [[E_CUR:%.*]] = phi [[A]]* [ [[E2]], {{%.*}} ], [ [[E_NEXT:%.*]], {{%.*}} ]
  // CHECK-NEXT: invoke void @_ZN5test01AC1Ev([[A]]* {{[^,]*}} [[E_CUR]])
  // CHECK:      [[E_NEXT]] = getelementptr inbounds [[A]], [[A]]* [[E_CUR]], i64 1
  // CHECK-NEXT: store [[A]]* [[E_NEXT]], [[A]]** [[ENDVAR]]
  // CHECK-NEXT: [[T0:%.*]] = icmp eq [[A]]* [[E_NEXT]], [[E_END]]
  // CHECK-NEXT: br i1 [[T0]],

  // Run.
  // CHECK:      invoke void @_Z6opaquev()

  // Normal destroy.
  // CHECK:      [[ED_BEGIN:%.*]] = getelementptr inbounds [10 x [[A]]], [10 x [[A]]]* [[AS]], i32 0, i32 0
  // CHECK-NEXT: [[ED_END:%.*]] = getelementptr inbounds [[A]], [[A]]* [[ED_BEGIN]], i64 10
  // CHECK-NEXT: br label
  // CHECK:      [[ED_AFTER:%.*]] = phi [[A]]* [ [[ED_END]], {{%.*}} ], [ [[ED_CUR:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[ED_CUR]] = getelementptr inbounds [[A]], [[A]]* [[ED_AFTER]], i64 -1
  // CHECKv03-NEXT: invoke void @_ZN5test01AD1Ev([[A]]* {{[^,]*}} [[ED_CUR]])
  // CHECKv11-NEXT: call   void @_ZN5test01AD1Ev([[A]]* {{[^,]*}} [[ED_CUR]])
  // CHECK:      [[T0:%.*]] = icmp eq [[A]]* [[ED_CUR]], [[ED_BEGIN]]
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      ret void

  // Partial destroy for initialization.
  // CHECK:      landingpad { i8*, i32 }
  // CHECK-NEXT:   cleanup
  // CHECK:      [[PARTIAL_END:%.*]] = load [[A]]*, [[A]]** [[ENDVAR]]
  // CHECK-NEXT: [[T0:%.*]] = icmp eq [[A]]* [[E_BEGIN]], [[PARTIAL_END]]
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[E_AFTER:%.*]] = phi [[A]]* [ [[PARTIAL_END]], {{%.*}} ], [ [[E_CUR:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[E_CUR]] = getelementptr inbounds [[A]], [[A]]* [[E_AFTER]], i64 -1
  // CHECKv03-NEXT: invoke void @_ZN5test01AD1Ev([[A]]* {{[^,]*}} [[E_CUR]])
  // CHECKv11-NEXT: call   void @_ZN5test01AD1Ev([[A]]* {{[^,]*}} [[E_CUR]])
  // CHECK:      [[T0:%.*]] = icmp eq [[A]]* [[E_CUR]], [[E_BEGIN]]
  // CHECK-NEXT: br i1 [[T0]],

  // Primary EH destructor.
  // CHECK:      landingpad { i8*, i32 }
  // CHECK-NEXT:   cleanup
  // CHECK:      [[E0:%.*]] = getelementptr inbounds [10 x [[A]]], [10 x [[A]]]* [[AS]], i32 0, i32 0
  // CHECK-NEXT: [[E_END:%.*]] = getelementptr inbounds [[A]], [[A]]* [[E0]], i64 10
  // CHECK-NEXT: br label

  // Partial destructor for primary normal destructor.
  // FIXME: There's some really bad block ordering here which causes
  // the partial destroy for the primary normal destructor to fall
  // within the primary EH destructor.
  // CHECKv03:      landingpad { i8*, i32 }
  // CHECKv03-NEXT:   cleanup
  // CHECKv03:      [[T0:%.*]] = icmp eq [[A]]* [[ED_BEGIN]], [[ED_CUR]]
  // CHECKv03-NEXT: br i1 [[T0]]
  // CHECKv03:      [[EDD_AFTER:%.*]] = phi [[A]]* [ [[ED_CUR]], {{%.*}} ], [ [[EDD_CUR:%.*]], {{%.*}} ]
  // CHECKv03-NEXT: [[EDD_CUR]] = getelementptr inbounds [[A]], [[A]]* [[EDD_AFTER]], i64 -1
  // CHECKv03-NEXT: invoke void @_ZN5test01AD1Ev([[A]]* {{[^,]*}} [[EDD_CUR]])
  // CHECKv03:      [[T0:%.*]] = icmp eq [[A]]* [[EDD_CUR]], [[ED_BEGIN]]
  // CHECKv03-NEXT: br i1 [[T0]]

  // Back to the primary EH destructor.
  // CHECK:      [[E_AFTER:%.*]] = phi [[A]]* [ [[E_END]], {{%.*}} ], [ [[E_CUR:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[E_CUR]] = getelementptr inbounds [[A]], [[A]]* [[E_AFTER]], i64 -1
  // CHECKv03-NEXT: invoke void @_ZN5test01AD1Ev([[A]]* {{[^,]*}} [[E_CUR]])
  // CHECKv11-NEXT: call   void @_ZN5test01AD1Ev([[A]]* {{[^,]*}} [[E_CUR]])
  // CHECK:      [[T0:%.*]] = icmp eq [[A]]* [[E_CUR]], [[E0]]
  // CHECK-NEXT: br i1 [[T0]],

}

namespace test1 {
  struct A { A(); A(int); ~A(); };
  struct B { A x, y, z; int w; };

  void test() {
    B v = { 5, 6, 7, 8 };
  }
  // CHECK-LABEL:    define{{.*}} void @_ZN5test14testEv()
  // CHECK-SAME: personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
  // CHECK:      [[V:%.*]] = alloca [[B:%.*]], align 4
  // CHECK-NEXT: alloca i8*
  // CHECK-NEXT: alloca i32
  // CHECK-NEXT: [[X:%.*]] = getelementptr inbounds [[B]], [[B]]* [[V]], i32 0, i32 0
  // CHECK-NEXT: call void @_ZN5test11AC1Ei([[A:%.*]]* {{[^,]*}} [[X]], i32 5)
  // CHECK-NEXT: [[Y:%.*]] = getelementptr inbounds [[B]], [[B]]* [[V]], i32 0, i32 1
  // CHECK-NEXT: invoke void @_ZN5test11AC1Ei([[A]]* {{[^,]*}} [[Y]], i32 6)
  // CHECK:      [[Z:%.*]] = getelementptr inbounds [[B]], [[B]]* [[V]], i32 0, i32 2
  // CHECK-NEXT: invoke void @_ZN5test11AC1Ei([[A]]* {{[^,]*}} [[Z]], i32 7)
  // CHECK:      [[W:%.*]] = getelementptr inbounds [[B]], [[B]]* [[V]], i32 0, i32 3
  // CHECK-NEXT: store i32 8, i32* [[W]], align 4
  // CHECK-NEXT: call void @_ZN5test11BD1Ev([[B]]* {{[^,]*}} [[V]])
  // CHECK-NEXT: ret void

  // FIXME: again, the block ordering is pretty bad here
  // CHECK:      landingpad { i8*, i32 }
  // CHECK-NEXT:   cleanup
  // CHECK:      landingpad { i8*, i32 }
  // CHECK-NEXT:   cleanup
  // CHECKv03:      invoke void @_ZN5test11AD1Ev([[A]]* {{[^,]*}} [[Y]])
  // CHECKv03:      invoke void @_ZN5test11AD1Ev([[A]]* {{[^,]*}} [[X]])
  // CHECKv11:      call   void @_ZN5test11AD1Ev([[A]]* {{[^,]*}} [[Y]])
  // CHECKv11:      call   void @_ZN5test11AD1Ev([[A]]* {{[^,]*}} [[X]])
}

namespace test2 {
  struct A { A(); ~A(); };

  void test() {
    A v[4][7];

    // CHECK-LABEL:    define{{.*}} void @_ZN5test24testEv()
    // CHECK-SAME: personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
    // CHECK:      [[V:%.*]] = alloca [4 x [7 x [[A:%.*]]]], align 1
    // CHECK-NEXT: alloca i8*
    // CHECK-NEXT: alloca i32

    // Main initialization loop.
    // CHECK-NEXT: [[BEGIN:%.*]] = getelementptr inbounds [4 x [7 x [[A]]]], [4 x [7 x [[A]]]]* [[V]], i32 0, i32 0, i32 0
    // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds [[A]], [[A]]* [[BEGIN]], i64 28
    // CHECK-NEXT: br label
    // CHECK:      [[CUR:%.*]] = phi [[A]]* [ [[BEGIN]], {{%.*}} ], [ [[NEXT:%.*]], {{%.*}} ]
    // CHECK-NEXT: invoke void @_ZN5test21AC1Ev([[A]]* {{[^,]*}} [[CUR]])
    // CHECK:      [[NEXT:%.*]] = getelementptr inbounds [[A]], [[A]]* [[CUR]], i64 1
    // CHECK-NEXT: [[DONE:%.*]] = icmp eq [[A]]* [[NEXT]], [[END]]
    // CHECK-NEXT: br i1 [[DONE]],

    // Partial destruction landing pad.
    // CHECK:      landingpad { i8*, i32 }
    // CHECK-NEXT:   cleanup
    // CHECK:      [[EMPTY:%.*]] = icmp eq [[A]]* [[BEGIN]], [[CUR]]
    // CHECK-NEXT: br i1 [[EMPTY]],
    // CHECK:      [[PAST:%.*]] = phi [[A]]* [ [[CUR]], {{%.*}} ], [ [[DEL:%.*]], {{%.*}} ]
    // CHECK-NEXT: [[DEL]] = getelementptr inbounds [[A]], [[A]]* [[PAST]], i64 -1
    // CHECKv03-NEXT: invoke void @_ZN5test21AD1Ev([[A]]* {{[^,]*}} [[DEL]])
    // CHECKv11-NEXT: call   void @_ZN5test21AD1Ev([[A]]* {{[^,]*}} [[DEL]])
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

namespace test4 {
  struct A { A(unsigned i); ~A(); };
  void test() {
    A v[2][3] = { { A(0), A(1), A(2) }, { A(3), A(4), A(5) } };
  }
}
// CHECK-LABEL: define{{.*}} void @_ZN5test44testEv()
// CHECK:       [[ARRAY:%.*]] = alloca [2 x [3 x [[A:%.*]]]], align
// CHECK:       [[A0:%.*]] = getelementptr inbounds [2 x [3 x [[A]]]], [2 x [3 x [[A]]]]* [[ARRAY]], i64 0, i64 0
// CHECK-NEXT:  store [3 x [[A]]]* [[A0]],
// CHECK-NEXT:  [[A00:%.*]] = getelementptr inbounds [3 x [[A]]], [3 x [[A]]]* [[A0]], i64 0, i64 0
// CHECK-NEXT:  store [[A]]* [[A00]],
// CHECK-NEXT:  invoke void @_ZN5test41AC1Ej([[A]]* {{[^,]*}} [[A00]], i32 0)
// CHECK:       [[A01:%.*]] = getelementptr inbounds [[A]], [[A]]* [[A00]], i64 1
// CHECK-NEXT:  store [[A]]* [[A01]],
// CHECK-NEXT:  invoke void @_ZN5test41AC1Ej([[A]]* {{[^,]*}} [[A01]], i32 1)
// CHECK:       [[A02:%.*]] = getelementptr inbounds [[A]], [[A]]* [[A01]], i64 1
// CHECK-NEXT:  store [[A]]* [[A02]],
// CHECK-NEXT:  invoke void @_ZN5test41AC1Ej([[A]]* {{[^,]*}} [[A02]], i32 2)
// CHECK:       [[A1:%.*]] = getelementptr inbounds [3 x [[A]]], [3 x [[A]]]* [[A0]], i64 1
// CHECK-NEXT:  store [3 x [[A]]]* [[A1]],
// CHECK-NEXT:  [[A10:%.*]] = getelementptr inbounds [3 x [[A]]], [3 x [[A]]]* [[A1]], i64 0, i64 0
// CHECK-NEXT:  store [[A]]* [[A10]],
// CHECK-NEXT:  invoke void @_ZN5test41AC1Ej([[A]]* {{[^,]*}} [[A10]], i32 3)
// CHECK:       [[A11:%.*]] = getelementptr inbounds [[A]], [[A]]* [[A10]], i64 1
// CHECK-NEXT:  store [[A]]* [[A11]],
// CHECK-NEXT:  invoke void @_ZN5test41AC1Ej([[A]]* {{[^,]*}} [[A11]], i32 4)
// CHECK:       [[A12:%.*]] = getelementptr inbounds [[A]], [[A]]* [[A11]], i64 1
// CHECK-NEXT:  store [[A]]* [[A12]],
// CHECK-NEXT:  invoke void @_ZN5test41AC1Ej([[A]]* {{[^,]*}} [[A12]], i32 5)
