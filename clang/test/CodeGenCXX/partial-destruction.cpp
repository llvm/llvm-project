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
  // CHECK-SAME: personality ptr @__gxx_personality_v0
  // CHECK:      [[AS:%.*]] = alloca [10 x [[A:%.*]]], align
  // CHECK-NEXT: [[ENDVAR:%.*]] = alloca ptr
  // CHECK-NEXT: [[EXN:%.*]] = alloca ptr
  // CHECK-NEXT: [[SEL:%.*]] = alloca i32

  // Initialize.
  // CHECK-NEXT: store ptr [[AS]], ptr [[ENDVAR]]
  // CHECK-NEXT: invoke void @_ZN5test01AC1Ei(ptr {{[^,]*}} [[AS]], i32 noundef 5)
  // CHECK:      [[E1:%.*]] = getelementptr inbounds [[A]], ptr [[AS]], i64 1
  // CHECK-NEXT: store ptr [[E1]], ptr [[ENDVAR]]
  // CHECK-NEXT: invoke void @_ZN5test01AC1Ei(ptr {{[^,]*}} [[E1]], i32 noundef 7)
  // CHECK:      [[E2:%.*]] = getelementptr inbounds [[A]], ptr [[AS]], i64 2
  // CHECK-NEXT: store ptr [[E2]], ptr [[ENDVAR]]
  // CHECK-NEXT: [[E_END:%.*]] = getelementptr inbounds [[A]], ptr [[AS]], i64 10
  // CHECK-NEXT: br label
  // CHECK:      [[E_CUR:%.*]] = phi ptr [ [[E2]], {{%.*}} ], [ [[E_NEXT:%.*]], {{%.*}} ]
  // CHECK-NEXT: invoke void @_ZN5test01AC1Ev(ptr {{[^,]*}} [[E_CUR]])
  // CHECK:      [[E_NEXT]] = getelementptr inbounds [[A]], ptr [[E_CUR]], i64 1
  // CHECK-NEXT: store ptr [[E_NEXT]], ptr [[ENDVAR]]
  // CHECK-NEXT: [[T0:%.*]] = icmp eq ptr [[E_NEXT]], [[E_END]]
  // CHECK-NEXT: br i1 [[T0]],

  // Run.
  // CHECK:      invoke void @_Z6opaquev()

  // Normal destroy.
  // CHECK:      [[ED_BEGIN:%.*]] = getelementptr inbounds [10 x [[A]]], ptr [[AS]], i32 0, i32 0
  // CHECK-NEXT: [[ED_END:%.*]] = getelementptr inbounds [[A]], ptr [[ED_BEGIN]], i64 10
  // CHECK-NEXT: br label
  // CHECK:      [[ED_AFTER:%.*]] = phi ptr [ [[ED_END]], {{%.*}} ], [ [[ED_CUR:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[ED_CUR]] = getelementptr inbounds [[A]], ptr [[ED_AFTER]], i64 -1
  // CHECKv03-NEXT: invoke void @_ZN5test01AD1Ev(ptr {{[^,]*}} [[ED_CUR]])
  // CHECKv11-NEXT: call   void @_ZN5test01AD1Ev(ptr {{[^,]*}} [[ED_CUR]])
  // CHECK:      [[T0:%.*]] = icmp eq ptr [[ED_CUR]], [[ED_BEGIN]]
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      ret void

  // Partial destroy for initialization.
  // CHECK:      landingpad { ptr, i32 }
  // CHECK-NEXT:   cleanup
  // CHECK:      [[PARTIAL_END:%.*]] = load ptr, ptr [[ENDVAR]]
  // CHECK-NEXT: [[T0:%.*]] = icmp eq ptr [[AS]], [[PARTIAL_END]]
  // CHECK-NEXT: br i1 [[T0]],
  // CHECK:      [[E_AFTER:%.*]] = phi ptr [ [[PARTIAL_END]], {{%.*}} ], [ [[E_CUR:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[E_CUR]] = getelementptr inbounds [[A]], ptr [[E_AFTER]], i64 -1
  // CHECKv03-NEXT: invoke void @_ZN5test01AD1Ev(ptr {{[^,]*}} [[E_CUR]])
  // CHECKv11-NEXT: call   void @_ZN5test01AD1Ev(ptr {{[^,]*}} [[E_CUR]])
  // CHECK:      [[T0:%.*]] = icmp eq ptr [[E_CUR]], [[AS]]
  // CHECK-NEXT: br i1 [[T0]],

  // Primary EH destructor.
  // CHECK:      landingpad { ptr, i32 }
  // CHECK-NEXT:   cleanup
  // CHECK:      [[E0:%.*]] = getelementptr inbounds [10 x [[A]]], ptr [[AS]], i32 0, i32 0
  // CHECK-NEXT: [[E_END:%.*]] = getelementptr inbounds [[A]], ptr [[E0]], i64 10
  // CHECK-NEXT: br label

  // Partial destructor for primary normal destructor.
  // FIXME: There's some really bad block ordering here which causes
  // the partial destroy for the primary normal destructor to fall
  // within the primary EH destructor.
  // CHECKv03:      landingpad { ptr, i32 }
  // CHECKv03-NEXT:   cleanup
  // CHECKv03:      [[T0:%.*]] = icmp eq ptr [[ED_BEGIN]], [[ED_CUR]]
  // CHECKv03-NEXT: br i1 [[T0]]
  // CHECKv03:      [[EDD_AFTER:%.*]] = phi ptr [ [[ED_CUR]], {{%.*}} ], [ [[EDD_CUR:%.*]], {{%.*}} ]
  // CHECKv03-NEXT: [[EDD_CUR]] = getelementptr inbounds [[A]], ptr [[EDD_AFTER]], i64 -1
  // CHECKv03-NEXT: invoke void @_ZN5test01AD1Ev(ptr {{[^,]*}} [[EDD_CUR]])
  // CHECKv03:      [[T0:%.*]] = icmp eq ptr [[EDD_CUR]], [[ED_BEGIN]]
  // CHECKv03-NEXT: br i1 [[T0]]

  // Back to the primary EH destructor.
  // CHECK:      [[E_AFTER:%.*]] = phi ptr [ [[E_END]], {{%.*}} ], [ [[E_CUR:%.*]], {{%.*}} ]
  // CHECK-NEXT: [[E_CUR]] = getelementptr inbounds [[A]], ptr [[E_AFTER]], i64 -1
  // CHECKv03-NEXT: invoke void @_ZN5test01AD1Ev(ptr {{[^,]*}} [[E_CUR]])
  // CHECKv11-NEXT: call   void @_ZN5test01AD1Ev(ptr {{[^,]*}} [[E_CUR]])
  // CHECK:      [[T0:%.*]] = icmp eq ptr [[E_CUR]], [[E0]]
  // CHECK-NEXT: br i1 [[T0]],

}

namespace test1 {
  struct A { A(); A(int); ~A(); };
  struct B { A x, y, z; int w; };

  void test() {
    B v = { 5, 6, 7, 8 };
  }
  // CHECK-LABEL:    define{{.*}} void @_ZN5test14testEv()
  // CHECK-SAME: personality ptr @__gxx_personality_v0
  // CHECK:      [[V:%.*]] = alloca [[B:%.*]], align 4
  // CHECK-NEXT: alloca ptr
  // CHECK-NEXT: alloca i32
  // CHECK-NEXT: [[X:%.*]] = getelementptr inbounds [[B]], ptr [[V]], i32 0, i32 0
  // CHECK-NEXT: call void @_ZN5test11AC1Ei(ptr {{[^,]*}} [[X]], i32 noundef 5)
  // CHECK-NEXT: [[Y:%.*]] = getelementptr inbounds [[B]], ptr [[V]], i32 0, i32 1
  // CHECK-NEXT: invoke void @_ZN5test11AC1Ei(ptr {{[^,]*}} [[Y]], i32 noundef 6)
  // CHECK:      [[Z:%.*]] = getelementptr inbounds [[B]], ptr [[V]], i32 0, i32 2
  // CHECK-NEXT: invoke void @_ZN5test11AC1Ei(ptr {{[^,]*}} [[Z]], i32 noundef 7)
  // CHECK:      [[W:%.*]] = getelementptr inbounds [[B]], ptr [[V]], i32 0, i32 3
  // CHECK-NEXT: store i32 8, ptr [[W]], align 4
  // CHECK-NEXT: call void @_ZN5test11BD1Ev(ptr {{[^,]*}} [[V]])
  // CHECK-NEXT: ret void

  // FIXME: again, the block ordering is pretty bad here
  // CHECK:      landingpad { ptr, i32 }
  // CHECK-NEXT:   cleanup
  // CHECK:      landingpad { ptr, i32 }
  // CHECK-NEXT:   cleanup
  // CHECKv03:      invoke void @_ZN5test11AD1Ev(ptr {{[^,]*}} [[Y]])
  // CHECKv03:      invoke void @_ZN5test11AD1Ev(ptr {{[^,]*}} [[X]])
  // CHECKv11:      call   void @_ZN5test11AD1Ev(ptr {{[^,]*}} [[Y]])
  // CHECKv11:      call   void @_ZN5test11AD1Ev(ptr {{[^,]*}} [[X]])
}

namespace test2 {
  struct A { A(); ~A(); };

  void test() {
    A v[4][7];

    // CHECK-LABEL:    define{{.*}} void @_ZN5test24testEv()
    // CHECK-SAME: personality ptr @__gxx_personality_v0
    // CHECK:      [[V:%.*]] = alloca [4 x [7 x [[A:%.*]]]], align 1
    // CHECK-NEXT: alloca ptr
    // CHECK-NEXT: alloca i32

    // Main initialization loop.
    // CHECK-NEXT: [[BEGIN:%.*]] = getelementptr inbounds [4 x [7 x [[A]]]], ptr [[V]], i32 0, i32 0, i32 0
    // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds [[A]], ptr [[BEGIN]], i64 28
    // CHECK-NEXT: br label
    // CHECK:      [[CUR:%.*]] = phi ptr [ [[BEGIN]], {{%.*}} ], [ [[NEXT:%.*]], {{%.*}} ]
    // CHECK-NEXT: invoke void @_ZN5test21AC1Ev(ptr {{[^,]*}} [[CUR]])
    // CHECK:      [[NEXT:%.*]] = getelementptr inbounds [[A]], ptr [[CUR]], i64 1
    // CHECK-NEXT: [[DONE:%.*]] = icmp eq ptr [[NEXT]], [[END]]
    // CHECK-NEXT: br i1 [[DONE]],

    // Partial destruction landing pad.
    // CHECK:      landingpad { ptr, i32 }
    // CHECK-NEXT:   cleanup
    // CHECK:      [[EMPTY:%.*]] = icmp eq ptr [[BEGIN]], [[CUR]]
    // CHECK-NEXT: br i1 [[EMPTY]],
    // CHECK:      [[PAST:%.*]] = phi ptr [ [[CUR]], {{%.*}} ], [ [[DEL:%.*]], {{%.*}} ]
    // CHECK-NEXT: [[DEL]] = getelementptr inbounds [[A]], ptr [[PAST]], i64 -1
    // CHECKv03-NEXT: invoke void @_ZN5test21AD1Ev(ptr {{[^,]*}} [[DEL]])
    // CHECKv11-NEXT: call   void @_ZN5test21AD1Ev(ptr {{[^,]*}} [[DEL]])
    // CHECK:      [[T0:%.*]] = icmp eq ptr [[DEL]], [[BEGIN]]
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
// CHECK:       store ptr [[ARRAY]],
// CHECK-NEXT:  store ptr [[ARRAY]],
// CHECK-NEXT:  invoke void @_ZN5test41AC1Ej(ptr {{[^,]*}} [[ARRAY]], i32 noundef 0)
// CHECK:       [[A01:%.*]] = getelementptr inbounds [[A]], ptr [[ARRAY]], i64 1
// CHECK-NEXT:  store ptr [[A01]],
// CHECK-NEXT:  invoke void @_ZN5test41AC1Ej(ptr {{[^,]*}} [[A01]], i32 noundef 1)
// CHECK:       [[A02:%.*]] = getelementptr inbounds [[A]], ptr [[ARRAY]], i64 2
// CHECK-NEXT:  store ptr [[A02]],
// CHECK-NEXT:  invoke void @_ZN5test41AC1Ej(ptr {{[^,]*}} [[A02]], i32 noundef 2)
// CHECK:       [[A1:%.*]] = getelementptr inbounds [3 x [[A]]], ptr [[ARRAY]], i64 1
// CHECK-NEXT:  store ptr [[A1]],
// CHECK-NEXT:  store ptr [[A1]],
// CHECK-NEXT:  invoke void @_ZN5test41AC1Ej(ptr {{[^,]*}} [[A1]], i32 noundef 3)
// CHECK:       [[A11:%.*]] = getelementptr inbounds [[A]], ptr [[A1]], i64 1
// CHECK-NEXT:  store ptr [[A11]],
// CHECK-NEXT:  invoke void @_ZN5test41AC1Ej(ptr {{[^,]*}} [[A11]], i32 noundef 4)
// CHECK:       [[A12:%.*]] = getelementptr inbounds [[A]], ptr [[A1]], i64 2
// CHECK-NEXT:  store ptr [[A12]],
// CHECK-NEXT:  invoke void @_ZN5test41AC1Ej(ptr {{[^,]*}} [[A12]], i32 noundef 5)
