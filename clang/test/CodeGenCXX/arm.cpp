// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=thumbv7-apple-ios6.0 -fno-use-cxa-atexit -target-abi apcs-gnu -emit-llvm -std=gnu++98 -o - -fexceptions | FileCheck -check-prefix=CHECK -check-prefix=CHECK98 %s
// RUN: %clang_cc1 -no-enable-noundef-analysis %s -triple=thumbv7-apple-ios6.0 -fno-use-cxa-atexit -target-abi apcs-gnu -emit-llvm -std=gnu++11 -o - -fexceptions | FileCheck -check-prefix=CHECK -check-prefix=CHECK11 %s

// CHECK: @_ZZN5test74testEvE1x = internal global i32 0, align 4
// CHECK: @_ZGVZN5test74testEvE1x = internal global i32 0
// CHECK: @_ZZN5test84testEvE1x = internal global [[TEST8A:.*]] zeroinitializer, align 1
// CHECK: @_ZGVZN5test84testEvE1x = internal global i32 0

typedef typeof(sizeof(int)) size_t;

class foo {
public:
    foo();
    virtual ~foo();
};

class bar : public foo {
public:
	bar();
};

// The global dtor needs the right calling conv with -fno-use-cxa-atexit
bar baz;

// PR9593
// Make sure atexit(3) is used for global dtors.

// CHECK:      call ptr @_ZN3barC1Ev(
// CHECK-NEXT: call i32 @atexit(ptr @__dtor_baz)

// CHECK-NOT: @_GLOBAL__D_a()
// CHECK-LABEL: define internal void @__dtor_baz()
// CHECK: call ptr @_ZN3barD1Ev(ptr @baz)

// Destructors and constructors must return this.
namespace test1 {
  void foo();

  struct A {
    A(int i) { foo(); }
    ~A() { foo(); }
    void bar() { foo(); }
  };

  // CHECK-LABEL: define{{.*}} void @_ZN5test14testEv()
  void test() {
    // CHECK: [[AV:%.*]] = alloca [[A:%.*]], align 1
    // CHECK: call ptr @_ZN5test11AC1Ei(ptr {{[^,]*}} [[AV]], i32 10)
    // CHECK: invoke void @_ZN5test11A3barEv(ptr {{[^,]*}} [[AV]])
    // CHECK: call ptr @_ZN5test11AD1Ev(ptr {{[^,]*}} [[AV]])
    // CHECK: ret void
    A a = 10;
    a.bar();
  }

  // CHECK: define linkonce_odr ptr @_ZN5test11AC1Ei(ptr {{[^,]*}} returned {{[^,]*}} %this, i32 %i) unnamed_addr
  // CHECK:   [[THIS:%.*]] = alloca ptr, align 4
  // CHECK:   store ptr {{.*}}, ptr [[THIS]]
  // CHECK:   [[THIS1:%.*]] = load ptr, ptr [[THIS]]
  // CHECK:   {{%.*}} = call ptr @_ZN5test11AC2Ei(
  // CHECK:   ret ptr [[THIS1]]

  // CHECK: define linkonce_odr ptr @_ZN5test11AD1Ev(ptr {{[^,]*}} returned {{[^,]*}} %this) unnamed_addr
  // CHECK:   [[THIS:%.*]] = alloca ptr, align 4
  // CHECK:   store ptr {{.*}}, ptr [[THIS]]
  // CHECK:   [[THIS1:%.*]] = load ptr, ptr [[THIS]]
  // CHECK:   {{%.*}} = call ptr @_ZN5test11AD2Ev(
  // CHECK:   ret ptr [[THIS1]]
}

// Awkward virtual cases.
namespace test2 {
  void foo();

  struct A {
    int x;

    A(int);
    virtual ~A() { foo(); }
  };

  struct B {
    int y;
    int z;

    B(int);
    virtual ~B() { foo(); }
  };

  struct C : A, virtual B {
    int q;

    C(int i) : A(i), B(i) { foo(); }
    ~C() { foo(); }
  };

  void test() {
    C c = 10;
  }

  // Tests at eof
}

namespace test3 {
  struct A {
    int x;
    ~A();
  };

  void a() {
    // CHECK-LABEL: define{{.*}} void @_ZN5test31aEv()
    // CHECK: call noalias nonnull ptr @_Znam(i32 48)
    // CHECK: store i32 4
    // CHECK: store i32 10
    A *x = new A[10];
  }

  void b(int n) {
    // CHECK-LABEL: define{{.*}} void @_ZN5test31bEi(
    // CHECK: [[N:%.*]] = load i32, ptr
    // CHECK: @llvm.umul.with.overflow.i32(i32 [[N]], i32 4)
    // CHECK: @llvm.uadd.with.overflow.i32(i32 {{.*}}, i32 8)
    // CHECK: [[OR:%.*]] = or i1
    // CHECK: [[SZ:%.*]] = select i1 [[OR]]
    // CHECK: call noalias nonnull ptr @_Znam(i32 [[SZ]])
    // CHECK: store i32 4
    // CHECK: store i32 [[N]]
    A *x = new A[n];
  }

  void c() {
    // CHECK-LABEL: define{{.*}} void @_ZN5test31cEv()
    // CHECK: call noalias nonnull ptr @_Znam(i32 808)
    // CHECK: store i32 4
    // CHECK: store i32 200
    A (*x)[20] = new A[10][20];
  }

  void d(int n) {
    // CHECK-LABEL: define{{.*}} void @_ZN5test31dEi(
    // CHECK: [[N:%.*]] = load i32, ptr
    // CHECK: @llvm.umul.with.overflow.i32(i32 [[N]], i32 80)
    // CHECK: [[NE:%.*]] = mul i32 [[N]], 20
    // CHECK: @llvm.uadd.with.overflow.i32(i32 {{.*}}, i32 8)
    // CHECK: [[SZ:%.*]] = select
    // CHECK: call noalias nonnull ptr @_Znam(i32 [[SZ]])
    // CHECK: store i32 4
    // CHECK: store i32 [[NE]]
    A (*x)[20] = new A[n][20];
  }

  void e(A *x) {
    // CHECK-LABEL: define{{.*}} void @_ZN5test31eEPNS_1AE(
    // CHECK: icmp eq {{.*}}, null
    // CHECK: getelementptr {{.*}}, i32 -8
    // CHECK: getelementptr {{.*}}, i32 4
    // CHECK: load
    // CHECK98: invoke {{.*}} @_ZN5test31AD1Ev
    // CHECK11: call {{.*}} @_ZN5test31AD1Ev
    // CHECK: call void @_ZdaPv
    delete [] x;
  }

  void f(A (*x)[20]) {
    // CHECK-LABEL: define{{.*}} void @_ZN5test31fEPA20_NS_1AE(
    // CHECK: icmp eq {{.*}}, null
    // CHECK: getelementptr {{.*}}, i32 -8
    // CHECK: getelementptr {{.*}}, i32 4
    // CHECK: load
    // CHECK98: invoke {{.*}} @_ZN5test31AD1Ev
    // CHECK11: call {{.*}} @_ZN5test31AD1Ev
    // CHECK: call void @_ZdaPv
    delete [] x;
  }
}

namespace test4 {
  struct A {
    int x;
    void operator delete[](void *, size_t sz);
  };

  void a() {
    // CHECK-LABEL: define{{.*}} void @_ZN5test41aEv()
    // CHECK: call noalias nonnull ptr @_Znam(i32 48)
    // CHECK: store i32 4
    // CHECK: store i32 10
    A *x = new A[10];
  }

  void b(int n) {
    // CHECK-LABEL: define{{.*}} void @_ZN5test41bEi(
    // CHECK: [[N:%.*]] = load i32, ptr
    // CHECK: @llvm.umul.with.overflow.i32(i32 [[N]], i32 4)
    // CHECK: @llvm.uadd.with.overflow.i32(i32 {{.*}}, i32 8)
    // CHECK: [[SZ:%.*]] = select
    // CHECK: call noalias nonnull ptr @_Znam(i32 [[SZ]])
    // CHECK: store i32 4
    // CHECK: store i32 [[N]]
    A *x = new A[n];
  }

  void c() {
    // CHECK-LABEL: define{{.*}} void @_ZN5test41cEv()
    // CHECK: call noalias nonnull ptr @_Znam(i32 808)
    // CHECK: store i32 4
    // CHECK: store i32 200
    A (*x)[20] = new A[10][20];
  }

  void d(int n) {
    // CHECK-LABEL: define{{.*}} void @_ZN5test41dEi(
    // CHECK: [[N:%.*]] = load i32, ptr
    // CHECK: @llvm.umul.with.overflow.i32(i32 [[N]], i32 80)
    // CHECK: [[NE:%.*]] = mul i32 [[N]], 20
    // CHECK: @llvm.uadd.with.overflow.i32(i32 {{.*}}, i32 8)
    // CHECK: [[SZ:%.*]] = select
    // CHECK: call noalias nonnull ptr @_Znam(i32 [[SZ]])
    // CHECK: store i32 4
    // CHECK: store i32 [[NE]]
    A (*x)[20] = new A[n][20];
  }

  void e(A *x) {
    // CHECK-LABEL: define{{.*}} void @_ZN5test41eEPNS_1AE(
    // CHECK: [[ALLOC:%.*]] = getelementptr inbounds {{.*}}, i32 -8
    // CHECK: getelementptr inbounds {{.*}}, i32 4
    // CHECK: [[T0:%.*]] = load i32, ptr
    // CHECK: [[T1:%.*]] = mul i32 4, [[T0]]
    // CHECK: [[T2:%.*]] = add i32 [[T1]], 8
    // CHECK: call void @_ZN5test41AdaEPvm(ptr [[ALLOC]], i32 [[T2]])
    delete [] x;
  }

  void f(A (*x)[20]) {
    // CHECK-LABEL: define{{.*}} void @_ZN5test41fEPA20_NS_1AE(
    // CHECK: [[ALLOC:%.*]] = getelementptr inbounds {{.*}}, i32 -8
    // CHECK: getelementptr inbounds {{.*}}, i32 4
    // CHECK: [[T0:%.*]] = load i32, ptr
    // CHECK: [[T1:%.*]] = mul i32 4, [[T0]]
    // CHECK: [[T2:%.*]] = add i32 [[T1]], 8
    // CHECK: call void @_ZN5test41AdaEPvm(ptr [[ALLOC]], i32 [[T2]])
    delete [] x;
  }
}

namespace test5 {
  struct A {
    ~A();
  };

  // CHECK-LABEL: define{{.*}} void @_ZN5test54testEPNS_1AE
  void test(A *a) {
    // CHECK:      [[PTR:%.*]] = alloca ptr, align 4
    // CHECK-NEXT: store ptr {{.*}}, ptr [[PTR]], align 4
    // CHECK-NEXT: [[TMP:%.*]] = load ptr, ptr [[PTR]], align 4
    // CHECK-NEXT: call ptr @_ZN5test51AD1Ev(ptr {{[^,]*}} [[TMP]])
    // CHECK-NEXT: ret void
    a->~A();
  }
}

namespace test6 {
  struct A {
    virtual ~A();
  };

  // CHECK-LABEL: define{{.*}} void @_ZN5test64testEPNS_1AE
  void test(A *a) {
    // CHECK:      [[AVAR:%.*]] = alloca ptr, align 4
    // CHECK-NEXT: store ptr {{.*}}, ptr [[AVAR]], align 4
    // CHECK-NEXT: [[V:%.*]] = load ptr, ptr [[AVAR]], align 4
    // CHECK-NEXT: [[ISNULL:%.*]] = icmp eq ptr [[V]], null
    // CHECK-NEXT: br i1 [[ISNULL]]
    // CHECK: [[T1:%.*]] = load ptr, ptr [[V]]
    // CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds ptr, ptr [[T1]], i64 1
    // CHECK-NEXT: [[T3:%.*]] = load ptr, ptr [[T2]]
    // CHECK-NEXT: call void [[T3]](ptr {{[^,]*}} [[V]])
    // CHECK-NEXT: br label
    // CHECK:      ret void
    delete a;
  }
}

namespace test7 {
  int foo();

  // Static and guard tested at top of file

  // CHECK-LABEL: define{{.*}} void @_ZN5test74testEv() {{.*}} personality ptr @__gxx_personality_v0
  void test() {
    // CHECK:      [[T0:%.*]] = load atomic i8, ptr @_ZGVZN5test74testEvE1x acquire, align 4
    // CHECK-NEXT: [[T1:%.*]] = and i8 [[T0]], 1
    // CHECK-NEXT: [[T2:%.*]] = icmp eq i8 [[T1]], 0
    // CHECK-NEXT: br i1 [[T2]]
    //   -> fallthrough, end
    // CHECK:      [[T3:%.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVZN5test74testEvE1x)
    // CHECK-NEXT: [[T4:%.*]] = icmp ne i32 [[T3]], 0
    // CHECK-NEXT: br i1 [[T4]]
    //   -> fallthrough, end
    // CHECK:      [[INIT:%.*]] = invoke i32 @_ZN5test73fooEv()
    // CHECK:      store i32 [[INIT]], ptr @_ZZN5test74testEvE1x, align 4
    // CHECK-NEXT: call void @__cxa_guard_release(ptr @_ZGVZN5test74testEvE1x)
    // CHECK-NEXT: br label
    //   -> end
    // end:
    // CHECK:      ret void
    static int x = foo();

    // CHECK:      landingpad { ptr, i32 }
    // CHECK-NEXT:   cleanup
    // CHECK:      call void @__cxa_guard_abort(ptr @_ZGVZN5test74testEvE1x)
    // CHECK:      resume { ptr, i32 }
  }
}

namespace test8 {
  struct A {
    A();
    ~A();
  };

  // Static and guard tested at top of file

  // CHECK-LABEL: define{{.*}} void @_ZN5test84testEv() {{.*}} personality ptr @__gxx_personality_v0
  void test() {
    // CHECK:      [[T0:%.*]] = load atomic i8, ptr @_ZGVZN5test84testEvE1x acquire, align 4
    // CHECK-NEXT: [[T1:%.*]] = and i8 [[T0]], 1
    // CHECK-NEXT: [[T2:%.*]] = icmp eq i8 [[T1]], 0
    // CHECK-NEXT: br i1 [[T2]]
    //   -> fallthrough, end
    // CHECK:      [[T3:%.*]] = call i32 @__cxa_guard_acquire(ptr @_ZGVZN5test84testEvE1x)
    // CHECK-NEXT: [[T4:%.*]] = icmp ne i32 [[T3]], 0
    // CHECK-NEXT: br i1 [[T4]]
    //   -> fallthrough, end
    // CHECK:      [[INIT:%.*]] = invoke ptr @_ZN5test81AC1Ev(ptr {{[^,]*}} @_ZZN5test84testEvE1x)

    // FIXME: Here we register a global destructor that
    // unconditionally calls the destructor.  That's what we've always
    // done for -fno-use-cxa-atexit here, but that's really not
    // semantically correct at all.

    // CHECK:      call void @__cxa_guard_release(ptr @_ZGVZN5test84testEvE1x)
    // CHECK-NEXT: br label
    //   -> end
    // end:
    // CHECK:      ret void
    static A x;

    // CHECK:      landingpad { ptr, i32 }
    // CHECK-NEXT:   cleanup
    // CHECK:      call void @__cxa_guard_abort(ptr @_ZGVZN5test84testEvE1x)
    // CHECK:      resume { ptr, i32 }
  }
}

// Use a larger-than-mandated array cookie when allocating an
// array whose type is overaligned.
namespace test9 {
  class __attribute__((aligned(16))) A {
    float data[4];
  public:
    A();
    ~A();
  };

  A *testNew(unsigned n) {
    return new A[n];
  }
// CHECK:    define{{.*}} ptr @_ZN5test97testNewEj(i32
// CHECK:      [[N_VAR:%.*]] = alloca i32, align 4
// CHECK:      [[N:%.*]] = load i32, ptr [[N_VAR]], align 4
// CHECK-NEXT: [[T0:%.*]] = call { i32, i1 } @llvm.umul.with.overflow.i32(i32 [[N]], i32 16)
// CHECK-NEXT: [[O0:%.*]] = extractvalue { i32, i1 } [[T0]], 1
// CHECK-NEXT: [[T1:%.*]] = extractvalue { i32, i1 } [[T0]], 0
// CHECK-NEXT: [[T2:%.*]] = call { i32, i1 } @llvm.uadd.with.overflow.i32(i32 [[T1]], i32 16)
// CHECK-NEXT: [[O1:%.*]] = extractvalue { i32, i1 } [[T2]], 1
// CHECK-NEXT: [[OVERFLOW:%.*]] = or i1 [[O0]], [[O1]]
// CHECK-NEXT: [[T3:%.*]] = extractvalue { i32, i1 } [[T2]], 0
// CHECK-NEXT: [[T4:%.*]] = select i1 [[OVERFLOW]], i32 -1, i32 [[T3]]
// CHECK-NEXT: [[ALLOC:%.*]] = call noalias nonnull ptr @_Znam(i32 [[T4]])
// CHECK-NEXT: store i32 16, ptr [[ALLOC]]
// CHECK-NEXT: [[T1:%.*]] = getelementptr inbounds i32, ptr [[ALLOC]], i32 1
// CHECK-NEXT: store i32 [[N]], ptr [[T1]]
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds i8, ptr [[ALLOC]], i32 16
//   Array allocation follows.

  void testDelete(A *array) {
    delete[] array;
  }
// CHECK-LABEL:    define{{.*}} void @_ZN5test910testDeleteEPNS_1AE(
// CHECK:      [[BEGIN:%.*]] = load ptr, ptr
// CHECK-NEXT: [[T0:%.*]] = icmp eq ptr [[BEGIN]], null
// CHECK-NEXT: br i1 [[T0]],
// CHECK: [[ALLOC:%.*]] = getelementptr inbounds i8, ptr [[BEGIN]], i32 -16
// CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds i8, ptr [[ALLOC]], i32 4
// CHECK-NEXT: [[N:%.*]] = load i32, ptr [[T0]]
// CHECK-NEXT: [[END:%.*]] = getelementptr inbounds [[TEST9:%.*]], ptr [[BEGIN]], i32 [[N]]
// CHECK-NEXT: [[T0:%.*]] = icmp eq ptr [[BEGIN]], [[END]]
// CHECK-NEXT: br i1 [[T0]],
//   Array deallocation follows.
}

  // CHECK: define linkonce_odr ptr @_ZTv0_n12_N5test21CD1Ev(
  // CHECK:   call ptr @_ZN5test21CD1Ev(
  // CHECK:   ret ptr undef

  // CHECK-LABEL: define linkonce_odr void @_ZTv0_n12_N5test21CD0Ev(
  // CHECK:   call void @_ZN5test21CD0Ev(
  // CHECK:   ret void
