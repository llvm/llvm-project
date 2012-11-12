// RUN: %clang_cc1 %s -triple=thumbv7-apple-ios3.0 -fno-use-cxa-atexit -target-abi apcs-gnu -emit-llvm -o - -fexceptions | FileCheck %s

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
// rdar://7817590
bar baz;

// PR9593
// Make sure atexit(3) is used for global dtors.

// CHECK:      call [[BAR:%.*]]* @_ZN3barC1Ev(
// CHECK-NEXT: call i32 @atexit(void ()* @__dtor_baz)

// CHECK: define internal void @__dtor_baz()
// CHECK: call [[BAR]]* @_ZN3barD1Ev([[BAR]]* @baz)

// Destructors and constructors must return this.
namespace test1 {
  void foo();

  struct A {
    A(int i) { foo(); }
    ~A() { foo(); }
    void bar() { foo(); }
  };

  // CHECK: define void @_ZN5test14testEv()
  void test() {
    // CHECK: [[AV:%.*]] = alloca [[A:%.*]], align 1
    // CHECK: call [[A]]* @_ZN5test11AC1Ei([[A]]* [[AV]], i32 10)
    // CHECK: invoke void @_ZN5test11A3barEv([[A]]* [[AV]])
    // CHECK: call [[A]]* @_ZN5test11AD1Ev([[A]]* [[AV]])
    // CHECK: ret void
    A a = 10;
    a.bar();
  }

  // CHECK: define linkonce_odr [[A]]* @_ZN5test11AC1Ei([[A]]* %this, i32 %i) unnamed_addr
  // CHECK:   [[THIS:%.*]] = alloca [[A]]*, align 4
  // CHECK:   store [[A]]* {{.*}}, [[A]]** [[THIS]]
  // CHECK:   [[THIS1:%.*]] = load [[A]]** [[THIS]]
  // CHECK:   call [[A]]* @_ZN5test11AC2Ei(
  // CHECK:   ret [[A]]* [[THIS1]]

  // CHECK: define linkonce_odr [[A]]* @_ZN5test11AD1Ev([[A]]* %this) unnamed_addr
  // CHECK:   [[THIS:%.*]] = alloca [[A]]*, align 4
  // CHECK:   store [[A]]* {{.*}}, [[A]]** [[THIS]]
  // CHECK:   [[THIS1:%.*]] = load [[A]]** [[THIS]]
  // CHECK:   call [[A]]* @_ZN5test11AD2Ev(
  // CHECK:   ret [[A]]* [[THIS1]]
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
    // CHECK: define void @_ZN5test31aEv()
    // CHECK: call noalias i8* @_Znam(i32 48)
    // CHECK: store i32 4
    // CHECK: store i32 10
    A *x = new A[10];
  }

  void b(int n) {
    // CHECK: define void @_ZN5test31bEi(
    // CHECK: [[N:%.*]] = load i32*
    // CHECK: @llvm.umul.with.overflow.i32(i32 [[N]], i32 4)
    // CHECK: @llvm.uadd.with.overflow.i32(i32 {{.*}}, i32 8)
    // CHECK: [[OR:%.*]] = or i1
    // CHECK: [[SZ:%.*]] = select i1 [[OR]]
    // CHECK: call noalias i8* @_Znam(i32 [[SZ]])
    // CHECK: store i32 4
    // CHECK: store i32 [[N]]
    A *x = new A[n];
  }

  void c() {
    // CHECK: define void @_ZN5test31cEv()
    // CHECK: call  noalias i8* @_Znam(i32 808)
    // CHECK: store i32 4
    // CHECK: store i32 200
    A (*x)[20] = new A[10][20];
  }

  void d(int n) {
    // CHECK: define void @_ZN5test31dEi(
    // CHECK: [[N:%.*]] = load i32*
    // CHECK: @llvm.umul.with.overflow.i32(i32 [[N]], i32 80)
    // CHECK: [[NE:%.*]] = mul i32 [[N]], 20
    // CHECK: @llvm.uadd.with.overflow.i32(i32 {{.*}}, i32 8)
    // CHECK: [[SZ:%.*]] = select
    // CHECK: call noalias i8* @_Znam(i32 [[SZ]])
    // CHECK: store i32 4
    // CHECK: store i32 [[NE]]
    A (*x)[20] = new A[n][20];
  }

  void e(A *x) {
    // CHECK: define void @_ZN5test31eEPNS_1AE(
    // CHECK: icmp eq {{.*}}, null
    // CHECK: getelementptr {{.*}}, i64 -8
    // CHECK: getelementptr {{.*}}, i64 4
    // CHECK: bitcast {{.*}} to i32*
    // CHECK: load
    // CHECK: invoke {{.*}} @_ZN5test31AD1Ev
    // CHECK: call void @_ZdaPv
    delete [] x;
  }

  void f(A (*x)[20]) {
    // CHECK: define void @_ZN5test31fEPA20_NS_1AE(
    // CHECK: icmp eq {{.*}}, null
    // CHECK: getelementptr {{.*}}, i64 -8
    // CHECK: getelementptr {{.*}}, i64 4
    // CHECK: bitcast {{.*}} to i32*
    // CHECK: load
    // CHECK: invoke {{.*}} @_ZN5test31AD1Ev
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
    // CHECK: define void @_ZN5test41aEv()
    // CHECK: call noalias i8* @_Znam(i32 48)
    // CHECK: store i32 4
    // CHECK: store i32 10
    A *x = new A[10];
  }

  void b(int n) {
    // CHECK: define void @_ZN5test41bEi(
    // CHECK: [[N:%.*]] = load i32*
    // CHECK: @llvm.umul.with.overflow.i32(i32 [[N]], i32 4)
    // CHECK: @llvm.uadd.with.overflow.i32(i32 {{.*}}, i32 8)
    // CHECK: [[SZ:%.*]] = select
    // CHECK: call noalias i8* @_Znam(i32 [[SZ]])
    // CHECK: store i32 4
    // CHECK: store i32 [[N]]
    A *x = new A[n];
  }

  void c() {
    // CHECK: define void @_ZN5test41cEv()
    // CHECK: call  noalias i8* @_Znam(i32 808)
    // CHECK: store i32 4
    // CHECK: store i32 200
    A (*x)[20] = new A[10][20];
  }

  void d(int n) {
    // CHECK: define void @_ZN5test41dEi(
    // CHECK: [[N:%.*]] = load i32*
    // CHECK: @llvm.umul.with.overflow.i32(i32 [[N]], i32 80)
    // CHECK: [[NE:%.*]] = mul i32 [[N]], 20
    // CHECK: @llvm.uadd.with.overflow.i32(i32 {{.*}}, i32 8)
    // CHECK: [[SZ:%.*]] = select
    // CHECK: call noalias i8* @_Znam(i32 [[SZ]])
    // CHECK: store i32 4
    // CHECK: store i32 [[NE]]
    A (*x)[20] = new A[n][20];
  }

  void e(A *x) {
    // CHECK: define void @_ZN5test41eEPNS_1AE(
    // CHECK: [[ALLOC:%.*]] = getelementptr inbounds {{.*}}, i64 -8
    // CHECK: getelementptr inbounds {{.*}}, i64 4
    // CHECK: bitcast
    // CHECK: [[T0:%.*]] = load i32*
    // CHECK: [[T1:%.*]] = mul i32 4, [[T0]]
    // CHECK: [[T2:%.*]] = add i32 [[T1]], 8
    // CHECK: call void @_ZN5test41AdaEPvm(i8* [[ALLOC]], i32 [[T2]])
    delete [] x;
  }

  void f(A (*x)[20]) {
    // CHECK: define void @_ZN5test41fEPA20_NS_1AE(
    // CHECK: [[ALLOC:%.*]] = getelementptr inbounds {{.*}}, i64 -8
    // CHECK: getelementptr inbounds {{.*}}, i64 4
    // CHECK: bitcast
    // CHECK: [[T0:%.*]] = load i32*
    // CHECK: [[T1:%.*]] = mul i32 4, [[T0]]
    // CHECK: [[T2:%.*]] = add i32 [[T1]], 8
    // CHECK: call void @_ZN5test41AdaEPvm(i8* [[ALLOC]], i32 [[T2]])
    delete [] x;
  }
}

// <rdar://problem/8386802>: don't crash
namespace test5 {
  struct A {
    ~A();
  };

  // CHECK: define void @_ZN5test54testEPNS_1AE
  void test(A *a) {
    // CHECK:      [[PTR:%.*]] = alloca [[A:%.*]]*, align 4
    // CHECK-NEXT: store [[A]]* {{.*}}, [[A]]** [[PTR]], align 4
    // CHECK-NEXT: [[TMP:%.*]] = load [[A]]** [[PTR]], align 4
    // CHECK-NEXT: call [[A]]* @_ZN5test51AD1Ev([[A]]* [[TMP]])
    // CHECK-NEXT: ret void
    a->~A();
  }
}

namespace test6 {
  struct A {
    virtual ~A();
  };

  // CHECK: define void @_ZN5test64testEPNS_1AE
  void test(A *a) {
    // CHECK:      [[AVAR:%.*]] = alloca [[A:%.*]]*, align 4
    // CHECK-NEXT: store [[A]]* {{.*}}, [[A]]** [[AVAR]], align 4
    // CHECK-NEXT: [[V:%.*]] = load [[A]]** [[AVAR]], align 4
    // CHECK-NEXT: [[ISNULL:%.*]] = icmp eq [[A]]* [[V]], null
    // CHECK-NEXT: br i1 [[ISNULL]]
    // CHECK:      [[T0:%.*]] = bitcast [[A]]* [[V]] to [[A]]* ([[A]]*)***
    // CHECK-NEXT: [[T1:%.*]] = load [[A]]* ([[A]]*)*** [[T0]]
    // CHECK-NEXT: [[T2:%.*]] = getelementptr inbounds [[A]]* ([[A]]*)** [[T1]], i64 1
    // CHECK-NEXT: [[T3:%.*]] = load [[A]]* ([[A]]*)** [[T2]]
    // CHECK-NEXT: call [[A]]* [[T3]]([[A]]* [[V]])
    // CHECK-NEXT: br label
    // CHECK:      ret void
    delete a;
  }
}

namespace test7 {
  int foo();

  // Static and guard tested at top of file

  // CHECK: define void @_ZN5test74testEv()
  void test() {
    // CHECK:      [[T0:%.*]] = load i32* @_ZGVZN5test74testEvE1x
    // CHECK-NEXT: [[T1:%.*]] = and i32 [[T0]], 1
    // CHECK-NEXT: [[T2:%.*]] = icmp eq i32 [[T1]], 0
    // CHECK-NEXT: br i1 [[T2]]
    //   -> fallthrough, end
    // CHECK:      [[T3:%.*]] = call i32 @__cxa_guard_acquire(i32* @_ZGVZN5test74testEvE1x)
    // CHECK-NEXT: [[T4:%.*]] = icmp ne i32 [[T3]], 0
    // CHECK-NEXT: br i1 [[T4]]
    //   -> fallthrough, end
    // CHECK:      [[INIT:%.*]] = invoke i32 @_ZN5test73fooEv()
    // CHECK:      store i32 [[INIT]], i32* @_ZZN5test74testEvE1x, align 4
    // CHECK-NEXT: call void @__cxa_guard_release(i32* @_ZGVZN5test74testEvE1x)
    // CHECK-NEXT: br label
    //   -> end
    // end:
    // CHECK:      ret void
    static int x = foo();

    // CHECK:      landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
    // CHECK-NEXT:   cleanup
    // CHECK:      call void @__cxa_guard_abort(i32* @_ZGVZN5test74testEvE1x)
    // CHECK:      resume { i8*, i32 }
  }
}

namespace test8 {
  struct A {
    A();
    ~A();
  };

  // Static and guard tested at top of file

  // CHECK: define void @_ZN5test84testEv()
  void test() {
    // CHECK:      [[T0:%.*]] = load i32* @_ZGVZN5test84testEvE1x
    // CHECK-NEXT: [[T1:%.*]] = and i32 [[T0]], 1
    // CHECK-NEXT: [[T2:%.*]] = icmp eq i32 [[T1]], 0
    // CHECK-NEXT: br i1 [[T2]]
    //   -> fallthrough, end
    // CHECK:      [[T3:%.*]] = call i32 @__cxa_guard_acquire(i32* @_ZGVZN5test84testEvE1x)
    // CHECK-NEXT: [[T4:%.*]] = icmp ne i32 [[T3]], 0
    // CHECK-NEXT: br i1 [[T4]]
    //   -> fallthrough, end
    // CHECK:      [[INIT:%.*]] = invoke [[TEST8A]]* @_ZN5test81AC1Ev([[TEST8A]]* @_ZZN5test84testEvE1x)

    // FIXME: Here we register a global destructor that
    // unconditionally calls the destructor.  That's what we've always
    // done for -fno-use-cxa-atexit here, but that's really not
    // semantically correct at all.

    // CHECK:      call void @__cxa_guard_release(i32* @_ZGVZN5test84testEvE1x)
    // CHECK-NEXT: br label
    //   -> end
    // end:
    // CHECK:      ret void
    static A x;

    // CHECK:      landingpad { i8*, i32 } personality i8* bitcast (i32 (...)* @__gxx_personality_v0 to i8*)
    // CHECK-NEXT:   cleanup
    // CHECK:      call void @__cxa_guard_abort(i32* @_ZGVZN5test84testEvE1x)
    // CHECK:      resume { i8*, i32 }
  }
}

  // CHECK: define linkonce_odr [[C:%.*]]* @_ZTv0_n12_N5test21CD1Ev(
  // CHECK:   call [[C]]* @_ZN5test21CD1Ev(
  // CHECK:   ret [[C]]* undef

  // CHECK: define linkonce_odr void @_ZTv0_n12_N5test21CD0Ev(
  // CHECK:   call void @_ZN5test21CD0Ev(
  // CHECK:   ret void

// CH_ECK: @_GLOBAL__D_a()
// CH_ECK: call %class.bar* @_ZN3barD1Ev(%class.bar* @baz)
