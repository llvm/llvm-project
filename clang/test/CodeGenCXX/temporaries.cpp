// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 -std=c++11 | FileCheck %s -check-prefixes=CHECK,NULL-INVALID,CHECK-CXX11
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 -std=c++17 | FileCheck %s -check-prefixes=CHECK,NULL-INVALID,CHECK-CXX17
// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 -std=c++11 -fno-delete-null-pointer-checks | FileCheck %s -check-prefixes=CHECK,NULL-VALID,CHECK-CXX11

namespace PR16263 {
  const unsigned int n = 1234;
  extern const int &r = (const int&)n;
  // CHECK: @_ZGRN7PR162631rE_ = internal constant i32 1234,
  // CHECK: @_ZN7PR162631rE ={{.*}} constant ptr @_ZGRN7PR162631rE_,

  extern const int &s = reinterpret_cast<const int&>(n);
  // CHECK: @_ZN7PR16263L1nE = internal constant i32 1234, align 4
  // CHECK: @_ZN7PR162631sE ={{.*}} constant ptr @_ZN7PR16263L1nE, align 8

  struct A { int n; };
  struct B { int n; };
  struct C : A, B {};
  extern const A &&a = (A&&)(A&&)(C&&)(C{});
  // CHECK: @_ZGRN7PR162631aE_ = internal global {{.*}} zeroinitializer,
  // CHECK: @_ZN7PR162631aE ={{.*}} constant {{.*}} @_ZGRN7PR162631aE_

  extern const int &&t = ((B&&)C{}).n;
  // CHECK: @_ZGRN7PR162631tE_ = internal global {{.*}} zeroinitializer,
  // CHECK: @_ZN7PR162631tE ={{.*}} constant ptr {{.*}} @_ZGRN7PR162631tE_, {{.*}} 4

  struct D { double d; C c; };
  extern const int &&u = (123, static_cast<B&&>(0, ((D&&)D{}).*&D::c).n);
  // CHECK: @_ZGRN7PR162631uE_ = internal global {{.*}} zeroinitializer
  // CHECK: @_ZN7PR162631uE ={{.*}} constant ptr {{.*}} @_ZGRN7PR162631uE_, {{.*}} 12
}

namespace PR20227 {
  struct A { ~A(); };
  struct B { virtual ~B(); };
  struct C : B {};

  A &&a = dynamic_cast<A&&>(A{});
  // CHECK: @_ZGRN7PR202271aE_ = internal global

  B &&b = dynamic_cast<C&&>(dynamic_cast<B&&>(C{}));
  // CHECK: @_ZGRN7PR202271bE_ = internal global

  B &&c = static_cast<C&&>(static_cast<B&&>(C{}));
  // CHECK: @_ZGRN7PR202271cE_ = internal global
}

namespace BraceInit {
  typedef const int &CIR;
  CIR x = CIR{3};
  // CHECK-CXX11: @_ZGRN9BraceInit1xE_ = internal constant i32 3
  // FIXME: This should still be emitted as 'constant' in C++17.
  // CHECK-CXX17: @_ZGRN9BraceInit1xE_ = internal global i32 3
  // CHECK: @_ZN9BraceInit1xE ={{.*}} constant ptr @_ZGRN9BraceInit1xE_
}

namespace RefTempSubobject {
  struct SelfReferential {
    int *p = ints;
    int ints[3] = {1, 2, 3};
  };

  // CHECK: @_ZGRN16RefTempSubobject2srE_ = internal global { ptr, [3 x i32] } { {{.*}} getelementptr {{.*}} @_ZGRN16RefTempSubobject2srE_, {{.*}}, [3 x i32] [i32 1, i32 2, i32 3] }
  // CHECK: @_ZN16RefTempSubobject2srE = constant {{.*}} @_ZGRN16RefTempSubobject2srE_
  constexpr const SelfReferential &sr = SelfReferential();
}

struct A {
  A();
  ~A();
  void f();
};

void f1() {
  // CHECK: call void @_ZN1AC1Ev
  // CHECK: call void @_ZN1AD1Ev
  (void)A();

  // CHECK: call void @_ZN1AC1Ev
  // CHECK: call void @_ZN1AD1Ev
  A().f();
}

// Function calls
struct B {
  B();
  ~B();
};

B g();

void f2() {
  // CHECK-NOT: call void @_ZN1BC1Ev
  // CHECK: call void @_ZN1BD1Ev
  (void)g();
}

// Member function calls
struct C {
  C();
  ~C();

  C f();
};

void f3() {
  // CHECK: call void @_ZN1CC1Ev
  // CHECK: call void @_ZN1CD1Ev
  // CHECK: call void @_ZN1CD1Ev
  C().f();
}

// Function call operator
struct D {
  D();
  ~D();

  D operator()();
};

void f4() {
  // CHECK: call void @_ZN1DC1Ev
  // CHECK: call void @_ZN1DD1Ev
  // CHECK: call void @_ZN1DD1Ev
  D()();
}

// Overloaded operators
struct E {
  E();
  ~E();
  E operator+(const E&);
  E operator!();
};

void f5() {
  // CHECK: call void @_ZN1EC1Ev
  // CHECK: call void @_ZN1EC1Ev
  // CHECK: call void @_ZN1ED1Ev
  // CHECK: call void @_ZN1ED1Ev
  // CHECK: call void @_ZN1ED1Ev
  E() + E();

  // CHECK: call void @_ZN1EC1Ev
  // CHECK: call void @_ZN1ED1Ev
  // CHECK: call void @_ZN1ED1Ev
  !E();
}

struct F {
  F();
  ~F();
  F& f();
};

void f6() {
  // CHECK: call void @_ZN1FC1Ev
  // CHECK: call void @_ZN1FD1Ev
  F().f();
}

struct G {
  G();
  G(A);
  ~G();
  operator A();
};

void a(const A&);

void f7() {
  // CHECK: call void @_ZN1AC1Ev
  // CHECK: call void @_Z1aRK1A
  // CHECK: call void @_ZN1AD1Ev
  a(A());

  // CHECK: call void @_ZN1GC1Ev
  // CHECK: call void @_ZN1Gcv1AEv
  // CHECK: call void @_Z1aRK1A
  // CHECK: call void @_ZN1AD1Ev
  // CHECK: call void @_ZN1GD1Ev
  a(G());
}

namespace PR5077 {

struct A {
  A();
  ~A();
  int f();
};

void f();
int g(const A&);

struct B {
  int a1;
  int a2;
  B();
  ~B();
};

B::B()
  // CHECK: call void @_ZN6PR50771AC1Ev
  // CHECK: call noundef i32 @_ZN6PR50771A1fEv
  // CHECK: call void @_ZN6PR50771AD1Ev
  : a1(A().f())
  // CHECK: call void @_ZN6PR50771AC1Ev
  // CHECK: call noundef i32 @_ZN6PR50771gERKNS_1AE
  // CHECK: call void @_ZN6PR50771AD1Ev
  , a2(g(A()))
{
  // CHECK: call void @_ZN6PR50771fEv
  f();
}

}

A f8() {
  // CHECK: call void @_ZN1AC1Ev
  // CHECK-NOT: call void @_ZN1AD1Ev
  return A();
  // CHECK: ret void
}

struct H {
  H();
  ~H();
  H(const H&);
};

void f9(H h) {
  // CHECK: call void @_ZN1HC1Ev
  // CHECK: call void @_Z2f91H
  // CHECK: call void @_ZN1HD1Ev
  f9(H());

  // CHECK: call void @_ZN1HC1ERKS_
  // CHECK: call void @_Z2f91H
  // CHECK: call void @_ZN1HD1Ev
  f9(h);
}

void f10(const H&);

void f11(H h) {
  // CHECK: call void @_ZN1HC1Ev
  // CHECK: call void @_Z3f10RK1H
  // CHECK: call void @_ZN1HD1Ev
  f10(H());

  // CHECK: call void @_Z3f10RK1H
  // CHECK-NOT: call void @_ZN1HD1Ev
  // CHECK: ret void
  f10(h);
}

// PR5808
struct I {
  I(const char *);
  ~I();
};

// CHECK: _Z3f12v
I f12() {
  // CHECK: call void @_ZN1IC1EPKc
  // CHECK-NOT: call void @_ZN1ID1Ev
  // CHECK: ret void
  return "Hello";
}

// PR5867
namespace PR5867 {
  struct S {
    S();
    S(const S &);
    ~S();
  };

  void f(S, int);
  // CHECK-LABEL: define{{.*}} void @_ZN6PR58671gEv
  void g() {
    // CHECK: call void @_ZN6PR58671SC1Ev
    // CHECK-NEXT: call void @_ZN6PR58671fENS_1SEi
    // CHECK-NEXT: call void @_ZN6PR58671SD1Ev
    // CHECK-NEXT: ret void
    (f)(S(), 0);
  }

  // CHECK-LABEL: define linkonce_odr void @_ZN6PR58672g2IiEEvT_
  template<typename T>
  void g2(T) {
    // CHECK: call void @_ZN6PR58671SC1Ev
    // CHECK-NEXT: call void @_ZN6PR58671fENS_1SEi
    // CHECK-NEXT: call void @_ZN6PR58671SD1Ev
    // CHECK-NEXT: ret void
    (f)(S(), 0);
  }

  void h() {
    g2(17);
  }
}

// PR6199
namespace PR6199 {
  struct A { ~A(); };

  struct B { operator A(); };

  // CHECK-LABEL: define weak_odr void @_ZN6PR61992f2IiEENS_1AET_
  template<typename T> A f2(T) {
    B b;
    // CHECK: call void @_ZN6PR61991BcvNS_1AEEv
    // CHECK-NEXT: ret void
    return b;
  }

  template A f2<int>(int);

}

namespace T12 {

struct A {
  A();
  ~A();
  int f();
};

int& f(int);

// CHECK-LABEL: define{{.*}} void @_ZN3T121gEv
void g() {
  // CHECK: call void @_ZN3T121AC1Ev
  // CHECK-NEXT: call noundef i32 @_ZN3T121A1fEv(
  // CHECK-NEXT: call noundef {{(nonnull )?}}align {{[0-9]+}} dereferenceable({{[0-9]+}}) ptr @_ZN3T121fEi(
  // CHECK-NEXT: call void @_ZN3T121AD1Ev(
  int& i = f(A().f());
}

}

namespace PR6648 {
  struct B {
    ~B();
  };
  B foo;
  struct D;
  D& zed(B);
  void foobar() {
    // NULL-INVALID: call noundef nonnull align 1 ptr @_ZN6PR66483zedENS_1BE
    // NULL-VALID: call noundef align 1 ptr @_ZN6PR66483zedENS_1BE
    zed(foo);
  }
}

namespace UserConvertToValue {
  struct X {
    X(int);
    X(const X&);
    ~X();
  };

  void f(X);

  // CHECK: void @_ZN18UserConvertToValue1gEv()
  void g() {
    // CHECK: call void @_ZN18UserConvertToValue1XC1Ei
    // CHECK: call void @_ZN18UserConvertToValue1fENS_1XE
    // CHECK: call void @_ZN18UserConvertToValue1XD1Ev
    // CHECK: ret void
    f(1);
  }
}

namespace PR7556 {
  struct A { ~A(); };
  struct B { int i; ~B(); };
  struct C { int C::*pm; ~C(); };
  // CHECK-LABEL: define{{.*}} void @_ZN6PR75563fooEv()
  void foo() {
    // CHECK: call void @_ZN6PR75561AD1Ev
    A();
    // CHECK: call void @llvm.memset.p0.i64
    // CHECK: call void @_ZN6PR75561BD1Ev
    B();
    // CHECK: call void @llvm.memcpy.p0.p0.i64
    // CHECK: call void @_ZN6PR75561CD1Ev
    C();
    // CHECK-NEXT: ret void
  }
}

namespace Elision {
  struct A {
    A(); A(const A &); ~A();
    void *p;
    void foo() const;
  };

  void foo();
  A fooA();
  void takeA(A a);

  // CHECK-LABEL: define{{.*}} void @_ZN7Elision5test0Ev()
  void test0() {
    // CHECK:      [[I:%.*]] = alloca [[A:%.*]], align 8
    // CHECK-NEXT: [[J:%.*]] = alloca [[A]], align 8
    // CHECK-NEXT: [[T0:%.*]] = alloca [[A]], align 8
    // CHECK-NEXT: [[K:%.*]] = alloca [[A]], align 8
    // CHECK-NEXT: [[T1:%.*]] = alloca [[A]], align 8

    // CHECK-NEXT: call void @_ZN7Elision3fooEv()
    // CHECK-NEXT: call void @_ZN7Elision1AC1Ev(ptr {{[^,]*}} [[I]])
    A i = (foo(), A());

    // CHECK-NEXT: call void @_ZN7Elision4fooAEv(ptr sret([[A]]) align 8 [[T0]])
    // CHECK-NEXT: call void @_ZN7Elision1AC1Ev(ptr {{[^,]*}} [[J]])
    // CHECK-NEXT: call void @_ZN7Elision1AD1Ev(ptr {{[^,]*}} [[T0]])
    A j = (fooA(), A());

    // CHECK-NEXT: call void @_ZN7Elision1AC1Ev(ptr {{[^,]*}} [[T1]])
    // CHECK-NEXT: call void @_ZN7Elision4fooAEv(ptr sret([[A]]) align 8 [[K]])
    // CHECK-NEXT: call void @_ZN7Elision1AD1Ev(ptr {{[^,]*}} [[T1]])
    A k = (A(), fooA());

    // CHECK-NEXT: call void @_ZN7Elision1AD1Ev(ptr {{[^,]*}} [[K]])
    // CHECK-NEXT: call void @_ZN7Elision1AD1Ev(ptr {{[^,]*}} [[J]])
    // CHECK-NEXT: call void @_ZN7Elision1AD1Ev(ptr {{[^,]*}} [[I]])
  }


  // CHECK-LABEL: define{{.*}} void @_ZN7Elision5test1EbNS_1AE(
  void test1(bool c, A x) {
    // CHECK:      [[I:%.*]] = alloca [[A]], align 8
    // CHECK-NEXT: [[J:%.*]] = alloca [[A]], align 8

    // CHECK:      call void @_ZN7Elision1AC1Ev(ptr {{[^,]*}} [[I]])
    // CHECK:      call void @_ZN7Elision1AC1ERKS0_(ptr {{[^,]*}} [[I]], ptr noundef {{(nonnull )?}}align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[X:%.*]])
    A i = (c ? A() : x);

    // CHECK:      call void @_ZN7Elision1AC1ERKS0_(ptr {{[^,]*}} [[J]], ptr noundef {{(nonnull )?}}align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[X]])
    // CHECK:      call void @_ZN7Elision1AC1Ev(ptr {{[^,]*}} [[J]])
    A j = (c ? x : A());

    // CHECK:      call void @_ZN7Elision1AD1Ev(ptr {{[^,]*}} [[J]])
    // CHECK-NEXT: call void @_ZN7Elision1AD1Ev(ptr {{[^,]*}} [[I]])
  }

  // CHECK: define{{.*}} void @_ZN7Elision5test2Ev(ptr noalias sret([[A]]) align 8
  A test2() {
    // CHECK:      call void @_ZN7Elision3fooEv()
    // CHECK-NEXT: call void @_ZN7Elision1AC1Ev(ptr {{[^,]*}} [[RET:%.*]])
    // CHECK-NEXT: ret void
    return (foo(), A());
  }

  // CHECK: define{{.*}} void @_ZN7Elision5test3EiNS_1AE(ptr noalias sret([[A]]) align 8
  A test3(int v, A x) {
    if (v < 5)
    // CHECK:      call void @_ZN7Elision1AC1Ev(ptr {{[^,]*}} [[RET:%.*]])
    // CHECK:      call void @_ZN7Elision1AC1ERKS0_(ptr {{[^,]*}} [[RET]], ptr noundef {{(nonnull )?}}align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[X:%.*]])
      return (v < 0 ? A() : x);
    else
    // CHECK:      call void @_ZN7Elision1AC1ERKS0_(ptr {{[^,]*}} [[RET]], ptr noundef {{(nonnull )?}}align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[X]])
    // CHECK:      call void @_ZN7Elision1AC1Ev(ptr {{[^,]*}} [[RET]])
      return (v > 10 ? x : A());

    // CHECK:      ret void
  }

  // CHECK-LABEL: define{{.*}} void @_ZN7Elision5test4Ev()
  void test4() {
    // CHECK:      [[X:%.*]] = alloca [[A]], align 8
    // CHECK-NEXT: [[XS:%.*]] = alloca [2 x [[A]]], align 16

    // CHECK-NEXT: call void @_ZN7Elision1AC1Ev(ptr {{[^,]*}} [[X]])
    A x;

    // CHECK-NEXT: [[XS0:%.*]] = getelementptr inbounds [2 x [[A]]], ptr [[XS]], i64 0, i64 0
    // CHECK-NEXT: call void @_ZN7Elision1AC1Ev(ptr {{[^,]*}} [[XS0]])
    // CHECK-NEXT: [[XS1:%.*]] = getelementptr inbounds [[A]], ptr [[XS0]], i64 1
    // CHECK-NEXT: call void @_ZN7Elision1AC1ERKS0_(ptr {{[^,]*}} [[XS1]], ptr noundef {{(nonnull )?}}align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[X]])
    A xs[] = { A(), x };

    // CHECK-NEXT: [[BEGIN:%.*]] = getelementptr inbounds [2 x [[A]]], ptr [[XS]], i32 0, i32 0
    // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds [[A]], ptr [[BEGIN]], i64 2
    // CHECK-NEXT: br label
    // CHECK:      [[AFTER:%.*]] = phi ptr
    // CHECK-NEXT: [[CUR:%.*]] = getelementptr inbounds [[A]], ptr [[AFTER]], i64 -1
    // CHECK-NEXT: call void @_ZN7Elision1AD1Ev(ptr {{[^,]*}} [[CUR]])
    // CHECK-NEXT: [[T0:%.*]] = icmp eq ptr [[CUR]], [[BEGIN]]
    // CHECK-NEXT: br i1 [[T0]],

    // CHECK:      call void @_ZN7Elision1AD1Ev(ptr {{[^,]*}} [[X]])
  }

  // rdar://problem/8433352
  // CHECK: define{{.*}} void @_ZN7Elision5test5Ev(ptr noalias sret([[A]]) align 8
  struct B { A a; B(); };
  A test5() {
    // CHECK:      [[AT0:%.*]] = alloca [[A]], align 8
    // CHECK-NEXT: [[BT0:%.*]] = alloca [[B:%.*]], align 8
    // CHECK-NEXT: [[X:%.*]] = alloca [[A]], align 8
    // CHECK-NEXT: [[BT1:%.*]] = alloca [[B]], align 8
    // CHECK-NEXT: [[BT2:%.*]] = alloca [[B]], align 8

    // CHECK:      call void @_ZN7Elision1BC1Ev(ptr {{[^,]*}} [[BT0]])
    // CHECK-NEXT: [[AM:%.*]] = getelementptr inbounds [[B]], ptr [[BT0]], i32 0, i32 0
    // CHECK-NEXT: call void @_ZN7Elision1AC1ERKS0_(ptr {{[^,]*}} [[AT0]], ptr noundef {{(nonnull )?}}align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[AM]])
    // CHECK-NEXT: call void @_ZN7Elision5takeAENS_1AE(ptr noundef [[AT0]])
    // CHECK-NEXT: call void @_ZN7Elision1AD1Ev(ptr {{[^,]*}} [[AT0]])
    // CHECK-NEXT: call void @_ZN7Elision1BD1Ev(ptr {{[^,]*}} [[BT0]])
    takeA(B().a);

    // CHECK-NEXT: call void @_ZN7Elision1BC1Ev(ptr {{[^,]*}} [[BT1]])
    // CHECK-NEXT: [[AM:%.*]] = getelementptr inbounds [[B]], ptr [[BT1]], i32 0, i32 0
    // CHECK-NEXT: call void @_ZN7Elision1AC1ERKS0_(ptr {{[^,]*}} [[X]], ptr noundef {{(nonnull )?}}align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[AM]])
    // CHECK-NEXT: call void @_ZN7Elision1BD1Ev(ptr {{[^,]*}} [[BT1]])
    A x = B().a;

    // CHECK-NEXT: call void @_ZN7Elision1BC1Ev(ptr {{[^,]*}} [[BT2]])
    // CHECK-NEXT: [[AM:%.*]] = getelementptr inbounds [[B]], ptr [[BT2]], i32 0, i32 0
    // CHECK-NEXT: call void @_ZN7Elision1AC1ERKS0_(ptr {{[^,]*}} [[RET:%.*]], ptr noundef {{(nonnull )?}}align {{[0-9]+}} dereferenceable({{[0-9]+}}) [[AM]])
    // CHECK-NEXT: call void @_ZN7Elision1BD1Ev(ptr {{[^,]*}} [[BT2]])
    return B().a;

    // CHECK:      call void @_ZN7Elision1AD1Ev(ptr {{[^,]*}} [[X]])
  }

  // Reduced from webkit.
  // CHECK: define{{.*}} void @_ZN7Elision5test6EPKNS_1CE(ptr
  struct C { operator A() const; };
  void test6(const C *x) {
    // CHECK:      [[T0:%.*]] = alloca [[A]], align 8
    // CHECK:      [[X:%.*]] = load ptr, ptr {{%.*}}, align 8
    // CHECK-NEXT: call void @_ZNK7Elision1CcvNS_1AEEv(ptr sret([[A]]) align 8 [[T0]], ptr {{[^,]*}} [[X]])
    // CHECK-NEXT: call void @_ZNK7Elision1A3fooEv(ptr {{[^,]*}} [[T0]])
    // CHECK-NEXT: call void @_ZN7Elision1AD1Ev(ptr {{[^,]*}} [[T0]])
    // CHECK-NEXT: ret void
    A(*x).foo();
  }
}

namespace PR8623 {
  struct A { A(int); ~A(); };

  // CHECK-LABEL: define{{.*}} void @_ZN6PR86233fooEb(
  void foo(bool b) {
    // CHECK:      [[TMP:%.*]] = alloca [[A:%.*]], align 1
    // CHECK-NEXT: [[LCONS:%.*]] = alloca i1
    // CHECK-NEXT: [[RCONS:%.*]] = alloca i1
    // CHECK:      store i1 false, ptr [[LCONS]]
    // CHECK-NEXT: store i1 false, ptr [[RCONS]]
    // CHECK-NEXT: br i1
    // CHECK:      call void @_ZN6PR86231AC1Ei(ptr {{[^,]*}} [[TMP]], i32 noundef 2)
    // CHECK-NEXT: store i1 true, ptr [[LCONS]]
    // CHECK-NEXT: br label
    // CHECK:      call void @_ZN6PR86231AC1Ei(ptr {{[^,]*}} [[TMP]], i32 noundef 3)
    // CHECK-NEXT: store i1 true, ptr [[RCONS]]
    // CHECK-NEXT: br label
    // CHECK:      load i1, ptr [[RCONS]]
    // CHECK-NEXT: br i1
    // CHECK:      call void @_ZN6PR86231AD1Ev(ptr {{[^,]*}} [[TMP]])
    // CHECK-NEXT: br label
    // CHECK:      load i1, ptr [[LCONS]]
    // CHECK-NEXT: br i1
    // CHECK:      call void @_ZN6PR86231AD1Ev(ptr {{[^,]*}} [[TMP]])
    // CHECK-NEXT: br label
    // CHECK:      ret void
    b ? A(2) : A(3);
  }
}

namespace PR11365 {
  struct A { A(); ~A(); };

  // CHECK-LABEL: define{{.*}} void @_ZN7PR113653fooEv(
  void foo() {
    // CHECK: [[BEGIN:%.*]] = getelementptr inbounds [3 x [[A:%.*]]], ptr {{.*}}, i32 0, i32 0
    // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds [[A]], ptr [[BEGIN]], i64 3
    // CHECK-NEXT: br label

    // CHECK: [[PHI:%.*]] = phi
    // CHECK-NEXT: [[ELEM:%.*]] = getelementptr inbounds [[A]], ptr [[PHI]], i64 -1
    // CHECK-NEXT: call void @_ZN7PR113651AD1Ev(ptr {{[^,]*}} [[ELEM]])
    // CHECK-NEXT: icmp eq ptr [[ELEM]], [[BEGIN]]
    // CHECK-NEXT: br i1
    (void) (A [3]) {};
  }
}

namespace AssignmentOp {
  struct A { ~A(); };
  struct B { A operator=(const B&); };
  struct C : B { B b1, b2; };
  // CHECK-LABEL: define{{.*}} void @_ZN12AssignmentOp1fE
  void f(C &c1, const C &c2) {
    // CHECK: call {{.*}} @_ZN12AssignmentOp1CaSERKS0_(
    c1 = c2;
  }

  // Ensure that each 'A' temporary is destroyed before the next subobject is
  // copied.
  // CHECK: define {{.*}} @_ZN12AssignmentOp1CaSERKS0_(
  // CHECK: call {{.*}} @_ZN12AssignmentOp1BaSERKS
  // CHECK: call {{.*}} @_ZN12AssignmentOp1AD1Ev(
  // CHECK: call {{.*}} @_ZN12AssignmentOp1BaSERKS
  // CHECK: call {{.*}} @_ZN12AssignmentOp1AD1Ev(
  // CHECK: call {{.*}} @_ZN12AssignmentOp1BaSERKS
  // CHECK: call {{.*}} @_ZN12AssignmentOp1AD1Ev(
}

namespace BindToSubobject {
  struct A {
    A();
    ~A();
    int a;
  };

  void f(), g();

  // CHECK: call void @_ZN15BindToSubobject1AC1Ev({{.*}} @_ZGRN15BindToSubobject1aE_)
  // CHECK: call i32 @__cxa_atexit({{.*}} @_ZN15BindToSubobject1AD1Ev, ptr @_ZGRN15BindToSubobject1aE_, ptr @__dso_handle)
  // CHECK: store ptr @_ZGRN15BindToSubobject1aE_, ptr @_ZN15BindToSubobject1aE, align 8
  int &&a = A().a;

  // CHECK: call void @_ZN15BindToSubobject1fEv()
  // CHECK: call void @_ZN15BindToSubobject1AC1Ev({{.*}} @_ZGRN15BindToSubobject1bE_)
  // CHECK: call i32 @__cxa_atexit({{.*}} @_ZN15BindToSubobject1AD1Ev, ptr @_ZGRN15BindToSubobject1bE_, ptr @__dso_handle)
  // CHECK: store ptr @_ZGRN15BindToSubobject1bE_, ptr @_ZN15BindToSubobject1bE, align 8
  int &&b = (f(), A().a);

  int A::*h();

  // CHECK: call void @_ZN15BindToSubobject1fEv()
  // CHECK: call void @_ZN15BindToSubobject1gEv()
  // CHECK: call void @_ZN15BindToSubobject1AC1Ev({{.*}} @_ZGRN15BindToSubobject1cE_)
  // CHECK: call i32 @__cxa_atexit({{.*}} @_ZN15BindToSubobject1AD1Ev, ptr @_ZGRN15BindToSubobject1cE_, ptr @__dso_handle)
  // CHECK: call {{.*}} @_ZN15BindToSubobject1hE
  // CHECK: getelementptr
  // CHECK: store ptr {{.*}}, ptr @_ZN15BindToSubobject1cE, align 8
  int &&c = (f(), (g(), A().*h()));

  struct B {
    int padding;
    A a;
  };

  // CHECK: call void @_ZN15BindToSubobject1BC1Ev({{.*}} @_ZGRN15BindToSubobject1dE_)
  // CHECK: call i32 @__cxa_atexit({{.*}} @_ZN15BindToSubobject1BD1Ev, ptr @_ZGRN15BindToSubobject1dE_, ptr @__dso_handle)
  // CHECK: call {{.*}} @_ZN15BindToSubobject1hE
  // CHECK: getelementptr {{.*}} getelementptr
  // CHECK: store ptr {{.*}}, ptr @_ZN15BindToSubobject1dE, align 8
  int &&d = (B().a).*h();
}

namespace Bitfield {
  struct S { int a : 5; ~S(); };

  // Do not lifetime extend the S() temporary here.
  // CHECK: alloca
  // CHECK: call {{.*}}memset
  // CHECK: store i32 {{.*}}, ptr @_ZGRN8Bitfield1rE_
  // CHECK: call void @_ZN8Bitfield1SD1
  // CHECK: store ptr @_ZGRN8Bitfield1rE_, ptr @_ZN8Bitfield1rE, align 8
  int &&r = S().a;
}

namespace Vector {
  typedef __attribute__((vector_size(16))) int vi4a;
  typedef __attribute__((ext_vector_type(4))) int vi4b;
  struct S {
    vi4a v;
    vi4b w;
  };
  // CHECK: alloca
  // CHECK: extractelement
  // CHECK: store i32 {{.*}}, ptr @_ZGRN6Vector1rE_
  // CHECK: store ptr @_ZGRN6Vector1rE_, ptr @_ZN6Vector1rE,
  int &&r = S().v[1];

  // CHECK: alloca
  // CHECK: extractelement
  // CHECK: store i32 {{.*}}, ptr @_ZGRN6Vector1sE_
  // CHECK: store ptr @_ZGRN6Vector1sE_, ptr @_ZN6Vector1sE,
  int &&s = S().w[1];
  // FIXME PR16204: The following code leads to an assertion in Sema.
  //int &&s = S().w.y;
}

namespace ImplicitTemporaryCleanup {
  struct A { A(int); ~A(); };
  void g();

  // CHECK-LABEL: define{{.*}} void @_ZN24ImplicitTemporaryCleanup1fEv(
  void f() {
    // CHECK: call {{.*}} @_ZN24ImplicitTemporaryCleanup1AC1Ei(
    A &&a = 0;

    // CHECK: call {{.*}} @_ZN24ImplicitTemporaryCleanup1gEv(
    g();

    // CHECK: call {{.*}} @_ZN24ImplicitTemporaryCleanup1AD1Ev(
  }
}

namespace MultipleExtension {
  struct A { A(); ~A(); };
  struct B { B(); ~B(); };
  struct C { C(); ~C(); };
  struct D { D(); ~D(); int n; C c; };
  struct E { const A &a; B b; const C &c; ~E(); };

  E &&e1 = { A(), B(), D().c };

  // CHECK: call void @_ZN17MultipleExtension1AC1Ev({{.*}} @[[TEMPA:_ZGRN17MultipleExtension2e1E.*]])
  // CHECK: call i32 @__cxa_atexit({{.*}} @_ZN17MultipleExtension1AD1Ev, {{.*}} @[[TEMPA]]
  // CHECK: store {{.*}} @[[TEMPA]], {{.*}} @[[TEMPE:_ZGRN17MultipleExtension2e1E.*]],

  // CHECK: call void @_ZN17MultipleExtension1BC1Ev({{.*}} getelementptr inbounds ({{.*}} @[[TEMPE]], i32 0, i32 1))

  // CHECK: call void @_ZN17MultipleExtension1DC1Ev({{.*}} @[[TEMPD:_ZGRN17MultipleExtension2e1E.*]])
  // CHECK: call i32 @__cxa_atexit({{.*}} @_ZN17MultipleExtension1DD1Ev, {{.*}} @[[TEMPD]]
  // CHECK: store {{.*}} @[[TEMPD]], {{.*}} getelementptr inbounds ({{.*}} @[[TEMPE]], i32 0, i32 2)
  // CHECK: call i32 @__cxa_atexit({{.*}} @_ZN17MultipleExtension1ED1Ev, {{.*}} @[[TEMPE]]
  // CHECK: store {{.*}} @[[TEMPE]], ptr @_ZN17MultipleExtension2e1E, align 8

  E e2 = { A(), B(), D().c };

  // CHECK: call void @_ZN17MultipleExtension1AC1Ev({{.*}} @[[TEMPA:_ZGRN17MultipleExtension2e2E.*]])
  // CHECK: call i32 @__cxa_atexit({{.*}} @_ZN17MultipleExtension1AD1Ev, {{.*}} @[[TEMPA]]
  // CHECK: store {{.*}} @[[TEMPA]], {{.*}} @[[E:_ZN17MultipleExtension2e2E]]

  // CHECK: call void @_ZN17MultipleExtension1BC1Ev({{.*}} getelementptr inbounds ({{.*}} @[[E]], i32 0, i32 1))

  // CHECK: call void @_ZN17MultipleExtension1DC1Ev({{.*}} @[[TEMPD:_ZGRN17MultipleExtension2e2E.*]])
  // CHECK: call i32 @__cxa_atexit({{.*}} @_ZN17MultipleExtension1DD1Ev, {{.*}} @[[TEMPD]]
  // CHECK: store {{.*}} @[[TEMPD]], {{.*}} getelementptr inbounds ({{.*}} @[[E]], i32 0, i32 2)
  // CHECK: call i32 @__cxa_atexit({{.*}} @_ZN17MultipleExtension1ED1Ev, {{.*}} @[[E]]


  void g();
  // CHECK: define{{.*}} void @[[NS:_ZN17MultipleExtension]]1fEv(
  void f() {
    E &&e1 = { A(), B(), D().c };
    // CHECK: %[[TEMPE1_A:.*]] = getelementptr inbounds {{.*}} %[[TEMPE1:.*]], i32 0, i32 0
    // CHECK: call void @[[NS]]1AC1Ev({{.*}} %[[TEMPA1:.*]])
    // CHECK: store {{.*}} %[[TEMPA1]], {{.*}} %[[TEMPE1_A]]
    // CHECK: %[[TEMPE1_B:.*]] = getelementptr inbounds {{.*}} %[[TEMPE1]], i32 0, i32 1
    // CHECK: call void @[[NS]]1BC1Ev({{.*}} %[[TEMPE1_B]])
    // CHECK: %[[TEMPE1_C:.*]] = getelementptr inbounds {{.*}} %[[TEMPE1]], i32 0, i32 2
    // CHECK: call void @[[NS]]1DC1Ev({{.*}} %[[TEMPD1:.*]])
    // CHECK: %[[TEMPD1_C:.*]] = getelementptr inbounds {{.*}} %[[TEMPD1]], i32 0, i32 1
    // CHECK: store {{.*}} %[[TEMPD1_C]], {{.*}} %[[TEMPE1_C]]
    // CHECK: store {{.*}} %[[TEMPE1]], {{.*}} %[[E1:.*]]

    g();
    // CHECK: call void @[[NS]]1gEv()

    E e2 = { A(), B(), D().c };
    // CHECK: %[[TEMPE2_A:.*]] = getelementptr inbounds {{.*}} %[[E2:.*]], i32 0, i32 0
    // CHECK: call void @[[NS]]1AC1Ev({{.*}} %[[TEMPA2:.*]])
    // CHECK: store {{.*}} %[[TEMPA2]], {{.*}} %[[TEMPE2_A]]
    // CHECK: %[[TEMPE2_B:.*]] = getelementptr inbounds {{.*}} %[[E2]], i32 0, i32 1
    // CHECK: call void @[[NS]]1BC1Ev({{.*}} %[[TEMPE2_B]])
    // CHECK: %[[TEMPE2_C:.*]] = getelementptr inbounds {{.*}} %[[E2]], i32 0, i32 2
    // CHECK: call void @[[NS]]1DC1Ev({{.*}} %[[TEMPD2:.*]])
    // CHECK: %[[TEMPD2_C:.*]] = getelementptr inbounds {{.*}} %[[TEMPD2]], i32 0, i32 1
    // CHECK: store {{.*}} %[[TEMPD2_C]], ptr %[[TEMPE2_C]]

    g();
    // CHECK: call void @[[NS]]1gEv()

    // CHECK: call void @[[NS]]1ED1Ev({{.*}} %[[E2]])
    // CHECK: call void @[[NS]]1DD1Ev({{.*}} %[[TEMPD2]])
    // CHECK: call void @[[NS]]1AD1Ev({{.*}} %[[TEMPA2]])
    // CHECK: call void @[[NS]]1ED1Ev({{.*}} %[[TEMPE1]])
    // CHECK: call void @[[NS]]1DD1Ev({{.*}} %[[TEMPD1]])
    // CHECK: call void @[[NS]]1AD1Ev({{.*}} %[[TEMPA1]])
  }
}

namespace ArrayAccess {
  struct A { A(int); ~A(); };
  void g();
  void f() {
    using T = A[3];

    // CHECK: call void @_ZN11ArrayAccess1AC1Ei({{.*}}, i32 noundef 1
    // CHECK-NOT: @_ZN11ArrayAccess1AD
    // CHECK: call void @_ZN11ArrayAccess1AC1Ei({{.*}}, i32 noundef 2
    // CHECK-NOT: @_ZN11ArrayAccess1AD
    // CHECK: call void @_ZN11ArrayAccess1AC1Ei({{.*}}, i32 noundef 3
    // CHECK-NOT: @_ZN11ArrayAccess1AD
    A &&a = T{1, 2, 3}[1];

    // CHECK: call void @_ZN11ArrayAccess1AC1Ei({{.*}}, i32 noundef 4
    // CHECK-NOT: @_ZN11ArrayAccess1AD
    // CHECK: call void @_ZN11ArrayAccess1AC1Ei({{.*}}, i32 noundef 5
    // CHECK-NOT: @_ZN11ArrayAccess1AD
    // CHECK: call void @_ZN11ArrayAccess1AC1Ei({{.*}}, i32 noundef 6
    // CHECK-NOT: @_ZN11ArrayAccess1AD
    A &&b = 2[T{4, 5, 6}];

    // CHECK: call void @_ZN11ArrayAccess1gEv(
    g();

    // CHECK: call void @_ZN11ArrayAccess1AD
    // CHECK: call void @_ZN11ArrayAccess1AD
  }
}

namespace PR14130 {
  struct S { S(int); };
  struct U { S &&s; };
  U v { { 0 } };
  // CHECK: call void @_ZN7PR141301SC1Ei({{.*}} @_ZGRN7PR141301vE_, i32 noundef 0)
  // CHECK: store {{.*}} @_ZGRN7PR141301vE_, {{.*}} @_ZN7PR141301vE
}

namespace Conditional {
  struct A {};
  struct B : A { B(); ~B(); };
  struct C : A { C(); ~C(); };

  void g();

  // CHECK-LABEL: define {{.*}} @_ZN11Conditional1fEb(
  void f(bool b) {
    // CHECK: store i1 false, ptr %[[CLEANUP_B:.*]],
    // CHECK: store i1 false, ptr %[[CLEANUP_C:.*]],
    // CHECK: br i1
    //
    // CHECK: call {{.*}} @_ZN11Conditional1BC1Ev(
    // CHECK: store i1 true, ptr %[[CLEANUP_B]],
    // CHECK: br label
    //
    // CHECK: call {{.*}} @_ZN11Conditional1CC1Ev(
    // CHECK: store i1 true, ptr %[[CLEANUP_C]],
    // CHECK: br label
    A &&r = b ? static_cast<A&&>(B()) : static_cast<A&&>(C());

    // CHECK: call {{.*}} @_ZN11Conditional1gEv(
    g();

    // CHECK: load {{.*}} %[[CLEANUP_C]]
    // CHECK: br i1
    // CHECK: call {{.*}} @_ZN11Conditional1CD1Ev(
    // CHECK: br label

    // CHECK: load {{.*}} %[[CLEANUP_B]]
    // CHECK: br i1
    // CHECK: call {{.*}} @_ZN11Conditional1BD1Ev(
    // CHECK: br label
  }

  struct D { A &&a; };
  // CHECK-LABEL: define {{.*}} @_ZN11Conditional10f_indirectEb(
  void f_indirect(bool b) {
    // CHECK: store i1 false, ptr %[[CLEANUP_B:.*]],
    // CHECK: store i1 false, ptr %[[CLEANUP_C:.*]],
    // CHECK: br i1
    //
    // CHECK: call {{.*}} @_ZN11Conditional1BC1Ev(
    // CHECK: store i1 true, ptr %[[CLEANUP_B]],
    // CHECK: br label
    //
    // CHECK: call {{.*}} @_ZN11Conditional1CC1Ev(
    // CHECK: store i1 true, ptr %[[CLEANUP_C]],
    // CHECK: br label
    D d = b ? D{B()} : D{C()};

    // In C++17, the expression D{...} directly initializes the 'd' object, so
    // lifetime-extending the temporaries to the lifetime of the D object
    // extends them past the call to g().
    //
    // In C++14 and before, D is move-constructed from the result of the
    // conditional expression, so no lifetime extension occurs.

    // CHECK-CXX17: call {{.*}} @_ZN11Conditional1gEv(

    // CHECK: load {{.*}} %[[CLEANUP_C]]
    // CHECK: br i1
    // CHECK: call {{.*}} @_ZN11Conditional1CD1Ev(
    // CHECK: br label

    // CHECK: load {{.*}} %[[CLEANUP_B]]
    // CHECK: br i1
    // CHECK: call {{.*}} @_ZN11Conditional1BD1Ev(
    // CHECK: br label

    // CHECK-CXX11: call {{.*}} @_ZN11Conditional1gEv(
    g();
  }

  extern bool b;
  // CHECK: load {{.*}} @_ZN11Conditional1b
  // CHECK: br i1
  //
  // CHECK: call {{.*}} @_ZN11Conditional1BC1Ev({{.*}} @_ZGRN11Conditional1rE_)
  // CHECK: call {{.*}} @__cxa_atexit({{.*}} @_ZN11Conditional1BD1Ev, ptr @_ZGRN11Conditional1rE_,
  // CHECK: br label
  //
  // CHECK: call {{.*}} @_ZN11Conditional1CC1Ev({{.*}} @_ZGRN11Conditional1rE0_)
  // CHECK: call {{.*}} @__cxa_atexit({{.*}} @_ZN11Conditional1CD1Ev, ptr @_ZGRN11Conditional1rE0_,
  // CHECK: br label
  A &&r = b ? static_cast<A&&>(B()) : static_cast<A&&>(C());
}

#if __cplusplus >= 201703L
namespace PR42220 {
  struct X { X(); ~X(); };
  struct A { X &&x; };
  struct B : A {};
  void g() noexcept;
  // CHECK-CXX17-LABEL: define{{.*}} @_ZN7PR422201fEv(
  void f() {
    // CHECK-CXX17: call{{.*}} @_ZN7PR422201XC1Ev(
    B &&b = {X()};
    // CHECK-CXX17-NOT: call{{.*}} @_ZN7PR422201XD1Ev(
    // CHECK-CXX17: call{{.*}} @_ZN7PR422201gEv(
    g();
    // CHECK-CXX17: call{{.*}} @_ZN7PR422201XD1Ev(
  }
}
#endif
