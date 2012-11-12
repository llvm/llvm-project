// RUN: %clang_cc1 -emit-llvm %s -o - -triple=x86_64-apple-darwin9 | FileCheck %s
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
  // CHECK: call i32 @_ZN6PR50771A1fEv
  // CHECK: call void @_ZN6PR50771AD1Ev
  : a1(A().f())
  // CHECK: call void @_ZN6PR50771AC1Ev
  // CHECK: call i32 @_ZN6PR50771gERKNS_1AE
  // CHECK: call void @_ZN6PR50771AD1Ev
  , a2(g(A()))
{
  // CHECK: call void @_ZN6PR50771fEv
  f();
}
  
struct C {
  C();
  
  const B& b;
};

C::C() 
  // CHECK: call void @_ZN6PR50771BC1Ev
  : b(B()) {
  // CHECK: call void @_ZN6PR50771fEv
  f();
  
  // CHECK: call void @_ZN6PR50771BD1Ev
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
  // CHECK: define void @_ZN6PR58671gEv
  void g() {
    // CHECK: call void @_ZN6PR58671SC1Ev
    // CHECK-NEXT: call void @_ZN6PR58671fENS_1SEi
    // CHECK-NEXT: call void @_ZN6PR58671SD1Ev
    // CHECK-NEXT: ret void
    (f)(S(), 0);
  }

  // CHECK: define linkonce_odr void @_ZN6PR58672g2IiEEvT_
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

  // CHECK: define weak_odr void @_ZN6PR61992f2IiEENS_1AET_
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

// CHECK: define void @_ZN3T121gEv
void g() {
  // CHECK: call void @_ZN3T121AC1Ev
  // CHECK-NEXT: call i32 @_ZN3T121A1fEv(
  // CHECK-NEXT: call i32* @_ZN3T121fEi(
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
    // CHECK: call %"struct.PR6648::D"* @_ZN6PR66483zedENS_1BE
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
  // CHECK: define void @_ZN6PR75563fooEv()
  void foo() { 
    // CHECK: call void @_ZN6PR75561AD1Ev
    A(); 
    // CHECK: call void @llvm.memset.p0i8.i64
    // CHECK: call void @_ZN6PR75561BD1Ev
    B();
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
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

  // CHECK: define void @_ZN7Elision5test0Ev()
  void test0() {
    // CHECK:      [[I:%.*]] = alloca [[A:%.*]], align 8
    // CHECK-NEXT: [[J:%.*]] = alloca [[A]], align 8
    // CHECK-NEXT: [[T0:%.*]] = alloca [[A]], align 8
    // CHECK-NEXT: [[K:%.*]] = alloca [[A]], align 8
    // CHECK-NEXT: [[T1:%.*]] = alloca [[A]], align 8

    // CHECK-NEXT: call void @_ZN7Elision3fooEv()
    // CHECK-NEXT: call void @_ZN7Elision1AC1Ev([[A]]* [[I]])
    A i = (foo(), A());

    // CHECK-NEXT: call void @_ZN7Elision4fooAEv([[A]]* sret [[T0]])
    // CHECK-NEXT: call void @_ZN7Elision1AC1Ev([[A]]* [[J]])
    // CHECK-NEXT: call void @_ZN7Elision1AD1Ev([[A]]* [[T0]])
    A j = (fooA(), A());

    // CHECK-NEXT: call void @_ZN7Elision1AC1Ev([[A]]* [[T1]])
    // CHECK-NEXT: call void @_ZN7Elision4fooAEv([[A]]* sret [[K]])
    // CHECK-NEXT: call void @_ZN7Elision1AD1Ev([[A]]* [[T1]])
    A k = (A(), fooA());

    // CHECK-NEXT: call void @_ZN7Elision1AD1Ev([[A]]* [[K]])
    // CHECK-NEXT: call void @_ZN7Elision1AD1Ev([[A]]* [[J]])
    // CHECK-NEXT: call void @_ZN7Elision1AD1Ev([[A]]* [[I]])
  }


  // CHECK: define void @_ZN7Elision5test1EbNS_1AE(
  void test1(bool c, A x) {
    // CHECK:      [[I:%.*]] = alloca [[A]], align 8
    // CHECK-NEXT: [[J:%.*]] = alloca [[A]], align 8

    // CHECK:      call void @_ZN7Elision1AC1Ev([[A]]* [[I]])
    // CHECK:      call void @_ZN7Elision1AC1ERKS0_([[A]]* [[I]], [[A]]* [[X:%.*]])
    A i = (c ? A() : x);

    // CHECK:      call void @_ZN7Elision1AC1ERKS0_([[A]]* [[J]], [[A]]* [[X]])
    // CHECK:      call void @_ZN7Elision1AC1Ev([[A]]* [[J]])
    A j = (c ? x : A());

    // CHECK:      call void @_ZN7Elision1AD1Ev([[A]]* [[J]])
    // CHECK-NEXT: call void @_ZN7Elision1AD1Ev([[A]]* [[I]])
  }

  // CHECK: define void @_ZN7Elision5test2Ev([[A]]* noalias sret
  A test2() {
    // CHECK:      call void @_ZN7Elision3fooEv()
    // CHECK-NEXT: call void @_ZN7Elision1AC1Ev([[A]]* [[RET:%.*]])
    // CHECK-NEXT: ret void
    return (foo(), A());
  }

  // CHECK: define void @_ZN7Elision5test3EiNS_1AE([[A]]* noalias sret
  A test3(int v, A x) {
    if (v < 5)
    // CHECK:      call void @_ZN7Elision1AC1Ev([[A]]* [[RET:%.*]])
    // CHECK:      call void @_ZN7Elision1AC1ERKS0_([[A]]* [[RET]], [[A]]* [[X:%.*]])
      return (v < 0 ? A() : x);
    else
    // CHECK:      call void @_ZN7Elision1AC1ERKS0_([[A]]* [[RET]], [[A]]* [[X]])
    // CHECK:      call void @_ZN7Elision1AC1Ev([[A]]* [[RET]])
      return (v > 10 ? x : A());

    // CHECK:      ret void
  }

  // CHECK: define void @_ZN7Elision5test4Ev()
  void test4() {
    // CHECK:      [[X:%.*]] = alloca [[A]], align 8
    // CHECK-NEXT: [[XS:%.*]] = alloca [2 x [[A]]], align 16

    // CHECK-NEXT: call void @_ZN7Elision1AC1Ev([[A]]* [[X]])
    A x;

    // CHECK-NEXT: [[XS0:%.*]] = getelementptr inbounds [2 x [[A]]]* [[XS]], i64 0, i64 0
    // CHECK-NEXT: call void @_ZN7Elision1AC1Ev([[A]]* [[XS0]])
    // CHECK-NEXT: [[XS1:%.*]] = getelementptr inbounds [[A]]* [[XS0]], i64 1
    // CHECK-NEXT: call void @_ZN7Elision1AC1ERKS0_([[A]]* [[XS1]], [[A]]* [[X]])
    A xs[] = { A(), x };

    // CHECK-NEXT: [[BEGIN:%.*]] = getelementptr inbounds [2 x [[A]]]* [[XS]], i32 0, i32 0
    // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds [[A]]* [[BEGIN]], i64 2
    // CHECK-NEXT: br label
    // CHECK:      [[AFTER:%.*]] = phi [[A]]*
    // CHECK-NEXT: [[CUR:%.*]] = getelementptr inbounds [[A]]* [[AFTER]], i64 -1
    // CHECK-NEXT: call void @_ZN7Elision1AD1Ev([[A]]* [[CUR]])
    // CHECK-NEXT: [[T0:%.*]] = icmp eq [[A]]* [[CUR]], [[BEGIN]]
    // CHECK-NEXT: br i1 [[T0]],

    // CHECK:      call void @_ZN7Elision1AD1Ev([[A]]* [[X]])
  }

  // rdar://problem/8433352
  // CHECK: define void @_ZN7Elision5test5Ev([[A]]* noalias sret
  struct B { A a; B(); };
  A test5() {
    // CHECK:      [[AT0:%.*]] = alloca [[A]], align 8
    // CHECK-NEXT: [[BT0:%.*]] = alloca [[B:%.*]], align 8
    // CHECK-NEXT: [[X:%.*]] = alloca [[A]], align 8
    // CHECK-NEXT: [[BT1:%.*]] = alloca [[B]], align 8
    // CHECK-NEXT: [[BT2:%.*]] = alloca [[B]], align 8

    // CHECK:      call void @_ZN7Elision1BC1Ev([[B]]* [[BT0]])
    // CHECK-NEXT: [[AM:%.*]] = getelementptr inbounds [[B]]* [[BT0]], i32 0, i32 0
    // CHECK-NEXT: call void @_ZN7Elision1AC1ERKS0_([[A]]* [[AT0]], [[A]]* [[AM]])
    // CHECK-NEXT: call void @_ZN7Elision5takeAENS_1AE([[A]]* [[AT0]])
    // CHECK-NEXT: call void @_ZN7Elision1AD1Ev([[A]]* [[AT0]])
    // CHECK-NEXT: call void @_ZN7Elision1BD1Ev([[B]]* [[BT0]])
    takeA(B().a);

    // CHECK-NEXT: call void @_ZN7Elision1BC1Ev([[B]]* [[BT1]])
    // CHECK-NEXT: [[AM:%.*]] = getelementptr inbounds [[B]]* [[BT1]], i32 0, i32 0
    // CHECK-NEXT: call void @_ZN7Elision1AC1ERKS0_([[A]]* [[X]], [[A]]* [[AM]])
    // CHECK-NEXT: call void @_ZN7Elision1BD1Ev([[B]]* [[BT1]])
    A x = B().a;

    // CHECK-NEXT: call void @_ZN7Elision1BC1Ev([[B]]* [[BT2]])
    // CHECK-NEXT: [[AM:%.*]] = getelementptr inbounds [[B]]* [[BT2]], i32 0, i32 0
    // CHECK-NEXT: call void @_ZN7Elision1AC1ERKS0_([[A]]* [[RET:%.*]], [[A]]* [[AM]])
    // CHECK-NEXT: call void @_ZN7Elision1BD1Ev([[B]]* [[BT2]])
    return B().a;

    // CHECK:      call void @_ZN7Elision1AD1Ev([[A]]* [[X]])
  }

  // Reduced from webkit.
  // CHECK: define void @_ZN7Elision5test6EPKNS_1CE([[C:%.*]]*
  struct C { operator A() const; };
  void test6(const C *x) {
    // CHECK:      [[T0:%.*]] = alloca [[A]], align 8
    // CHECK:      [[X:%.*]] = load [[C]]** {{%.*}}, align 8
    // CHECK-NEXT: call void @_ZNK7Elision1CcvNS_1AEEv([[A]]* sret [[T0]], [[C]]* [[X]])
    // CHECK-NEXT: call void @_ZNK7Elision1A3fooEv([[A]]* [[T0]])
    // CHECK-NEXT: call void @_ZN7Elision1AD1Ev([[A]]* [[T0]])
    // CHECK-NEXT: ret void
    A(*x).foo();
  }
}

namespace PR8623 {
  struct A { A(int); ~A(); };

  // CHECK: define void @_ZN6PR86233fooEb(
  void foo(bool b) {
    // CHECK:      [[TMP:%.*]] = alloca [[A:%.*]], align 1
    // CHECK-NEXT: [[LCONS:%.*]] = alloca i1
    // CHECK-NEXT: [[RCONS:%.*]] = alloca i1
    // CHECK:      store i1 false, i1* [[LCONS]]
    // CHECK-NEXT: store i1 false, i1* [[RCONS]]
    // CHECK-NEXT: br i1
    // CHECK:      call void @_ZN6PR86231AC1Ei([[A]]* [[TMP]], i32 2)
    // CHECK-NEXT: store i1 true, i1* [[LCONS]]
    // CHECK-NEXT: br label
    // CHECK:      call void @_ZN6PR86231AC1Ei([[A]]* [[TMP]], i32 3)
    // CHECK-NEXT: store i1 true, i1* [[RCONS]]
    // CHECK-NEXT: br label
    // CHECK:      load i1* [[RCONS]]
    // CHECK-NEXT: br i1
    // CHECK:      call void @_ZN6PR86231AD1Ev([[A]]* [[TMP]])
    // CHECK-NEXT: br label
    // CHECK:      load i1* [[LCONS]]
    // CHECK-NEXT: br i1
    // CHECK:      call void @_ZN6PR86231AD1Ev([[A]]* [[TMP]])
    // CHECK-NEXT: br label
    // CHECK:      ret void
    b ? A(2) : A(3);
  }
}

namespace PR11365 {
  struct A { A(); ~A(); };

  // CHECK: define void @_ZN7PR113653fooEv(
  void foo() {
    // CHECK: [[BEGIN:%.*]] = getelementptr inbounds [3 x [[A:%.*]]]* {{.*}}, i32 0, i32 0
    // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds [[A]]* [[BEGIN]], i64 3
    // CHECK-NEXT: br label

    // CHECK: [[PHI:%.*]] = phi
    // CHECK-NEXT: [[ELEM:%.*]] = getelementptr inbounds [[A]]* [[PHI]], i64 -1
    // CHECK-NEXT: call void @_ZN7PR113651AD1Ev([[A]]* [[ELEM]])
    // CHECK-NEXT: icmp eq [[A]]* [[ELEM]], [[BEGIN]]
    // CHECK-NEXT: br i1
    (void) (A [3]) {};
  }
}
