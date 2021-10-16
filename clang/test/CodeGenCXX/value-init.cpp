// RUN: %clang_cc1 -std=c++98 %s -triple x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-CXX98
// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-apple-darwin10 -emit-llvm -o - | FileCheck %s --check-prefix=CHECK --check-prefix=CHECK-CXX17

struct A {
  virtual ~A();
};

struct B : A { };

struct C {
  int i;
  B b;
};

// CHECK: _Z15test_value_initv
void test_value_init() {
  // This value initialization requires zero initialization of the 'B'
  // subobject followed by a call to its constructor.
  // PR5800

  // CHECK: store i32 17
  // CHECK: call void @llvm.memset.p0i8.i64
  // CHECK: call void @_ZN1BC1Ev
  C c = { 17 } ;
  // CHECK: call void @_ZN1CD1Ev
}

enum enum_type { negative_number = -1, magic_number = 42 };

class enum_holder
{
  enum_type m_enum;

public:
  enum_holder() : m_enum(magic_number) { }
};

struct enum_holder_and_int
{
  enum_holder e;
  int i;
};

// CHECK: _Z24test_enum_holder_and_intv()
void test_enum_holder_and_int() {
  // CHECK: alloca
  // CHECK-NEXT: bitcast
  // CHECK-NEXT: call void @llvm.memset
  // CHECK-NEXT: call void @_ZN19enum_holder_and_intC1Ev
  enum_holder_and_int();
  // CHECK-NEXT: ret void
}

// PR7834: don't crash.
namespace test1 {
  struct A {
    int A::*f;
    A();
    A(const A&);
    A &operator=(const A &);
  };

  struct B {
    A base;
  };

  void foo() {
    B();
  }
}

namespace ptrmem {
  struct S {
    int mem1;
    int S::*mem2;
  };

  // CHECK-LABEL: define{{.*}} i32 @_ZN6ptrmem4testEPNS_1SE
  int test(S *s) {
    // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
    // CHECK: getelementptr
    // CHECK: ret
    return s->*S().mem2;
  }
}

namespace PR9801 {

struct Test {
  Test() : i(10) {}
  Test(int i) : i(i) {}
  int i;
private:
  int j;
};

struct Test2 {
  Test t;
};

struct Test3 : public Test { };

// CHECK-LABEL: define{{.*}} void @_ZN6PR98011fEv
void f() {
  // CHECK-NOT: call void @llvm.memset.p0i8.i64
  // CHECK: call void @_ZN6PR98014TestC1Ei
  // CHECK-NOT: call void @llvm.memset.p0i8.i64
  // CHECK: call void @_ZN6PR98014TestC1Ev
  Test partial[3] = { 1 };

  // CHECK-NOT: call void @llvm.memset.p0i8.i64
  // CHECK: call void @_ZN6PR98014TestC1Ev
  // CHECK-NOT: call void @_ZN6PR98014TestC1Ev
  Test empty[3] = {};

  // CHECK: call void @llvm.memset.p0i8.i64
  // CHECK-NOT: call void @llvm.memset.p0i8.i64
  // CHECK-CXX98: call void @_ZN6PR98015Test2C1Ev
  // CHECK-CXX17: call void @_ZN6PR98014TestC1Ev
  // CHECK-NOT: call void @_ZN6PR98015Test2C1Ev
  Test2 empty2[3] = {};

  // CHECK: call void @llvm.memset.p0i8.i64
  // CHECK-NOT: call void @llvm.memset.p0i8.i64
  // CHECK-CXX98: call void @_ZN6PR98015Test3C1Ev
  // CHECK-CXX17: call void @_ZN6PR98014TestC2Ev
  // CHECK-NOT: call void @llvm.memset.p0i8.i64
  // CHECK-NOT: call void @_ZN6PR98015Test3C1Ev
  Test3 empty3[3] = {};
}

}

namespace zeroinit {
  struct S { int i; };

  // CHECK-LABEL: define{{.*}} i32 @_ZN8zeroinit4testEv()
  int test() {
    // CHECK: call void @llvm.memset.p0i8.i64
    // CHECK: ret i32 0
    return S().i;
  }

  struct X0 {
    X0() { }
    int x;
  };

  struct X1 : X0 {
    int x1;
    void f();
  };

  // CHECK-LABEL: define{{.*}} void @_ZN8zeroinit9testX0_X1Ev
  void testX0_X1() {
    // CHECK: call void @llvm.memset.p0i8.i64
    // CHECK-NEXT: call void @_ZN8zeroinit2X1C1Ev
    // CHECK-NEXT: call void @_ZN8zeroinit2X11fEv
    X1().f();
  }

  template<typename>
  struct X2 : X0 {
    int x2;
    void f();
  };

  template<typename>
  struct X3 : X2<int> { 
    X3() : X2<int>() { }
    int i;
  };
  

  // CHECK-LABEL: define{{.*}} void @_ZN8zeroinit9testX0_X3Ev
  void testX0_X3() {
    // CHECK-NOT: call void @llvm.memset
    // CHECK: call void @_ZN8zeroinit2X3IiEC1Ev
    // CHECK: call void @_ZN8zeroinit2X2IiE1fEv
    // CHECK-NEXT: ret void
    X3<int>().f();
  }

  // More checks at EOF
}

namespace PR8726 {
class C;
struct S {
  const C &c1;
  int i;
  const C &c2;
};
void f(const C& c) {
  S s = {c, 42, c};
}

}

// rdar://problem/9355931
namespace test6 {
  struct A { A(); A(int); };

  void test() {
    A arr[10][20] = { 5 };
  };
  // CHECK-LABEL:    define{{.*}} void @_ZN5test64testEv()
  // CHECK:      [[ARR:%.*]] = alloca [10 x [20 x [[A:%.*]]]],

  // CHECK-NEXT: [[INNER:%.*]] = getelementptr inbounds [10 x [20 x [[A]]]], [10 x [20 x [[A]]]]* [[ARR]], i64 0, i64 0
  // CHECK-NEXT: [[T0:%.*]] = getelementptr inbounds [20 x [[A]]], [20 x [[A]]]* [[INNER]], i64 0, i64 0
  // CHECK-NEXT: call void @_ZN5test61AC1Ei([[A]]* {{[^,]*}} [[T0]], i32 noundef 5)
  // CHECK-NEXT: [[BEGIN:%.*]] = getelementptr inbounds [[A]], [[A]]* [[T0]], i64 1
  // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds [[A]], [[A]]* [[T0]], i64 20
  // CHECK-NEXT: br label
  // CHECK:      [[CUR:%.*]] = phi [[A]]* [ [[BEGIN]], {{%.*}} ], [ [[NEXT:%.*]], {{%.*}} ]
  // CHECK-NEXT: call void @_ZN5test61AC1Ev([[A]]* {{[^,]*}} [[CUR]])
  // CHECK-NEXT: [[NEXT]] = getelementptr inbounds [[A]], [[A]]* [[CUR]], i64 1
  // CHECK-NEXT: [[T0:%.*]] = icmp eq [[A]]* [[NEXT]], [[END]]
  // CHECK-NEXT: br i1

  // CHECK:      [[BEGIN:%.*]] = getelementptr inbounds [20 x [[A]]], [20 x [[A]]]* [[INNER]], i64 1
  // CHECK-NEXT: [[END:%.*]] = getelementptr inbounds [20 x [[A]]], [20 x [[A]]]* [[INNER]], i64 10
  // CHECK-NEXT: br label
  // CHECK:      [[CUR:%.*]] = phi [20 x [[A]]]* [ [[BEGIN]], {{%.*}} ], [ [[NEXT:%.*]], {{%.*}} ]

  // Inner loop.
  // CHECK-NEXT: [[IBEGIN:%.*]] = getelementptr inbounds [20 x [[A]]], [20 x [[A]]]* [[CUR]], i{{32|64}} 0, i{{32|64}} 0
  // CHECK-NEXT: [[IEND:%.*]] = getelementptr inbounds [[A]], [[A]]* [[IBEGIN]], i64 20
  // CHECK-NEXT: br label
  // CHECK:      [[ICUR:%.*]] = phi [[A]]* [ [[IBEGIN]], {{%.*}} ], [ [[INEXT:%.*]], {{%.*}} ]
  // CHECK-NEXT: call void @_ZN5test61AC1Ev([[A]]* {{[^,]*}} [[ICUR]])
  // CHECK-NEXT: [[INEXT:%.*]] = getelementptr inbounds [[A]], [[A]]* [[ICUR]], i64 1
  // CHECK-NEXT: [[T0:%.*]] = icmp eq [[A]]* [[INEXT]], [[IEND]]
  // CHECK-NEXT: br i1 [[T0]],

  // CHECK:      [[NEXT]] = getelementptr inbounds [20 x [[A]]], [20 x [[A]]]* [[CUR]], i64 1
  // CHECK-NEXT: [[T0:%.*]] = icmp eq [20 x [[A]]]* [[NEXT]], [[END]]
  // CHECK-NEXT: br i1 [[T0]]
  // CHECK:      ret void
}

namespace PR11124 {
  // Make sure C::C doesn't overwrite parts of A while it is zero-initializing B
  struct A { int a; A(); A(int); };
  struct B : virtual A { int b; };
  struct C : B { C(); };      
  C::C() : A(3), B() {}
  // CHECK-LABEL: define{{.*}} void @_ZN7PR111241CC1Ev
  // CHECK: call void @llvm.memset.p0i8.i64(i8* align 8 {{.*}}, i8 0, i64 12, i1 false)
  // CHECK-NEXT: call void @_ZN7PR111241BC2Ev
  // Make sure C::C doesn't overwrite parts of A while it is zero-initializing B

  struct B2 : virtual A { int B::*b; };
  struct C2 : B2 { C2(); };      
  C2::C2() : A(3), B2() {}
  // CHECK-LABEL: define{{.*}} void @_ZN7PR111242C2C1Ev
  // CHECK: call void @llvm.memcpy.p0i8.p0i8.i64(i8* align 8 %{{.*}}, i8* align 8 {{.*}}, i64 16, i1 false)
  // CHECK-NEXT: call void @_ZN7PR111242B2C2Ev
}

// Ensure we produce an i1 here, and don't assert.
// CHECK-LABEL: define{{.*}} void @_Z9r170806_bv(
// CHECK: call void @_Z9r170806_ab(i1 noundef zeroext false)
void r170806_a(bool b = bool());
void r170806_b() { r170806_a(); }

namespace PR20256 {
  struct data { int i; };

  template<typename T = int>
  data g() {
    data d; // not value-init
    return d;
  }
  template data g();
  // CHECK-LABEL: define {{.*}} @_ZN7PR202561gIiEENS_4dataEv(
  // CHECK-NOT: store
  // CHECK-NOT: memset
  // CHECK: }

  template<typename ...T>
  data h(T ...t) {
    data d(t...); // value-init
    return d;
  }
  template data h();
  // CHECK-LABEL: define {{.*}} @_ZN7PR202561hIJEEENS_4dataEDpT_(
  // CHECK: call void @llvm.memset
  // CHECK: }


  template<typename T = int>
  data j() {
    data d = {}; // value-init
    return d;
  }
  template data j();
  // CHECK-LABEL: define {{.*}} @_ZN7PR202561jIiEENS_4dataEv(
  // CHECK: call void @llvm.memset
  // CHECK: }

  data f() {
    data d; // not value-init
    return d;
  }
  // CHECK-LABEL: define {{.*}} @_ZN7PR202561fEv(
  // CHECK-NOT: store
  // CHECK-NOT: memset
  // CHECK: }

  data i() {
    data d = {}; // value-init
    return d;
  }
  // CHECK-LABEL: define {{.*}} @_ZN7PR202561iEv(
  // CHECK: call void @llvm.memset
  // CHECK: }
}

// CHECK-LABEL: define {{.*}}@_Z20explicitly_defaultedv
int explicitly_defaulted() {
  struct A { A() = default; int n; };
  // CHECK: call void @llvm.memset
  A a = A();
  return a.n;
} // CHECK-LABEL: }

// CHECK-LABEL: define linkonce_odr void @_ZN8zeroinit2X3IiEC2Ev(%"struct.zeroinit::X3"* {{[^,]*}} %this) unnamed_addr
// CHECK: call void @llvm.memset.p0i8.i64
// CHECK-NEXT: call void @_ZN8zeroinit2X2IiEC2Ev
// CHECK-NEXT: ret void

#if __cplusplus >= 201103L
namespace transparent_init_list {
  struct optional_assign_base {};
  struct optional_data_dtor_base { char dummy_[24]; };
  struct optional : optional_data_dtor_base, optional_assign_base {};
  optional f(optional a) { return {optional(a)}; }
}
#endif
