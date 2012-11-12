// RUN: %clang_cc1 %s -triple=x86_64-pc-linux-gnu -munwind-tables -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-pc-linux-gnu -munwind-tables -fhidden-weak-vtables -emit-llvm -o - | FileCheck -check-prefix=HIDDEN %s

namespace Test1 {

// Check that we emit a non-virtual thunk for C::f.

struct A {
  virtual void f();
};

struct B {
  virtual void f();
};

struct C : A, B {
  virtual void c();
  
  virtual void f();
};

// CHECK: define void @_ZThn8_N5Test11C1fEv(
void C::f() { }

}

namespace Test2 {

// Check that we emit a thunk for B::f since it's overriding a virtual base.

struct A {
  virtual void f();
};

struct B : virtual A {
  virtual void b();
  virtual void f();
};

// CHECK: define void @_ZTv0_n24_N5Test21B1fEv(
void B::f() { }

}

namespace Test3 {

// Check that we emit a covariant thunk for B::f.

struct V1 { };
struct V2 : virtual V1 { };

struct A {
  virtual V1 *f();
};

struct B : A {
  virtual void b();
  
  virtual V2 *f();
};

// CHECK: define %{{.*}}* @_ZTch0_v0_n24_N5Test31B1fEv(
V2 *B::f() { return 0; }

}

namespace Test4 {

// Check that the thunk for 'C::f' has the same visibility as the function itself.

struct A {
  virtual void f();
};

struct B {
  virtual void f();
};

struct __attribute__((visibility("protected"))) C : A, B {
  virtual void c();
  
  virtual void f();
};

// CHECK: define protected void @_ZThn8_N5Test41C1fEv(
void C::f() { }

}

// Check that the thunk gets internal linkage.
namespace Test4B {
  struct A {
    virtual void f();
  };

  struct B {
    virtual void f();
  };

  namespace {
    struct C : A, B {
      virtual void c();
      virtual void f();
    };
  }
  void C::c() {}
  void C::f() {}

  // Force C::f to be used.
  void f() { 
    C c; 
    c.f();
  }
}

namespace Test5 {

// Check that the thunk for 'B::f' gets the same linkage as the function itself.
struct A {
  virtual void f();
};

struct B : virtual A {
  virtual void f() { }
};

void f(B b) {
  b.f();
}
}

namespace Test6 {
  struct X {
    X();
    X(const X&);
    X &operator=(const X&);
    ~X();
  };

  struct P {
    P();
    P(const P&);
    ~P();
    X first;
    X second;
  };

  P getP();

  struct Base1 {
    int i;

    virtual X f() { return X(); }
  };

  struct Base2 {
    float real;

    virtual X f() { return X(); }
  };

  struct Thunks : Base1, Base2 {
    long l;

    virtual X f();
  };

  // CHECK: define void @_ZThn16_N5Test66Thunks1fEv
  // CHECK-NOT: memcpy
  // CHECK: {{call void @_ZN5Test66Thunks1fEv.*sret}}
  // CHECK: ret void
  X Thunks::f() { return X(); }
}

namespace Test7 {
  // PR7188
  struct X {
    X();
    X(const X&);
    X &operator=(const X&);
    ~X();
  };

  struct Small { short s; };
  struct Large {
    char array[1024];
  };

  class A {
  protected:
    virtual void foo() = 0;
  };

  class B : public A {
  protected:
    virtual void bar() = 0;
  };

  class C : public A  {
  protected:
    virtual void baz(X, X&, _Complex float, Small, Small&, Large) = 0;
  };

  class D : public B,
            public C {

    void foo() {}
    void bar() {}
    void baz(X, X&, _Complex float, Small, Small&, Large);
  };

  void D::baz(X, X&, _Complex float, Small, Small&, Large) { }

  // CHECK: define void @_ZThn8_N5Test71D3bazENS_1XERS1_CfNS_5SmallERS4_NS_5LargeE(
  // CHECK-NOT: memcpy
  // CHECK: ret void
  void testD() { D d; }
}

namespace Test8 {
  struct NonPOD { ~NonPOD(); int x, y, z; };
  struct A { virtual void foo(); };
  struct B { virtual void bar(NonPOD); };
  struct C : A, B { virtual void bar(NonPOD); static void helper(NonPOD); };

  // CHECK: define void @_ZN5Test81C6helperENS_6NonPODE([[NONPODTYPE:%.*]]*
  void C::helper(NonPOD var) {}

  // CHECK: define void @_ZThn8_N5Test81C3barENS_6NonPODE(
  // CHECK-NOT: load [[NONPODTYPE]]*
  // CHECK-NOT: memcpy
  // CHECK: ret void
  void C::bar(NonPOD var) {}
}

// PR7241: Emitting thunks for a method shouldn't require the vtable for
// that class to be emitted.
namespace Test9 {
  struct A { virtual ~A() { } };
  struct B : A { virtual void test() const {} };
  struct C : B { C(); ~C(); };
  struct D : C { D() {} };
  void test() {
    D d;
  }
}

namespace Test10 {
  struct A { virtual void foo(); };
  struct B { virtual void foo(); };
  struct C : A, B { void foo() {} };

  // CHECK-HIDDEN: define linkonce_odr void @_ZN6Test101C3fooEv
  // CHECK-HIDDEN: define linkonce_odr hidden void @_ZThn8_N6Test101C3fooEv

  void test() {
    C c;
  }
}

// PR7611
namespace Test11 {
  struct A {             virtual A* f(); };
  struct B : virtual A { virtual A* f(); };
  struct C : B         { virtual C* f(); };
  C* C::f() { return 0; }

  //  C::f itself.
  // CHECK: define {{.*}} @_ZN6Test111C1fEv(

  //  The this-adjustment and return-adjustment thunk required when
  //  C::f appears in a vtable where A is at a nonzero offset from C.
  // CHECK: define {{.*}} @_ZTcv0_n24_v0_n32_N6Test111C1fEv(

  //  The return-adjustment thunk required when C::f appears in a vtable
  //  where A is at a zero offset from C.
  // CHECK: define {{.*}} @_ZTch0_v0_n32_N6Test111C1fEv(
}

// Varargs thunk test.
namespace Test12 {
  struct A {
    virtual A* f(int x, ...);
  };
  struct B {
    virtual B* f(int x, ...);
  };
  struct C : A, B {
    virtual void c();
    virtual C* f(int x, ...);
  };
  C* C::f(int x, ...) { return this; }

  // C::f
  // CHECK: define {{.*}} @_ZN6Test121C1fEiz

  // Varargs thunk; check that both the this and covariant adjustments
  // are generated.
  // CHECK: define {{.*}} @_ZTchn8_h8_N6Test121C1fEiz
  // CHECK: getelementptr inbounds i8* {{.*}}, i64 -8
  // CHECK: getelementptr inbounds i8* {{.*}}, i64 8
}

// PR13832
namespace Test13 {
  struct B1 {
    virtual B1 &foo1();
  };
  struct Pad1 {
    virtual ~Pad1();
  };
  struct Proxy1 : Pad1, B1 {
    virtual ~Proxy1();
  };
  struct D : virtual Proxy1 {
    virtual ~D();
    virtual D &foo1();
  };
  D& D::foo1() {
    return *this;
  }
  // CHECK: define {{.*}} @_ZTcvn8_n32_v8_n24_N6Test131D4foo1Ev
  // CHECK: getelementptr inbounds i8* {{.*}}, i64 -8
  // CHECK: getelementptr inbounds i8* {{.*}}, i64 -32
  // CHECK: getelementptr inbounds i8* {{.*}}, i64 -24
  // CHECK: getelementptr inbounds i8* {{.*}}, i64 8
  // CHECK: ret %"struct.Test13::D"*
}

namespace Test14 {
  class A {
    virtual void f();
  };
  class B {
    virtual void f();
  };
  class C : public A, public B  {
    virtual void f();
  };
  void C::f() {
  }
  // CHECK: define void @_ZThn8_N6Test141C1fEv({{.*}}) {{.*}} uwtable
}

/**** The following has to go at the end of the file ****/

// This is from Test5:
// CHECK: define linkonce_odr void @_ZTv0_n24_N5Test51B1fEv
// CHECK: define internal void @_ZThn8_N6Test4B12_GLOBAL__N_11C1fEv(
