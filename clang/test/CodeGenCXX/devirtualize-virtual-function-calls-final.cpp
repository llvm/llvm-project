// RUN: %clang_cc1 -triple i386-unknown-unknown -std=c++11 %s -emit-llvm -o - | FileCheck %s

namespace Test1 {
  struct A {
    virtual int f() final;
  };

  // CHECK-LABEL: define{{.*}} i32 @_ZN5Test11fEPNS_1AE
  int f(A *a) {
    // CHECK: call i32 @_ZN5Test11A1fEv
    return a->f();
  }
}

namespace Test2 {
  struct A final {
    virtual int f();
  };

  // CHECK-LABEL: define{{.*}} i32 @_ZN5Test21fEPNS_1AE
  int f(A *a) {
    // CHECK: call i32 @_ZN5Test21A1fEv
    return a->f();
  }
}

namespace Test2a {
  struct A {
    virtual ~A() final {}
    virtual int f();
  };

  // CHECK-LABEL: define{{.*}} i32 @_ZN6Test2a1fEPNS_1AE
  int f(A *a) {
    // CHECK: call i32 @_ZN6Test2a1A1fEv
    return a->f();
  }
}


namespace Test3 {
  struct A {
    virtual int f();  };

  struct B final : A { };

  // CHECK-LABEL: define{{.*}} i32 @_ZN5Test31fEPNS_1BE
  int f(B *b) {
    // CHECK: call i32 @_ZN5Test31A1fEv
    return b->f();
  }

  // CHECK-LABEL: define{{.*}} i32 @_ZN5Test31fERNS_1BE
  int f(B &b) {
    // CHECK: call i32 @_ZN5Test31A1fEv
    return b.f();
  }

  // CHECK-LABEL: define{{.*}} i32 @_ZN5Test31fEPv
  int f(void *v) {
    // CHECK: call i32 @_ZN5Test31A1fEv
    return static_cast<B*>(v)->f();
  }
}

namespace Test4 {
  struct A {
    virtual void f();
    virtual int operator-();
  };

  struct B final : A {
    virtual void f();
    virtual int operator-();
  };

  // CHECK-LABEL: define{{.*}} void @_ZN5Test41fEPNS_1BE
  void f(B* d) {
    // CHECK: call void @_ZN5Test41B1fEv
    static_cast<A*>(d)->f();
    // CHECK: call i32 @_ZN5Test41BngEv
    -static_cast<A&>(*d);
  }
}

namespace Test5 {
  struct A {
    virtual void f();
    virtual int operator-();
  };

  struct B : A {
    virtual void f();
    virtual int operator-();
  };

  struct C final : B {
  };

  // CHECK-LABEL: define{{.*}} void @_ZN5Test51fEPNS_1CE
  void f(C* d) {
    // FIXME: It should be possible to devirtualize this case, but that is
    // not implemented yet.
    // CHECK: getelementptr
    // CHECK-NEXT: %[[FUNC:.*]] = load
    // CHECK-NEXT: call void %[[FUNC]]
    static_cast<A*>(d)->f();
  }
  // CHECK-LABEL: define{{.*}} void @_ZN5Test53fopEPNS_1CE
  void fop(C* d) {
    // FIXME: It should be possible to devirtualize this case, but that is
    // not implemented yet.
    // CHECK: getelementptr
    // CHECK-NEXT: %[[FUNC:.*]] = load
    // CHECK-NEXT: call i32 %[[FUNC]]
    -static_cast<A&>(*d);
  }
}

namespace Test6 {
  struct A {
    virtual ~A();
  };

  struct B : public A {
    virtual ~B();
  };

  struct C {
    virtual ~C();
  };

  struct D final : public C, public B {
  };

  // CHECK-LABEL: define{{.*}} void @_ZN5Test61fEPNS_1DE
  void f(D* d) {
    // CHECK: call void @_ZN5Test61DD1Ev
    static_cast<A*>(d)->~A();
  }
}

namespace Test7 {
  struct foo {
    virtual void g() {}
  };

  struct bar {
    virtual int f() { return 0; }
  };

  struct zed final : public foo, public bar {
    int z;
    virtual int f() {return z;}
  };

  // CHECK-LABEL: define{{.*}} i32 @_ZN5Test71fEPNS_3zedE
  int f(zed *z) {
    // CHECK: alloca
    // CHECK-NEXT: store
    // CHECK-NEXT: load
    // CHECK-NEXT: call i32 @_ZN5Test73zed1fEv
    // CHECK-NEXT: ret
    return static_cast<bar*>(z)->f();
  }
}

namespace Test8 {
  struct A { virtual ~A() {} };
  struct B {
    int b;
    virtual int foo() { return b; }
  };
  struct C final : A, B {  };
  // CHECK-LABEL: define{{.*}} i32 @_ZN5Test84testEPNS_1CE
  int test(C *c) {
    // CHECK: %[[THIS:.*]] = phi
    // CHECK-NEXT: call i32 @_ZN5Test81B3fooEv(%"struct.Test8::B"* {{[^,]*}} %[[THIS]])
    return static_cast<B*>(c)->foo();
  }
}

namespace Test9 {
  struct A {
    int a;
  };
  struct B {
    int b;
  };
  struct C : public B, public A {
  };
  struct RA {
    virtual A *f() {
      return 0;
    }
    virtual A *operator-() {
      return 0;
    }
  };
  struct RC final : public RA {
    virtual C *f() {
      C *x = new C();
      x->a = 1;
      x->b = 2;
      return x;
    }
    virtual C *operator-() {
      C *x = new C();
      x->a = 1;
      x->b = 2;
      return x;
    }
  };
  // CHECK: define {{.*}} @_ZN5Test91fEPNS_2RCE
  A *f(RC *x) {
    // FIXME: It should be possible to devirtualize this case, but that is
    // not implemented yet.
    // CHECK: load
    // CHECK: bitcast
    // CHECK: [[F_PTR_RA:%.+]] = bitcast
    // CHECK: [[VTABLE:%.+]] = load {{.+}} [[F_PTR_RA]]
    // CHECK: [[VFN:%.+]] = getelementptr inbounds {{.+}} [[VTABLE]], i{{[0-9]+}} 0
    // CHECK-NEXT: %[[FUNC:.*]] = load {{.+}} [[VFN]]
    // CHECK-NEXT: = call {{.*}} %[[FUNC]]
    return static_cast<RA*>(x)->f();
  }
  // CHECK: define {{.*}} @_ZN5Test93fopEPNS_2RCE
  A *fop(RC *x) {
    // FIXME: It should be possible to devirtualize this case, but that is
    // not implemented yet.
    // CHECK: load
    // CHECK: bitcast
    // CHECK: [[F_PTR_RA:%.+]] = bitcast
    // CHECK: [[VTABLE:%.+]] = load {{.+}} [[F_PTR_RA]]
    // CHECK: [[VFN:%.+]] = getelementptr inbounds {{.+}} [[VTABLE]], i{{[0-9]+}} 1
    // CHECK-NEXT: %[[FUNC:.*]] = load {{.+}} [[VFN]]
    // CHECK-NEXT: = call {{.*}} %[[FUNC]]
    return -static_cast<RA&>(*x);
  }
}

namespace Test10 {
  struct A {
    virtual int f();
  };

  struct B : A {
    int f() final;
  };

  // CHECK-LABEL: define{{.*}} i32 @_ZN6Test101fEPNS_1BE
  int f(B *b) {
    // CHECK: call i32 @_ZN6Test101B1fEv
    return static_cast<A *>(b)->f();
  }
}

namespace TestVBase {
  struct A { virtual void f(); };
  struct B : virtual A {};
  struct C : virtual A { void f() override; };

  extern struct BC final : B, C {} &bc;
  extern struct BCusingA final : B, C { using A::f; } &bc_using_a;
  extern struct BCusingB final : B, C { using B::f; } &bc_using_b;
  extern struct BCusingC final : B, C { using C::f; } &bc_using_c;

  extern struct CB final : C, B {} &cb;
  extern struct CBusingA final : C, B { using A::f; } &cb_using_a;
  extern struct CBusingB final : C, B { using B::f; } &cb_using_b;
  extern struct CBusingC final : C, B { using C::f; } &cb_using_c;

  // CHECK-LABEL: @_ZN9TestVBase4testEv(
  void test() {
    // FIXME: The 'using A' case can be devirtualized to call A's virtual
    // adjustment thunk for C::f.
    // FIXME: The 'using B' case can be devirtualized, but requires us to emit
    // a derived-to-base or base-to-derived conversion as part of
    // devirtualization.

    // CHECK: call void @_ZN9TestVBase1C1fEv(
    bc.f();
    // CHECK: call void %
    bc_using_a.f();
    // CHECK: call void %
    bc_using_b.f();
    // CHECK: call void @_ZN9TestVBase1C1fEv(
    bc_using_c.f();

    // CHECK: call void @_ZN9TestVBase1C1fEv(
    cb.f();
    // CHECK: call void %
    cb_using_a.f();
    // CHECK: call void %
    cb_using_b.f();
    // CHECK: call void @_ZN9TestVBase1C1fEv(
    cb_using_c.f();
  }
}

namespace Test11 {
  // Check that the definitions of Derived's operators are emitted.

  // CHECK-LABEL: define linkonce_odr void @_ZN6Test111SIiE4foo1Ev(
  // CHECK: call void @_ZN6Test111SIiE7DerivedclEv(
  // CHECK: call zeroext i1 @_ZN6Test111SIiE7DerivedeqERKNS_4BaseE(
  // CHECK: call zeroext i1 @_ZN6Test111SIiE7DerivedntEv(
  // CHECK: call nonnull align 4 dereferenceable(4) %"class.Test11::Base"* @_ZN6Test111SIiE7DerivedixEi(
  // CHECK: define linkonce_odr void @_ZN6Test111SIiE7DerivedclEv(
  // CHECK: define linkonce_odr zeroext i1 @_ZN6Test111SIiE7DerivedeqERKNS_4BaseE(
  // CHECK: define linkonce_odr zeroext i1 @_ZN6Test111SIiE7DerivedntEv(
  // CHECK: define linkonce_odr nonnull align 4 dereferenceable(4) %"class.Test11::Base"* @_ZN6Test111SIiE7DerivedixEi(
  class Base {
  public:
    virtual void operator()() {}
    virtual bool operator==(const Base &other) { return false; }
    virtual bool operator!() { return false; }
    virtual Base &operator[](int i) { return *this; }
  };

  template<class T>
  struct S {
    class Derived final : public Base {
    public:
      void operator()() override {}
      bool operator==(const Base &other) override { return true; }
      bool operator!() override { return true; }
      Base &operator[](int i) override { return *this; }
    };

    Derived *ptr = nullptr, *ptr2 = nullptr;

    void foo1() {
      if (ptr && ptr2) {
        // These calls get devirtualized. Linkage fails if the definitions of
        // the called functions are not emitted.
        (*ptr)();
        (void)(*ptr == *ptr2);
        (void)(!(*ptr));
        (void)((*ptr)[1]);
      }
    }
  };

  void foo2() {
    S<int> *s = new S<int>;
    s->foo1();
  }
}
