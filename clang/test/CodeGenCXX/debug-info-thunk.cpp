// RUN: %clang_cc1 %s -triple=x86_64-pc-windows-msvc -debug-info-kind=limited -S -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -triple %itanium_abi_triple -debug-info-kind=limited -S -emit-llvm -o - | FileCheck %s -check-prefix=ITANIUM
//
// Validate we emit a "DIFlagThunk" flag on DISubprogram entries for thunks.
// This flag is used for emitting S_THUNK32 symbols for CodeView debugging.
//
// NOTE:
// Because thunks are compiler generated and don't exist in the source, this
// test is dependent upon the linkage name to identify the thunk.  Any changes
// in the name mangling may require this test to be updated.
//
// NOTE:
// The FileCheck directives below use CHECK-DAG because the thunks may not be
// emitted in source order.
//

namespace Test1 {
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
  
// CHECK-DAG: DISubprogram{{.*}}linkageName: "?f@C@Test1@@W7EAAXXZ"{{.*}} flags: {{.*}}DIFlagThunk
  void C::f() { }
}

namespace Test2 {
  struct V1 { };
  struct V2 : virtual V1 { };
  
  struct A {
    virtual V1 *f();
  };
  
  struct B : A {
    virtual void b();
    
    virtual V2 *f();
  };
  
// CHECK-DAG: DISubprogram{{.*}}linkageName: "?f@B@Test2@@QEAAPEAUV1@2@XZ"{{.*}} flags: {{.*}}DIFlagThunk
  V2 *B::f() { return 0; }
}

namespace Test3 {
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
  
// CHECK-DAG: DISubprogram{{.*}}linkageName: "?f@C@Test3@@W7EAAXXZ"{{.*}} flags: {{.*}}DIFlagThunk
  void C::f() { }
}

namespace Test4 {
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
// CHECK-DAG: DISubprogram{{.*}}linkageName: "?f@C@?A0x{{[^@]*}}@Test4@@W7EAAXXZ"{{.*}} flags: {{.*}}DIFlagThunk
  void C::f() {}

  // Force C::f to be used.
  void f() { 
    C c; 
    c.f();
  }
}

namespace Test5 {
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

// CHECK-DAG: DISubprogram{{.*}}linkageName: "?f@Thunks@Test5@@WBA@EAA?AUX@2@XZ"{{.*}} flags: {{.*}}DIFlagThunk
  X Thunks::f() { return X(); }
}

namespace Test6 {
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
// CHECK-DAG: DISubprogram{{.*}}linkageName: "?foo@D@Test6@@G7EAAXXZ"{{.*}} flags: {{.*}}DIFlagThunk
    void foo() {}
    void bar() {}
    void baz(X, X&, _Complex float, Small, Small&, Large);
  };

  void D::baz(X, X&, _Complex float, Small, Small&, Large) { }

  void testD() { D d; }
}

namespace Test7 {
  struct A { virtual void foo(); };
  struct B { virtual void foo(); };
// CHECK-DAG: DISubprogram{{.*}}linkageName: "?foo@C@Test7@@W7EAAXXZ"{{.*}} flags: {{.*}}DIFlagThunk
  struct C : A, B { void foo() {} };

  // Test later.
  void test() {
    C c;
  }
}

namespace Test8 {
  struct A {             virtual A* f(); };
  struct B : virtual A { virtual A* f(); };
  struct C : B         { virtual C* f(); };
// CHECK-DAG: DISubprogram{{.*}}linkageName: "?f@C@Test8@@QEAAPEAUA@2@XZ"{{.*}} flags: {{.*}}DIFlagThunk
  C* C::f() { return 0; }
}

namespace Test9 {
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
// CHECK-DAG: DISubprogram{{.*}}linkageName: "?foo1@D@Test9@@$4PPPPPPPE@A@EAAAEAUB1@2@XZ"{{.*}} flags: {{.*}}DIFlagThunk
// CHECK-DAG: DISubprogram{{.*}}linkageName: "?foo1@D@Test9@@$4PPPPPPPE@A@EAAAEAU12@XZ"{{.*}} flags: {{.*}}DIFlagThunk
  D& D::foo1() {
    return *this;
  }
}

namespace Test10 {
  class A {
    virtual void f();
  };
  class B {
    virtual void f();
  };
  class C : public A, public B  {
    virtual void f();
  };
// CHECK-DAG: DISubprogram{{.*}}linkageName: "?f@C@Test10@@G7EAAXXZ"{{.*}} flags: {{.*}}DIFlagThunk
  void C::f() {
  }
}

namespace Test11 {
  class A {
  public:
    virtual void f();
  };

  void test() {
// CHECK-DAG: DISubprogram{{.*}}linkageName: "??_9A@Test11@@$BA@AA"{{.*}} flags: {{.*}}DIFlagThunk
    void (A::*p)() = &A::f;
  }
}

namespace Test12 {
  struct A {
    virtual void f();
  };
  
  struct B {
    virtual void f();
  };
  
  struct C : A, B {
    virtual void f();
  };
  
  void C::f() { }
  // ITANIUM: define {{.*}}void @_ZThn{{[48]}}_N6Test121C1fEv
  // ITANIUM-SAME: !dbg ![[SP:[0-9]+]]
  // ITANIUM-NOT: {{ret }}
  // ITANIUM: = load{{.*}} !dbg ![[DBG:[0-9]+]]
  // ITANIUM-NOT: {{ret }}
  // ITANIUM: ret void, !dbg ![[DBG]]
  //
  // ITANIUM: ![[SP]] = distinct !DISubprogram(linkageName: "_ZThn{{[48]}}_N6Test121C1fEv"
  // ITANIUM-SAME:          line: 261
  // ITANIUM-SAME:          DIFlagArtificial
  // ITANIUM-SAME:          DIFlagThunk
  // ITANIUM-SAME:          DISPFlagDefinition
  // ITANIUM-SAME:          ){{$}}
  //
  // ITANIUM: ![[DBG]] = !DILocation(line: 0
}
