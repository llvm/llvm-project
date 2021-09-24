// Sparc64 doesn't support musttail (yet), so it uses method cloning for
// variadic thunks. Use it for testing.
// RUN: %clang_cc1 %s -triple=sparc64-pc-linux-gnu -funwind-tables=2 -emit-llvm -o - \
// RUN:     | FileCheck --check-prefixes=CHECK,CHECK-CLONE,CHECK-NONOPT %s
// RUN: %clang_cc1 %s -triple=sparc64-pc-linux-gnu -debug-info-kind=standalone -dwarf-version=5 -funwind-tables=2 -emit-llvm -o - \
// RUN:     | FileCheck --check-prefixes=CHECK,CHECK-CLONE,CHECK-NONOPT,CHECK-DBG %s
// RUN: %clang_cc1 %s -triple=sparc64-pc-linux-gnu -funwind-tables=2 -emit-llvm -o - -O1 -disable-llvm-passes \
// RUN:     | FileCheck --check-prefixes=CHECK,CHECK-CLONE,CHECK-OPT %s

// Test x86_64, which uses musttail for variadic thunks.
// RUN: %clang_cc1 %s -triple=x86_64-pc-linux-gnu -funwind-tables=2 -emit-llvm -o - -O1 -disable-llvm-passes \
// RUN:     | FileCheck --check-prefixes=CHECK,CHECK-TAIL,CHECK-OPT %s

// Finally, reuse these tests for the MS ABI.
// RUN: %clang_cc1 %s -triple=x86_64-windows-msvc -funwind-tables=2 -emit-llvm -o - -O1 -disable-llvm-passes \
// RUN:     | FileCheck --check-prefixes=WIN64 %s


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

// CHECK-LABEL: define{{.*}} void @_ZThn8_N5Test11C1fEv(
// CHECK-DBG-NOT: dbg.declare
// CHECK: ret void
//
// WIN64-LABEL: define dso_local void @"?f@C@Test1@@UEAAXXZ"(
// WIN64-LABEL: define linkonce_odr dso_local void @"?f@C@Test1@@W7EAAXXZ"(
// WIN64: getelementptr i8, i8* {{.*}}, i32 -8
// WIN64: ret void
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

// CHECK-LABEL: define{{.*}} void @_ZTv0_n24_N5Test21B1fEv(
// CHECK-DBG-NOT: dbg.declare
// CHECK: ret void
void B::f() { }

// No thunk is used for this case in the MS ABI.
// WIN64-LABEL: define dso_local void @"?f@B@Test2@@UEAAXXZ"(
// WIN64-NOT: define {{.*}} void @"?f@B@Test2

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

// CHECK: define{{.*}} %{{.*}}* @_ZTch0_v0_n24_N5Test31B1fEv(
// WIN64: define weak_odr dso_local %{{.*}} @"?f@B@Test3@@QEAAPEAUV1@2@XZ"(
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

// CHECK-LABEL: define protected void @_ZThn8_N5Test41C1fEv(
// CHECK-DBG-NOT: dbg.declare
// CHECK: ret void
void C::f() { }

// Visibility doesn't matter on COFF, but whatever. We could add an ELF test
// mode later.
// WIN64-LABEL: define protected void @"?f@C@Test4@@UEAAXXZ"(
// WIN64-LABEL: define linkonce_odr protected void @"?f@C@Test4@@W7EAAXXZ"(
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
// Not sure why this isn't delayed like in Itanium.
// WIN64-LABEL: define internal void @"?f@C@?A{{.*}}@Test4B@@UEAAXXZ"(

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
// No thunk in MS ABI in this case.
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

  // CHECK-LABEL: define{{.*}} void @_ZThn16_N5Test66Thunks1fEv
	// CHECK-DBG-NOT: dbg.declare
  // CHECK-NOT: memcpy
  // CHECK: {{call void @_ZN5Test66Thunks1fEv.*sret(.+) align 1}}
  // CHECK: ret void
  X Thunks::f() { return X(); }

  // WIN64-LABEL: define linkonce_odr dso_local void @"?f@Thunks@Test6@@WBA@EAA?AUX@2@XZ"({{.*}} sret({{.*}}) align 1 %{{.*}})
  // WIN64-NOT: memcpy
  // WIN64: tail call void @"?f@Thunks@Test6@@UEAA?AUX@2@XZ"({{.*}} sret({{.*}}) align 1 %{{.*}})
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

  // CHECK-LABEL: define{{.*}} void @_ZThn8_N5Test71D3bazENS_1XERS1_CfNS_5SmallERS4_NS_5LargeE(
  // CHECK-DBG-NOT: dbg.declare
  // CHECK-NOT: memcpy
  // CHECK: ret void
  void testD() { D d; }

  // MS C++ ABI doesn't use a thunk, so this case isn't interesting.
}

namespace Test8 {
  struct NonPOD { ~NonPOD(); int x, y, z; };
  struct A { virtual void foo(); };
  struct B { virtual void bar(NonPOD); };
  struct C : A, B { virtual void bar(NonPOD); static void helper(NonPOD); };

  // CHECK: define{{.*}} void @_ZN5Test81C6helperENS_6NonPODE([[NONPODTYPE:%.*]]*
  void C::helper(NonPOD var) {}

  // CHECK-LABEL: define{{.*}} void @_ZThn8_N5Test81C3barENS_6NonPODE(
  // CHECK-DBG-NOT: dbg.declare
  // CHECK-NOT: load [[NONPODTYPE]], [[NONPODTYPE]]*
  // CHECK-NOT: memcpy
  // CHECK: ret void
  void C::bar(NonPOD var) {}

  // MS C++ ABI doesn't use a thunk, so this case isn't interesting.
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

  // Test later.
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
  // CHECK-DBG-NOT: dbg.declare
  // CHECK: ret

  //  The return-adjustment thunk required when C::f appears in a vtable
  //  where A is at a zero offset from C.
  // CHECK: define {{.*}} @_ZTch0_v0_n32_N6Test111C1fEv(
  // CHECK-DBG-NOT: dbg.declare
  // CHECK: ret

  // WIN64-LABEL: define dso_local %{{.*}}* @"?f@C@Test11@@UEAAPEAU12@XZ"(i8*

  // WIN64-LABEL: define weak_odr dso_local %{{.*}}* @"?f@C@Test11@@QEAAPEAUA@2@XZ"(i8*
  // WIN64: call %{{.*}}* @"?f@C@Test11@@UEAAPEAU12@XZ"(i8* %{{.*}})
  //
  // Match the vbtable return adjustment.
  // WIN64: load i32*, i32** %{{[^,]*}}, align 8
  // WIN64: getelementptr inbounds i32, i32* %{{[^,]*}}, i32 1
  // WIN64: load i32, i32* %{{[^,]*}}, align 4
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
  C* makeC();
  C* C::f(int x, ...) { return makeC(); }

  // C::f
  // CHECK: define {{.*}} @_ZN6Test121C1fEiz

  // Varargs thunk; check that both the this and covariant adjustments
  // are generated.
  // CHECK: define {{.*}} @_ZTchn8_h8_N6Test121C1fEiz
  // CHECK-DBG-NOT: dbg.declare
  // CHECK: getelementptr inbounds i8, i8* {{.*}}, i64 -8
  // CHECK: getelementptr inbounds i8, i8* {{.*}}, i64 8

  // The vtable layout goes:
  // C vtable in A:
  // - f impl, no adjustment
  // C vtable in B:
  // - f thunk 2, covariant, clone
  // - f thunk 2, musttail this adjust to impl
  // FIXME: The weak_odr linkage is probably not necessary and just an artifact
  // of Itanium ABI details.
  // WIN64-LABEL: define dso_local {{.*}} @"?f@C@Test12@@UEAAPEAU12@HZZ"(
  // WIN64: call %{{.*}}* @"?makeC@Test12@@YAPEAUC@1@XZ"()
  //
  // This thunk needs return adjustment, clone.
  // WIN64-LABEL: define weak_odr dso_local {{.*}} @"?f@C@Test12@@W7EAAPEAUB@2@HZZ"(
  // WIN64: call %{{.*}}* @"?makeC@Test12@@YAPEAUC@1@XZ"()
  // WIN64: getelementptr inbounds i8, i8* %{{.*}}, i32 8
  //
  // Musttail call back to the A implementation after this adjustment from B to A.
  // WIN64-LABEL: define linkonce_odr dso_local %{{.*}}* @"?f@C@Test12@@W7EAAPEAU12@HZZ"(
  // WIN64: getelementptr i8, i8* %{{[^,]*}}, i32 -8
  // WIN64: musttail call {{.*}} @"?f@C@Test12@@UEAAPEAU12@HZZ"(
  C c;
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
  // CHECK-DBG-NOT: dbg.declare
  // CHECK: getelementptr inbounds i8, i8* {{.*}}, i64 -8
  // CHECK: getelementptr inbounds i8, i8* {{.*}}, i64 -32
  // CHECK: getelementptr inbounds i8, i8* {{.*}}, i64 -24
  // CHECK: getelementptr inbounds i8, i8* {{.*}}, i64 8
  // CHECK: ret %"struct.Test13::D"*

  // WIN64-LABEL: define weak_odr dso_local %"struct.Test13::D"* @"?foo1@D@Test13@@$4PPPPPPPE@A@EAAAEAUB1@2@XZ"(
  //    This adjustment.
  // WIN64: getelementptr inbounds i8, i8* {{.*}}, i64 -12
  //    Call implementation.
  // WIN64: call {{.*}} @"?foo1@D@Test13@@UEAAAEAU12@XZ"(i8* {{.*}})
  //    Virtual + nonvirtual return adjustment.
  // WIN64: load i32*, i32** %{{[^,]*}}, align 8
  // WIN64: getelementptr inbounds i32, i32* %{{[^,]*}}, i32 1
  // WIN64: load i32, i32* %{{[^,]*}}, align 4
  // WIN64: getelementptr inbounds i8, i8* %{{[^,]*}}, i32 %{{[^,]*}}
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
  // CHECK: define{{.*}} void @_ZThn8_N6Test141C1fEv({{.*}}) unnamed_addr [[NUW:#[0-9]+]]
  // CHECK-DBG-NOT: dbg.declare
  // CHECK: ret void
}

// Varargs non-covariant thunk test.
// PR18098
namespace Test15 {
  struct A {
    virtual ~A();
  };
  struct B {
    virtual void f(int x, ...);
  };
  struct C : A, B {
    virtual void c();
    virtual void f(int x, ...);
  };
  void C::c() {}

  // C::c
  // CHECK-CLONE: declare void @_ZN6Test151C1fEiz
  // non-virtual thunk to C::f
  // CHECK-CLONE: declare void @_ZThn8_N6Test151C1fEiz

  // If we have musttail, then we emit the thunk as available_externally.
  // CHECK-TAIL: declare void @_ZN6Test151C1fEiz
  // CHECK-TAIL: define available_externally void @_ZThn8_N6Test151C1fEiz({{.*}})
  // CHECK-TAIL: musttail call void (%"struct.Test15::C"*, i32, ...) @_ZN6Test151C1fEiz({{.*}}, ...)

  // MS C++ ABI doesn't use a thunk, so this case isn't interesting.
}

namespace Test16 {
struct A {
  virtual ~A();
};
struct B {
  virtual void foo();
};
struct C : public A, public B {
  void foo() {}
};
struct D : public C {
  ~D();
};
D::~D() {}
// CHECK: define linkonce_odr void @_ZThn8_N6Test161C3fooEv({{.*}}) {{.*}} comdat
// CHECK-DBG-NOT: dbg.declare
// CHECK: ret void
}

namespace Test17 {
class A {
  virtual void f(const char *, ...);
};
class B {
  virtual void f(const char *, ...);
};
class C : A, B {
  virtual void anchor();
  void f(const char *, ...) override;
};
// Key method and object anchor vtable for Itanium and MSVC.
void C::anchor() {}
C c;

// CHECK-CLONE-LABEL: declare void @_ZThn8_N6Test171C1fEPKcz(

// CHECK-TAIL-LABEL: define available_externally void @_ZThn8_N6Test171C1fEPKcz(
// CHECK-TAIL: getelementptr inbounds i8, i8* %{{.*}}, i64 -8
// CHECK-TAIL: musttail call {{.*}} @_ZN6Test171C1fEPKcz({{.*}}, ...)

// MSVC-LABEL: define linkonce_odr dso_local void @"?f@C@Test17@@G7EAAXPEBDZZ"
// MSVC-SAME: (%"class.Test17::C"* %this, i8* %[[ARG:[^,]+]], ...)
// MSVC: getelementptr i8, i8* %{{.*}}, i32 -8
// MSVC: musttail call void (%"class.Test17::C"*, i8*, ...) @"?f@C@Test17@@EEAAXPEBDZZ"(%"class.Test17::C"* %{{.*}}, i8* %[[ARG]], ...)
}

/**** The following has to go at the end of the file ****/

// checking without opt
// CHECK-NONOPT-LABEL: define internal void @_ZThn8_N6Test4B12_GLOBAL__N_11C1fEv(
// CHECK-NONOPT-NOT: comdat

// This is from Test5:
// CHECK-NONOPT-LABEL: define linkonce_odr void @_ZTv0_n24_N5Test51B1fEv

// This is from Test10:
// CHECK-NONOPT-LABEL: define linkonce_odr void @_ZN6Test101C3fooEv
// CHECK-NONOPT-LABEL: define linkonce_odr void @_ZThn8_N6Test101C3fooEv

// Checking with opt
// CHECK-OPT-LABEL: define internal void @_ZThn8_N6Test4B12_GLOBAL__N_11C1fEv(%"struct.Test4B::(anonymous namespace)::C"* %this) unnamed_addr #1 align 2

// This is from Test5:
// CHECK-OPT-LABEL: define linkonce_odr void @_ZTv0_n24_N5Test51B1fEv

// This is from Test10:
// CHECK-OPT-LABEL: define linkonce_odr void @_ZN6Test101C3fooEv
// CHECK-OPT-LABEL: define linkonce_odr void @_ZThn8_N6Test101C3fooEv

// This is from Test10:
// WIN64-LABEL: define linkonce_odr dso_local void @"?foo@C@Test10@@UEAAXXZ"(
// WIN64-LABEL: define linkonce_odr dso_local void @"?foo@C@Test10@@W7EAAXXZ"(

// CHECK-NONOPT: attributes [[NUW]] = { noinline nounwind optnone uwtable{{.*}} }
// CHECK-OPT: attributes [[NUW]] = { nounwind uwtable{{.*}} }
