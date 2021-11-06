// RUN: %clang_cc1 %s -I%S -triple=x86_64-apple-darwin10 -std=c++98 -emit-llvm -o %t
// RUN: %clang_cc1 %s -I%S -triple=x86_64-apple-darwin10 -std=c++98 -O2 -disable-llvm-passes -emit-llvm -o %t.opt
// RUN: %clang_cc1 %s -I%S -triple=x86_64-apple-darwin10 -std=c++98 -O2 -disable-llvm-passes -emit-llvm -o %t.vtable -fforce-emit-vtables -fstrict-vtable-pointers -mconstructor-aliases
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefix=CHECK-TEST1 %s < %t
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefix=CHECK-TEST2 %s < %t
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefix=CHECK-TEST5 %s < %t
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefix=CHECK-TEST8 %s < %t.opt
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefix=CHECK-TEST9 %s < %t.opt
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefix=CHECK-TEST10 %s < %t.opt
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefix=CHECK-TEST11 %s < %t.opt
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefix=CHECK-TEST12 %s < %t.opt
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefix=CHECK-TEST13 %s < %t.opt
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefix=CHECK-TEST14 %s < %t.opt
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefix=CHECK-TEST15 %s < %t.opt
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefix=CHECK-TEST16 %s < %t.opt
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefix=CHECK-TEST17 %s < %t.opt
// RUN: FileCheck -allow-deprecated-dag-overlap --check-prefix=CHECK-FORCE-EMIT %s < %t.vtable


#include <typeinfo>

// CHECK-TEST1: @_ZTVN5Test11AE = external unnamed_addr constant
// CHECK-FORCE-EMIT-DAG: @_ZTVN5Test11AE = available_externally unnamed_addr constant
namespace Test1 {

struct A {
  A();
  virtual void f();
  virtual ~A() { }
};

A::A() { }

void f(A* a) {
  a->f();
};

// CHECK-LABEL: define{{.*}} void @_ZN5Test11gEv
// CHECK: call void @_ZN5Test11A1fEv
void g() {
  A a;
  f(&a);
}

}

// Test2::A's key function (f) is defined in this translation unit, but when
// we're doing codegen for the typeid(A) call, we don't know that yet.
// This tests mainly that the typeinfo and typename constants have their linkage
// updated correctly.

// CHECK-TEST2: @_ZTSN5Test21AE ={{.*}} constant
// CHECK-TEST2: @_ZTIN5Test21AE ={{.*}} constant
// CHECK-TEST2: @_ZTVN5Test21AE ={{.*}} unnamed_addr constant
namespace Test2 {
  struct A {
    virtual void f();
  };

  const std::type_info &g() {
    return typeid(A);
  };

  void A::f() { }
}

// Test that we don't assert on this test.
namespace Test3 {

struct A {
  virtual void f();
  virtual ~A() { }
};

struct B : A {
  B();
  virtual void f();
};

B::B() { }

void g(A* a) {
  a->f();
};

}

// PR9114, test that we don't try to instantiate RefPtr<Node>.
namespace Test4 {

template <class T> struct RefPtr {
  T* p;
  ~RefPtr() {
    p->deref();
  }
};

struct A {
  virtual ~A();
};

struct Node;

struct B : A {
  virtual void deref();
  RefPtr<Node> m;
};

void f() {
  RefPtr<B> b;
}

}

// PR9130, test that we emit a definition of A::f.
// CHECK-TEST5-LABEL: define linkonce_odr void @_ZN5Test51A1fEv
namespace Test5 {

struct A {
  virtual void f() { }
};

struct B : A { 
  virtual ~B();
};

B::~B() { }

}

// Check that we don't assert on this test.
namespace Test6 {

struct A {
  virtual ~A();
  int a;
};

struct B {
  virtual ~B();
  int b;
};

struct C : A, B { 
  C();
};

struct D : C {
  virtual void f();
  D();
};

D::D() { }

}

namespace Test7 {

struct c1 {};
struct c10 : c1{
  virtual void foo ();
};
struct c11 : c10, c1{
  virtual void f6 ();
};
struct c28 : virtual c11{
  void f6 ();
};
}

namespace Test8 {
// CHECK-TEST8: @_ZTVN5Test81YE = available_externally unnamed_addr constant
// vtable for X is not generated because there are no stores here
struct X {
  X();
  virtual void foo();
};
struct Y : X {
  void foo();
};

void g(X* p) { p->foo(); }
void f() {
  Y y;
  g(&y);
  X x;
  g(&x);
}

}  // Test8

namespace Test9 {
// All virtual functions are outline, so we can assume that it will
// be generated in translation unit where foo is defined.
// CHECK-TEST9-DAG: @_ZTVN5Test91AE = available_externally unnamed_addr constant
// CHECK-TEST9-DAG: @_ZTVN5Test91BE = available_externally unnamed_addr constant
struct A {
  virtual void foo();
  virtual void bar();
};
void A::bar() {}

struct B : A {
  void foo();
};

void g() {
  A a;
  a.foo();
  B b;
  b.foo();
}

}  // Test9

namespace Test10 {

// because A's key function is defined here, vtable is generated in this TU
// CHECK-TEST10-DAG: @_ZTVN6Test101AE ={{.*}} unnamed_addr constant
// CHECK-FORCE-EMIT-DAG: @_ZTVN6Test101AE ={{.*}} unnamed_addr constant
struct A {
  virtual void foo();
  virtual void bar();
};
void A::foo() {}

// Because key function is inline we will generate vtable as linkonce_odr.
// CHECK-TEST10-DAG: @_ZTVN6Test101DE = linkonce_odr unnamed_addr constant
// CHECK-FORCE-EMIT-DAG: @_ZTVN6Test101DE = linkonce_odr unnamed_addr constant
struct D : A {
  void bar();
};
inline void D::bar() {}

// Because B has outline all virtual functions, we can refer to them.
// CHECK-TEST10-DAG: @_ZTVN6Test101BE = available_externally unnamed_addr constant
struct B : A {
  void foo();
  void bar();
};

// C's key function (car) is outline, but C has inline virtual function so we
// can't guarantee that we will be able to refer to bar from name
// so (at the moment) we can't emit vtable available_externally.
// CHECK-TEST10-DAG: @_ZTVN6Test101CE = external unnamed_addr constant
// CHECK-FORCE-EMIT-DAG: @_ZTVN6Test101CE = available_externally unnamed_addr constant
struct C : A {
  void bar() {}               // defined in body - not key function
  virtual inline void gar();  // inline in body - not key function
  virtual void car();
};

// no key function, vtable will be generated everywhere it will be used
// CHECK-TEST10-DAG: @_ZTVN6Test101EE = linkonce_odr unnamed_addr constant
// CHECK-FORCE-EMIT-DAG: @_ZTVN6Test101EE = linkonce_odr unnamed_addr constant

struct E : A {};

void g(A& a) {
  a.foo();
  a.bar();
}

void f() {
  A a;
  g(a);
  B b;
  g(b);
  C c;
  g(c);
  D d;
  g(d);
  E e;
  g(e);
}

}  // Test10

namespace Test11 {
struct D;
// Can emit C's vtable available_externally.
// CHECK-TEST11: @_ZTVN6Test111CE = available_externally unnamed_addr constant
struct C {
  virtual D& operator=(const D&);
};

// Can emit D's vtable available_externally.
// CHECK-TEST11: @_ZTVN6Test111DE = available_externally unnamed_addr constant
struct D : C {
  virtual void key();
};
D f();

void g(D& a) {
  C c;
  c = a;
  a.key();
  a.key();
}
void g() {
  D d;
  d = f();
  g(d);
}
}  // Test 11

namespace Test12 {

// CHECK-TEST12: @_ZTVN6Test121AE = external unnamed_addr constant
// CHECK-FORCE-EMIT-DAG: @_ZTVN6Test121AE = available_externally unnamed_addr constant
struct A {
  virtual void foo();
  virtual ~A() {}
};
// CHECK-TEST12: @_ZTVN6Test121BE = external unnamed_addr constant
// CHECK-FORCE-EMIT-DAG: @_ZTVN6Test121BE = available_externally unnamed_addr constant
struct B : A {
  void foo();
};

void g() {
  A a;
  a.foo();
  B b;
  b.foo();
}
}

namespace Test13 {

// CHECK-TEST13-DAG: @_ZTVN6Test131AE = available_externally unnamed_addr constant
// CHECK-TEST13-DAG: @_ZTVN6Test131BE = external unnamed_addr constant
// CHECK-FORCE-EMIT-DAG: @_ZTVN6Test131AE = available_externally unnamed_addr constant
// CHECK-FORCE-EMIT-DAG: @_ZTVN6Test131BE = available_externally unnamed_addr constant

struct A {
  virtual ~A();
};
struct B : A {
  virtual void f();
  void operator delete(void *);
  ~B() {}
};

void g() {
  A *b = new B;
}
}

namespace Test14 {

// CHECK-TEST14: @_ZTVN6Test141AE = available_externally unnamed_addr constant
struct A {
  virtual void f();
  void operator delete(void *);
  ~A();
};

void g() {
  A *b = new A;
  delete b;
}
}

namespace Test15 {
// In this test D's vtable has two slots for function f(), but uses only one,
// so the second slot is set to null.
// CHECK-TEST15: @_ZTVN6Test151DE = available_externally unnamed_addr constant
struct A { virtual void f() {} };
struct B : virtual A {};
struct C : virtual A {};
struct D : B, C {
  virtual void g();
  void f();
};

void test() {
  D * d = new D;
  d->f();
}
}

namespace Test16 {
// S has virtual method that is hidden, because of it we can't
// generate available_externally vtable for it.
// CHECK-TEST16-DAG: @_ZTVN6Test161SE = external unnamed_addr constant
// CHECK-TEST16-DAG: @_ZTVN6Test162S2E = available_externally
// CHECK-FORCE-EMIT-DAG: @_ZTVN6Test161SE = external unnamed_addr constant
// CHECK-FORCE-EMIT-DAG: @_ZTVN6Test162S2E = available_externally

struct S {
  __attribute__((visibility("hidden"))) virtual void doStuff();
};

struct S2 {
  virtual void doStuff();
  __attribute__((visibility("hidden"))) void unused();

};

void test() {
  S *s = new S;
  s->doStuff();

  S2 *s2 = new S2;
  s2->doStuff();
}
}

namespace Test17 {
// This test checks if we emit vtables opportunistically.
// CHECK-TEST17-DAG: @_ZTVN6Test171AE = available_externally
// CHECK-TEST17-DAG: @_ZTVN6Test171BE = external
// CHECK-FORCE-EMIT-DAG: @_ZTVN6Test171AE = available_externally
// CHECK-FORCE-EMIT-DAG: @_ZTVN6Test171BE = available_externally
// CHECK-FORCE-EMIT-DAG: define linkonce_odr void @_ZN6Test171BD2Ev(
// CHECK-FORCE-EMIT-DAG: define linkonce_odr void @_ZN6Test171BD0Ev(

struct A {
  virtual void key();
  virtual void bar() {}
};

// We won't gonna use deleting destructor for this type, which will disallow
// emitting vtable as available_externally
struct B {
  virtual void key();
  virtual ~B() {}
};

void testcaseA() {
  A a;
  a.bar(); // this forces to emit definition of bar
}

void testcaseB() {
  B b; // This only forces emitting of complete object destructor
}

} // namespace Test17

namespace Test18 {
// Here vtable will be only emitted because it is referenced by assume-load
// after the Derived construction.
// CHECK-FORCE-EMIT-DAG: @_ZTVN6Test187DerivedE = linkonce_odr unnamed_addr constant {{.*}} @_ZTIN6Test187DerivedE {{.*}} @_ZN6Test184Base3funEv {{.*}} @_ZN6Test184BaseD2Ev {{.*}} @_ZN6Test187DerivedD0Ev
// CHECK-FORCE-EMIT-DAG: define linkonce_odr void @_ZN6Test187DerivedD0Ev
// CHECK-FORCE-EMIT-DAG: define linkonce_odr void @_ZN6Test184BaseD2Ev
// CHECK-FORCE-EMIT-DAG: define linkonce_odr i32 @_ZN6Test184Base3funEv
// CHECK-FORCE-EMIT-DAG: @_ZTIN6Test187DerivedE = linkonce_odr constant

struct Base {
  virtual int fun() { return 42; }
  virtual ~Base() { }
};

struct Derived : Base {
  Derived();
};

int foo() {
  Derived *der = new Derived();
  return der->fun();
}
}

namespace TestTemplates {

// CHECK-FORCE-EMIT-DAG: @_ZTVN13TestTemplates8TemplateIiEE = linkonce_odr unnamed_addr constant {{.*}} @_ZTIN13TestTemplates8TemplateIiEE {{.*}} @_ZN13TestTemplates8TemplateIiE3fooEi {{.*}}@_ZN13TestTemplates8TemplateIiE22thisShouldBeEmittedTooEi {{.*}}@_ZN13TestTemplates8TemplateIiED1Ev {{.*}}@_ZN13TestTemplates8TemplateIiED0Ev
// CHECK-FORCE-EMIT-DAG: define linkonce_odr i32 @_ZN13TestTemplates8TemplateIiE22thisShouldBeEmittedTooEi

template<class T>
struct Template {
  Template();
  virtual T foo(T val);
  // CHECK-FORCE-EMIT-DAG: define linkonce_odr i32 @_ZN13TestTemplates8TemplateIiE22thisShouldBeEmittedTooEi
  virtual T thisShouldBeEmittedToo(T val) { return val; }
  virtual ~Template();
};


struct NonTemplate {
  typedef int T;
  NonTemplate();
  virtual T foo(T val);
  // CHECK-FORCE-EMIT-DAG: define linkonce_odr i32 @_ZN13TestTemplates11NonTemplate22thisShouldBeEmittedTooEi
  virtual T thisShouldBeEmittedToo(T val) { return val; }
  virtual ~NonTemplate();
};

// CHECK-FORCE-EMIT-DAG: @_ZTVN13TestTemplates16OuterNonTemplate27NestedTemplateInNonTemplateIiEE = linkonce_odr {{.*}} @_ZTIN13TestTemplates16OuterNonTemplate27NestedTemplateInNonTemplateIiEE {{.*}} @_ZN13TestTemplates16OuterNonTemplate27NestedTemplateInNonTemplateIiE3fooEi {{.*}} @_ZN13TestTemplates16OuterNonTemplate27NestedTemplateInNonTemplateIiE22thisShouldBeEmittedTooEi {{.*}} @_ZN13TestTemplates16OuterNonTemplate27NestedTemplateInNonTemplateIiED1Ev {{.*}} @_ZN13TestTemplates16OuterNonTemplate27NestedTemplateInNonTemplateIiED0Ev

struct OuterNonTemplate {
  template<class T>
  struct NestedTemplateInNonTemplate {
    NestedTemplateInNonTemplate();
    virtual T foo(T val);
    // CHECK-FORCE-EMIT-DAG: define linkonce_odr i32 @_ZN13TestTemplates16OuterNonTemplate27NestedTemplateInNonTemplateIiE22thisShouldBeEmittedTooEi
    virtual T thisShouldBeEmittedToo(T val) { return val; }
    virtual ~NestedTemplateInNonTemplate();
  };

  struct NestedNonTemplateInNonTemplate {
    typedef int T;
    NestedNonTemplateInNonTemplate();
    virtual T foo(T val);
    // CHECK-FORCE-EMIT-DAG: define linkonce_odr i32 @_ZN13TestTemplates16OuterNonTemplate30NestedNonTemplateInNonTemplate22thisShouldBeEmittedTooEi
    virtual T thisShouldBeEmittedToo(T val) { return val; }
    virtual ~NestedNonTemplateInNonTemplate();
  };
};

template<class>
struct OuterTemplate {
  template<class T>
  struct NestedTemplateInTemplate {
    NestedTemplateInTemplate();
    virtual T foo(T val);
    // CHECK-FORCE-EMIT-DAG: define linkonce_odr i32 @_ZN13TestTemplates13OuterTemplateIlE24NestedTemplateInTemplateIiE22thisShouldBeEmittedTooEi
    virtual T thisShouldBeEmittedToo(T val) { return val; }
    virtual ~NestedTemplateInTemplate();
  };

  struct NestedNonTemplateInTemplate {
    typedef int T;
    NestedNonTemplateInTemplate();
    virtual T foo(T val);
    // CHECK-FORCE-EMIT-DAG: define linkonce_odr i32 @_ZN13TestTemplates13OuterTemplateIlE27NestedNonTemplateInTemplate22thisShouldBeEmittedTooEi
    virtual T thisShouldBeEmittedToo(T val) { return val; }
    virtual ~NestedNonTemplateInTemplate();
  };
};

template<class T>
int use() {
  T *ptr = new T();
  return ptr->foo(42);
}

void test() {
  use<Template<int> >();
  use<OuterTemplate<long>::NestedTemplateInTemplate<int> >();
  use<OuterNonTemplate::NestedTemplateInNonTemplate<int> >();

  use<NonTemplate>();
  use<OuterTemplate<long>::NestedNonTemplateInTemplate>();
  use<OuterNonTemplate::NestedNonTemplateInNonTemplate>();
}
}
