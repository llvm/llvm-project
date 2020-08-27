// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11 -Wabstract-vbase-init

#ifndef __GXX_EXPERIMENTAL_CXX0X__
#define __CONCAT(__X, __Y) __CONCAT1(__X, __Y)
#define __CONCAT1(__X, __Y) __X ## __Y

#define static_assert(__b, __m) \
  typedef int __CONCAT(__sa, __LINE__)[__b ? 1 : -1]
#endif

union IncompleteUnion;

static_assert(!__is_abstract(IncompleteUnion), "unions are never abstract");

class C {
  virtual void f() = 0; // expected-note {{unimplemented pure virtual method 'f'}}
};

static_assert(__is_abstract(C), "C has a pure virtual function");

class D : C {
};

static_assert(__is_abstract(D), "D inherits from an abstract class");

class E : D {
  virtual void f();
};

static_assert(!__is_abstract(E), "E inherits from an abstract class but implements f");

C *d = new C; // expected-error {{allocating an object of abstract class type 'C'}}

C c; // expected-error {{variable type 'C' is an abstract class}}
void t1(C c); // expected-error {{parameter type 'C' is an abstract class}}
void t2(C); // expected-error {{parameter type 'C' is an abstract class}}

struct S {
  C c; // expected-error {{field type 'C' is an abstract class}}
};

void t3(const C&);

void f() {
  C(); // expected-error {{allocating an object of abstract class type 'C'}}
  t3(C()); // expected-error {{allocating an object of abstract class type 'C'}}
}

C e1[2]; // expected-error {{array of abstract class type 'C'}}
C (*e2)[2]; // expected-error {{array of abstract class type 'C'}}
C (**e3)[2]; // expected-error {{array of abstract class type 'C'}}

void t4(C c[2]); // expected-error {{array of abstract class type 'C'}}

void t5(void (*)(C)); // expected-error {{parameter type 'C' is an abstract class}}

typedef void (*Func)(C); // expected-error {{parameter type 'C' is an abstract class}}
void t6(Func);

class F {
  F a() { while (1) {} } // expected-error {{return type 'F' is an abstract class}}
    
  class D {
    void f(F c); // expected-error {{parameter type 'F' is an abstract class}}
  };

  union U {
    void u(F c); // expected-error {{parameter type 'F' is an abstract class}}
  };
    
  virtual void f() = 0; // expected-note {{unimplemented pure virtual method 'f'}}
};

// Diagnosing in these cases is prohibitively expensive.  We still
// diagnose at the function definition, of course.

class Abstract;

void t7(Abstract a);

void t8() {
  void h(Abstract a);
}

namespace N {
void h(Abstract a);
}

class Abstract {
  virtual void f() = 0;
};

// <rdar://problem/6854087>
class foo {
public:
  virtual foo *getFoo() = 0;
};

class bar : public foo {
public:
  virtual bar *getFoo();
};

bar x;

// <rdar://problem/6902298>
class A {
public:
  virtual void release() = 0;
  virtual void release(int count) = 0;
  virtual void retain() = 0;
};

class B : public A {
public:
  virtual void release();
  virtual void release(int count);
  virtual void retain();
};

void foo(void) {
  B b;
}

struct K {
 int f;
 virtual ~K();
};

struct L : public K {
 void f();
};

// PR5222
namespace PR5222 {
  struct A {
    virtual A *clone() = 0;
  };
  struct B : public A {
    virtual B *clone() = 0;
  };
  struct C : public B {
    virtual C *clone();
  };

  C c;  
}

// PR5550 - instantiating template didn't track overridden methods
namespace PR5550 {
  struct A {
    virtual void a() = 0;
    virtual void b() = 0;
  };
  template<typename T> struct B : public A {
    virtual void b();
    virtual void c() = 0;
  };
  struct C : public B<int> {
    virtual void a();
    virtual void c();
  }; 
  C x;
}

namespace PureImplicit {
  // A pure virtual destructor should be implicitly overridden.
  struct A { virtual ~A() = 0; };
  struct B : A {};
  B x;

  // A pure virtual assignment operator should be implicitly overridden.
  struct D;
  struct C { virtual D& operator=(const D&) = 0; };
  struct D : C {};
  D y;
}

namespace test1 {
  struct A {
    virtual void foo() = 0;
  };

  struct B : A {
    using A::foo;
  };

  struct C : B {
    void foo();
  };

  void test() {
    C c;
  }
}

// rdar://problem/8302168
namespace test2 {
  struct X1 {
    virtual void xfunc(void) = 0;  // expected-note {{unimplemented pure virtual method}}
    void g(X1 parm7);        // expected-error {{parameter type 'test2::X1' is an abstract class}}
    void g(X1 parm8[2]);     // expected-error {{array of abstract class type 'test2::X1'}}
  };

  template <int N>
  struct X2 {
    virtual void xfunc(void) = 0;  // expected-note {{unimplemented pure virtual method}}
    void g(X2 parm10);        // expected-error {{parameter type 'X2<N>' is an abstract class}}
    void g(X2 parm11[2]);     // expected-error {{array of abstract class type 'X2<N>'}}
  };
}

namespace test3 {
  struct A { // expected-note {{not complete until}}
    A x; // expected-error {{field has incomplete type}}
    virtual void abstract() = 0;
  };

  struct B { // expected-note {{not complete until}}
    virtual void abstract() = 0;
    B x; // expected-error {{field has incomplete type}}
  };

  struct C {
    static C x; // expected-error {{abstract class}}
    virtual void abstract() = 0; // expected-note {{unimplemented pure virtual method}}
  };

  struct D {
    virtual void abstract() = 0; // expected-note {{unimplemented pure virtual method}}
    static D x; // expected-error {{abstract class}}
  };
}

namespace test4 {
  template <class T> struct A {
    A x; // expected-error {{abstract class}}
    virtual void abstract() = 0; // expected-note {{unimplemented pure virtual method}}
  };

  template <class T> struct B {
    virtual void abstract() = 0; // expected-note {{unimplemented pure virtual method}}
    B x; // expected-error {{abstract class}}
  };

  template <class T> struct C {
    static C x; // expected-error {{abstract class}}
    virtual void abstract() = 0; // expected-note {{unimplemented pure virtual method}}
  };

  template <class T> struct D {
    virtual void abstract() = 0; // expected-note {{unimplemented pure virtual method}}
    static D x; // expected-error {{abstract class}}
  };
}

namespace test5 {
  struct A { A(int); virtual ~A() = 0; }; // expected-note {{pure virtual method}}
  const A &a = 0; // expected-error {{abstract class}}
  void f(const A &a = 0); // expected-error {{abstract class}}
  void g(const A &a);
  void h() { g(0); } // expected-error {{abstract class}}
}

// PR9247: Crash on invalid in clang::Sema::ActOnFinishCXXMemberSpecification
namespace pr9247 {
  struct A {
    virtual void g(const A& input) = 0;
    struct B {
      C* f(int foo);
    };
  };
}

namespace pr12658 {
  class C {
    public:
      C(int v){}
      virtual void f() = 0; // expected-note {{unimplemented pure virtual method 'f' in 'C'}}
  };

  void foo(const C& c ) {}

  void bar( void ) {
    foo(C(99)); // expected-error {{allocating an object of abstract class type 'pr12658::C'}}
  }
}

namespace pr16659 {
  struct A {
    A(int);
    virtual void x() = 0; // expected-note {{unimplemented pure virtual method 'x' in 'RedundantInit'}}
  };
  struct B : virtual A {};
  struct C : B {
    C() : A(37) {}
    void x() override {}
  };

  struct X {
    friend class Z;
  private:
    X &operator=(const X&);
  };
  struct Y : virtual X { // expected-note {{::X' has an inaccessible copy assignment}}
    virtual ~Y() = 0;
  };
  struct Z : Y {}; // expected-note {{::Y' has a deleted copy assignment}}
  void f(Z &a, const Z &b) { a = b; } // expected-error {{copy assignment operator is implicitly deleted}}

  struct RedundantInit : virtual A {
    RedundantInit() : A(0) {} // expected-warning {{initializer for virtual base class 'pr16659::A' of abstract class 'RedundantInit' will never be used}}
  };
}
