// RUN: %clang_cc1 -fsyntax-only -triple %itanium_abi_triple -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -triple %ms_abi_triple -DMSABI -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -triple %itanium_abi_triple -verify -std=c++11 %s
// RUN: %clang_cc1 -fsyntax-only -triple %ms_abi_triple -DMSABI -verify -std=c++11 %s

typedef __typeof(sizeof(int)) size_t;

// PR7803
namespace test0 {
  class A {
  public:
    static void operator delete(void *p) {};
    virtual ~A();
  };

  class B : protected A {
  public:
    ~B();
  };

  class C : protected B {
  public:
    using B::operator delete;
    ~C();
  };

  // Shouldn't have an error.
  C::~C() {}
}

namespace test1 {
  class A {
  public:
    static void operator delete(void *p) {};
    virtual ~A();
  };

  class B : protected A {
  public:
    static void operator delete(void *, size_t) {};
    ~B();
  };

  class C : protected B {
  public:
    using A::operator delete;
    using B::operator delete;

    ~C();
  };

  // We assume that the intent is to treat C::operator delete(void*, size_t) as
  // /not/ being a usual deallocation function, as it would be if it were
  // declared with in C directly.
  C::~C() {}

  struct D {
    void operator delete(void*); // expected-note {{member 'operator delete' declared here}}
    void operator delete(void*, ...); // expected-note {{member 'operator delete' declared here}}
    virtual ~D();
  };
  // FIXME: The standard doesn't say this is ill-formed, but presumably either
  // it should be or the variadic operator delete should not be a usual
  // deallocation function.
  D::~D() {} // expected-error {{multiple suitable 'operator delete' functions in 'D'}}
}

// ...at the point of definition of a virtual destructor...
namespace test2 {
  struct A {
    virtual ~A();
    static void operator delete(void*, const int &);
  };

  struct B {
    virtual ~B();
    static void operator delete(void*, const int &); // expected-note {{declared here}}
  };
  B::~B() {} // expected-error {{no suitable member 'operator delete' in 'B'}}

#if __cplusplus < 201103L
  struct CBase { virtual ~CBase(); };
  struct C : CBase { // expected-error {{no suitable member 'operator delete' in 'C'}}
    static void operator delete(void*, const int &); // expected-note {{declared here}}
  };
  void test() {
    C c; // expected-note {{first required here}}
  }
#else
  struct CBase { virtual ~CBase(); }; // expected-note {{overridden virtual function is here}}
  struct C : CBase { // expected-error {{deleted function '~C' cannot override a non-deleted function}} expected-note 2{{requires an unambiguous, accessible 'operator delete'}}
    static void operator delete(void*, const int &);
  };
  void test() {
    C c; // expected-error {{attempt to use a deleted function}}
  }
#endif
}

// PR7346
namespace test3 {
  struct A {
#ifdef MSABI
    // expected-error@+2 {{no suitable member 'operator delete' in 'A'}}
#endif
    virtual ~A();
#ifdef MSABI
    // expected-note@+2 {{declared here}}
#endif
    static void operator delete(void*, const int &);
  };

  struct B : A {
    virtual ~B() {}
    static void operator delete(void*);
  };

  void f() {
#ifdef MSABI
    // expected-note@+2 {{implicit default constructor for 'test3::B' first required here}}
#endif
    B use_vtable;
  }
}
