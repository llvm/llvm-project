// RUN: %clang_cc1 -std=c++11 -triple %itanium_abi_triple -fsyntax-only -Wnon-virtual-dtor -Wdelete-non-virtual-dtor -fcxx-exceptions -verify %s
// RUN: %clang_cc1 -std=c++11 -triple %ms_abi_triple -DMSABI -fsyntax-only -Wnon-virtual-dtor -Wdelete-non-virtual-dtor -verify %s

#if defined(BE_THE_HEADER)

// Wdelete-non-virtual-dtor should warn about the delete from smart pointer
// classes in system headers (std::unique_ptr...) too.

#pragma clang system_header
namespace dnvd {

struct SystemB {
  virtual void foo();
};

template <typename T>
class simple_ptr {
public:
  simple_ptr(T* t): _ptr(t) {}
  ~simple_ptr() { delete _ptr; } // \
    // expected-warning {{delete called on non-final 'dnvd::B' that has virtual functions but non-virtual destructor}} \
    // expected-warning {{delete called on non-final 'dnvd::D' that has virtual functions but non-virtual destructor}}
  T& operator*() const { return *_ptr; }
private:
  T* _ptr;
};
}

#else

#define BE_THE_HEADER
#include __FILE__

class A {
public:
  ~A();
};

class B {
public:
  ~B() { }
};

class C {
public:
  (~C)() { }
};

struct D {
  static void ~D(int, ...) const { } //                          \
    // expected-error{{static member function cannot have 'const' qualifier}} \
    // expected-error{{destructor cannot be declared 'static'}}  \
    // expected-error{{destructor cannot have any parameters}}   \
    // expected-error{{destructor cannot be variadic}} \
    // expected-error{{destructor cannot have a return type}} \
    // expected-error{{'const' qualifier is not allowed on a destructor}}
};

struct D2 {
  void ~D2() { } //                          \
  // expected-error{{destructor cannot have a return type}}  
};


struct E;

typedef E E_typedef;
struct E {
  ~E_typedef(); // expected-error{{destructor cannot be declared using a typedef 'E_typedef' (aka 'E') of the class name}}
};

struct F {
  (~F)(); // expected-note {{previous declaration is here}}
  ~F(); // expected-error {{destructor cannot be redeclared}}
};

~; // expected-error {{expected a class name after '~' to name a destructor}}
~undef(); // expected-error {{expected the class name after '~' to name a destructor}}
~operator+(int, int);  // expected-error {{expected a class name after '~' to name a destructor}}
~F(){} // expected-error {{destructor must be a non-static member function}}

struct G {
  ~G();
};

G::~G() { }

// <rdar://problem/6841210>
struct H {
  ~H(void) { } 
};

struct X {};

struct Y {
  ~X(); // expected-error {{expected the class name after '~' to name the enclosing class}}
};

namespace PR6421 {
  class T; // expected-note{{forward declaration}}

  class QGenericArgument
  {
    template<typename U>
    void foo(T t) // expected-error{{variable has incomplete type}}
    { }
    
    void disconnect()
    {
      T* t;
      bob<QGenericArgument>(t); // expected-error{{undeclared identifier 'bob'}}
    }
  };
}

namespace PR6709 {
#ifdef MSABI
  // This bug, "Clang instantiates destructor for function argument" is intended
  // behaviour in the Microsoft ABI because the callee needs to destruct the arguments.
  // expected-error@+3 {{indirection requires pointer operand ('int' invalid)}}
  // expected-note@+3 {{in instantiation of member function 'PR6709::X<int>::~X' requested here}}
#endif
  template<class T> class X { T v; ~X() { ++*v; } };
  void a(X<int> x) {}
}

struct X0 { virtual ~X0() throw(); };
struct X1 : public X0 { };

// Make sure we instantiate operator deletes when building a virtual
// destructor.
namespace test6 {
  template <class T> class A {
  public:
    void *operator new(__SIZE_TYPE__);
    void operator delete(void *p) {
      T::deleteIt(p); // expected-error {{type 'int' cannot be used prior to '::'}}
    }

#ifdef MSABI
    // expected-note@+2 {{in instantiation of member function 'test6::A<int>::operator delete' requested here}}
#endif
    virtual ~A() {}
  };

#ifndef MSABI
    // expected-note@+2 {{in instantiation of member function 'test6::A<int>::operator delete' requested here}}
#endif
  class B : A<int> { B(); };
  B::B() {}
}

// Make sure classes are marked invalid when they have invalid
// members.  This avoids a crash-on-invalid.
namespace test7 {
  struct A {
    ~A() const; // expected-error {{'const' qualifier is not allowed on a destructor}}
  };
  struct B : A {};

  void test() {
    B *b;
    b->~B();
  }
}

namespace nonvirtualdtor {
struct S1 { // expected-warning {{has virtual functions but non-virtual destructor}}
  virtual void m();
};

struct S2 {
  ~S2(); // expected-warning {{has virtual functions but non-virtual destructor}}
  virtual void m();
};

struct S3 : public S1 {  // expected-warning {{has virtual functions but non-virtual destructor}}
  virtual void m();
};

struct S4 : public S2 {  // expected-warning {{has virtual functions but non-virtual destructor}}
  virtual void m();
};

struct B {
  virtual ~B();
  virtual void m();
};

struct S5 : public B {
  virtual void m();
};

struct S6 {
  virtual void m();
private:
  ~S6();
};

struct S7 {
  virtual void m();
protected:
  ~S7();
};

struct S8 {} s8;

UnknownType S8::~S8() { // expected-error {{unknown type name 'UnknownType'}}
  s8.~S8();
}

template<class T> class TS : public B {
  virtual void m();
};

TS<int> baz;

template<class T> class TS2 { // expected-warning {{'nonvirtualdtor::TS2<int>' has virtual functions but non-virtual destructor}}
  virtual void m();
};

TS2<int> foo; // expected-note {{instantiation}}
}

namespace dnvd { // delete-non-virtual-dtor warning
struct NP {};

struct B { // expected-warning {{has virtual functions but non-virtual destructor}}
  virtual void foo();
};

struct D: B {}; // expected-warning {{has virtual functions but non-virtual destructor}}

struct F final : B {};

struct VB {
  virtual void foo();
  virtual ~VB();
};

struct VD: VB {};

struct VF final: VB {};

template <typename T>
class simple_ptr2 {
public:
  simple_ptr2(T* t): _ptr(t) {}
  ~simple_ptr2() { delete _ptr; } // expected-warning {{delete called on non-final 'dnvd::B' that has virtual functions but non-virtual destructor}}
  T& operator*() const { return *_ptr; }
private:
  T* _ptr;
};

void use(B&);
void use(SystemB&);
void use(VB&);

void nowarnstack() {
  B b; use(b);
  D d; use(d);
  F f; use(f);
  VB vb; use(vb);
  VD vd; use(vd);
  VF vf; use(vf);
}

void nowarnnonpoly() {
  {
    NP* np = new NP();
    delete np;
  }
  {
    NP* np = new NP[4];
    delete[] np;
  }
}

// FIXME: Why are these supposed to not warn?
void nowarnarray() {
  {
    B* b = new B[4];
    delete[] b;
  }
  {
    D* d = new D[4];
    delete[] d;
  }
  {
    VB* vb = new VB[4];
    delete[] vb;
  }
  {
    VD* vd = new VD[4];
    delete[] vd;
  }
}

template <typename T>
void nowarntemplate() {
  {
    T* t = new T();
    delete t;
  }
  {
    T* t = new T[4];
    delete[] t;
  }
}

void nowarn0() {
  {
    F* f = new F();
    delete f;
  }
  {
    VB* vb = new VB();
    delete vb;
  }
  {
    VB* vb = new VD();
    delete vb;
  }
  {
    VD* vd = new VD();
    delete vd;
  }
  {
    VF* vf = new VF();
    delete vf;
  }
}

void nowarn0_explicit_dtor(F* f, VB* vb, VD* vd, VF* vf) {
  f->~F();
  f->~F();
  vb->~VB();
  vd->~VD();
  vf->~VF();
}

void warn0() {
  {
    B* b = new B();
    delete b; // expected-warning {{delete called on non-final 'dnvd::B' that has virtual functions but non-virtual destructor}}
  }
  {
    B* b = new D();
    delete b; // expected-warning {{delete called on non-final 'dnvd::B' that has virtual functions but non-virtual destructor}}
  }
  {
    D* d = new D();
    delete d; // expected-warning {{delete called on non-final 'dnvd::D' that has virtual functions but non-virtual destructor}}
  }
}

// Taken from libc++, slightly simplified.
template <class>
struct __is_destructible_apply { typedef int type; };
struct __two {char __lx[2];};
template <typename _Tp>
struct __is_destructor_wellformed {
  template <typename _Tp1>
  static char __test(typename __is_destructible_apply<
                       decltype(_Tp1().~_Tp1())>::type);
  template <typename _Tp1>
  static __two __test (...);
              
  static const bool value = sizeof(__test<_Tp>(12)) == sizeof(char);
};

void warn0_explicit_dtor(B* b, B& br, D* d) {
  b->~B(); // expected-warning {{destructor called on non-final 'dnvd::B' that has virtual functions but non-virtual destructor}} expected-note{{qualify call to silence this warning}}
  b->B::~B(); // No warning when the call isn't virtual.

  // No warning in unevaluated contexts.
  (void)__is_destructor_wellformed<B>::value;

  br.~B(); // expected-warning {{destructor called on non-final 'dnvd::B' that has virtual functions but non-virtual destructor}} expected-note{{qualify call to silence this warning}}
  br.B::~B();

  d->~D(); // expected-warning {{destructor called on non-final 'dnvd::D' that has virtual functions but non-virtual destructor}} expected-note{{qualify call to silence this warning}}
  d->D::~D();
}

void nowarn1() {
  {
    simple_ptr<F> f(new F());
    use(*f);
  }
  {
    simple_ptr<VB> vb(new VB());
    use(*vb);
  }
  {
    simple_ptr<VB> vb(new VD());
    use(*vb);
  }
  {
    simple_ptr<VD> vd(new VD());
    use(*vd);
  }
  {
    simple_ptr<VF> vf(new VF());
    use(*vf);
  }
  {
    simple_ptr<SystemB> sb(new SystemB());
    use(*sb);
  }
}

void warn1() {
  {
    simple_ptr<B> b(new B()); // expected-note {{in instantiation of member function 'dnvd::simple_ptr<dnvd::B>::~simple_ptr' requested here}}
    use(*b);
  }
  {
    simple_ptr2<B> b(new D()); // expected-note {{in instantiation of member function 'dnvd::simple_ptr2<dnvd::B>::~simple_ptr2' requested here}}
    use(*b);
  }
  {
    simple_ptr<D> d(new D()); // expected-note {{in instantiation of member function 'dnvd::simple_ptr<dnvd::D>::~simple_ptr' requested here}}
    use(*d);
  }
}
}

namespace PR9238 {
  class B { public: ~B(); };
  class C : virtual B { public: ~C() { } };
}

namespace PR7900 {
  struct A { // expected-note 2{{type 'PR7900::A' is declared here}}
  };
  struct B : public A {
  };
  void foo() {
    B b;
    b.~B();
    b.~A(); // expected-error{{destructor type 'PR7900::A' in object destruction expression does not match the type 'PR7900::B' of the object being destroyed}}
    (&b)->~A(); // expected-error{{destructor type 'PR7900::A' in object destruction expression does not match the type 'PR7900::B' of the object being destroyed}}
  }
}

namespace PR16892 {
  auto p = &A::~A; // expected-error{{taking the address of a destructor}}
}

namespace PR20238 {
struct S {
  volatile ~S() { } // expected-error{{destructor cannot have a return type}}
};
}

namespace PR22668 {
struct S {
};
void f(S s) {
  (s.~S)();
}
void g(S s) {
  (s.~S); // expected-error{{reference to destructor must be called}}
}
}

class Invalid {
    ~Invalid();
    UnknownType xx; // expected-error{{unknown type name}}
};

// The constructor definition should not have errors
Invalid::~Invalid() {}

namespace PR30361 {
template <typename T>
struct C1 {
  ~C1() {}
  operator C1<T>* () { return nullptr; }
  void foo1();
};

template<typename T>
void C1<T>::foo1() {
  C1::operator C1<T>*();
  C1::~C1();
}

void foo1() {
  C1<int> x;
  x.foo1();
}
}
#endif // BE_THE_HEADER
