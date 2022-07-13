// RUN: %clang_cc1 -fsyntax-only -verify -Wno-unused-comparison %s

// PR8439
class A
{
};

class B
{
public:
  A & m;
};

class Base
{
public:
  B &f();
};

class Derived1 : public Base { };

class Derived2 : public Base { };

class X : public B, public Derived2, public Derived1
{
public:
  virtual void g();
};

void X::g()
{
  m.f<int>(); // expected-error{{no member named 'f' in 'A'}} \
  // expected-error{{expected '(' for function-style cast}} \
  // expected-error{{expected expression}}
}

namespace PR11134 {
  template<typename Derived> class A;
  template<typename Derived> class B : A<Derived> {
    typedef A<Derived> Base;
    using Base::member;
    int member;
  };
}

namespace AddrOfMember {
  struct A { int X; };
  typedef int (A::*P);
  template<typename T> struct S : T {
    void f() {
      P(&T::X) // expected-error {{cannot cast from type 'int *' to member pointer type 'AddrOfMember::P'}}
          == &A::X;
    }
  };

  void g() {
    S<A>().f(); // ok, &T::X is 'int (A::*)', not 'int *', even though T is a base class
  }

  struct B : A { static int X; };
  void h() {
    S<B>().f(); // expected-note {{here}}
  }
}
