// RUN: %clang_cc1 -fsyntax-only -verify %s

// FIXME: Access control checks

namespace PR5820 {
  // also <rdar://problem/7535045>
  struct Base {
    void Foo();
    int Member;
  };

  struct D1 : public Base {};
  struct D2 : public Base {};

  struct Derived : public D1, public D2 {
    void Inner();
  };

  void Test() {
    Derived d;
    d.D1::Foo();
    d.D1::Member = 17;
  }

  void Derived::Inner() {
    D1::Foo();
    D1::Member = 42;
    this->D1::Foo();
    this->D1::Member = 42;
  }
}

template<typename T>
struct BaseT {
  void Foo(); // expected-note{{found by ambiguous name lookup}}
  int Member;
};

template<typename T> struct Derived1T : BaseT<T> { };
template<typename T> struct Derived2T : BaseT<T> { };

template<typename T>
struct DerivedT : public Derived1T<T>, public Derived2T<T> {
  void Inner();
};

template<typename T>
void DerivedT<T>::Inner() {
  Derived1T<T>::Foo();
  Derived2T<T>::Member = 42;
  this->Derived1T<T>::Foo();
  this->Derived2T<T>::Member = 42;
  this->Foo(); // expected-error{{non-static member 'Foo' found in multiple base-class subobjects of type 'BaseT<int>'}}
}

template<typename T>
void Test(DerivedT<T> d) {
  d.template Derived1T<T>::Foo();
  d.template Derived2T<T>::Member = 17;
  d.Inner(); // expected-note{{in instantiation}}
}

template void Test(DerivedT<int>);
