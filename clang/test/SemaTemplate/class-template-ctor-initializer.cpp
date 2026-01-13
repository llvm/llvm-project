// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

template<class X> struct A {};

template<class X> struct B : A<X> {
  B() : A<X>() {}
};
B<int> x;

template<class X> struct B1 : A<X> {
  typedef A<X> Base;
  B1() : Base() {}
};
B1<int> x1;


template<typename T> struct Tmpl { };

template<typename T> struct TmplB { };

struct TmplC : Tmpl<int> {
   TmplC() :
             Tmpl<int>(),
             TmplB<int>() { } // expected-error {{type 'TmplB<int>' is not a direct or virtual base of 'TmplC'}}
};


struct TmplD : Tmpl<char>, TmplB<char> {
    TmplD():
            Tmpl<int>(), // expected-error {{type 'Tmpl<int>' is not a direct or virtual base of 'TmplD'}}
            TmplB<char>() {}
};

namespace PR7259 {
  class Base {
  public:
    Base() {}
  };

  template <class ParentClass>
  class Derived : public ParentClass {
  public:
    Derived() : Base() {}
  };

  class Final : public Derived<Base> {
  };

  int
  main (void)
  {
    Final final;
    return 0;
  }
}

namespace NonDependentError {
  struct Base { Base(int); }; // expected-note {{candidate constructor not viable}}
// expected-note@-1 {{candidate constructor (the implicit copy constructor) not viable}}
#if __cplusplus >= 201103L // C++11 or later
// expected-note@-3 {{candidate constructor (the implicit move constructor) not viable}}
#endif

  template<typename T>
  struct Derived1 : Base {
    Derived1() : Base(1, 2) {} // expected-error {{no matching constructor}}
  };

  template<typename T>
  struct Derived2 : Base {
    Derived2() : BaseClass(1) {} // expected-error {{does not name a non-static data member or base}}
  };

  Derived1<void> d1;
  Derived2<void> d2;
}

namespace UnresolvedUsing {
  template <class T> class A {
    using typename T::B;
    struct C : B {
      C() : B() {}
    };
  };
} // namespace UnresolvedUsing
