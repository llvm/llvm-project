// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T>
struct X {
  X<T*> *ptr;
};

X<int> x;

template<>
struct X<int***> {
  typedef X<int***> *ptr;
};

X<float>::X<int> xi = x; // expected-error{{qualified reference to 'X' is a constructor name rather than a template name}}
void f() {
  X<float>::X<int> xi = x; // expected-error{{qualified reference to 'X' is a constructor name rather than a template name}}
}

// [temp.local]p1:

// FIXME: test template template parameters
template<typename T, typename U>
struct X0 {
  typedef T type;
  typedef U U_type;
  typedef U_type U_type2;

  void f0(const X0&); // expected-note{{here}}
  void f0(X0&);
  void f0(const X0<T, U>&); // expected-error{{redecl}}

  void f1(const X0&); // expected-note{{here}}
  void f1(X0&);
  void f1(const X0<type, U_type2>&); // expected-error{{redecl}}

  void f2(const X0&); // expected-note{{here}}
  void f2(X0&);
  void f2(const ::X0<type, U_type2>&); // expected-error{{redecl}}
};

template<typename T, T N>
struct X1 {
  void f0(const X1&); // expected-note{{here}}
  void f0(X1&);
  void f0(const X1<T, N>&); // expected-error{{redecl}}
};

namespace pr6326 {
  template <class T> class A {
    friend class A;
  };
  template class A<int>;
}

namespace ForwardDecls {
  template<typename T>
  struct X;

  template<typename T>
  struct X {
    typedef T foo;
    typedef X<T> xt;
    typename xt::foo *t;
  };
}

namespace ConflictingRedecl {
  template<typename> struct Nested {
    template<typename> struct Nested; // expected-error {{member 'Nested' has the same name as its class}}
  };
}

namespace TwoLevels {
  template <class> struct A {
    template <bool> A f();
  };
  template <class T> template <bool> A<T> A<T>::f() {}
} // namespace TwoLevels

namespace TwoLevelsAlias1 {
  template <class> struct A;
  template <class T> using alias = T;
  template <class T> struct B {
    using type = alias<T>;
    template <class> A<type> f();
  };
  template <class T> template <class>
  A<typename B<T>::type> B<T>::f() {}
} // namespace TwoLevelsAlias

namespace TwoLevelsAlias2 {
  template <class> struct A;
  template <class T> using alias = typename A<T>::type;
  template <class T> struct B {
    template <class> typename A<T>::type f();
  };
  template <class T> template <class> alias<T> B<T>::f() {}
} // namespace TwoLevelsAlias2

namespace TwoLevelsAlias3 {
  template <class T> using void_t = void;
  template <class T> struct A { // expected-note {{defined here}}
    template <int> void_t<typename T::type> f();
  };
  template <class T> template <int> void A<T>::f() {} // expected-error {{does not match}}
} // namspace TwoLevelsAlias3

namespace Unique {
  template <class T> struct A {
    template <class> A<T> f1();
    template <class> A<T> f2();
  };
  template <class T> template <class> A<T> A<T>::f1() {}
  template <class T> template <class> A<T> A<T>::f2() {}
} // namespace Unique
