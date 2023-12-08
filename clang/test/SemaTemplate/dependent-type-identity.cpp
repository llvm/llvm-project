// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

// This test concerns the identity of dependent types within the
// canonical type system. This corresponds to C++ [temp.type], which
// specifies type equivalence within a template.
//
// FIXME: template template parameters

namespace N {
  template<typename T>
  struct X2 {
    template<typename U>
    struct apply {
      typedef U* type;
    };
  };
}

namespace Nalias = N;

template<typename T>
struct X0 { };

using namespace N;

template<typename T, typename U>
struct X1 {
  typedef T type;
  typedef U U_type;

  void f0(T); // expected-note{{previous}}
  void f0(U);
  void f0(type); // expected-error{{redeclar}}

  void f1(T*); // expected-note{{previous}}
  void f1(U*);
  void f1(type*); // expected-error{{redeclar}}

  void f2(X0<T>*); // expected-note{{previous}}
  void f2(X0<U>*);
  void f2(X0<type>*); // expected-error{{redeclar}}

  void f3(X0<T>*); // expected-note{{previous}}
  void f3(X0<U>*);
  void f3(::X0<type>*); // expected-error{{redeclar}}  

  void f4(typename T::template apply<U>*); // expected-note{{previous}}
  void f4(typename U::template apply<U>*);
  void f4(typename type::template apply<T>*);
  void f4(typename type::template apply<U_type>*); // expected-error{{redeclar}}

  void f5(typename T::template apply<U>::type*); // expected-note{{previous}}
  void f5(typename U::template apply<U>::type*);
  void f5(typename U::template apply<T>::type*);
  void f5(typename type::template apply<T>::type*);
  void f5(typename type::template apply<U_type>::type*); // expected-error{{redeclar}}

  void f6(typename N::X2<T>::template apply<U> *); // expected-note{{previous}}
  void f6(typename N::X2<U>::template apply<U> *);
  void f6(typename N::X2<U>::template apply<T> *);
  void f6(typename ::N::X2<type>::template apply<U_type> *); // expected-error{{redeclar}}
  
  void f7(typename N::X2<T>::template apply<U> *); // expected-note{{previous}}
  void f7(typename N::X2<U>::template apply<U> *);
  void f7(typename N::X2<U>::template apply<T> *);
  void f7(typename X2<type>::template apply<U_type> *); // expected-error{{redeclar}}

  void f8(typename N::X2<T>::template apply<U> *); // expected-note{{previous}}
  void f8(typename N::X2<U>::template apply<U> *);
  void f8(typename N::X2<U>::template apply<T> *);
  void f8(typename ::Nalias::X2<type>::template apply<U_type> *); // expected-error{{redeclar}}
};

namespace PR6851 {
  template <bool v>
  struct S;

  struct N {
    template <bool w>
    S< S<w>::cond && 1 > foo();
  };

  struct Arrow { Arrow *operator->(); int n; };
  template<typename T> struct M {
    Arrow a;
    auto f() -> M<decltype(a->n)>;
  };

  struct Alien;
  bool operator&&(const Alien&, const Alien&);

  template <bool w>
  S< S<w>::cond && 1 > N::foo() { }

  template<typename T>
  auto M<T>::f() -> M<decltype(a->n)> {}
}

namespace PR7460 {
  template <typename T>
  struct TemplateClass2
  {
    enum { SIZE = 100 };
    static T member[SIZE];
  };

  template <typename T>
  T TemplateClass2<T>::member[TemplateClass2<T>::SIZE];
}

namespace PR18275 {
  template<typename T> struct A {
    void f(const int);
    void g(int);
    void h(const T);
    void i(T);
  };

  template<typename T>
  void A<T>::f(int x) { x = 0; }

  template<typename T>
  void A<T>::g(const int x) {  // expected-note {{declared const here}}
    x = 0; // expected-error {{cannot assign to variable 'x'}}
  }

  template<typename T>
  void A<T>::h(T) {} // FIXME: Should reject this. Type is different from prior decl if T is an array type.

  template<typename T>
  void A<T>::i(const T) {} // FIXME: Should reject this. Type is different from prior decl if T is an array type.

  template struct A<int>;
  template struct A<int[1]>;
}

namespace PR21289 {
  template<typename T> using X = int;
  template<typename T, decltype(sizeof(0))> using Y = int;
  template<typename ...Ts> struct S {};
  template<typename ...Ts> void f() {
    // This is a dependent type. It is *not* S<int>, even though it canonically
    // contains no template parameters.
    using Type = S<X<Ts>...>;
    Type s;
    using Type = S<int, int, int>;
  }
  void g() { f<void, void, void>(); }

  template<typename ...Ts> void h(S<int>) {}
  // Pending a core issue, it's not clear if these are redeclarations, but they
  // are probably intended to be... even though substitution can succeed for one
  // of them but fail for the other!
  template<typename ...Ts> void h(S<X<Ts>...>) {} // expected-note {{previous}}
  template<typename ...Ts> void h(S<Y<Ts, sizeof(Ts)>...>) {} // expected-error {{redefinition}}
}
