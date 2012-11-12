// RUN: %clang_cc1 -fsyntax-only -verify %s

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

  struct Alien;
  bool operator&&(const Alien&, const Alien&);

  template <bool w>
  S< S<w>::cond && 1 > N::foo() { }
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
