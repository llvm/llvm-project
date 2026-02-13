// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++98 %s
// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s
template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

template<typename MetaFun, typename T1, typename T2>
struct metafun_apply2 {
  typedef typename MetaFun::template apply<T1, T2> inner;
  typedef typename inner::type type;
};

template<typename T, typename U> struct pair;

struct make_pair {
  template<typename T1, typename T2>
  struct apply {
    typedef pair<T1, T2> type;
  };
};

int a0[is_same<metafun_apply2<make_pair, int, float>::type,
               pair<int, float> >::value? 1 : -1];
int a1[is_same<
         typename make_pair::template apply<int, float>,
#if __cplusplus <= 199711L // C++03 and earlier modes
         // expected-warning@-2 {{'template' keyword outside of a template}}
         // expected-warning@-3 {{'typename' outside of a template is a C++11 extension}}
#endif
         make_pair::apply<int, float>
       >::value? 1 : -1];

template<typename MetaFun>
struct swap_and_apply2 {
  template<typename T1, typename T2>
  struct apply {
    typedef typename MetaFun::template apply<T2, T1> new_metafun;
    typedef typename new_metafun::type type;
  };
};

int a2[is_same<swap_and_apply2<make_pair>::apply<int, float>::type,
               pair<float, int> >::value? 1 : -1];

template<typename MetaFun>
struct swap_and_apply2b {
  template<typename T1, typename T2>
  struct apply {
    typedef typename MetaFun::template apply<T2, T1>::type type;
  };
};

int a3[is_same<swap_and_apply2b<make_pair>::apply<int, float>::type,
               pair<float, int> >::value? 1 : -1];

template<typename T>
struct X0 {
  template<typename U, typename V>
  struct Inner;

  void f0(X0<T>::Inner<T*, T&>); // expected-note{{here}}
  void f0(typename X0<T>::Inner<T*, T&>); // expected-error{{redecl}}

  void f1(X0<T>::Inner<T*, T&>); // expected-note{{here}}
  void f1(typename X0<T>::template Inner<T*, T&>); // expected-error{{redecl}}

  void f2(typename X0<T>::Inner<T*, T&>::type); // expected-note{{here}}
  void f2(typename X0<T>::template Inner<T*, T&>::type); // expected-error{{redecl}}
};

namespace PR6236 {
  template<typename T, typename U> struct S { };

  template<typename T> struct S<T, T> {
    template<typename U> struct K { };

    void f() {
      typedef typename S<T, T>::template K<T> Foo;
    }
  };
}

namespace PR6268 {
  template <typename T>
  struct Outer {
    template <typename U>
    struct Inner {};

    template <typename U>
    typename Outer<T>::template Inner<U>
    foo(typename Outer<T>::template Inner<U>);
  };

  template <typename T>
  template <typename U>
  typename Outer<T>::template Inner<U>
  Outer<T>::foo(typename Outer<T>::template Inner<U>) {
    return Inner<U>();
  }
}

namespace PR6463 {
  struct B { typedef int type; }; // expected-note 2{{member type 'int' found by ambiguous name lookup}}
  struct C { typedef const int type; }; // expected-note 2{{member type 'const int' found by ambiguous name lookup}}

  template<typename T>
  struct A : B, C {
    type& a(); // expected-error{{found in multiple base classes}}
    int x;
  };

  // FIXME: Improve source location info here.
  template<typename T>
  typename A<T>::type& A<T>::a() { // expected-error{{found in multiple base classes}}
    return x;
  }
}

namespace PR7419 {
  template <typename T> struct S {
    typedef typename T::Y T2;
    typedef typename T2::Z T3;
    typedef typename T3::W T4;
    T4 *f();

    typedef typename T::template Y<int> TT2;
    typedef typename TT2::template Z<float> TT3;
    typedef typename TT3::template W<double> TT4;
    TT4 g();
  };

  template <typename T> typename T::Y::Z::W *S<T>::f() { }
  template <typename T> typename T::template Y<int>::template Z<float>::template W<double> S<T>::g() { }
}

namespace rdar8740998 {
  template<typename T>
  struct X : public T {
    using T::iterator; // expected-note{{add 'typename' to treat this using declaration as a type}} \
    // expected-error{{dependent using declaration resolved to type without 'typename'}}

    void f() {
      typename X<T>::iterator i; // expected-error{{typename specifier refers to a dependent using declaration for a value 'iterator' in 'rdar8740998::X<T>'}}
    }
  };

  struct HasIterator {
    typedef int *iterator; // expected-note{{target of using declaration}}
  };

  void test_X(X<HasIterator> xi) { // expected-note{{in instantiation of template class}}
    xi.f();
  }
}

namespace rdar9068589 {
  // From GCC PR c++/13950
  template <class T> struct Base {};
  template <class T> struct Derived: public Base<T> {
    typename Derived::template Base<double>* p1;
  };
}
