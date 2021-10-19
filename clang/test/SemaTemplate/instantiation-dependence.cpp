// RUN: %clang_cc1 -std=c++2b -verify %s

// Ensure we substitute into instantiation-dependent but non-dependent
// constructs. The poster-child for this is...
template<class ...> using void_t = void;

namespace PR24076 {
  template<class T> T declval();
  struct s {};

  template<class T,
           class = void_t<decltype(declval<T>() + 1)>>
    void foo(T) {} // expected-note {{invalid operands to binary expression}}

  void f() {
    foo(s{}); // expected-error {{no matching function}}
  }

  template<class T,
           class = void_t<decltype(declval<T>() + 1)>> // expected-error {{invalid operands to binary expression}}
  struct bar {};

  bar<s> bar; // expected-note {{in instantiation of}}
}

namespace PR33655 {
  struct One { using x = int; };
  struct Two { using y = int; };

  template<typename T, void_t<typename T::x> * = nullptr> int &func() {}
  template<typename T, void_t<typename T::y> * = nullptr> float &func() {}

  int &test1 = func<One>();
  float &test2 = func<Two>();

  template<class ...Args> struct indirect_void_t_imp { using type = void; };
  template<class ...Args> using indirect_void_t = typename indirect_void_t_imp<Args...>::type;

  template<class T> void foo() {
    static_assert(!__is_void(indirect_void_t<T>)); // "ok", dependent
    static_assert(!__is_void(void_t<T>)); // expected-error {{failed}}
  }
}

namespace PR46791 { // also PR45782
  template<typename T, typename = void>
  struct trait {
    static constexpr int specialization = 0;
  };

  // FIXME: Per a strict interpretation of the C++ rules, the two void_t<...>
  // types below are equivalent -- we only (effectively) do token-by-token
  // comparison for *expressions* appearing within types. But all other
  // implementations accept this, using rules that are unclear.
  template<typename T>
  struct trait<T, void_t<typename T::value_type>> { // expected-note {{previous}} FIXME-note {{matches}}
    static constexpr int specialization = 1;
  };

  template<typename T>
  struct trait<T, void_t<typename T::element_type>> { // expected-error {{redefinition}} FIXME-note {{matches}}
    static constexpr int specialization = 2;
  };

  struct A {};
  struct B { typedef int value_type; };
  struct C { typedef int element_type; };
  struct D : B, C {};

  static_assert(trait<A>::specialization == 0);
  static_assert(trait<B>::specialization == 1); // FIXME expected-error {{failed}}
  static_assert(trait<C>::specialization == 2); // FIXME expected-error {{failed}}
  static_assert(trait<D>::specialization == 0); // FIXME-error {{ambiguous partial specialization}}
}

namespace TypeQualifier {
  // Ensure that we substitute into an instantiation-dependent but
  // non-dependent qualifier.
  template<int> struct A { using type = int; };
  template<typename T> A<sizeof(sizeof(T::error))>::type f() {} // expected-note {{'int' cannot be used prior to '::'}}
  int k = f<int>(); // expected-error {{no matching}}
}

namespace MemberOfInstantiationDependentBase {
  template<typename T> struct A { template<int> void f(int); };
  template<typename T> struct B { using X = A<T>; };
  template<typename T> struct C1 : B<int> {
    using X = typename C1::X;
    void f(X *p) {
      p->f<0>(0);
      p->template f<0>(0);
    }
  };
  template<typename T> struct C2 : B<int> {
    using X = typename C2<T>::X;
    void f(X *p) {
      p->f<0>(0);
      p->template f<0>(0);
    }
  };
  void q(C1<int> *c) { c->f(0); }
  void q(C2<int> *c) { c->f(0); }
}
