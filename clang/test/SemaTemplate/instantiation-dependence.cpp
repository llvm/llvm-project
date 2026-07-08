// RUN: %clang_cc1 -std=c++26 -verify %s

// Ensure we substitute into instantiation-dependent but non-dependent
// constructs. The poster-child for this is...
template<class ...> using void_t = void;
template<class T, class> using alias = T;

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
    int check1[__is_void(indirect_void_t<T>) == 0 ? 1 : -1]; // "ok", dependent
    int check2[__is_void(void_t<T>) == 0 ? 1 : -1]; // expected-error {{array with a negative size}}
  }
}

namespace PR46791 { // also PR45782
  template<typename T, typename = void>
  struct trait {
    static constexpr int specialization = 0;
  };

  template<typename T>
  struct trait<T, void_t<typename T::value_type>> { // expected-note {{matches}}
    static constexpr int specialization = 1;
  };

  template<typename T>
  struct trait<T, void_t<typename T::element_type>> { // expected-note {{matches}}
    static constexpr int specialization = 2;
  };

  struct A {};
  struct B { typedef int value_type; };
  struct C { typedef int element_type; };
  struct D : B, C {};

  static_assert(trait<A>::specialization == 0);
  static_assert(trait<B>::specialization == 1);
  static_assert(trait<C>::specialization == 2);
  static_assert(trait<D>::specialization == 0); // expected-error {{ambiguous partial specialization}}
}

namespace TypeQualifier {
  // Ensure that we substitute into an instantiation-dependent but
  // non-dependent qualifier.
  template<int> struct A { using type = int; };
  template<typename T> A<sizeof(sizeof(T::error))>::type f() {} // expected-note {{'int' cannot be used prior to '::'}}
  int k = f<int>(); // expected-error {{no matching}}
}

namespace BaseClass1 {
  struct A {};
  template <class> using B = A;
  template <class T> struct C : B<T*> {};
  // expected-error@-1 {{'type name' declared as a pointer to a reference of type 'int &'}}
  template struct C<int&>; // expected-note {{requested here}}
} // namespace BaseClass1

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

namespace GH8740 {
  struct A { typedef int T; };
  template<int> struct U { typedef int T; };
  template<typename> struct S {
    A a;
    int n = decltype(a)::T();
    int m = U<sizeof(a)>::T();
  };
  S<char> s;
} // namespace GH8740

namespace NonInstDependentArgs1 {
  template<class T, class = void> struct X;
  template<class T> struct X<T, void_t<char>> {}; // expected-note  {{previous}}
  template<class T> struct X<T, void_t<void>> {}; // expected-error {{redefinition}}

  template<class T, class = void> bool x;
  template<class T> bool x<T, void_t<char>>;  // expected-note  {{previous}}
  template<class T> bool x<T, void_t<void>>; // expected-error {{redefinition}}
} // namespace NonInstDependentArgs1

namespace NonInstDependentArgs2 {
  template<class T, class = void> struct X;
  template<class T> struct X<T, void_t<T, void>> {};
  template<class T> struct X<T, void_t<T, char>> {};

  template<class T, class = void> bool x;
  template<class T> bool x<T, void_t<T, char>>;
  template<class T> bool x<T, void_t<T, void>>;
} // namespace NonInstDependentArgs2

namespace Level1 {
  template<class T, class = void> struct X;
  template<class T> struct X<T, void_t<T>> {};
  template<class T> struct X<T, void_t<T*>> {};
} // namespace Level1

namespace Level2 {
  template<class T, class = void> struct X;
  template<class T> struct X<T, void_t<void_t<T>>> {};
  template<class T> struct X<T, void_t<void_t<T*>>> {};
} // namespace Level2

namespace IndirectAlias1 {
  template<class T> using alias2 = void_t<T>;
  template<class T, class = void> struct X;
  template<class T> struct X<T, void_t<T>> {}; // expected-note  {{previous}}
  template<class T> struct X<T, alias2<T>> {}; // expected-error {{redefinition}}
} // namspace IndirectAlias1

namespace IndirectAlias2 {
  template<class T, class U> using alias2 = alias<T, U>;
  template<class T, class = void> struct X;
  template<class T> struct X<T, T> {};
  template<class T> struct X<T, alias<T, T>> {};  // expected-note  {{previous}}
  template<class T> struct X<T, alias2<T, T>> {}; // expected-error {{redefinition}}
} // namespace IndirectAlias2

namespace PackIndexing1 {
  // FIXME: This should not be a redefinition.
  template<class ...Ts> using aliaspack = Ts...[0];
  template<class T, class = void> struct X;
  template<class T> struct X<T, T> {};                              // expected-note  {{previous}}
  template<class T> struct X<T, aliaspack<T, typename T::type>> {}; // expected-error {{redefinition}}
} // namespace PackIndexing1

namespace DeclType1 {
  template<class T, class = void> struct X;
  template<class T> struct X<T, decltype(void())> {};         // expected-note  {{previous}}
  template<class T> struct X<T, decltype(void_t<char>())> {}; // expected-error {{redefinition}}
} // namespace DeclType1

namespace DeclType2 {
  template<class T, class = void> struct X;
  template<class T> struct X<T, decltype(void())> {};
  template<class T> struct X<T, decltype(void_t<typename T::type>())> {}; // expected-note  {{previous}}
  template<class T> struct X<T, decltype(void_t<typename T::type>())> {}; // expected-error {{redefinition}}
} // namespace DeclType2

namespace DeclType3 {
  template<class T, class = void> struct X;
  template<class T> struct X<T, decltype(void())> {};
  template<class T> struct X<T, decltype(void_t<typename T::type>())> {};
  template<class T> struct X<T, decltype(void_t<typename T::bar>())> {};
} // namespace DeclType3

namespace DeclType4 {
  template<class T> using X = decltype(void());                     // expected-note  {{previous}}
  template<class T> using X = decltype(void_t<typename T::type>()); // expected-error {{redefinition with different types}}
} // namespace DeclType4

namespace DeclType5 {
  template<class T> using X = decltype(void_t<typename T::bar>());  // expected-note  {{previous}}
  template<class T> using X = decltype(void_t<typename T::type>()); // expected-error {{redefinition with different types}}
} // namespace DeclType5

namespace TempArg1 {
  template<void*> struct A;
  template<class T> using X = A<(void*){}>;                     // expected-note  {{previous}}
  template<class T> using X = A<(void_t<typename T::type>*){}>; // expected-error {{redefinition with different types}}
} // namespace TempArg1

namespace TempArg2 {
  template<void*> struct A;
  template<class T> using X = A<(void_t<typename T::bar>*){}>;  // expected-note  {{previous}}
  template<class T> using X = A<(void_t<typename T::type>*){}>; // expected-error {{redefinition with different types}}
} // namespace TempArg2

namespace ConstTempParam1 {
  // FIXME: Create a disambiguation rule for this.
  //        Maybe: If one candidate has functional types in the signature, the
  //        other does not, the former is more specialized.
  template<class T, T N> struct Test0; // expected-note {{declared here}}
  template<class T, alias<T, typename T::type> N>
    struct Test0<T, N>;
  // expected-error@-1 {{not more specialized than the primary template}}
} // namespace ConstTempParam1

namespace NestedClassSpec1 {
  template<class T> struct A {
    template<class> struct Cls;
    template<class U> struct Cls<alias<U, typename U::type>> {};
    template<class U> struct Cls<alias<U, typename U::bar>> {};
  };
} // nestedClassSpec1

namespace NestedClassSpec2 {
  template<class T> struct A {
    template<class> struct Cls;
    template<class U> struct Cls<alias<U, void>> {};
    // expected-error@-1 {{does not specialize any template argument}}
    template<class U> struct Cls<alias<U, typename U::type>> {}; // expected-note {{previous definition is here}}
    template<class U> struct Cls<alias<U, typename U::type>> {};
    // expected-error@-1 {{redefinition}}
  };
} // nestedClassSpec2

namespace NestedClassSpec3 {
  template<template<class...> class TT> struct A {
    template<class,class> struct Cls;
    template<class T> struct Cls<int, TT<T, typename T::type>> {};
    template<class T> struct Cls<int, TT<T, typename T::bar>> {};
  };
  template struct A<alias>;
} // nestedClassSpec3

namespace NestedVarSpec1 {
  template<class T> struct A {
    template<class> static int Var;
    template<class U> static int Var<alias<U, typename U::type>>;
    template<class U> static int Var<alias<U, typename U::bar>>;
  };
} // namespace NestedVarSpec1

namespace NestedVarSpec2 {
  template<class T> struct A {
    template<class> static int Var;
    template<class U> static int Var<alias<U, void>>;
    // expected-error@-1 {{does not specialize any template argument}}
    template<class U> static int Var<alias<U, typename U::type>>; // expected-note {{previous declaration is here}}
    template<class U> static int Var<alias<U, typename U::type>>;
    // expected-error@-1 {{duplicate member 'Var'}}
  };
} // namespace NestedVarSpec2

namespace NestedVarSpec3 {
  template<template<class...> class TT> struct A {
    template<class,class> static int Var;
    template<class T> static int Var<int, TT<T, typename T::type>>;
    template<class T> static int Var<int, TT<T, typename T::bar>>;
  };
  template struct A<alias>;
} // nestedClassSpec3

namespace VarRedecl1 {
  template<class T> extern void *x; // expected-note {{previous declaration}}
  template<class T> extern void_t<T> *x;
  // expected-error@-1 {{redeclaration of 'x' with a different type}}
} // namespace VarRedecl1

namespace VarRedecl2 {
  template<class T> extern void_t<T> *x;
  template<class T> extern void_t<T> *x;
} // namespace VarRedecl2

namespace VarRedecl3 {
  template<class T> extern void_t<T, typename T::foo> *x; // expected-note {{previous declaration}}
  template<class T> extern void_t<T, typename T::bar> *x;
  // expected-error@-1 {{redeclaration of 'x' with a different type}}
} // namespace VarRedecl3

namespace FuncParamDecay1 {
  template<int N> constexpr int f(char *) { return 1; }
  template<int N> constexpr int f(char[N]) = delete;
  static_assert(f<0>(nullptr) == 1);
} // namespace FuncParamDecay1

namespace ArrayQualifiers1 {
  template<class T, class...> using alias = const int[5];
  template<class T> void f() {
    using X = const alias<T>;
    using X = alias<T>;
  };
} // namespace ArrayQualifiers1

namespace ArrayQualifiers2 {
  template<class T, class...> using alias = const int;
  template<class T> void f() {
    using X = const alias<T>[5];
    using X = alias<T>[5];
  };
} // namespace ArrayQualifiers2

namespace TempTempParam1 {
  template<class T, template<void*> class>      void f() {}
  template<class T, template<void_t<T>*> class> void f() {}

  template<void*> struct A;

  template<class, class> struct B;
  template<class T, void* V>      struct B<T, A<V>> {};
  template<class T, void_t<T>* V> struct B<T, A<V>> {};

  template<class,class> struct C;
  template<class T, template<void*> class TT>      struct C<T, TT<nullptr>> {};
  template<class T, template<void_t<T>*> class TT> struct C<T, TT<nullptr>> {};
} // namespace TempTempParam1

namespace OverloadExpr1 {
  template <class T> T &&declval();

  int g(int);
  template <class T> typename T::type g(T);

  template <class T, class = void> struct X;

  template <class T>
  struct X<T, decltype((g(declval<T>()), void()))> {
    static constexpr auto val = 1;
  };

  template <class T>
  struct X<T, decltype((g<>(declval<T>()), void()))> {
    static constexpr auto val = 2;
  };

  static_assert(X<int>::val == 1);
} // namespace OverloadExpr1

namespace NameQualifierUsing {
  struct Base {
    using type = int;
  };
  template <class T> struct S : Base {
    using Base::type;

    void f(S<alias<T, typename T::foo>>::type) {}
    void f(S<alias<T, typename T::bar>>::type) {}
  };
} // namespace QualfierUsing

namespace NameQualifierUnresolvedUsing1 {
  template <class T> struct A : T {
    using typename T::foo;
    void f(foo) {}
    void f(A<alias<T, typename T::bar>>::foo) {}
    void f(A<alias<T, typename T::baz>>::foo) {}
  };
} // namespace NameQualifierUnresolvedUsing1

namespace RequiresExpr1 {
  template<class T>
    requires (alias<T, typename T::foo>() == 0)
    void f() {}

  template<class T>
    requires (alias<T, typename T::bar>() == 0)
    void f() {}

  template<typename T> requires (sizeof(alias<T, typename T::foo>) >= 4)
  bool a = false; // expected-note{{template is declared here}}

  template<typename T> requires (sizeof(alias<T, typename T::bar>) >= 4 && sizeof(T) <= 10)
  bool a<T> = true; // expected-error{{variable template partial specialization is not more specialized than the primary template}}
} // namespace RequiresExpr1

namespace NoexceptExpr1 {
  template<class T>
    void f() noexcept(alias<T, typename T::foo>() == 0) {}
  // expected-note@-1 {{previous declaration is here}}

  template<class T>
    void f() noexcept(alias<T, typename T::bar>() == 0) {}
  // expected-error@-1 {{exception specification in declaration does not match previous declaration}}
} // namespace NoexceptExpr1

namespace SimpleAlias1 {
  template<class T> void foo(void_t<T*>*) = delete;
  template<class T> void foo(void_t<T>*) {}
  template void foo<int&>(void*);
} // namespace SimpleAlias1

namespace OutOfLinePartial1 {
  template<class> struct A {
    template<class, class> struct B {};
    template<class U> struct B<U, typename U::type1>;
    template<class U> struct B<U, typename U::type2>;
  };
  template<> template<class U> struct A<int>::B<U, typename U::type1> {};
  template<> template<class U> struct A<int>::B<U, typename U::type2> {};
} // namespace OutOfLinePartial1

// FIXME: This is a flaky test
#if 0
namespace UnaryTransformDecay {
  template<class T, class U = void> struct X;
  template<class T> struct X<T, __decay(int[T()])> {}; // FIXME-note  {{previous}}
  template<class T> struct X<T, __decay(int[T()])> {}; // FIXME-error {{redefinition}}
} // namespace UnaryTransformDecay
#endif
