// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++1z

// Template argument deduction with template template parameters.
template<typename T, template<T> class A> 
struct X0 {
  static const unsigned value = 0;
};

template<template<int> class A>
struct X0<int, A> {
  static const unsigned value = 1;
};

template<int> struct X0i;
template<long> struct X0l;
int array_x0a[X0<long, X0l>::value == 0? 1 : -1];
int array_x0b[X0<int, X0i>::value == 1? 1 : -1];

template<typename T, typename U>
struct is_same {
  static const bool value = false;
};

template<typename T>
struct is_same<T, T> {
  static const bool value = true;
};

template<typename T> struct allocator { };
template<typename T, typename Alloc = allocator<T> > struct vector {};

// Fun with meta-lambdas!
struct _1 {};
struct _2 {};

// Replaces all occurrences of _1 with Arg1 and _2 with Arg2 in T.
template<typename T, typename Arg1, typename Arg2>
struct Replace {
  typedef T type;
};

// Replacement of the whole type.
template<typename Arg1, typename Arg2>
struct Replace<_1, Arg1, Arg2> {
  typedef Arg1 type;
};

template<typename Arg1, typename Arg2>
struct Replace<_2, Arg1, Arg2> {
  typedef Arg2 type;
};

// Replacement through cv-qualifiers
template<typename T, typename Arg1, typename Arg2>
struct Replace<const T, Arg1, Arg2> {
  typedef typename Replace<T, Arg1, Arg2>::type const type;
};

// Replacement of templates
template<template<typename> class TT, typename T1, typename Arg1, typename Arg2>
struct Replace<TT<T1>, Arg1, Arg2> {
  typedef TT<typename Replace<T1, Arg1, Arg2>::type> type;
};

template<template<typename, typename> class TT, typename T1, typename T2,
         typename Arg1, typename Arg2>
struct Replace<TT<T1, T2>, Arg1, Arg2> {
  typedef TT<typename Replace<T1, Arg1, Arg2>::type,
             typename Replace<T2, Arg1, Arg2>::type> type;
};

// Just for kicks...
template<template<typename, typename> class TT, typename T1,
         typename Arg1, typename Arg2>
struct Replace<TT<T1, _2>, Arg1, Arg2> {
  typedef TT<typename Replace<T1, Arg1, Arg2>::type, Arg2> type;
};

int array0[is_same<Replace<_1, int, float>::type, int>::value? 1 : -1];
int array1[is_same<Replace<const _1, int, float>::type, const int>::value? 1 : -1];
int array2[is_same<Replace<vector<_1>, int, float>::type, vector<int> >::value? 1 : -1];
int array3[is_same<Replace<vector<const _1>, int, float>::type, vector<const int> >::value? 1 : -1];
int array4[is_same<Replace<vector<int, _2>, double, float>::type, vector<int, float> >::value? 1 : -1];

// PR5911
template <typename T, int N> void f(const T (&a)[N]);
int iarr[] = { 1 };
void test_PR5911() { f(iarr); }

// Must not examine base classes of incomplete type during template argument
// deduction.
namespace PR6257 {
  template <typename T> struct X {
    template <typename U> X(const X<U>& u);
  };
  struct A;
  void f(A& a);
  void f(const X<A>& a);
  void test(A& a) { (void)f(a); }
}

// PR7463
namespace PR7463 {
  const int f ();
  template <typename T_> void g (T_&); // expected-note{{T_ = int}}
  void h (void) { g(f()); } // expected-error{{no matching function for call}}
}

namespace test0 {
  template <class T> void make(const T *(*fn)()); // expected-note {{candidate template ignored: cannot deduce a type for 'T' that would make 'const T' equal 'char'}}
  char *char_maker();
  void test() {
    make(char_maker); // expected-error {{no matching function for call to 'make'}}
  }
}

namespace test1 {
  template<typename T> void foo(const T a[3][3]);
  void test() {
    int a[3][3];
    foo(a);
  }
}

// PR7708
namespace test2 {
  template<typename T> struct Const { typedef void const type; };

  template<typename T> void f(T, typename Const<T>::type*);
  template<typename T> void f(T, void const *);

  void test() {
    void *p = 0;
    f(0, p);
  }
}

// rdar://problem/8537391
namespace test3 {
  struct Foo {
    template <void F(char)> static inline void foo();
  };

  class Bar {
    template<typename T> static inline void wobble(T ch);

  public:
    static void madness() {
      Foo::foo<wobble<char> >();
    }
  };
}

// Verify that we can deduce enum-typed arguments correctly.
namespace test14 {
  enum E { E0, E1 };
  template <E> struct A {};
  template <E e> void foo(const A<e> &a) {}

  void test() {
    A<E0> a;
    foo(a);
  }
}

namespace PR21536 {
  template<typename ...T> struct X;
  template<typename A, typename ...B> struct S {
    static_assert(sizeof...(B) == 1, "");
    void f() {
      using T = A;
      using T = int;

      using U = X<B...>;
      using U = X<int>;
    }
  };
  template<typename ...T> void f(S<T...>);
  void g() { f(S<int, int>()); }
}

namespace PR19372 {
  template <template<typename...> class C, typename ...Us> struct BindBack {
    template <typename ...Ts> using apply = C<Ts..., Us...>;
  };
  template <typename, typename...> struct Y;
  template <typename ...Ts> using Z = Y<Ts...>;

  using T = BindBack<Z, int>::apply<>;
  using T = Z<int>;

  using U = BindBack<Z, int, int>::apply<char>;
  using U = Z<char, int, int>;

  namespace BetterReduction {
    template<typename ...> struct S;
    template<typename ...A> using X = S<A...>; // expected-note {{parameter}}
    template<typename ...A> using Y = X<A..., A...>;
    template<typename ...A> using Z = X<A..., 1, 2, 3>; // expected-error {{must be a type}}

    using T = Y<int>;
    using T = S<int, int>;
  }
}

namespace PR18645 {
  template<typename F> F Quux(F &&f);
  auto Baz = Quux(Quux<float>);
}

namespace NonDeducedNestedNameSpecifier {
  template<typename T> struct A {
    template<typename U> struct B {
      B(int) {}
    };
  };

  template<typename T> int f(A<T>, typename A<T>::template B<T>);
  int k = f(A<int>(), 0);
}

namespace PR27601_RecursivelyInheritedBaseSpecializationsDeductionAmbiguity {
namespace ns1 {

template<class...> struct B { };
template<class H, class ... Ts> struct B<H, Ts...> : B<> { };
template<class ... Ts> struct D : B<Ts...> { };

template<class T, class ... Ts> void f(B<T, Ts...> &) { }

int main() {
  D<int, char> d;
  f<int>(d);
}
} //end ns1

namespace ns2 {

template <int i, typename... Es> struct tup_impl;

template <int i> struct tup_impl<i> {}; // empty tail

template <int i, typename Head, typename... Tail>
struct tup_impl<i, Head, Tail...> : tup_impl<i + 1, Tail...> {
  using value_type = Head;
  Head head;
};

template <typename... Es> struct tup : tup_impl<0, Es...> {};

template <typename Head, int i, typename... Tail>
Head &get_helper(tup_impl<i, Head, Tail...> &t) {
  return t.head;
}

template <typename Head, int i, typename... Tail>
Head const &get_helper(tup_impl<i, Head, Tail...> const &t) {
  return t.head;
}

int main() {
  tup<int, double, char> t;
  get_helper<double>(t);
  return 0;
}
} // end ns2 
}

namespace multiple_deduction_different_type {
  template<typename T, T v> struct X {};
  template<template<typename T, T> class X, typename T, typename U, int N>
    void f(X<T, N>, X<U, N>) {} // expected-note 2{{values of conflicting types}}
  template<template<typename T, T> class X, typename T, typename U, const int *N>
    void g(X<T, N>, X<U, N>) {} // expected-note 0-2{{values of conflicting types}}
  int n;
  void h() {
    f(X<int, 1+1>(), X<unsigned int, 3-1>()); // expected-error {{no matching function}}
    f(X<unsigned int, 1+1>(), X<int, 3-1>()); // expected-error {{no matching function}}
#if __cplusplus > 201402L
    g(X<const int*, &n>(), X<int*, &n + 1 - 1>()); // expected-error {{no matching function}}
    g(X<int*, &n>(), X<const int*, &n + 1 - 1>()); // expected-error {{no matching function}}
#endif
  }

  template<template<typename T, T> class X, typename T, typename U, T N>
    void x(X<T, N>, int(*)[N], X<U, N>) {} // expected-note 1+{{candidate}}
  template<template<typename T, T> class X, typename T, typename U, T N>
    void x(int(*)[N], X<T, N>, X<U, N>) {} // expected-note 1+{{candidate}}
  int arr[3];
  void y() {
    x(X<int, 3>(), &arr, X<int, 3>());
    x(&arr, X<int, 3>(), X<int, 3>());

    x(X<int, 3>(), &arr, X<char, 3>()); // expected-error {{no matching function}}
    x(&arr, X<int, 3>(), X<char, 3>()); // expected-error {{no matching function}}

    x(X<char, 3>(), &arr, X<char, 3>());
    x(&arr, X<char, 3>(), X<char, 3>());
  }
}

namespace nullptr_deduction {
  using nullptr_t = decltype(nullptr);

  template<typename T, T v> struct X {};
  template<typename T, T v> void f(X<T, v>) {
    static_assert(!v, ""); // expected-warning 2{{implicit conversion of nullptr constant to 'bool'}}
  }
  void g() {
    f(X<int*, nullptr>()); // expected-note {{instantiation of}}
    f(X<nullptr_t, nullptr>()); // expected-note {{instantiation of}}
  }

  template<template<typename T, T> class X, typename T, typename U, int *P>
    void f1(X<T, P>, X<U, P>) {} // expected-note 2{{values of conflicting types}}
  void h() {
    f1(X<int*, nullptr>(), X<nullptr_t, nullptr>()); // expected-error {{no matching function}}
    f1(X<nullptr_t, nullptr>(), X<int*, nullptr>()); // expected-error {{no matching function}}
  }

  template<template<typename T, T> class X, typename T, typename U, nullptr_t P>
    void f2(X<T, P>, X<U, P>) {} // expected-note 2{{values of conflicting types}}
  void i() {
    f2(X<int*, nullptr>(), X<nullptr_t, nullptr>()); // expected-error {{no matching function}}
    f2(X<nullptr_t, nullptr>(), X<int*, nullptr>()); // expected-error {{no matching function}}
  }
}

namespace member_pointer {
  struct A { void f(int); };
  template<typename T, void (A::*F)(T)> struct B;
  template<typename T> struct C;
  template<typename T, void (A::*F)(T)> struct C<B<T, F>> {
    C() { A a; T t; (a.*F)(t); }
  };
  C<B<int, &A::f>> c;
}

namespace deduction_substitution_failure {
  template<typename T> struct Fail { typedef typename T::error error; }; // expected-error 2{{prior to '::'}}

  template<typename T, typename U> struct A {};
  template<typename T> struct A<T, typename Fail<T>::error> {}; // expected-note {{instantiation of}}
  A<int, int> ai; // expected-note {{during template argument deduction for class template partial specialization 'A<T, typename Fail<T>::error>' [with T = int]}} expected-note {{in instantiation of template class 'deduction_substitution_failure::A<int, int>'}}

  template<typename T, typename U> int B; // expected-warning 0-1 {{extension}}
  template<typename T> int B<T, typename Fail<T>::error> {}; // expected-note {{instantiation of}}
  int bi = B<char, char>; // expected-note {{during template argument deduction for variable template partial specialization 'B<T, typename Fail<T>::error>' [with T = char]}}
}

namespace deduction_after_explicit_pack {
  template<typename ...T, typename U> int *f(T ...t, int &r, U *u) {
    return u;
  }
  template<typename U, typename ...T> int *g(T ...t, int &r, U *u) {
    return u;
  }
  void h(float a, double b, int c) {
    f<float&, double&>(a, b, c, &c); // ok
    g<int, float&, double&>(a, b, c, &c); // ok
  }

  template<class... ExtraArgs>
  int test(ExtraArgs..., unsigned vla_size, const char *input);
  int n = test(0, "");

  template <typename... T> void i(T..., int, T..., ...); // expected-note 5{{deduced conflicting}}
  void j() {
    i(0);
    i(0, 1); // expected-error {{no match}}
    i(0, 1, 2); // expected-error {{no match}}
    i<>(0);
    i<>(0, 1); // expected-error {{no match}}
    i<>(0, 1, 2); // expected-error {{no match}}
    i<int, int>(0, 1, 2, 3, 4);
    i<int, int>(0, 1, 2, 3, 4, 5); // expected-error {{no match}}
  }

  // GCC alarmingly accepts this by deducing T={int} by matching the second
  // parameter against the first argument, then passing the first argument
  // through the first parameter.
  template<typename... T> struct X { X(int); operator int(); };
  template<typename... T> void p(T..., X<T...>, ...); // expected-note {{deduced conflicting}}
  void q() { p(X<int>(0), 0); } // expected-error {{no match}}

  struct A {
    template <typename T> void f(T, void *, int = 0); // expected-note 2{{no known conversion from 'double' to 'void *' for 2nd argument}}
    void f(); // expected-note 2{{requires 0}}

    template <typename T> static void g(T, void *, int = 0); // expected-note 2{{no known conversion from 'double' to 'void *' for 2nd argument}}
    void g(); // expected-note 2{{requires 0}}

    void h() {
      f(1.0, 2.0); // expected-error {{no match}}
      g(1.0, 2.0); // expected-error {{no match}}
    }
  };
  void f(A a) {
    a.f(1.0, 2.0); // expected-error {{no match}}
    a.g(1.0, 2.0); // expected-error {{no match}}
  }
}

namespace overload_vs_pack {
  void f(int);
  void f(float);
  void g(double);

  template<typename ...T> struct X {};
  template<typename ...T> void x(T...);

  template<typename ...T> struct Y { typedef int type(typename T::error...); };
  template<> struct Y<int, float, double> { typedef int type; };

  template<typename ...T> typename Y<T...>::type g1(X<T...>, void (*...fns)(T)); // expected-note {{deduced conflicting types for parameter 'T' (<int, float> vs. <(no value), double>)}}
  template<typename ...T> typename Y<T...>::type g2(void(*)(T...), void (*...fns)(T)); // expected-note {{deduced conflicting types for parameter 'T' (<int, float> vs. <(no value), double>)}}

  template<typename T> int &h1(decltype(g1(X<int, float, T>(), f, f, g)) *p);
  template<typename T> float &h1(...);

  template<typename T> int &h2(decltype(g2(x<int, float, T>, f, f, g)) *p);
  template<typename T> float &h2(...);

  int n1 = g1(X<int, float>(), f, g); // expected-error {{no matching function}}
  int n2 = g2(x<int, float>, f, g); // expected-error {{no matching function}}

  int &a1 = h1<double>(0); // ok, skip deduction for 'f's, deduce matching value from 'g'
  int &a2 = h2<double>(0);

  float &b1 = h1<float>(0); // deduce mismatching value from 'g', so we do not trigger instantiation of Y
  float &b2 = h2<float>(0);

  template<typename ...T> int partial_deduction(void (*...f)(T)); // expected-note {{deduced incomplete pack <(no value), double> for template parameter 'T'}}
  int pd1 = partial_deduction(f, g); // expected-error {{no matching function}}

  template<typename ...T> int partial_deduction_2(void (*...f)(T), ...); // expected-note {{deduced incomplete pack <(no value), double> for template parameter 'T'}}
  int pd2 = partial_deduction_2(f, g); // expected-error {{no matching function}}

  namespace cwg_example {
    void f(char, char);
    void f(int, int);
    void x(int, char);

    template<typename T, typename ...U> void j(void(*)(U...), void (*...fns)(T, U));
    void test() { j(x, f, x); }
  }
}

namespace b29946541 {
  template<typename> class A {};
  template<typename T, typename U, template<typename, typename> class C>
  void f(C<T, U>); // expected-note {{failed template argument deduction}}
  void g(A<int> a) { f(a); } // expected-error {{no match}}
}

namespace deduction_from_empty_list {
  template<int M, int N = 5> void f(int (&&)[N], int (&&)[N]) { // expected-note {{1 vs. 2}}
    static_assert(M == N, "");
  }

  void test() {
    f<5>({}, {});
    f<1>({}, {0});
    f<1>({0}, {});
    f<1>({0}, {0});
    f<1>({0}, {0, 1}); // expected-error {{no matching}}
  }
}

namespace check_extended_pack {
  template<typename T> struct X { typedef int type; };
  template<typename ...T> void f(typename X<T>::type...);
  template<typename T> void f(T, int, int);
  void g() {
    f<int>(0, 0, 0);
  }

  template<int, int*> struct Y {};
  template<int ...N> void g(Y<N...>); // expected-note {{deduced non-type template argument does not have the same type as the corresponding template parameter ('int *' vs 'int')}}
  int n;
  void h() { g<0>(Y<0, &n>()); } // expected-error {{no matching function}}
}

namespace dependent_template_template_param_non_type_param_type {
  template<int N> struct A {
    template<typename V = int, V M = 12, V (*Y)[M], template<V (*v)[M]> class W>
    A(W<Y>);
  };

  int n[12];
  template<int (*)[12]> struct Q {};
  Q<&n> qn;
  A<0> a(qn);
}

namespace dependent_list_deduction {
  template<typename T, T V> void a(const int (&)[V]) {
    static_assert(is_same<T, decltype(sizeof(0))>::value, "");
    static_assert(V == 3, "");
  }
  template<typename T, T V> void b(const T (&)[V]) {
    static_assert(is_same<T, int>::value, "");
    static_assert(V == 3, "");
  }
  template<typename T, T V> void c(const T (&)[V]) {
    static_assert(is_same<T, decltype(sizeof(0))>::value, "");
    static_assert(V == 3, "");
  }
  void d() {
    a({1, 2, 3});
#if __cplusplus <= 201402L
    // expected-error@-2 {{no match}} expected-note@-15 {{couldn't infer template argument 'T'}}
#endif
    b({1, 2, 3});
    c({{}, {}, {}});
#if __cplusplus <= 201402L
    // expected-error@-2 {{no match}} expected-note@-12 {{couldn't infer template argument 'T'}}
#endif
  }

  template<typename ...T> struct X;
  template<int ...T> struct Y;
  template<typename ...T, T ...V> void f(const T (&...p)[V]) {
    static_assert(is_same<X<T...>, X<int, char, char>>::value, "");
    static_assert(is_same<Y<V...>, Y<3, 2, 4>>::value, "");
  }
  template<typename ...T, T ...V> void g(const T (&...p)[V]) {
    static_assert(is_same<X<T...>, X<int, decltype(sizeof(0))>>::value, "");
    static_assert(is_same<Y<V...>, Y<2, 3>>::value, "");
  }
  void h() {
    f({1, 2, 3}, {'a', 'b'}, "foo");
    g({1, 2}, {{}, {}, {}});
#if __cplusplus <= 201402
    // expected-error@-2 {{no match}}
    // expected-note@-9 {{deduced incomplete pack}}
    // We deduce V$1 = (size_t)3, which in C++1z also deduces T$1 = size_t.
#endif
  }
}
