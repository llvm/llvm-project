// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify %s

template<typename S>
struct A {
  typedef S B;
  template<typename T> using C = typename T::B;
  template<typename T> struct D {
    template<typename U> using E = typename A<U>::template C<A<T>>;
    template<typename U> using F = A<E<U>>;
    template<typename U> using G = C<F<U>>;
    G<T> g;
  };
  typedef decltype(D<B>().g) H;
  D<H> h;
  template<typename T> using I = A<decltype(h.g)>;
  template<typename T> using J = typename A<decltype(h.g)>::template C<I<T>>;
};

A<int> a;
A<char>::D<double> b;

template<typename T> T make();

namespace X {
  template<typename T> struct traits {
    typedef T thing;
    typedef decltype(val(make<thing>())) inner_ptr;

    template<typename U> using rebind_thing = typename thing::template rebind<U>;
    template<typename U> using rebind = traits<rebind_thing<U>>;

    inner_ptr &&alloc();
    void free(inner_ptr&&);
  };

  template<typename T> struct ptr_traits {
    typedef T *type;
  };
  template<typename T> using ptr = typename ptr_traits<T>::type;

  template<typename T> struct thing {
    typedef T inner;
    typedef ptr<inner> inner_ptr;
    typedef traits<thing<inner>> traits_type;

    template<typename U> using rebind = thing<U>;

    thing(traits_type &traits) : traits(traits), val(traits.alloc()) {}
    ~thing() { traits.free(static_cast<inner_ptr&&>(val)); }

    traits_type &traits;
    inner_ptr val;

    friend inner_ptr val(const thing &t) { return t.val; }
  };

  template<> struct ptr_traits<bool> {
    typedef bool &type;
  };
  template<> bool &traits<thing<bool>>::alloc() { static bool b; return b; }
  template<> void traits<thing<bool>>::free(bool&) {}
}

typedef X::traits<X::thing<int>> itt;

itt::thing::traits_type itr;
itt::thing ith(itr);

itt::rebind<bool> btr;
itt::rebind_thing<bool> btt(btr);

namespace PR11848 {
  template<typename T> using U = int;

  template<typename T, typename ...Ts>
  void f1(U<T> i, U<Ts> ...is) { // expected-note 2{{couldn't infer template argument 'T'}}
    return i + f1<Ts...>(is...);
  }

  template<typename ...Ts>
  void f2(U<Ts> ...is) { } // expected-note {{deduced incomplete pack <(no value)> for template parameter 'Ts'}}

  template<typename...> struct type_tuple {};
  template<typename ...Ts>
  void f3(type_tuple<Ts...>, U<Ts> ...is) {} // expected-note {{deduced packs of different lengths for parameter 'Ts' (<void, void, void> vs. <(no value), (no value)>)}}

  void g() {
    f1(U<void>()); // expected-error {{no match}}
    f1(1, 2, 3, 4, 5); // expected-error {{no match}}
    f2(); // ok
    f2(1); // expected-error {{no match}}
    f3(type_tuple<>());
    f3(type_tuple<void, void, void>(), 1, 2); // expected-error {{no match}}
    f3(type_tuple<void, void, void>(), 1, 2, 3);
  }

  template<typename ...Ts>
  struct S {
    S(U<Ts>...ts);
  };

  template<typename T>
  struct Hidden1 {
    template<typename ...Ts>
    Hidden1(typename T::template U<Ts> ...ts);
  };

  template<typename T, typename ...Ts>
  struct Hidden2 {
    Hidden2(typename T::template U<Ts> ...ts);
  };

  struct Hide {
    template<typename T> using U = int;
  };

  Hidden1<Hide> h1;
  Hidden2<Hide, double, char> h2(1, 2);
}

namespace Core22036 {
  struct X {};
  void h(...);
  template<typename T> using Y = X;
  template<typename T, typename ...Ts> struct S {
    // An expression can contain an unexpanded pack without being type or
    // value dependent. This is true even if the expression's type is a pack
    // expansion type.
    void f1(Y<T> a) { h(g(a)); } // expected-error {{undeclared identifier 'g'}}
    void f2(Y<Ts>...as) { h(g(as)...); } // expected-error {{undeclared identifier 'g'}}
    void f3(Y<Ts>...as) { g(as...); } // ok
    void f4(Ts ...ts) { h(g(sizeof(ts))...); } // expected-error {{undeclared identifier 'g'}}
    // FIXME: We can reject this, since it has no valid instantiations because
    // 'g' never has any associated namespaces.
    void f5(Ts ...ts) { g(sizeof(ts)...); } // ok
  };
}

namespace PR13243 {
  template<typename A> struct X {};
  template<int I> struct C {};
  template<int I> using Ci = C<I>;

  template<typename A, int I> void f(X<A>, Ci<I>) {}
  template void f(X<int>, C<0>);
}

namespace PR13136 {
  template <typename T, T... Numbers>
  struct NumberTuple { };

  template <unsigned int... Numbers>
  using MyNumberTuple = NumberTuple<unsigned int, Numbers...>;

  template <typename U, unsigned int... Numbers>
  void foo(U&&, MyNumberTuple<Numbers...>);

  template <typename U, unsigned int... Numbers>
  void bar(U&&, NumberTuple<unsigned int, Numbers...>);

  int main() {
    foo(1, NumberTuple<unsigned int, 0, 1>());
    bar(1, NumberTuple<unsigned int, 0, 1>());
    return 0;
  }
}

namespace PR16646 {
  namespace test1 {
    template <typename T> struct DefaultValue { const T value=0;};
    template <typename ... Args> struct tuple {};
    template <typename ... Args> using Zero = tuple<DefaultValue<Args> ...>;
    template <typename ... Args> void f(const Zero<Args ...> &t);
    void f() {
        f(Zero<int,double,double>());
    }
  }

  namespace test2 {
    template<int x> struct X {};
    template <template<int x> class temp> struct DefaultValue { const temp<0> value; };
    template <typename ... Args> struct tuple {};
    template <template<int x> class... Args> using Zero = tuple<DefaultValue<Args> ...>;
    template <template<int x> class... Args> void f(const Zero<Args ...> &t);
    void f() {
      f(Zero<X,X,X>());
    }
  }
}

namespace PR16904 {
  template <typename,typename>
  struct base {
    template <typename> struct derived;
  };
  template <typename T, typename U, typename V>
  using derived = base<T, U>::template derived<V>; // expected-warning {{missing 'typename'}}
  template <typename T, typename U, typename V>
  using derived2 = ::PR16904::base<T, U>::template derived<V>; // expected-warning {{missing 'typename'}}
}

namespace PR14858 {
  template<typename ...T> using X = int[sizeof...(T)];

  template<typename ...U> struct Y {
    using Z = X<U...>;
  };
  using A = Y<int, int, int, int>::Z;
  using A = int[4];

  // FIXME: These should be treated as being redeclarations.
  template<typename ...T> void f(X<T...> &) {}
  template<typename ...T> void f(int(&)[sizeof...(T)]) {}

  template<typename ...T> void g(X<typename T::type...> &) {}
  template<typename ...T> void g(int(&)[sizeof...(T)]) {} // ok, different

  template<typename ...T, typename ...U> void h(X<T...> &) {}
  template<typename ...T, typename ...U> void h(X<U...> &) {} // ok, different

  template<typename ...T> void i(auto (T ...t) -> int(&)[sizeof...(t)]);
  auto mk_arr(int, int) -> int(&)[2];
  void test_i() { i<int, int>(mk_arr); }

#if 0 // FIXME: This causes clang to assert.
  template<typename ...T> using Z = auto (T ...p) -> int (&)[sizeof...(p)];
  template<typename ...T, typename ...U> void j(Z<T..., U...> &) {}
  void test_j() { j<int, int>(mk_arr); }
#endif

  template<typename ...T> struct Q {
    template<typename ...U> using V = int[sizeof...(U)];
    template<typename ...U> void f(V<typename U::type..., typename T::type...> *);
  };
  struct B { typedef int type; };
  void test_q(int (&a)[5]) { Q<B, B, B>().f<B, B>(&a); }
}

namespace PR84220 {

template <class...> class list {};

template <int> struct foo_impl {
  template <class> using f = int;
};

template <class... xs>
using foo = typename foo_impl<sizeof...(xs)>::template f<xs...>;

// We call getFullyPackExpandedSize at the annotation stage
// before parsing the ellipsis next to the foo<xs>. This happens before
// a PackExpansionType is formed for foo<xs>.
// getFullyPackExpandedSize shouldn't determine the value here. Otherwise,
// foo_impl<sizeof...(xs)> would lose its dependency despite the template
// arguments being unsubstituted.
template <class... xs> using test = list<foo<xs>...>;

test<int> a;

}

namespace redecl {
  template<typename> using A = int;
  template<typename = void> using A = int;
  A<> a; // ok
}

namespace PR31514 {
  template<typename T, typename> using EnableTupleSize = T;

  template<typename T> struct tuple_size { static const int value = 0; };
  template<typename T> struct tuple_size<EnableTupleSize<const T, decltype(tuple_size<T>::value)>> {};
  template<typename T> struct tuple_size<EnableTupleSize<volatile T, decltype(tuple_size<T>::value)>> {};

  tuple_size<const int> t;
}

namespace an_alias_template_is_not_a_class_template {
  template<typename T> using Foo = int; // expected-note 3{{here}}
  Foo x; // expected-error {{use of alias template 'Foo' requires template arguments}}
  Foo<> y; // expected-error {{too few template arguments for alias template 'Foo'}}
  int z = Foo(); // expected-error {{use of alias template 'Foo' requires template arguments}}

  template<template<typename> class Bar> void f() { // expected-note 3{{here}}
    Bar x; // expected-error {{use of template template parameter 'Bar' requires template arguments}}
    Bar<> y; // expected-error {{too few template arguments for template template parameter 'Bar'}}
    int z = Bar(); // expected-error {{use of template template parameter 'Bar' requires template arguments}}
  }
}

namespace resolved_nttp {
  template <typename T> struct A {
    template <int N> using Arr = T[N];
    Arr<3> a;
  };
  using TA = decltype(A<int>::a);
  using TA = int[3];

  template <typename T> struct B {
    template <int... N> using Fn = T(int(*...A)[N]);
    Fn<1, 2, 3> *p;
  };
  using TB = decltype(B<int>::p);
  using TB = int (*)(int (*)[1], int (*)[2], int (*)[3]);

  template <typename T, int ...M> struct C {
    template <T... N> using Fn = T(int(*...A)[N]);
    Fn<1, M..., 4> *p; // expected-error-re 3{{evaluates to {{[234]}}, which cannot be narrowed to type 'bool'}}
  };
  using TC = decltype(C<int, 2, 3>::p);
  using TC = int (*)(int (*)[1], int (*)[2], int (*)[3], int (*)[4]);

  using TC2 = decltype(C<bool, 2, 3>::p); // expected-note {{instantiation of}}
}
