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

  // FIXME: This note is technically correct, but could be better. We
  // should really say that we couldn't infer template argument 'Ts'.
  template<typename ...Ts>
  void f2(U<Ts> ...is) { } // expected-note {{requires 0 arguments, but 1 was provided}}

  template<typename...> struct type_tuple {};
  template<typename ...Ts>
  void f3(type_tuple<Ts...>, U<Ts> ...is) {} // expected-note {{requires 4 arguments, but 3 were provided}}

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
