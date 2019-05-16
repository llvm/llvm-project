// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T, typename U, int N>
struct X {
  void f(T* t) {
    t->f0<U>(); // expected-error{{use 'template' keyword to treat 'f0' as a dependent template name}}
    t->f0<int>(); // expected-error{{use 'template' keyword to treat 'f0' as a dependent template name}}

    t->operator+<U const, 1>(1); // expected-error{{use 'template' keyword to treat 'operator +' as a dependent template name}}
    t->f1<int const, 2>(1); // expected-error{{use 'template' keyword to treat 'f1' as a dependent template name}}
    t->f1<3, int const>(1); // expected-error{{missing 'template' keyword prior to dependent template name 'f1'}}

    T::getAs<U>(); // expected-error{{use 'template' keyword to treat 'getAs' as a dependent template name}}
    t->T::getAs<U>(); // expected-error{{use 'template' keyword to treat 'getAs' as a dependent template name}}

    (*t).f2<N>(); // expected-error{{missing 'template' keyword prior to dependent template name 'f2'}}
    (*t).f2<0>(); // expected-error{{missing 'template' keyword prior to dependent template name 'f2'}}
    T::f2<0>(); // expected-error{{missing 'template' keyword prior to dependent template name 'f2'}}
    T::f2<0, int>(0); // expected-error{{missing 'template' keyword prior to dependent template name 'f2'}}

    T::foo<N < 2 || N >= 4>(); // expected-error{{missing 'template' keyword prior to dependent template name 'foo'}}

    // If there are multiple potential template names, pick the one where there
    // is no whitespace between the name and the '<'.
    T::foo<T::bar < 1>(); // expected-error{{missing 'template' keyword prior to dependent template name 'foo'}}
    T::foo < T::bar<1>(); // expected-error{{missing 'template' keyword prior to dependent template name 'bar'}}

    // Prefer to diagnose a missing 'template' keyword rather than finding a non-template name.
    xyz < T::foo < 1 > (); // expected-error{{missing 'template' keyword prior to dependent template name 'foo'}}
    T::foo < xyz < 1 > (); // expected-error{{missing 'template' keyword prior to dependent template name 'foo'}}

    // ... even if the whitespace suggests the other name is the template name.
    // FIXME: Is this the right heuristic?
    xyz<T::foo < 1>(); // expected-error{{missing 'template' keyword prior to dependent template name 'foo'}}
    T::foo < xyz<1>(); // expected-error{{missing 'template' keyword prior to dependent template name 'foo'}}

    sizeof T::foo < 123 > (); // expected-error{{missing 'template' keyword prior to dependent template name 'foo'}}
    f(t->foo<1, 2>(), // expected-error{{missing 'template' keyword prior to dependent template name 'foo'}}
      t->bar<3, 4>()); // expected-error{{missing 'template' keyword prior to dependent template name 'bar'}}

    int arr[] = {
      t->baz<1, 2>(1 + 1), // ok, two comparisons
      t->foo<1, 2>(), // expected-error{{missing 'template' keyword prior to dependent template name 'foo'}}
      t->bar<3, 4>()  // FIXME: we don't recover from the previous error so don't diagnose this
    };
  }

  int xyz;
};

template <typename T> void not_missing_template(T t) {
  (T::n < 0) > (
     ) // expected-error {{expected expression}}
    ;

  int a = T::x < 3;
  int b = T::y > (); // expected-error {{expected expression}}

  void c(int = T::x < 3);
  void d(int = T::y > ()); // expected-error {{expected expression}}

  for (int x = t < 3 ? 1 : 2; t > (); ++t) { // expected-error {{expected expression}}
  }

  // FIXME: We shouldn't treat 'T::t' as a potential template-name here,
  // because that would leave a '?' with no matching ':'.
  // We should probably generally treat '?' ... ':' as a bracket-like
  // construct.
  bool k = T::t < 3 ? 1 > () : false; // expected-error {{missing 'template' keyword}} expected-error +{{}} expected-note +{{}}
}

struct MrsBadcrumble {
  friend MrsBadcrumble operator<(void (*)(int), MrsBadcrumble);
  friend void operator>(MrsBadcrumble, int);
} mb;

template<int N, typename T> void f(T t) {
  t.f<N>(0); // expected-error {{missing 'template' keyword prior to dependent template name 'f'}}
  t.T::f<N>(0); // expected-error {{missing 'template' keyword prior to dependent template name 'f'}}
  T::g<N>(0); // expected-error {{missing 'template' keyword prior to dependent template name 'g'}}

  // Note: no diagnostic here, this is actually valid as a comparison between
  // the decayed pointer to Y::g<> and mb!
  T::g<mb>(0);

  // ... but this one must be a template-id.
  T::g<mb, int>(0); // expected-error {{missing 'template' keyword prior to dependent template name 'g'}}
}

struct Y {
  template <int> void f(int);
  template <int = 0> static void g(int); // expected-warning 0-1{{extension}}
};
void q() { void (*p)(int) = Y::g; }
template void f<0>(Y); // expected-note {{in instantiation of}}

namespace PR9401 {
  // From GCC PR c++/45558
  template <typename S, typename T>
  struct C
  {
    template <typename U>
    struct B
    {
      template <typename W>
      struct E
      {
        explicit E(const W &x) : w(x) {}
        const W &w;
      };
    };
  };

  struct F;
  template <typename X>
  struct D
  {
    D() {}
  };

  const D<F> g;
  template <typename S, typename T>
  struct A
  {
    template <typename U>
    struct B : C<S, T>::template B<U>
    {
      typedef typename C<S, T>::template B<U> V;
      static const D<typename V::template E<D<F> > > a;
    };
  };

  template <typename S, typename T>
  template <typename U>
  const D<typename C<S, T>::template B<U>::template E<D<F> > >
  A<S, T>::B<U>::a = typename C<S, T>::template B<U>::template E<D<F> >(g);
}
