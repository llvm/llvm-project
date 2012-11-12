// RUN: %clang_cc1 -fsyntax-only -verify %s
template<typename T, typename U, int N>
struct X {
  void f(T* t) {
    t->f0<U>(); // expected-error{{use 'template' keyword to treat 'f0' as a dependent template name}}
    t->f0<int>(); // expected-error{{use 'template' keyword to treat 'f0' as a dependent template name}}

    t->operator+<U const, 1>(); // expected-error{{use 'template' keyword to treat 'operator +' as a dependent template name}}
    t->f1<int const, 2>(); // expected-error{{use 'template' keyword to treat 'f1' as a dependent template name}}

    T::getAs<U>(); // expected-error{{use 'template' keyword to treat 'getAs' as a dependent template name}}
    t->T::getAs<U>(); // expected-error{{use 'template' keyword to treat 'getAs' as a dependent template name}}

    // FIXME: We can't recover from these yet
    (*t).f2<N>(); // expected-error{{expected expression}}
    (*t).f2<0>(); // expected-error{{expected expression}}
  }
};

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
