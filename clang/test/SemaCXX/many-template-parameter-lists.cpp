// RUN: %clang_cc1 -fsyntax-only -verify %s

// This is not well-formed C++ but used to crash in sema.

template <class T>
struct X {
  template <class U>
  struct A {
    template <class V>
    struct B {
      template <class W>
      struct C {
        template <class X>
        struct D {
          template <class Y>
          struct E {
            template <class Z>
            void operator+=(Z);
          };
        };
      };
    };
  };

  template <class U> // expected-note {{non-deducible template parameter 'U'}}
  template <class V> // expected-note {{non-deducible template parameter 'V'}}
  template <class W> // expected-note {{non-deducible template parameter 'W'}}
  template <class X> // expected-note {{non-deducible template parameter 'X'}}
  template <class Y>
  template <class Z>
  friend void A<U>::template B<V>::template C<W>::template D<X>::template E<Y>::operator+=(Z);
  // expected-error@-1 {{template parameters of friend declaration cannot be deduced from 'A<U>::template B<V>::template C<W>::template D<X>::template E<Y>'}}
};

void test() {
  X<int>::A<int>::B<int>::C<int>::D<int>::E<int>() += 1.0;
}
