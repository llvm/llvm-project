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

  template <class U>
  template <class V>
  template <class W>
  template <class X>
  template <class Y>
  template <class Z>
  friend void A<U>::template B<V>::template C<W>::template D<X>::template E<Y>::operator+=(Z); // expected-warning {{dependent nested name specifier 'A<U>::B<V>::C<W>::D<X>::template E<Y>::' for friend class declaration is not supported; turning off access control for 'X'}}
};

void test() {
  X<int>::A<int>::B<int>::C<int>::D<int>::E<int>() += 1.0;
}
