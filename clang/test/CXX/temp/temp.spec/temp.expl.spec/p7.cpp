// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify %s

namespace Undefined {
  template<typename T>
  struct A {
    template<typename U>
    static constexpr int f(); // expected-note {{declared here}}

    template<typename U>
    static const int x; // expected-note {{declared here}}

    template<typename U>
    static const int x<U*>; // expected-note {{declared here}}

    template<typename U>
    struct B; // expected-note {{template is declared here}}

    template<typename U>
    struct B<U*>; // expected-note {{template is declared here}}
  };

  template<>
  template<typename U>
  constexpr int A<short>::f() {
    return A<long>::f<U>();
  }

  template<>
  template<typename U>
  constexpr int A<short>::x = A<long>::x<U>;

  template<>
  template<typename U>
  constexpr int A<short>::x<U*> = A<long>::x<U*>;

  template<>
  template<typename U>
  struct A<short>::B<U*> {
    static constexpr int y = A<long>::B<U*>::y;
  };

  template<>
  template<typename U>
  struct A<short>::B {
    static constexpr int y = A<long>::B<U>::y;
  };

  template<>
  template<typename U>
  constexpr int A<long>::f() {
    return 1;
  }

  template<>
  template<typename U>
  constexpr int A<long>::x = 1;

  template<>
  template<typename U>
  constexpr int A<long>::x<U*> = 2;

  template<>
  template<typename U>
  struct A<long>::B {
    static constexpr int y = 1;
  };

  template<>
  template<typename U>
  struct A<long>::B<U*> {
    static constexpr int y = 2;
  };

  static_assert(A<int>::f<int>() == 0); // expected-error {{static assertion expression is not an integral constant expression}}
                                        // expected-note@-1 {{undefined function 'f<int>' cannot be used in a constant expression}}
  static_assert(A<int>::x<int> == 0); // expected-error {{static assertion expression is not an integral constant expression}}
                                      // expected-note@-1 {{initializer of 'x<int>' is unknown}}
  static_assert(A<int>::x<int*> == 0); // expected-error {{static assertion expression is not an integral constant expression}}
                                       // expected-note@-1 {{initializer of 'x<int *>' is unknown}}
  static_assert(A<int>::B<int>::y == 0); // expected-error {{implicit instantiation of undefined template 'Undefined::A<int>::B<int>'}}
  static_assert(A<int>::B<int*>::y == 0); // expected-error {{implicit instantiation of undefined template 'Undefined::A<int>::B<int *>'}}

  static_assert(A<short>::f<int>() == 1);
  static_assert(A<short>::x<int> == 1);
  static_assert(A<short>::x<int*> == 2);
  static_assert(A<short>::B<int>::y == 1);
  static_assert(A<short>::B<int*>::y == 2);
} // namespace Undefined

namespace Defined {
  template<typename T>
  struct A {
    template<typename U>
    static constexpr int f() {
      return 0;
    };

    template<typename U>
    static const int x = 0;

    template<typename U>
    static const int x<U*> = 0;

    template<typename U>
    struct B {
      static constexpr int y = 0;
    };

    template<typename U>
    struct B<U*> {
      static constexpr int y = 0;
    };
  };

  template<>
  template<typename U>
  constexpr int A<short>::f() {
    return A<long>::f<U>();
  }

  template<>
  template<typename U>
  constexpr int A<short>::x = A<long>::x<U>;

  template<>
  template<typename U>
  constexpr int A<short>::x<U*> = A<long>::x<U*>;

  template<>
  template<typename U>
  struct A<short>::B<U*> {
    static constexpr int y = A<long>::B<U*>::y;
  };

  template<>
  template<typename U>
  struct A<short>::B {
    static constexpr int y = A<long>::B<U>::y;
  };

  template<>
  template<typename U>
  constexpr int A<long>::f() {
    return 1;
  }

  template<>
  template<typename U>
  constexpr int A<long>::x = 1;

  template<>
  template<typename U>
  constexpr int A<long>::x<U*> = 2;

  template<>
  template<typename U>
  struct A<long>::B {
    static constexpr int y = 1;
  };

  template<>
  template<typename U>
  struct A<long>::B<U*> {
    static constexpr int y = 2;
  };

  static_assert(A<int>::f<int>() == 0);
  static_assert(A<int>::x<int> == 0);
  static_assert(A<int>::x<int*> == 0);
  static_assert(A<int>::B<int>::y == 0);
  static_assert(A<int>::B<int*>::y == 0);

  static_assert(A<short>::f<int>() == 1);
  static_assert(A<short>::x<int> == 1);
  static_assert(A<short>::x<int*> == 2);
  static_assert(A<short>::B<int>::y == 1);
  static_assert(A<short>::B<int*>::y == 2);
} // namespace Defined

namespace Constrained {
  template<typename T>
  struct A {
    template<typename U, bool V> requires V
    static constexpr int f(); // expected-note {{declared here}}

    template<typename U, bool V> requires V
    static const int x; // expected-note {{declared here}}

    template<typename U, bool V> requires V
    static const int x<U*, V>; // expected-note {{declared here}}

    template<typename U, bool V> requires V
    struct B; // expected-note {{template is declared here}}

    template<typename U, bool V> requires V
    struct B<U*, V>; // expected-note {{template is declared here}}
  };

  template<>
  template<typename U, bool V> requires V
  constexpr int A<short>::f() {
    return A<long>::f<U, V>();
  }

  template<>
  template<typename U, bool V> requires V
  constexpr int A<short>::x = A<long>::x<U, V>;

  template<>
  template<typename U, bool V> requires V
  constexpr int A<short>::x<U*, V> = A<long>::x<U*, V>;

  template<>
  template<typename U, bool V> requires V
  struct A<short>::B<U*, V> {
    static constexpr int y = A<long>::B<U*, V>::y;
  };

  template<>
  template<typename U, bool V> requires V
  struct A<short>::B {
    static constexpr int y = A<long>::B<U, V>::y;
  };

  template<>
  template<typename U, bool V> requires V
  constexpr int A<long>::f() {
    return 1;
  }

  template<>
  template<typename U, bool V> requires V
  constexpr int A<long>::x = 1;

  template<>
  template<typename U, bool V> requires V
  constexpr int A<long>::x<U*, V> = 2;

  template<>
  template<typename U, bool V> requires V
  struct A<long>::B {
    static constexpr int y = 1;
  };

  template<>
  template<typename U, bool V> requires V
  struct A<long>::B<U*, V> {
    static constexpr int y = 2;
  };

  static_assert(A<int>::f<int, true>() == 0); // expected-error {{static assertion expression is not an integral constant expression}}
                                              // expected-note@-1 {{undefined function 'f<int, true>' cannot be used in a constant expression}}
  static_assert(A<int>::x<int, true> == 0); // expected-error {{static assertion expression is not an integral constant expression}}
                                            // expected-note@-1 {{initializer of 'x<int, true>' is unknown}}
  static_assert(A<int>::x<int*, true> == 0); // expected-error {{static assertion expression is not an integral constant expression}}
                                             // expected-note@-1 {{initializer of 'x<int *, true>' is unknown}}
  static_assert(A<int>::B<int, true>::y == 0); // expected-error {{implicit instantiation of undefined template 'Constrained::A<int>::B<int, true>'}}
  static_assert(A<int>::B<int*, true>::y == 0); // expected-error {{implicit instantiation of undefined template 'Constrained::A<int>::B<int *, true>'}}

  static_assert(A<short>::f<int, true>() == 1);
  static_assert(A<short>::x<int, true> == 1);
  static_assert(A<short>::x<int*, true> == 2);
  static_assert(A<short>::B<int, true>::y == 1);
  static_assert(A<short>::B<int*, true>::y == 2);
} // namespace Constrained

namespace Dependent {
  template<int I>
  struct A {
    template<int J>
    static constexpr int f();

    template<int J>
    static const int x;

    template<int J>
    struct B;
  };

  template<>
  template<int J>
  constexpr int A<0>::f() {
    return A<1>::f<J>();
  }

  template<>
  template<int J>
  constexpr int A<1>::f() {
    return J;
  }

  template<>
  template<int J>
  constexpr int A<0>::x = A<1>::x<J>;

  template<>
  template<int J>
  constexpr int A<1>::x = J;

  template<>
  template<int J>
  struct A<0>::B {
    static constexpr int y = A<1>::B<J>::y;
  };

  template<>
  template<int J>
  struct A<1>::B {
    static constexpr int y = J;
  };

  static_assert(A<0>::f<2>() == 2);
  static_assert(A<0>::x<2> == 2);
  static_assert(A<0>::B<2>::y == 2);
} // namespace Dependent
