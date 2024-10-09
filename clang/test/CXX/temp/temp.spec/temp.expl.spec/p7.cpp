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
