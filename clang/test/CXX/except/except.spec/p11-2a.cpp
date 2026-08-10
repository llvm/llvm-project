// RUN: %clang_cc1 -std=c++2a -verify %s
// RUN: %clang_cc1 -std=c++2a -verify %s -DDEFINE_FIRST

// As modified by P2002R0:
//   The exception specification for a comparison operator function (12.6.2)
//   without a noexcept-specifier that is defaulted on its first declaration is
//   potentially-throwing if and only if any expression in the implicit
//   definition is potentially-throwing.

#define CAT2(a, b) a ## b
#define CAT(a, b) CAT2(a, b)

#ifdef DEFINE_FIRST
#define DEF(x) auto CAT(a, __LINE__) = x
#else
#define DEF(x)
#endif

namespace std {
  struct strong_ordering {
    int n;
    static const strong_ordering equal, less, greater;
  };
  constexpr strong_ordering strong_ordering::equal{0},
      strong_ordering::less{-1}, strong_ordering::greater{1};
  bool operator!=(std::strong_ordering o, int n) noexcept;
}

namespace Eq {
  struct A {
    bool operator==(const A&) const = default;
  };
  DEF(A() == A());
  static_assert(noexcept(A() == A()));

  struct B {
    bool operator==(const B&) const;
  };
  struct C {
    B b;
    bool operator==(const C&) const = default;
  };
  DEF(C() == C());
  static_assert(!noexcept(C() == C()));

  // Ensure we do not trigger odr-use from exception specification computation.
  template<typename T> struct D {
    bool operator==(const D &) const {
      typename T::error error; // expected-error {{no type}}
    }
  };
  struct E {
    D<E> d;
    bool operator==(const E&) const = default;
  };
  static_assert(!noexcept(E() == E()));

  // (but we do when defining the function).
  struct F {
    D<F> d;
    bool operator==(const F&) const = default; // expected-note {{in instantiation}}
  };
  bool equal = F() == F();
  static_assert(!noexcept(F() == F()));
}

namespace Spaceship {
  struct X {
    friend std::strong_ordering operator<=>(X, X);
  };
  struct Y : X {
    friend std::strong_ordering operator<=>(Y, Y) = default;
  };
  DEF(Y() <=> Y());
  static_assert(!noexcept(Y() <=> Y()));

  struct ThrowingCmpCat {
    ThrowingCmpCat(std::strong_ordering);
    operator std::strong_ordering();
  };
  bool operator!=(ThrowingCmpCat o, int n) noexcept;

  struct A {
    friend ThrowingCmpCat operator<=>(A, A) noexcept;
  };

  struct B {
    A a;
    std::strong_ordering operator<=>(const B&) const = default;
  };
  DEF(B() <=> B());
  static_assert(!noexcept(B() <=> B()));

  struct C {
    int n;
    ThrowingCmpCat operator<=>(const C&) const = default;
  };
  DEF(C() <=> C());
  static_assert(!noexcept(C() <=> C()));

  struct D {
    int n;
    std::strong_ordering operator<=>(const D&) const = default;
  };
  DEF(D() <=> D());
  static_assert(noexcept(D() <=> D()));


  struct ThrowingCmpCat2 {
    ThrowingCmpCat2(std::strong_ordering) noexcept;
    operator std::strong_ordering() noexcept;
  };
  bool operator!=(ThrowingCmpCat2 o, int n);

  struct E {
    friend ThrowingCmpCat2 operator<=>(E, E) noexcept;
  };

  struct F {
    E e;
    std::strong_ordering operator<=>(const F&) const = default;
  };
  DEF(F() <=> F());
  static_assert(noexcept(F() <=> F()));

  struct G {
    int n;
    ThrowingCmpCat2 operator<=>(const G&) const = default;
  };
  DEF(G() <=> G());
  static_assert(!noexcept(G() <=> G()));
}

namespace Synth {
  struct A {
    friend bool operator==(A, A) noexcept;
    friend bool operator<(A, A) noexcept;
  };
  struct B {
    A a;
    friend std::strong_ordering operator<=>(B, B) = default;
  };
  std::strong_ordering operator<=>(B, B) noexcept;

  struct C {
    friend bool operator==(C, C);
    friend bool operator<(C, C) noexcept;
  };
  struct D {
    C c;
    friend std::strong_ordering operator<=>(D, D) = default; // expected-note {{previous}}
  };
  std::strong_ordering operator<=>(D, D) noexcept; // expected-error {{does not match}}

  struct E {
    friend bool operator==(E, E) noexcept;
    friend bool operator<(E, E);
  };
  struct F {
    E e;
    friend std::strong_ordering operator<=>(F, F) = default; // expected-note {{previous}}
  };
  std::strong_ordering operator<=>(F, F) noexcept; // expected-error {{does not match}}
}

namespace Secondary {
  struct A {
    friend bool operator==(A, A);
    friend bool operator!=(A, A) = default; // expected-note {{previous}}

    friend int operator<=>(A, A);
    friend bool operator<(A, A) = default; // expected-note {{previous}}
    friend bool operator<=(A, A) = default; // expected-note {{previous}}
    friend bool operator>(A, A) = default; // expected-note {{previous}}
    friend bool operator>=(A, A) = default; // expected-note {{previous}}
  };
  bool operator!=(A, A) noexcept; // expected-error {{does not match}}
  bool operator<(A, A) noexcept; // expected-error {{does not match}}
  bool operator<=(A, A) noexcept; // expected-error {{does not match}}
  bool operator>(A, A) noexcept; // expected-error {{does not match}}
  bool operator>=(A, A) noexcept; // expected-error {{does not match}}

  struct B {
    friend bool operator==(B, B) noexcept;
    friend bool operator!=(B, B) = default;

    friend int operator<=>(B, B) noexcept;
    friend bool operator<(B, B) = default;
    friend bool operator<=(B, B) = default;
    friend bool operator>(B, B) = default;
    friend bool operator>=(B, B) = default;
  };
  bool operator!=(B, B) noexcept;
  bool operator<(B, B) noexcept;
  bool operator<=(B, B) noexcept;
  bool operator>(B, B) noexcept;
  bool operator>=(B, B) noexcept;
}

// Check that we attempt to define a defaulted comparison before trying to
// compute its exception specification.
namespace DefineBeforeComputingExceptionSpec {
  template<int> struct A {
    A();
    A(const A&) = delete; // expected-note 3{{here}}
    friend bool operator==(A, A); // expected-note 3{{passing}}
    friend bool operator!=(const A&, const A&) = default; // expected-error 3{{call to deleted constructor}}
  };

  bool a0 = A<0>() != A<0>(); // expected-note {{in defaulted equality comparison operator}}
  bool a1 = operator!=(A<1>(), A<1>()); // expected-note {{in defaulted equality comparison operator}}

  template struct A<2>;
  bool operator!=(const A<2>&, const A<2>&) noexcept; // expected-note {{in evaluation of exception specification}}

  template<int> struct B {
    B();
    B(const B&) = delete; // expected-note 3{{here}}
    friend bool operator==(B, B); // expected-note 3{{passing}}
    bool operator!=(const B&) const = default; // expected-error 3{{call to deleted constructor}}
  };

  bool b0 = B<0>() != B<0>(); // expected-note {{in defaulted equality comparison operator}}
  bool b1 = B<1>().operator!=(B<1>()); // expected-note {{in defaulted equality comparison operator}}
  int b2 = sizeof(&B<2>::operator!=); // expected-note {{in evaluation of exception specification}}
}
