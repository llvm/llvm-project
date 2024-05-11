// RUN: %clang_cc1 -std=c++20 -verify %s

template<int I>
concept C = I >= 4;

template<int I>
concept D = I < 8;

template<int I>
struct A {
  constexpr static int f() { return 0; }
  constexpr static int f() requires C<I> && D<I> { return 1; }
  constexpr static int f() requires C<I> { return 2; }

  constexpr static int g() requires C<I> { return 0; } // #candidate-0
  constexpr static int g() requires D<I> { return 1; } // #candidate-1

  constexpr static int h() requires C<I> { return 0; } // expected-note {{member declaration nearly matches}}
};

template<>
constexpr int A<2>::f() { return 3; }

template<>
constexpr int A<4>::f() { return 4; }

template<>
constexpr int A<8>::f() { return 5; }

static_assert(A<3>::f() == 0);
static_assert(A<5>::f() == 1);
static_assert(A<9>::f() == 2);
static_assert(A<2>::f() == 3);
static_assert(A<4>::f() == 4);
static_assert(A<8>::f() == 5);

template<>
constexpr int A<0>::g() { return 2; }

template<>
constexpr int A<8>::g() { return 3; }

template<>
constexpr int A<6>::g() { return 4; } // expected-error {{ambiguous member function specialization 'A<6>::g' of 'A::g'}}
                                      // expected-note@#candidate-0 {{member function specialization matches 'g'}}
                                      // expected-note@#candidate-1 {{member function specialization matches 'g'}}

static_assert(A<9>::g() == 0);
static_assert(A<1>::g() == 1);
static_assert(A<0>::g() == 2);
static_assert(A<8>::g() == 3);

template<>
constexpr int A<4>::h() { return 1; }

template<>
constexpr int A<0>::h() { return 2; } // expected-error {{out-of-line definition of 'h' does not match any declaration in 'A<0>'}}

static_assert(A<5>::h() == 0);
static_assert(A<4>::h() == 1);
