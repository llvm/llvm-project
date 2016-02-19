// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -fcxx-exceptions %s

// An explicitly-defaulted function may be declared constexpr only if it would
// have been implicitly declared as constexpr.
struct S1 {
  constexpr S1() = default; // expected-error {{defaulted definition of default constructor is not constexpr}}
  constexpr S1(const S1&) = default;
  constexpr S1(S1&&) = default;
  constexpr S1 &operator=(const S1&) const = default; // expected-error {{explicitly-defaulted copy assignment operator may not have}}
  constexpr S1 &operator=(S1&&) const = default; // expected-error {{explicitly-defaulted move assignment operator may not have}}
  constexpr ~S1() = default; // expected-error {{destructor cannot be marked constexpr}}
  int n;
};
struct NoCopyMove {
  constexpr NoCopyMove() {}
  NoCopyMove(const NoCopyMove&);
  NoCopyMove(NoCopyMove&&);
};
struct S2 {
  constexpr S2() = default;
  constexpr S2(const S2&) = default; // expected-error {{defaulted definition of copy constructor is not constexpr}}
  constexpr S2(S2&&) = default; // expected-error {{defaulted definition of move constructor is not constexpr}}
  NoCopyMove ncm;
};

// If a function is explicitly defaulted on its first declaration
//   -- it is implicitly considered to be constexpr if the implicit declaration
//      would be
struct S3 {
  S3() = default;
  S3(const S3&) = default;
  S3(S3&&) = default;
  constexpr S3(int n) : n(n) {}
  int n;
};
constexpr S3 s3a = S3(0);
constexpr S3 s3b = s3a;
constexpr S3 s3c = S3();
constexpr S3 s3d; // expected-error {{default initialization of an object of const type 'const S3' without a user-provided default constructor}}

struct S4 {
  S4() = default;
  S4(const S4&) = default; // expected-note {{here}}
  S4(S4&&) = default; // expected-note {{here}}
  NoCopyMove ncm;
};
constexpr S4 s4a{}; // ok
constexpr S4 s4b = S4(); // expected-error {{constant expression}} expected-note {{non-constexpr constructor}}
constexpr S4 s4c = s4a; // expected-error {{constant expression}} expected-note {{non-constexpr constructor}}

struct S5 {
  constexpr S5();
  int n = 1, m = n + 3;
};
constexpr S5::S5() = default;
static_assert(S5().m == 4, "");


// An explicitly-defaulted function may have an exception specification only if
// it is compatible with the exception specification on an implicit declaration.
struct E1 {
  E1() noexcept = default;
  E1(const E1&) noexcept = default;
  E1(E1&&) noexcept = default;
  E1 &operator=(const E1&) noexcept = default;
  E1 &operator=(E1&&) noexcept = default;
  ~E1() noexcept = default;
};
struct E2 {
  E2() noexcept(false) = default; // expected-error {{exception specification of explicitly defaulted default constructor does not match the calculated one}}
  E2(const E2&) noexcept(false) = default; // expected-error {{exception specification of explicitly defaulted copy constructor does not match the calculated one}}
  E2(E2&&) noexcept(false) = default; // expected-error {{exception specification of explicitly defaulted move constructor does not match the calculated one}}
  E2 &operator=(const E2&) noexcept(false) = default; // expected-error {{exception specification of explicitly defaulted copy assignment operator does not match the calculated one}}
  E2 &operator=(E2&&) noexcept(false) = default; // expected-error {{exception specification of explicitly defaulted move assignment operator does not match the calculated one}}
  ~E2() noexcept(false) = default; // expected-error {{exception specification of explicitly defaulted destructor does not match the calculated one}}
};

// If a function is explicitly defaulted on its first declaration
//   -- it is implicitly considered to have the same exception-specification as
//      if it had been implicitly declared
struct E3 {
  E3() = default;
  E3(const E3&) = default;
  E3(E3&&) = default;
  E3 &operator=(const E3&) = default;
  E3 &operator=(E3&&) = default;
  ~E3() = default;
};
E3 e3;
static_assert(noexcept(E3(), E3(E3()), E3(e3), e3 = E3(), e3 = e3), "");
struct E4 {
  E4() noexcept(false);
  E4(const E4&) noexcept(false);
  E4(E4&&) noexcept(false);
  E4 &operator=(const E4&) noexcept(false);
  E4 &operator=(E4&&) noexcept(false);
  ~E4() noexcept(false);
};
struct E5 {
  E5() = default;
  E5(const E5&) = default;
  E5(E5&&) = default;
  E5 &operator=(const E5&) = default;
  E5 &operator=(E5&&) = default;
  ~E5() = default;

  E4 e4;
};
E5 e5;
static_assert(!noexcept(E5()), "");
static_assert(!noexcept(E5(static_cast<E5&&>(e5))), "");
static_assert(!noexcept(E5(e5)), "");
static_assert(!noexcept(e5 = E5()), "");
static_assert(!noexcept(e5 = e5), "");

namespace PR13492 {
  struct B {
    B() = default;
    int field;
  };

  void f() {
    const B b; // expected-error {{default initialization of an object of const type 'const PR13492::B' without a user-provided default constructor}}
  }
}
