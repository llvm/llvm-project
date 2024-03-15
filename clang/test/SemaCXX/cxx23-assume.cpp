// RUN: %clang_cc1 -std=c++23  -x c++ %s -verify
// RUN: %clang_cc1 -std=c++20 -pedantic -x c++ %s -verify=ext,expected

struct A{};
struct B{ explicit operator bool() { return true; } };

template <bool cond>
void f() {
  [[assume(cond)]]; // ext-warning {{C++23 extension}}
}

template <bool cond>
struct S {
  void f() {
    [[assume(cond)]]; // ext-warning {{C++23 extension}}
  }

  template <typename T>
  constexpr bool g() {
    [[assume(cond == sizeof(T))]]; // expected-note {{assumption evaluated to false}} ext-warning {{C++23 extension}}
    return true;
  }
};

bool f2();

template <typename T>
constexpr void f3() {
  [[assume(T{})]]; // expected-error {{not contextually convertible to 'bool'}} expected-warning {{has side effects that will be discarded}} ext-warning {{C++23 extension}}
}

void g(int x) {
  f<true>();
  f<false>();
  S<true>{}.f();
  S<false>{}.f();
  S<true>{}.g<char>();
  S<true>{}.g<int>();
  [[assume(f2())]]; // expected-warning {{side effects that will be discarded}} ext-warning {{C++23 extension}}

  [[assume((x = 3))]]; // expected-warning {{has side effects that will be discarded}} // ext-warning {{C++23 extension}}
  [[assume(x++)]]; // expected-warning {{has side effects that will be discarded}} // ext-warning {{C++23 extension}}
  [[assume(++x)]]; // expected-warning {{has side effects that will be discarded}} // ext-warning {{C++23 extension}}
  [[assume([]{ return true; }())]]; // expected-warning {{has side effects that will be discarded}} // ext-warning {{C++23 extension}}
  [[assume(B{})]]; // expected-warning {{has side effects that will be discarded}} // ext-warning {{C++23 extension}}
  [[assume((1, 2))]]; // expected-warning {{has no effect}} // ext-warning {{C++23 extension}}

  f3<A>(); // expected-note {{in instantiation of}}
  f3<B>(); // expected-note {{in instantiation of}}
  [[assume]]; // expected-error {{takes one argument}}
  [[assume(z)]]; // expected-error {{undeclared identifier}}
  [[assume(A{})]]; // expected-error {{not contextually convertible to 'bool'}}
  [[assume(true)]] if (true) {} // expected-error {{only applies to empty statements}}
  [[assume(true)]] {} // expected-error {{only applies to empty statements}}
  [[assume(true)]] for (;false;) {} // expected-error {{only applies to empty statements}}
  [[assume(true)]] while (false) {} // expected-error {{only applies to empty statements}}
  [[assume(true)]] label:; // expected-error {{cannot be applied to a declaration}}
  [[assume(true)]] goto label; // expected-error {{only applies to empty statements}}
}

// Check that 'x' is ODR-used here.
constexpr int h(int x) { return sizeof([=] { [[assume(x)]]; }); } // ext-warning {{C++23 extension}}
static_assert(h(4) == sizeof(int));

static_assert(__has_cpp_attribute(assume) == 202207L);
static_assert(__has_attribute(assume));

constexpr bool i() { // ext-error {{never produces a constant expression}}
  [[assume(false)]]; // ext-note {{assumption evaluated to false}} expected-note {{assumption evaluated to false}} ext-warning {{C++23 extension}}
  return true;
}

constexpr bool j(bool b) {
  [[assume(b)]]; // expected-note {{assumption evaluated to false}} ext-warning {{C++23 extension}}
  return true;
}

static_assert(i()); // expected-error {{not an integral constant expression}} expected-note {{in call to}}
static_assert(j(true));
static_assert(j(false)); // expected-error {{not an integral constant expression}} expected-note {{in call to}}
static_assert(S<true>{}.g<char>());
static_assert(S<false>{}.g<A>()); // expected-error {{not an integral constant expression}} expected-note {{in call to}}


template <typename T>
constexpr bool f4() {
  [[assume(!T{})]]; // expected-error {{invalid argument type 'D'}} // expected-warning 2 {{side effects}} ext-warning {{C++23 extension}}
  return sizeof(T) == sizeof(int);
}

template <typename T>
concept C = f4<T>(); // expected-note 3 {{in instantiation of}}
                     // expected-note@-1 3 {{while substituting}}
                     // expected-error@-2 2 {{resulted in a non-constant expression}}

struct D {
  int x;
};

struct E {
  int x;
  constexpr explicit operator bool() { return false; }
};

struct F {
  int x;
  int y;
  constexpr explicit operator bool() { return false; }
};

template <typename T>
constexpr int f5() requires C<T> { return 1; } // expected-note {{while checking the satisfaction}}
                                               // expected-note@-1 {{while substituting template arguments}}
                                               // expected-note@-2 {{candidate template ignored}}

template <typename T>
constexpr int f5() requires (!C<T>) { return 2; } // expected-note 4 {{while checking the satisfaction}}
                                                  // expected-note@-1 4 {{while substituting template arguments}}
                                                  // expected-note@-2 {{candidate template ignored}}

static_assert(f5<int>() == 1);
static_assert(f5<D>() == 1); // expected-note 3 {{while checking constraint satisfaction}}
                             // expected-note@-1 3 {{in instantiation of}}
                             // expected-error@-2 {{no matching function for call}}

static_assert(f5<double>() == 2);
static_assert(f5<E>() == 1); // expected-note {{while checking constraint satisfaction}} expected-note {{in instantiation of}}
static_assert(f5<F>() == 2); // expected-note {{while checking constraint satisfaction}} expected-note {{in instantiation of}}
