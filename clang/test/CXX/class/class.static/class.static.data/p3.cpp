// RUN: %clang_cc1 -fsyntax-only -verify -std=c++11 %s

struct NonLit { // expected-note 3{{no constexpr constructors}}
  NonLit();
};

struct S {
  static constexpr int a = 0;
  static constexpr int b; // expected-error {{declaration of constexpr static data member 'b' requires an initializer}}

  static constexpr int c = 0;
  static const int d;
  static const int d2 = 0;

  static constexpr double e = 0.0; // ok
  static const double f = 0.0; // expected-warning {{extension}} expected-note {{use 'constexpr' specifier}}
  static char *const g = 0; // expected-error {{requires 'constexpr' specifier}}
  static const NonLit h = NonLit(); // expected-error {{must be initialized out of line}}
};

constexpr int S::a;
constexpr int S::b = 0;

const int S::c;
constexpr int S::d = 0;
constexpr int S::d2;

template<typename T>
struct U {
  static constexpr int a = 0;
  static constexpr int b; // expected-error {{declaration of constexpr static data member 'b' requires an initializer}}
  static constexpr NonLit h = NonLit(); // expected-error {{cannot have non-literal type 'const NonLit'}}
  static constexpr T c = T(); // expected-error {{cannot have non-literal type}}
  static const T d;
};

template<typename T> constexpr T U<T>::d = T(); // expected-error {{non-literal type 'const NonLit'}}

U<int> u1;
U<NonLit> u2; // expected-note {{here}}

static_assert(U<int>::a == 0, "");

constexpr int outofline = (U<NonLit>::d, 0); // expected-note {{here}} expected-warning {{unused}}
