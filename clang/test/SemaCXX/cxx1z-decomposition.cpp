// RUN: %clang_cc1 -std=c++1z -verify %s

void use_from_own_init() {
  auto [a] = a; // expected-error {{binding 'a' cannot appear in the initializer of its own decomposition declaration}}
}

// As a Clang extension, _Complex can be decomposed.
float decompose_complex(_Complex float cf) {
  static _Complex float scf;
  auto &[sre, sim] = scf;
  // ok, this is references initialized by constant expressions all the way down
  static_assert(&sre == &__real scf);
  static_assert(&sim == &__imag scf);

  auto [re, im] = cf;
  return re*re + im*im;
}

// As a Clang extension, vector types can be decomposed.
typedef float vf3 __attribute__((ext_vector_type(3)));
float decompose_vector(vf3 v) {
  auto [x, y, z] = v;
  auto *p = &x; // expected-error {{address of vector element requested}}
  return x + y + z;
}

struct S { int a, b; };
constexpr int f(S s) {
  auto &[a, b] = s;
  return a * 10 + b;
}
static_assert(f({1, 2}) == 12);

constexpr bool g(S &&s) { 
  auto &[a, b] = s;
  return &a == &s.a && &b == &s.b && &a != &b;
}
static_assert(g({1, 2}));

void enclosing() {
  struct S { int a; };
  auto [n] = S(); // expected-note 2{{'n' declared here}}

  struct Q { int f() { return n; } }; // expected-error {{reference to local binding 'n' declared in enclosing function}}
  // FIXME: This is probably supposed to be valid, but we do not have clear rules on how it's supposed to work.
  (void) [&] { return n; }; // expected-error {{reference to local binding 'n' declared in enclosing function}}
  (void) [n] {}; // expected-error {{'n' in capture list does not name a variable}}
}

void bitfield() {
  struct { int a : 3, : 4, b : 5; } a;
  auto &[x, y] = a;
  auto &[p, q, r] = a; // expected-error {{decomposes into 2 elements, but 3 names were provided}}
}

void for_range() {
  int x = 1;
  for (auto[a, b] : x) { // expected-error {{invalid range expression of type 'int'; no viable 'begin' function available}}
    a++;
  }

  int y[5];
  for (auto[c] : y) { // expected-error {{cannot decompose non-class, non-array type 'int'}}
    c++;
  }
}

int error_recovery() {
  auto [foobar]; // expected-error {{requires an initializer}}
  return foobar_; // expected-error {{undeclared identifier 'foobar_'}}
}

// FIXME: by-value array copies
