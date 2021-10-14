// RUN: %clang_cc1 -std=c++17 -verify %s

void use_from_own_init() {
  auto [a] = a; // expected-error {{binding 'a' cannot appear in the initializer of its own decomposition declaration}}
}

void num_elems() {
  struct A0 {} a0;
  int a1[1], a2[2];

  auto [] = a0; // expected-warning {{does not allow a decomposition group to be empty}}
  auto [v1] = a0; // expected-error {{type 'A0' decomposes into 0 elements, but 1 name was provided}}
  auto [] = a1; // expected-error {{type 'int [1]' decomposes into 1 element, but no names were provided}} expected-warning {{empty}}
  auto [v2] = a1;
  auto [v3, v4] = a1; // expected-error {{type 'int [1]' decomposes into 1 element, but 2 names were provided}}
  auto [] = a2; // expected-error {{type 'int [2]' decomposes into 2 elements, but no names were provided}} expected-warning {{empty}}
  auto [v5] = a2; // expected-error {{type 'int [2]' decomposes into 2 elements, but only 1 name was provided}}
  auto [v6, v7] = a2;
  auto [v8, v9, v10] = a2; // expected-error {{type 'int [2]' decomposes into 2 elements, but 3 names were provided}}
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

auto [outer1, outer2] = S{1, 2};
void enclosing() {
  struct S { int a = outer1; };
  auto [n] = S(); // expected-note 2{{'n' declared here}}

  struct Q { int f() { return n; } }; // expected-error {{reference to local binding 'n' declared in enclosing function}}
  (void) [&] { return n; }; // expected-error {{reference to local binding 'n' declared in enclosing function}}
  (void) [n] {}; // expected-error {{'n' in capture list does not name a variable}}

  static auto [m] = S(); // expected-warning {{extension}}
  struct R { int f() { return m; } };
  (void) [&] { return m; };
  (void) [m] {}; // expected-error {{'m' in capture list does not name a variable}}
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

// PR32172
template <class T> void dependent_foreach(T t) {
  for (auto [a,b,c] : t)
    a,b,c;
}

struct PR37352 {
  int n;
  void f() { static auto [a] = *this; } // expected-warning {{C++20 extension}}
};

namespace instantiate_template {

template <typename T1, typename T2>
struct pair {
  T1 a;
  T2 b;
};

const pair<int, int> &f1();

int f2() {
  const auto &[a, b] = f1();
  return a + b;
}

} // namespace instantiate_template

namespace lambdas {
  void f() {
    int n;
    auto [a] =  // expected-error {{cannot decompose lambda closure type}}
        [n] {}; // expected-note {{lambda expression}}
  }

  auto [] = []{}; // expected-warning {{ISO C++17 does not allow a decomposition group to be empty}}

  int g() {
    int n = 0;
    auto a = [=](auto &self) { // expected-note {{lambda expression}}
      auto &[capture] = self; // expected-error {{cannot decompose lambda closure type}}
      ++capture;
      return n;
    };
    return a(a); // expected-note {{in instantiation of}}
  }

  int h() {
    auto x = [] {};
    struct A : decltype(x) {
      int n;
    };
    auto &&[r] = A{x, 0}; // OK (presumably), non-capturing lambda has no non-static data members
    return r;
  }

  int i() {
    int n;
    auto x = [n] {};
    struct A : decltype(x) {
      int n;
    };
    auto &&[r] = A{x, 0}; // expected-error-re {{cannot decompose class type 'A': both it and its base class 'decltype(x)' (aka '(lambda {{.*}})') have non-static data members}}
    return r;
  }

  void j() {
    auto x = [] {};
    struct A : decltype(x) {};
    auto &&[] = A{x}; // expected-warning {{ISO C++17 does not allow a decomposition group to be empty}}
  }
}

// FIXME: by-value array copies
