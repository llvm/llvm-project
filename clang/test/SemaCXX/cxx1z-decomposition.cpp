// RUN: %clang_cc1 -std=c++17 -Wc++20-extensions -verify=expected %s
// RUN: %clang_cc1 -std=c++20 -Wpre-c++20-compat -verify=expected %s
// RUN: %clang_cc1 -std=c++20 -Wpre-c++20-compat -fexperimental-new-constant-interpreter -verify=expected %s

void use_from_own_init() {
  auto [a] = a; // expected-error {{binding 'a' cannot appear in the initializer of its own decomposition declaration}}
}

void num_elems() {
  struct A0 {} a0;
  int a1[1], a2[2];

  auto [] = a0; // expected-warning {{does not allow a decomposition group to be empty}}
  auto [v1] = a0; // expected-error {{type 'struct A0' decomposes into 0 elements, but 1 name was provided}}
  auto [] = a1; // expected-error {{type 'int[1]' decomposes into 1 element, but no names were provided}} expected-warning {{empty}}
  auto [v2] = a1;
  auto [v3, v4] = a1; // expected-error {{type 'int[1]' decomposes into 1 element, but 2 names were provided}}
  auto [] = a2; // expected-error {{type 'int[2]' decomposes into 2 elements, but no names were provided}} expected-warning {{empty}}
  auto [v5] = a2; // expected-error {{type 'int[2]' decomposes into 2 elements, but only 1 name was provided}}
  auto [v6, v7] = a2;
  auto [v8, v9, v10] = a2; // expected-error {{type 'int[2]' decomposes into 2 elements, but 3 names were provided}}
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

struct S1 {
  int a, b;
};
struct S2 {
  int a : 1; // expected-note 2{{bit-field is declared here}}
  int b;
};

auto [outer1, outer2] = S1{1, 2};
auto [outerbit1, outerbit2] = S1{1, 2}; // expected-note {{declared here}}

void enclosing() {
  struct S { int a = outer1; };
  auto [n] = S(); // expected-note 3{{'n' declared here}}

  struct Q {
    int f() { return n; } // expected-error {{reference to local binding 'n' declared in enclosing function 'enclosing'}}
  };

  (void)[&] { return n; }; // expected-warning {{C++20}}
  (void)[n] { return n; }; // expected-warning {{C++20}}

  static auto [m] = S(); // expected-note {{'m' declared here}} \
                         // expected-warning {{C++20}}

  struct R { int f() { return m; } };
  (void) [&] { return m; };
  (void)[m]{}; // expected-error {{'m' cannot be captured because it does not have automatic storage duration}}

  (void)[outerbit1]{}; // expected-error {{'outerbit1' cannot be captured because it does not have automatic storage duration}}

  auto [bit, var] = S2{-1, 1}; // expected-note 2{{'bit' declared here}}

  (void)[&bit] { // expected-error {{non-const reference cannot bind to bit-field 'a'}} \
                    // expected-warning {{C++20}}
    return bit;
  };

  union { // expected-note {{declared here}}
    int u;
  };

  (void)[&] { return bit + u; } // expected-error {{unnamed variable cannot be implicitly captured in a lambda expression}} \
                                // expected-error {{non-const reference cannot bind to bit-field 'a'}} \
                                // expected-warning {{C++20}}
  ();
}

void bitfield() {
  struct { int a : 3, : 4, b : 5; } a;
  auto &[x, y] = a;
  auto &[p, q, r] = a; // expected-error-re {{type 'struct (unnamed struct at {{.*}})' decomposes into 2 elements, but 3 names were provided}}
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
  void f() { static auto [a] = *this; } // expected-warning {{C++20}}
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

namespace by_value_array_copy {
  struct explicit_copy {
    explicit_copy() = default; // expected-note 2{{candidate constructor not viable: requires 0 arguments, but 1 was provided}}
    explicit explicit_copy(const explicit_copy&) = default; // expected-note 2{{explicit constructor is not a candidate}}
  };

  constexpr int direct_initialization_for_elements() {
    explicit_copy ec_arr[2];
    auto [a1, b1](ec_arr);

    int arr[3]{1, 2, 3};
    auto [a2, b2, c2](arr);
    arr[0]--;
    return a2 + b2 + c2 + arr[0];
  }
  static_assert(direct_initialization_for_elements() == 6);

  constexpr int copy_initialization_for_elements() {
    int arr[2]{4, 5};
    auto [a1, b1] = arr;
    auto [a2, b2]{arr}; // GH31813
    arr[0] = 0;
    return a1 + b1 + a2 + b2 + arr[0];
  }
  static_assert(copy_initialization_for_elements() == 18);

  void copy_initialization_for_elements_with_explicit_copy_ctor() {
    explicit_copy ec_arr[2];
    auto [a1, b1] = ec_arr; // expected-error {{no matching constructor for initialization of 'explicit_copy[2]'}}
    auto [a2, b2]{ec_arr}; // expected-error {{no matching constructor for initialization of 'explicit_copy[2]'}}

    // Test prvalue
    using T = explicit_copy[2];
    auto [a3, b3] = T{};
    auto [a4, b4]{T{}};
  }
} // namespace by_value_array_copy
