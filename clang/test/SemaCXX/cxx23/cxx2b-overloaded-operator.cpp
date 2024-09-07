// RUN: %clang_cc1 -verify -std=c++23 %s

namespace N {

void empty() {
  struct S {
    int operator[](); // expected-note{{not viable: requires 0 arguments, but 1 was provided}}
  };

  S{}[];
  S{}[1]; // expected-error {{no viable overloaded operator[] for type 'S'}}
}

void default_var() {
  struct S {
    constexpr int operator[](int i = 42) { return i; } // expected-note {{not viable: allows at most single argument 'i'}}
  };
  static_assert(S{}[] == 42);
  static_assert(S{}[1] == 1);
  static_assert(S{}[1, 2] == 1); // expected-error {{no viable overloaded operator[] for type 'S'}}
}

struct Variadic {
  constexpr int operator[](auto... i) { return (42 + ... + i); }
};

void variadic() {

  static_assert(Variadic{}[] == 42);
  static_assert(Variadic{}[1] == 43);
  static_assert(Variadic{}[1, 2] == 45);
}

void multiple() {
  struct S {
    constexpr int operator[]() { return 0; }
    constexpr int operator[](int) { return 1; };
    constexpr int operator[](int, int) { return 2; };
  };
  static_assert(S{}[] == 0);
  static_assert(S{}[1] == 1);
  static_assert(S{}[1, 1] == 2);
}

void ambiguous() {
  struct S {
    constexpr int operator[]() { return 0; }         // expected-note{{candidate function}}
    constexpr int operator[](int = 0) { return 1; }; // expected-note{{candidate function}}
  };

  static_assert(S{}[] == 0); // expected-error{{call to subscript operator of type 'S' is ambiguous}}
}
} // namespace N

template <typename... T>
struct T1 {
  constexpr auto operator[](T... arg) { // expected-note {{candidate function not viable: requires 2 arguments, but 1 was provided}}
    return (1 + ... + arg);
  }
};

static_assert(T1<>{}[] == 1);
static_assert(T1<int>{}[1] == 2);
static_assert(T1<int, int>{}[1, 1] == 3);
static_assert(T1<int, int>{}[1] == 3); // expected-error {{no viable overloaded operator[] for type 'T1<int, int>'}}

struct T2 {
  constexpr auto operator[](auto... arg) {
    return (1 + ... + arg);
  }
};

static_assert(T2{}[] == 1);
static_assert(T2{}[1] == 2);
static_assert(T2{}[1, 1] == 3);

namespace test_packs {

struct foo_t {
template<typename... Ts>
constexpr int operator[](Ts... idx) {
    return (0 + ... + idx);
}
};

template<int... Is>
constexpr int cxx_subscript() {
  foo_t foo;
  return foo[Is...];
}

template<int... Is>
int cxx_subscript_unexpanded() {
  foo_t foo;
  return foo[Is]; // expected-error {{expression contains unexpanded parameter pack 'Is'}}
}

template<int... Is>
constexpr int c_array() {
  int arr[] = {1, 2, 3};
  return arr[Is...]; // expected-error 2{{type 'int[3]' does not provide a subscript operator}}
}

template<int... Is>
int c_array_unexpanded() {
  int arr[] = {1, 2, 3};
  return arr[Is]; // expected-error {{expression contains unexpanded parameter pack 'Is'}}
}

void test() {
  static_assert(cxx_subscript<1, 2, 3>() == 6);
  static_assert(c_array<1>() == 2);

  c_array<>(); // expected-note {{in instantiation}}
  c_array<1>();
  c_array<1, 2>(); // expected-note {{in instantiation}}
}

}
