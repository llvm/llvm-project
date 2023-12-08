// RUN: %clang_analyze_cc1 -std=c++17 -analyzer-checker=core,deadcode -verify %s

typedef unsigned long size_t;

// Machinery required for custom structured bindings decomposition.
namespace std {
template <class T> class tuple_size;
template <class T>
 constexpr size_t tuple_size_v = tuple_size<T>::value;
template <size_t I, class T> class tuple_element;

template<class T, T v>
struct integral_constant {
    static constexpr T value = v;
    typedef T value_type;
    typedef integral_constant type;
    constexpr operator value_type() const noexcept { return value; }
};
}

struct S {
  int a;
  double b;
  S(int a, double b) : a(a), b(b) {};
};

S GetNumbers();

int used_binding() {
    const auto [a, b] = GetNumbers(); // no-warning
    return a + b; 
}

void no_warning_on_copy(S s) {
  // Copy constructor might have side effects.
  const auto [a, b] = s; // no-warning
}


int unused_binding_ignored() {
    const auto [a, b] = GetNumbers(); // expected-warning{{Value stored to '[a, b]' during its initialization is never read}}
    return 0;
}

int unused_binding_liveness_required() {
    auto [a2, b2] = GetNumbers(); // expected-warning{{Value stored to '[a2, b2]' during its initialization is never read}}
    a2 = 10;
    b2 = 20;
    return a2 + b2;
}

int kill_one_binding() {
  auto [a, b] = GetNumbers(); // no-warning
  a = 100;
  return a + b;

}

int kill_one_binding2() {
  auto [a, b] = GetNumbers(); // expected-warning{{Value stored to '[a, b]' during its initialization is never read}}
  a = 100;
  return a;
}

void use_const_reference_bindings() {
  const auto &[a, b] = GetNumbers(); // no-warning
}

void use_reference_bindings() {
  S s(0, 0);
  auto &[a, b] = s; // no-warning
  a = 200;
}

int read_through_pointer() {
  auto [a, b] = GetNumbers(); // no-warning
  int *z = &a;
  return *z;
}

auto [globalA, globalB] = GetNumbers(); // no-warning, globals
auto [globalC, globalD] = GetNumbers(); // no-warning, globals

void use_globals() {
  globalA = 300; // no-warning
  globalB = 200;
}

struct Mytuple {
  int a;
  int b;

  template <size_t N>
  int get() const {
    if      constexpr (N == 0) return a;
    else if constexpr (N == 1) return b;
  }
};

namespace std {
    template<>
    struct tuple_size<Mytuple>
        : std::integral_constant<size_t, 2> {};

    template<size_t N>
    struct tuple_element<N, Mytuple> {
        using type = int;
    };
}

void no_warning_on_tuple_types_copy(Mytuple t) {
  auto [a, b] = t; // no-warning
}

Mytuple getMytuple();

void deconstruct_tuple_types_warning() {
  // The initializers reference the decomposed region, so the warning is not reported
  // FIXME: ideally we want to ignore that the initializers reference the decomposed region, and report the warning,
  // though the first step towards that is to handle DeadCode if the initializer is CXXConstructExpr.
  auto [a, b] = getMytuple(); // no-warning
}

int deconstruct_tuple_types_no_warning() {
  auto [a, b] = getMytuple(); // no-warning
  return a + b;
}
