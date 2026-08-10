// RUN: %clang_cc1 -std=c++20 -fsyntax-only -verify -Wno-unused-value %s
// expected-no-diagnostics

void discarded_value() {
  constexpr bool b = true;
  [] { b; };
  [] { static_cast<void>(b); };
  [] { +b; };
  [] { static_cast<bool>(b); };

  int i;
  [] { i; };

  enum E {};
  const auto e = static_cast<E>(42);
  [] { e; };
}

enum NonConstantE {};
NonConstantE make_enum();

void discarded_nonconstant_enum() {
  const auto e = make_enum();
  [] { e; };
}

struct PotentialResults {
  int x;
  mutable int y;
};

void side_effect();

void discarded_potential_results() {
  PotentialResults object{};
  int array[1]{};
  int i = 0;

  [] { (object); };
  [] { object.x; };
  [] { object.*&PotentialResults::x; };
  [] { true ? object.x : object.y; };
  [] { static_cast<void>(object.x); };
  [] { array[0]; };
  [] { (side_effect(), i); };
}

template <typename T>
void dependent_parameter(T t) {
  [] { t; };
}

void generic_lambda_parameter() {
  [](auto t) {
    [] { t; };
  }(0);
}

void capture_default(int i) {
  auto by_copy = [=] { i; };
  static_assert(sizeof(by_copy) >= sizeof(i));

  constexpr int constant = 42;
  auto without_capture = [=] { constant; };
  static_assert(sizeof(without_capture) == 1);
}

template <typename T>
void dependent_capture_default(T t) {
  auto l = [=](auto) { t; };
  static_assert(sizeof(l) >= sizeof(T));
  l(0);
}

struct Noncopyable {
  constexpr Noncopyable() = default;
  Noncopyable(const Noncopyable &) = delete;
};

template <typename T>
void dependent_discarded_constant() {
  constexpr Noncopyable n;
  [=](auto) { n; }(T());
}

template void dependent_parameter<int>(int);
template void dependent_capture_default<int>(int);
template void dependent_discarded_constant<int>();
