// RUN: %clang_cc1 -std=c++20 -Wno-unused-value -fsyntax-only -verify %s

namespace GH49266 {
struct X {
  X() = default;
  X(X const&) = delete; // expected-note {{'X' has been explicitly marked deleted here}}
};

void take_by_copy(auto &...args) {
  [...args = args] {}(); // expected-error {{call to deleted constructor}}
}

void take_by_ref(auto &...args) {
  [&...args = args] {}(); // args is passed by reference and not copied.
}

void foo() {
  X x;
  take_by_copy(x); // expected-note {{in instantiation of function template specialization}}
  take_by_ref(x);
}
}

namespace GH48937 {

template <typename... Ts>
consteval int f(Ts... ts) {
  return ([]<Ts a = 42>(){ return a;}, ...)();
}

static_assert(f(0, 42) == 42);

template <typename Ts>
int g(Ts ts) {
  return ([]<Ts a = 42>(){ return a;}, ...)();  // expected-error {{pack expansion does not contain any unexpanded parameter packs}}
}

template <typename... Ts>
int h(Ts... ts) {
  return ([]<Ts a = 42>(){ return a;})();  // expected-error {{expression contains unexpanded parameter pack 'Ts'}}
}

}
