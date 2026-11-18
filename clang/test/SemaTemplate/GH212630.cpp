// RUN: %clang_cc1 -std=c++26 -fsyntax-only -verify %s
// expected-no-diagnostics

namespace GH212630 {

void f(int g(const int&)) {
  template for (auto x : {g(1), g(2), g(3)})
    g(0);
}

struct M {
  int m(const int &x) const { return x; }
};

int overloaded(const int &);
long overloaded(const long &);

void related(int (*fp)(const int &), int (&fr)(const int &), M m) {
  template for (auto x : {fp(1), fr(2), m.m(3), overloaded(4), overloaded(5L)}) {}
}

constexpr int h(const int &x) { return x * 2; }

struct S {
  int v;
  constexpr S(int v) : v(v) {}
  constexpr ~S() {}
};

constexpr int direct() {
  int sum = 0;
  template for (auto x : {h(1), h(2), h(3)}) { sum += x; }
  template for (constexpr auto x : {h(1), h(2), h(3)}) { sum += x; }
  template for (auto s : {S(1), S(2)}) { sum += s.v; }
  return sum;
}
static_assert(direct() == 27);

// With a pack, the elements are rebuilt when the template is instantiated.
template <typename... Ts>
constexpr int pack(Ts... ts) {
  int sum = 0;
  template for (auto x : {h(1), h(ts)...}) { sum += x; }
  template for (auto s : {S(ts)...}) { sum += s.v; }
  return sum;
}
static_assert(pack(2, 3) == 17);

} // namespace GH212630
