// RUN: %clang_cc1 -std=c++14 -fsyntax-only -verify %s
// expected-no-diagnostics

template <class> auto fn0 = [] {};
template <typename> void foo0() { fn0<char>(); }

template<typename T> auto fn1 = [](auto a) { return a + T(1); };

template <typename X>
int foo2() {
  X a = 0x61;
  fn1<char>(a);
  return 0;
}

int main() {
  foo2<int>();
}
