// RUN: %clang_cc1 -std=c++20 -verify %s
// expected-no-diagnostics

template <typename T>
auto foo(T const& arg) -> T;

template <typename Fp, typename Vis>
auto dispatch(Fp fp, Vis vis) {
  return fp(vis);
}

auto baz(int v) {
  auto callable = []<typename Arg>(Arg const& arg) -> int {
    return foo(arg);
  };
  return dispatch(callable, v);
}

template <typename T>
auto foo(T const& arg) -> T {
  return arg;
}

int main() {
  return baz(5);
}