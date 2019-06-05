// RUN: %clang_cc1 -std=c++2a -verify %s
// expected-no-diagnostics

template<typename ...T, typename ...Lambda> void check_sizes(Lambda ...L) {
  static_assert(((sizeof(T) == sizeof(Lambda)) && ...));
}

template<typename ...T> void f(T ...v) {
  // Pack expansion of lambdas: each lambda captures only one pack element.
  check_sizes<T...>([=] { (void)&v; } ...);

  // Pack expansion inside lambda: captures all pack elements.
  auto l = [=] { ((void)&v, ...); };
  static_assert(sizeof(l) >= (sizeof(T) + ...));
}

template void f(int, char, double);
