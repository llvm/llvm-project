// RUN: %clang_cc1 -std=c++2a -verify %s

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

namespace PR41576 {
  template <class... Xs> constexpr int f(Xs ...xs) {
    return [&](auto ...ys) { // expected-note {{instantiation}}
      return ((xs + ys), ...); // expected-warning {{left operand of comma operator has no effect}}
    }(1, 2);
  }
  static_assert(f(3, 4) == 6); // expected-note {{instantiation}}
}
