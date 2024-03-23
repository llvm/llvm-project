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

namespace PR85667 {

template <class T>
struct identity {
  using type = T;
};

template <class = void> void f() {

  static_assert([]<class... Is>(Is... x) {
    return ([I(x)] {
      return I;
    }() + ...);
  }(1, 2) == 3);

  static_assert([]<class... Is>(Is... x) {
    return ([](auto y = Is()) { return y + 1; } + ...);
  }(0, 0, 0) == 3);

  []<class... Is>() {
    return ([]() noexcept(Is()) { return 0; }() + ...);
  }.template operator()<int, int>();

  static_assert(__is_same(decltype([]<class... Is>() {
                            return ([]() -> decltype(Is()) { return {}; }(),
                                    ...);
                          }.template operator()<int, char>()),
                          char));

  []<class... Is>() {
    return ([]<class... Ts>() -> decltype(Is()) { return Ts(); }() + ...);
    // expected-error@-1 {{unexpanded parameter pack 'Ts'}}
  }.template operator()<int, int>();

  // Note that GCC and EDG reject this case currently.
  // GCC says the fold expression "has no unexpanded parameter packs", while
  // EDG says the constraint is not allowed on a non-template function.
  // MSVC is happy with it.
  []<class... Is>() {
    ([]()
       requires(Is())
     {},
     ...);
  }.template operator()<bool, bool>();

  // https://github.com/llvm/llvm-project/issues/56852
  []<class... Is>(Is...) {
    ([] {
      using T = identity<Is>::type;
    }(), ...);
  }(1, 2);

  // https://github.com/llvm/llvm-project/issues/18873
  [](auto ...y) {
    ([y] { }, ...);
  }();

  [](auto ...x) {
    ([&](auto ...y) {
      // FIXME: This now hits assertion `PackIdx != -1 && "found declaration pack but not pack expanding"'
      // in Sema::FindInstantiatedDecl.
      // This is because the captured variable x has been expanded while transforming
      // the outermost lambda call, but the expansion is held off while transforming
      // the folded expression. Then, we would hit the assertion when instantiating the
      // captured variable in TransformLambdaExpr.
      // I think this is supposed to be ill-formed, but GCC and MSVC currently accept this.
      // However, if x gets expanded with non-empty arguments, then GCC and MSVC will reject it -
      // we probably need a diagnostic for it.
      // ([x, y] { }, ...);
      ([x..., y] { }, ...);
    })();
  }();
}

template void f();

}
