// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only %s

auto c1(auto f, auto ...fs) {
  constexpr bool a = true;
  return [](auto) requires a {
    constexpr bool b = true;
    return []() requires a && b {
      constexpr bool c = true;
      return [](auto) requires a && b && c {
        constexpr bool d = true;
        // expected-note@+2{{because substituted constraint expression is ill-formed: no matching function for call to 'c1'}}
        // expected-note@+1{{candidate function not viable: constraints not satisfied}}
        return []() requires a && b && c && d && (c1(fs...)) {};
      };
    }();
  }(1);
}

void foo() {
  c1(true)(1.0)(); // expected-error{{no matching function for call to object of type}}
}
