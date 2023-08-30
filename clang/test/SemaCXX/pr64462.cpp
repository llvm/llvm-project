// RUN: %clang_cc1 -verify -std=c++20 -fsyntax-only %s

auto c1(auto f, auto ...fs) {
  constexpr bool a = true;
  // expected-note@+2{{because substituted constraint expression is ill-formed: no matching function for call to 'c1'}}
  // expected-note@+1{{candidate template ignored: constraints not satisfied [with auto:1 = bool}}
  return [](auto) requires a && (c1(fs...)) {};
}

auto c2(auto f, auto ...fs) {
  constexpr bool a = true;
  // expected-note@+2{{because substituted constraint expression is ill-formed: no matching function for call to 'c2'}}
  // expected-note@+1{{candidate function not viable: constraints not satisfied}}
  return []() requires a && (c2(fs...)) {};
}

void foo() {
  c1(true)(true); // expected-error {{no matching function for call to object of type}}
  c2(true)(); // expected-error {{no matching function for call to object of type}}
}
