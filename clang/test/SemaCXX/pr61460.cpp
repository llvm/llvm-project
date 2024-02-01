// RUN: %clang_cc1 -std=c++17 %s -fsyntax-only -verify

template <typename... Ts> void g(Ts... p1s) {
  (void)[&](auto... p2s) { ([&] { p1s; p2s; }, ...); };
}

void f1() {
  g();
}

template <typename... Ts> void g2(Ts... p1s) {
  (void)[&](auto... p2s) { [&] { p1s; p2s; }; }; // expected-error {{expression contains unexpanded parameter pack 'p2s'}}
}
