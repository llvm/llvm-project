// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11

template<typename T>
struct only {
  only(T);
  template<typename U> only(U) = delete;
};

void f() {
  only<const int*> p = new const auto (0);
  only<double*> q = new (auto) (0.0);
  only<char*> r = new auto {'a'};

  new auto; // expected-error{{new expression for type 'auto' requires a constructor argument}}
  new (const auto)(); // expected-error{{new expression for type 'const auto' requires a constructor argument}}
  new (auto) (1,2,3); // expected-error{{new expression for type 'auto' contains multiple constructor arguments}}
  new auto {}; // expected-error{{new expression for type 'auto' requires a constructor argument}}
  new auto {1,2,3}; // expected-error{{new expression for type 'auto' contains multiple constructor arguments}}
  new auto ({1,2,3}); // expected-error{{new expression for type 'auto' contains multiple constructor arguments}}
}

void p2example() {
  only<int*> r = new auto(1);
  auto x = new auto('a');

  only<char*> testX = x;
}
