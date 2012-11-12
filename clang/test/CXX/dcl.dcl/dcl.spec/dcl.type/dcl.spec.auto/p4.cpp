// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++11
// RUN: %clang_cc1 -fsyntax-only -verify %s -std=c++98 -Wno-c++11-extensions

template<typename T>
struct only {
  only(T);
  template<typename U> only(U) = delete;
};

void f() {
  if (auto a = true) {
  }

  switch (auto a = 0) {
  }

  while (auto a = false) {
  }

  for (; auto a = false; ) {
  }

  new const auto (0);
  new (auto) (0.0);

  int arr[] = {1, 2, 3};
  for (auto i : arr) {
  }
}

class X {
  static const auto n = 'x';

  auto m = 0; // expected-error {{'auto' not allowed in non-static class member}}
};

struct S {
  static const auto a; // expected-error {{declaration of variable 'a' with type 'auto const' requires an initializer}}
  static const auto b = 0;
  static const int c;
};
const int S::b;
const auto S::c = 0;
