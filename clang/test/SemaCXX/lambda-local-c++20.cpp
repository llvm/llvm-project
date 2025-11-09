// RUN: %clang_cc1 -fsyntax-only -std=c++20 -verify %s

int main() {
  auto L = []() {
    struct LocalInLambda { // expected-error {{templates cannot be declared inside of a local class}}
      void qux(auto x) {}
    };
    (void)sizeof(LocalInLambda);
  };
  L();
}
