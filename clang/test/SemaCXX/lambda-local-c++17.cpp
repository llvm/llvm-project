// RUN: %clang_cc1 -fsyntax-only -verify %s

int main() {
  auto L = []() {
    struct LocalInLambda {
      void qux(auto x) {} // expected-error {{'auto' not allowed in function prototype}}
    };
    (void)sizeof(LocalInLambda);
  };
  L();
}