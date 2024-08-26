// RUN: %clang_cc1 -fopenmp -fsyntax-only %s

// expected-no-diagnostics
struct S {
  int i;
};

auto [a] = S{1};

void foo() {
    a;
}
