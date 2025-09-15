// RUN: %clang_cc1 -fsyntax-only %s -verify
// expected-no-diagnostics

template <int i>
int g() {
  return [] (auto) -> int {
    struct L {
      int m = i;
    };
    return 0;
  } (42);
}

int v = g<1>();
