// RUN: %clang_cc1 -fsyntax-only %s -verify
// expected-no-diagnostics
// PR5679

enum X { A = 3 };

struct ReadWrite {
  constexpr operator int() const { return 1; }
};

struct Locality {
  constexpr operator int() const { return 3; }
};

void Test() {
  char ch;
  __builtin_prefetch(&ch, 0, A);
  __builtin_prefetch(&ch, ReadWrite());
  __builtin_prefetch(&ch, 0, Locality());
}
