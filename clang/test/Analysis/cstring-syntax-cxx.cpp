// RUN: %clang_cc1 -analyze -analyzer-checker=unix.cstring.BadSizeArg -analyzer-store=region -verify %s
// expected-no-diagnostics

// Ensure we don't crash on C++ declarations with special names.
struct X {
  X(int i): i(i) {}
  int i;
};

X operator+(X a, X b) {
  return X(a.i + b.i);
}

void test(X a, X b) {
  X c = a + b;
}

