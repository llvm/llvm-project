// RUN: %clang_cc1 -fsyntax-only -verify %s
// expected-no-diagnostics
// PR5488

struct X {
  int x;
};

struct Iter {
  X* operator->();
};

template <typename T>
void Foo() {
  (void)Iter()->x;
}

void Func() {
  Foo<int>();
}

