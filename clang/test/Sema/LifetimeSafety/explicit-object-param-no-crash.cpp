// RUN: %clang_cc1 %s -std=c++23 -verify -fsyntax-only -Wlifetime-safety

// expected-no-diagnostics

// Explicit object member functions must not be treated as having an implicit
// object argument.
struct Foo {
  template <typename T>
  int get(this Foo &&self, T) {
    return self.field;
  }

  int field;
};

void call() {
  Foo().get(0);
}
