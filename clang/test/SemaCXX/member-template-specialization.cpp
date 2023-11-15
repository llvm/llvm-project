// RUN: %clang_cc1 -verify -fsyntax-only %s
// expected-no-diagnostics

// Verify that the inner template specialization can be found

template <typename Ty>
struct S {
  static void bar() {
    Ty t;
    t.foo();
  }

  static void take(Ty&) {}
};

template <typename P>
struct Outer {
  template <typename C>
  struct Inner;

  using U = S<Inner<P>>;

  template <>
  struct Inner<void> {
    void foo() {
      U::take(*this);
    }
  };
};

int main() {
  Outer<void>::U::bar();
}
