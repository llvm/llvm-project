// RUN: %clang_cc1 -verify -fsyntax-only %s
// Verify the absence of assertion failures when solving calls to unresolved
// template member functions.

struct A {
  template <typename T>
  static void bar(int) { } // expected-note {{candidate template ignored: couldn't infer template argument 'T'}}
};

struct B {
  template <int i>
  static void foo() {
    int array[i];
    A::template bar(array[0]); // expected-error {{no matching function for call to 'bar'}}
  }
};

int main() {
  B::foo<4>(); // expected-note {{in instantiation of function template specialization 'B::foo<4>'}}
  return 0;
}

namespace GH70375 {

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

void instantiate() {
  Outer<void>::U::bar();
}

}
