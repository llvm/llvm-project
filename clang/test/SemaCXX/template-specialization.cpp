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
    A::template bar(array[0]); // expected-error {{a template argument list is expected after a name prefixed by the template keyword}} expected-error {{no matching function for call to 'bar'}}
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

namespace GH89374 {

struct A {};

template <typename Derived>
struct MatrixBase { // #GH89374-MatrixBase
  template <typename OtherDerived>
  Derived &operator=(const MatrixBase<OtherDerived> &); // #GH89374-copy-assignment
};

template <typename>
struct solve_retval;

template <typename Rhs>
struct solve_retval<int> : MatrixBase<solve_retval<Rhs> > {};
// expected-error@-1 {{partial specialization of 'solve_retval' does not use any of its template parameters}}

void ApproximateChebyshev() {
  MatrixBase<int> c;
  c = solve_retval<int>();
  // expected-error@-1 {{no viable overloaded '='}}
  //   expected-note@#GH89374-copy-assignment {{candidate template ignored: could not match 'MatrixBase' against 'solve_retval'}}
  //   expected-note@#GH89374-MatrixBase {{candidate function (the implicit copy assignment operator) not viable: no known conversion from 'solve_retval<int>' to 'const MatrixBase<int>' for 1st argument}}
  //   expected-note@#GH89374-MatrixBase {{candidate function (the implicit move assignment operator) not viable: no known conversion from 'solve_retval<int>' to 'MatrixBase<int>' for 1st argument}}
}

} // namespace GH89374
