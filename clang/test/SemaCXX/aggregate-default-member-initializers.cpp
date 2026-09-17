// RUN: %clang_cc1 -std=c++20 -Wno-unused-value -verify=expected %s
// RUN: %clang_cc1 -std=c++23 -Wno-unused-value -verify=expected %s
// RUN: %clang_cc1 -std=c++20 -Wno-unused-value -verify=expected,bytecode %s \
// RUN:   -fexperimental-new-constant-interpreter

namespace lifetime {

struct A {
  int &x;
  constexpr ~A() { x = 0; }
};

struct AA {
  int &x;
  constexpr ~AA() { x = -1; }
};

struct B {
  int &x;
  // FIXME: The new constant interpreter does not give a discarded composite
  // prvalue a result object, so the default member initializer cannot read
  // 'x'. This predates this change; the legacy interpreter gets it right.
  const A &a = A{x}; // bytecode-note 3{{implicit use of 'this' pointer is only allowed within the evaluation of a call to a 'constexpr' member function}}
};

struct BB {
  int &x;
  const AA &a = AA{x};
};

constexpr int one() {
  int x = 1;
  B{x};
  return x;
}

constexpr int two() {
  int x = 1;
  B{x}, BB{x};
  return x;
}

constexpr int paren() {
  int x = 1;
  (B(x));
  return x;
}

static_assert(one() == 0); // bytecode-error {{static assertion expression is not an integral constant expression}} \
                           // bytecode-note {{in call to 'one()'}}
static_assert(two() == 0); // bytecode-error {{static assertion expression is not an integral constant expression}} \
                           // bytecode-note {{in call to 'two()'}}
static_assert(paren() == 0); // bytecode-error {{static assertion expression is not an integral constant expression}} \
                             // bytecode-note {{in call to 'paren()'}}

} // namespace lifetime

namespace unevaluated {

template <typename T> int noInstantiate() {
  static_assert(false);
  return 0;
}

struct S {
  int x = noInstantiate<int>();
};

int size = sizeof(S{});

} // namespace unevaluated

namespace immediate {

struct Inner {
  int a;
  static consteval int decrement(int &x) {
    return --x; // bytecode-note {{modification of object of const-qualified type 'const int' is not allowed in a constant expression}}
  }
  // FIXME: The aggregate result object does not exist yet when the immediate
  // invocation is checked, so reading 'a' fails. This is long-standing and is
  // independent of which full-expression the initializer belongs to.
  int b = decrement(a); // expected-error {{call to consteval function 'immediate::Inner::decrement' is not a constant expression}} \
                        // expected-note {{implicit use of 'this' pointer is only allowed within the evaluation of a call to a 'constexpr' member function}} \
                        // expected-note {{declared here}} \
                        // bytecode-note {{in call to 'decrement(Inner{1}.a)'}}
};

struct Outer {
  const Inner &inner = Inner{1}; // expected-note {{in the default initializer of 'b'}}
};

constexpr int value = Outer{}.inner.a; // bytecode-error {{constexpr variable 'value' must be initialized by a constant expression}} \
                                       // bytecode-note {{declared here}}
static_assert(value == 0); // bytecode-error {{static assertion expression is not an integral constant expression}} \
                           // bytecode-note {{initializer of 'value' is not a constant expression}}

consteval unsigned currentLine(unsigned line = __builtin_LINE()) {
  return line;
}

struct SourceAndRuntime {
  unsigned line = currentLine();
  int runtime;
};

void sourceAndRuntime(int n) {
  // The runtime initializer does not make currentLine() non-constant.
  SourceAndRuntime value{.runtime = n};
}

} // namespace immediate
