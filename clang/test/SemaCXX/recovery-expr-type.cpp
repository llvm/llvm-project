// RUN: %clang_cc1 -triple=x86_64-unknown-unknown -frecovery-ast -frecovery-ast-type -o - %s -fsyntax-only -verify

namespace test0 {
struct Indestructible {
  // Indestructible();
  ~Indestructible() = delete; // expected-note 2{{deleted}}
};
Indestructible make_indestructible();

void test() {
  // no crash.
  int s = sizeof(make_indestructible()); // expected-error {{deleted}}
  constexpr int ss = sizeof(make_indestructible()); // expected-error {{deleted}}
  static_assert(ss, "");
  int array[ss];
}
}

namespace test1 {
constexpr int foo() { return 1; } // expected-note {{candidate function not viable}}
// verify the "not an integral constant expression" diagnostic is suppressed.
static_assert(1 == foo(1), ""); // expected-error {{no matching function}}
}

namespace test2 {
void foo(); // expected-note 3{{requires 0 arguments}}
void func() {
  // verify that "field has incomplete type" diagnostic is suppressed.
  typeof(foo(42)) var; // expected-error {{no matching function}}

  // FIXME: suppress the "cannot initialize a variable" diagnostic.
  int a = foo(1); // expected-error {{no matching function}} \
                  // expected-error {{cannot initialize a variable of type}}

  // FIXME: suppress the "invalid application" diagnostic.
  int s = sizeof(foo(42)); // expected-error {{no matching function}} \
                           // expected-error {{invalid application of 'sizeof'}}
};
}

namespace test3 {
template <int N>
constexpr int templated() __attribute__((enable_if(N, ""))) { // expected-note {{candidate disabled}}
  return 1;
}
// verify that "constexpr variable must be initialized" diagnostic is suppressed.
constexpr int A = templated<0>(); // expected-error{{no matching function}}

template <typename T>
struct AA {
  template <typename U>
  static constexpr int getB() { // expected-note{{candidate template ignored}}
    return 2;
  }
  static constexpr int foo2() {
    return AA<T>::getB(); // expected-error{{no matching function for call to 'getB'}} \
                          // expected-note {{subexpression not valid in a constant expression}}
  }
};
// FIXME: should we suppress the "be initialized by a constant expression" diagnostic?
constexpr auto x2 = AA<int>::foo2(); // expected-error {{be initialized by a constant expression}} \
                                     // expected-note {{in instantiation of member function}} \
                                     // expected-note {{in call to}}
}

// verify no assertion failure on violating value category.
namespace test4 {
int &&f(int);  // expected-note {{candidate function not viable}}
int &&k = f(); // expected-error {{no matching function for call}}
}
