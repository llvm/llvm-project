// RUN: %clang_cc1 -triple=x86_64-unknown-unknown -frecovery-ast -frecovery-ast-type -o - %s -std=gnu++17 -fsyntax-only -verify

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
  typeof(foo(42)) var; // expected-error {{no matching function}} \

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
    return AA<T>::getB(); // expected-error{{no matching function for call to 'getB'}}
  }
};
// FIXME: should we suppress the "be initialized by a constant expression" diagnostic?
constexpr auto x2 = AA<int>::foo2(); // expected-error {{be initialized by a constant expression}} \
                                     // expected-note {{in instantiation of member function}}
}

// verify no assertion failure on violating value category.
namespace test4 {
int &&f(int);  // expected-note {{candidate function not viable}}
int &&k = f(); // expected-error {{no matching function for call}}
}

// verify that "type 'double' cannot bind to a value of unrelated type 'int'" diagnostic is suppressed.
namespace test5 {
  template<typename T> using U = T; // expected-note {{template parameter is declared here}}
  template<typename...Ts> U<Ts...>& f(); // expected-error {{pack expansion used as argument for non-pack parameter of alias template}}
  double &s1 = f(); // expected-error {{no matching function}}
}

namespace test6 {
struct T {
  T() = delete; // expected-note {{has been explicitly marked deleted here}}
};

void func() {
  // verify that no -Wunused-value diagnostic.
  (T(T())); // expected-error {{call to deleted constructor}}
}
}

// verify the secondary diagnostic "no matching function" is emitted.
namespace test7 {
struct C {
  C() = delete; // expected-note {{has been explicitly marked deleted}}
};
void f(C &); // expected-note {{candidate function not viable: expects an lvalue for 1st argument}}
void test() {
  f(C()); // expected-error {{call to deleted constructor}} \
             expected-error {{no matching function for call}}
}
}

// verify the secondary diagnostic "cannot initialize" is emitted.
namespace test8 {
typedef int arr[];
int v = arr(); // expected-error {{array types cannot be value-initialized}} \
                  expected-error {{cannot initialize a variable of type 'int' with an rvalue of type 'test8::arr'}}
}

namespace test9 {
auto f(); // expected-note {{candidate function not viable}}
// verify no crash on evaluating the size of undeduced auto type.
static_assert(sizeof(f(1)), ""); // expected-error {{no matching function for call to 'f'}}
}

namespace test10 {
// Ensure we don't assert here.
int f(); // expected-note {{candidate}}
template<typename T> const int k = f(T()); // expected-error {{no matching function}}
static_assert(k<int> == 1, ""); // expected-note {{instantiation of}}
}

namespace test11 {
// Verify we do not assert()-fail here.
template <class T> void foo(T &t);
template <typename T>
void bar(T t) {
  foo(t);
}

template <typename T = void *>
struct S { // expected-note {{candidate}}
  S(T t);  // expected-note {{candidate}}
  ~S();
};
template <typename T> S(T t) -> S<void *>;

void baz() {
  bar(S(123)); // expected-error {{no matching conversion}}
}
} // namespace test11

namespace test12 {
// Verify we do not crash.
int fun(int *foo = no_such_function()); // expected-error {{undeclared identifier}}
void crash1() { fun(); }
void crash2() { constexpr int s = fun(); }
} // namespace test12
