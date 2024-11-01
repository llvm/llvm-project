// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2a %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

namespace dr1111 { // dr1111: yes
namespace example1 {
template <typename> struct set;

struct X {
  template <typename T> void set(const T &value);
};
void foo() {
  X x;
#pragma clang diagnostic push
#if __cplusplus < 201103L
#pragma clang diagnostic ignored "-Wambiguous-member-template"
#endif
  x.set<double>(3.2);
#pragma clang diagnostic pop
}

struct Y {};
void bar() {
  Y y;
  y.set<double>(3.2); // expected-error {{no member named 'set' in 'dr1111::example1::Y'}}
}
} // namespace example1

namespace example2 {
struct A {};
namespace N {
struct A {
  void g() {}
  template <class T> operator T();
};
} // namespace N

void baz() {
  N::A a;
  a.operator A();
}
} // namespace example2
} // namespace dr1111

namespace dr1113 { // dr1113: partial
  namespace named {
    extern int a; // expected-note {{previous}}
    static int a; // expected-error {{static declaration of 'a' follows non-static}}
  }
  namespace {
    extern int a;
    static int a; // ok, both declarations have internal linkage
    int b = a;
  }

  // FIXME: Per DR1113 and DR4, this is ill-formed due to ambiguity: the second
  // 'f' has internal linkage, and so does not have C language linkage, so is
  // not a redeclaration of the first 'f'.
  //
  // To avoid a breaking change here, Clang ignores the "internal linkage" effect
  // of anonymous namespaces on declarations declared within an 'extern "C"'
  // linkage-specification.
  extern "C" void f();
  namespace {
    extern "C" void f();
  }
  void g() { f(); }
}
