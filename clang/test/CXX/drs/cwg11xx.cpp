// RUN: %clang_cc1 -std=c++98 %s -verify=expected,cxx98 -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2a %s -verify=expected -fexceptions -fcxx-exceptions -pedantic-errors

namespace cwg1110 { // cwg1110: 3.1
#if __cplusplus >= 201103L
template <typename T>
T return_T();

struct A;

template <typename>
struct B;

decltype(return_T<A>())* a;
decltype(return_T<B<int>>())* b;
#endif
} // namespace cwg1110

namespace cwg1111 { // cwg1111: 3.2
namespace example1 {
template <typename> struct set; // #cwg1111-struct-set

struct X {
  template <typename T> void set(const T &value); // #cwg1111-func-set
};
void foo() {
  X x;
  // FIXME: should we backport C++11 behavior?
  x.set<double>(3.2);
  // cxx98-error@-1 {{lookup of 'set' in member access expression is ambiguous; using member of 'X'}}
  //   cxx98-note@#cwg1111-func-set {{lookup in the object type 'X' refers here}}
  //   cxx98-note@#cwg1111-struct-set {{lookup from the current scope refers here}}
}

struct Y {};
void bar() {
  Y y;
  y.set<double>(3.2);
  // expected-error@-1 {{no member named 'set' in 'cwg1111::example1::Y'}}
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
} // namespace cwg1111

namespace cwg1113 { // cwg1113: partial
  namespace named {
    extern int a; // #cwg1113-a
    static int a;
    // expected-error@-1 {{static declaration of 'a' follows non-static}}
    //   expected-note@#cwg1113-a {{previous declaration is here}}
  }
  namespace {
    extern int a;
    static int a; // ok, both declarations have internal linkage
    int b = a;
  }

  // FIXME: Per CWG1113 and CWG4, this is ill-formed due to ambiguity: the second
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

// cwg1150: na
