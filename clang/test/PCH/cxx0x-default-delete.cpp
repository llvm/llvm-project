// Without PCH
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -include %s %s
// With PCH
// RUN: %clang_cc1 -x c++-header -std=c++11 -emit-pch -o %t %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -include-pch %t %s

// RUN: %clang_cc1 -x c++-header -std=c++11 -emit-pch -fpch-instantiate-templates -o %t %s
// RUN: %clang_cc1 -std=c++11 -fsyntax-only -verify -include-pch %t %s

#ifndef PASS1
#define PASS1

struct foo {
  foo() = default;
  void bar() = delete;
};

struct baz {
  ~baz() = delete;
};

class quux {
  ~quux() = default;
};

struct A {
  A(const A&) = default;
  template<typename T> A(T&&);
};

#else

foo::foo() { } // expected-error{{definition of explicitly defaulted default constructor}}
foo f;
void fn() {
  f.bar(); // expected-error{{deleted function}} expected-note@15{{deleted here}}
}

baz bz; // expected-error{{deleted function}} expected-note@19{{deleted here}}
quux qx; // expected-error{{private destructor}} expected-note@23{{private here}}

struct B { A a; };
struct C { mutable A a; };
static_assert(__is_trivially_constructible(B, const B&), "");
static_assert(!__is_trivially_constructible(B, B&&), "");
static_assert(!__is_trivially_constructible(C, const C&), "");
static_assert(!__is_trivially_constructible(C, C&&), "");

#endif
