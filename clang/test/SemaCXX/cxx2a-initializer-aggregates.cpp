// RUN: %clang_cc1 -std=c++20 %s -verify=cxx20,expected,pedantic,override,reorder -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify=expected,pedantic,override,reorder -Wno-c++20-designator -pedantic-errors
// RUN: %clang_cc1 -std=c++20 %s -verify=cxx20,expected,pedantic -Werror=c99-designator -Wno-reorder-init-list -Wno-initializer-overrides
// RUN: %clang_cc1 -std=c++20 %s -verify=cxx20,expected,reorder -Wno-c99-designator -Werror=reorder-init-list -Wno-initializer-overrides
// RUN: %clang_cc1 -std=c++20 %s -verify=cxx20,expected,override -Wno-c99-designator -Wno-reorder-init-list -Werror=initializer-overrides
// RUN: %clang_cc1 -std=c++20 %s -verify=cxx20,expected -Wno-c99-designator -Wno-reorder-init-list -Wno-initializer-overrides


namespace class_with_ctor {
  struct A { // cxx20-note 6{{candidate}}
    A() = default; // cxx20-note 3{{candidate}}
    int x;
    int y;
  };
  A a = {1, 2}; // cxx20-error {{no matching constructor}}

  struct B {
    int x;
    int y;
  };
  B b1 = B(); // trigger declaration of implicit ctors
  B b2 = {1, 2}; // ok

  struct C : A {
    A a;
  };
  C c1 = {{}, {}}; // ok, call default ctor twice
  C c2 = {{1, 2}, {3, 4}}; // cxx20-error 2{{no matching constructor}}
}

namespace designator {
struct A { int x, y; };
struct B { A a; };

A a1 = {
  .y = 1, // reorder-note {{previous initialization for field 'y' is here}}
  .x = 2 // reorder-error {{ISO C++ requires field designators to be specified in declaration order; field 'y' will be initialized after field 'x'}}
};
int arr[3] = {[1] = 5}; // pedantic-error {{array designators are a C99 extension}}
B b = {.a.x = 0}; // pedantic-error {{nested designators are a C99 extension}}
A a2 = {
  .x = 1, // pedantic-error {{mixture of designated and non-designated initializers in the same initializer list is a C99 extension}}
  2 // pedantic-note {{first non-designated initializer is here}}
};
A a3 = {
  1, // pedantic-note {{first non-designated initializer is here}}
  .y = 2 // pedantic-error {{mixture of designated and non-designated initializers in the same initializer list is a C99 extension}}
};
A a4 = {
  .x = 1, // override-note {{previous}}
  .x = 1 // override-error {{overrides prior initialization}}
};
A a5 = {
  .y = 1, // override-note {{previous}}
  .y = 1 // override-error {{overrides prior initialization}}
};
struct C { int :0, x, :0, y, :0; };
C c = {
  .x = 1, // override-note {{previous}}
  .x = 1, // override-error {{overrides prior initialization}} override-note {{previous}}
  .y = 1, // override-note {{previous}}
  .y = 1, // override-error {{overrides prior initialization}}
  .x = 1, // reorder-error {{declaration order}} override-error {{overrides prior initialization}} override-note {{previous}}
  .x = 1, // override-error {{overrides prior initialization}}
};
}

namespace base_class {
  struct base {
    int x;
  };
  struct derived : base {
    int y;
  };
  derived d = {.x = 1, .y = 2}; // expected-error {{'x' does not refer to any field}}
}

namespace union_ {
  union U { int a, b; };
  U u = {
    .a = 1, // override-note {{here}}
    .b = 2, // override-error {{overrides prior}}
  };
}

namespace overload_resolution {
  struct A { int x, y; };
  union B { int x, y; };

  void f(A a);
  void f(B b) = delete;
  void g() { f({.x = 1, .y = 2}); } // ok, calls non-union overload

  // As an extension of the union case, overload resolution won't pick any
  // candidate where a field initializer would be overridden.
  struct A2 { int x, other, y; };
  int f(A2);
  void g2() { int k = f({.x = 1, 2, .y = 3}); (void)k; } // pedantic-error {{mixture of designated and non-designated}} pedantic-note {{here}}

  struct C { int x; };
  void h(A a); // expected-note {{candidate}}
  void h(C c); // expected-note {{candidate}}
  void i() {
    h({.x = 1, .y = 2});
    h({.y = 1, .x = 2}); // reorder-error {{declaration order}} reorder-note {{previous}}
    h({.x = 1}); // expected-error {{ambiguous}}
  }

  struct D { int y, x; };
  void j(A a); // expected-note {{candidate}}
  void j(D d); // expected-note {{candidate}}
  void k() {
    j({.x = 1, .y = 2}); // expected-error {{ambiguous}}
  }
}

namespace deduction {
  struct A { int x, y; };
  union B { int x, y; };

  template<typename T, typename U> void f(decltype(T{.x = 1, .y = 2}) = {});
  template<typename T, typename U> void f(decltype(U{.x = 1, .y = 2}) = {}) = delete;
  void g() { f<A, B>(); } // ok, calls non-union overload

  struct C { int y, x; };
  template<typename T, typename U> void h(decltype(T{.y = 1, .x = 2}) = {}) = delete;
  template<typename T, typename U> void h(decltype(U{.y = 1, .x = 2}) = {});
  void i() {
    h<A, C>(); // ok, selects C overload by SFINAE
  }
}
