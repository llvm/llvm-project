// RUN: %clang_cc1 -fsyntax-only -verify %s
// RUN: not %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck %s

namespace ambig {
  struct Foo {
    operator int();
    operator const char *();
  };


  void func(const char*, long);
  void func(const char*, const char*);
  void func(int, int);

  bool doit(Foo x) {
    func(x, x); // expected-error {{call to 'func' is ambiguous}}
                // expected-note@* 3{{candidate}}
    // Check that two functions with best conversions are at the top.
    // CHECK: error: call to 'func' is ambiguous
    // CHECK-NEXT: func(x, x)
    // CHECK-NEXT: ^~~~
    // CHECK-NEXT: note: candidate function
    // CHECK-NEXT: void func(const char*, const char*)
    // CHECK-NEXT: ^
    // CHECK-NEXT: note: candidate function
    // CHECK-NEXT: void func(int, int)
  }
}

namespace bad_conversion {
  struct Foo {
    operator int();
    operator const char *();
  };


  void func(double*, const char*, long);
  void func(double*, const char*, const char*);
  void func(double*, int, int);

  bool doit(Foo x) {
    func((int*)0, x, x); // expected-error {{no matching function for call to 'func'}}
                         // expected-note@* 3{{candidate}}
    // Check that two functions with best conversions are at the top.
    // CHECK: error: no matching function for call to 'func'
    // CHECK-NEXT: func((int*)0, x, x)
    // CHECK-NEXT: ^~~~
    // CHECK-NEXT: note: candidate function
    // CHECK-NEXT: void func(double*, const char*, const char*)
    // CHECK-NEXT: ^
    // CHECK-NEXT: note: candidate function
    // CHECK-NEXT: void func(double*, int, int)
  }
}

namespace bad_deduction {
  template <class> struct templ {};
  template <class T> void func(templ<T>);
  template <class T> void func(T*);
  template <class T> auto func(T&) -> decltype(T().begin());
  template <class T> auto func(const T&) -> decltype(T().begin());

  bool doit() {
    struct record {} r;
    func(r); // expected-error {{no matching function for call to 'func'}}
             // expected-note@* 4{{candidate}}
  }
}
