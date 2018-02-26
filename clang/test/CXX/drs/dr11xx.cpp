// RUN: %clang_cc1 -std=c++98 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++11 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++14 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++17 %s -verify -fexceptions -fcxx-exceptions -pedantic-errors
// RUN: %clang_cc1 -std=c++2a %s -verify -fexceptions -fcxx-exceptions -pedantic-errors

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
