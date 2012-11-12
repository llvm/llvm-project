// RUN: %clang_cc1 -fsyntax-only -verify -fexceptions -fobjc-exceptions %s
// expected-no-diagnostics

// Note that we're specifically excluding -fcxx-exceptions in the command line above.

// That this should work even with -fobjc-exceptions is PR9358

// PR7243: redeclarations
namespace test0 {
  void foo() throw(int);
  void foo() throw();
}

// Overrides.
namespace test1 {
  struct A {
    virtual void foo() throw();
  };

  struct B : A {
    virtual void foo() throw(int);
  };
}

// Calls from less permissive contexts.  We don't actually do this
// check, but if we did it should also be disabled under
// -fno-exceptions.
namespace test2 {
  void foo() throw(int);
  void bar() throw() {
    foo();
  }
}

