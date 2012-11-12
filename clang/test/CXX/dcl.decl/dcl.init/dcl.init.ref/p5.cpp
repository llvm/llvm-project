// RUN: %clang_cc1 -fsyntax-only -verify %s

namespace PR5909 {
  struct Foo {
    int x : 20;
  };
  
  bool Test(const int& foo);
  
  const Foo f = { 0 };  // It compiles without the 'const'.
  bool z = Test(f.x);
}

namespace PR6264 {
  typedef int (&T)[3];
  struct S
  {
    operator T ();
  };
  void f()
  {
    T bar = S();
  }
}

namespace PR6066 {
  struct B { };
  struct A : B {
    operator B*();
    operator B&(); // expected-warning{{conversion function converting 'PR6066::A' to its base class 'PR6066::B' will never be used}}
  };

  void f(B&); // no rvalues accepted
  void f(B*);

  int g() {
    f(A()); // calls f(B*)
    return 0;
  }
}
