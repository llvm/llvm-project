// RUN: %clangxx_msan %s -O0 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && %run %t

// RUN: %clangxx_msan %s -O1 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && %run %t

// RUN: %clangxx_msan %s -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && %run %t

// RUN: %clangxx_msan %s -DVPTRA=1 -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && not %run %t

// RUN: %clangxx_msan %s -DVPTRCA=1 -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && not %run %t

// RUN: %clangxx_msan %s -DVPTRCB=1 -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && not %run %t

// RUN: %clangxx_msan %s -DVPTRC=1 -O2 -fsanitize=memory -fsanitize-memory-use-after-dtor -o %t && not %run %t

// Expected to quit due to invalid access when invoking
// function using vtable.

#include <sanitizer/msan_interface.h>
#include <stdio.h>
#include <assert.h>

class A {
public:
  int x;
  ~A() {}
  virtual void A_Foo() {}
};

class B {
 public:
  int y;
  ~B() {}
  virtual void B_Foo() {}
};

class C : public A, public B {
 public:
  int z;
  ~C() {}
  virtual void C_Foo() {}
};

int main() {
  A *a = new A();
  a->~A();

  // Shouldn't be allowed to invoke function via vtable.
#ifdef VPTRA
  a->A_Foo();
#endif

  C *c = new C();
  c->~C();

#ifdef VPTRCA
  c->A_Foo();
#endif

#ifdef VPTRCB
  c->B_Foo();
#endif

#ifdef VPTRC
  c->C_Foo();
#endif

  return 0;
}
