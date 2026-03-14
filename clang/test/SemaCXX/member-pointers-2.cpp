// RUN: %clang_cc1 -emit-llvm-only %s

// Tests that Sema properly creates member-access expressions for
// these instead of bare FieldDecls.

struct Foo {
  int myvalue;

  // We have to override these to get something with an lvalue result.
  int &operator++(int);
  int &operator--(int);
};

struct Test0 {
  Foo memfoo;
  int memint;
  int memarr[10];
  Test0 *memptr;
  struct MemClass { int a; } memstruct;
  int &memfun();
  
  void test() {
    int *p;
    p = &Test0::memfoo++;
    p = &Test0::memfoo--;
    p = &Test0::memarr[1];
    p = &Test0::memptr->memint;
    p = &Test0::memstruct.a;
    p = &Test0::memfun();
  }
};

void test0() {
  Test0 mytest;
  mytest.test();
}

namespace rdar9065289 {
  typedef void (*FuncPtr)();
  struct X0 { };

  struct X1
  {
    X0* x0;
    FuncPtr X0::*fptr;
  };

  void f(X1 p) {
    (p.x0->*(p.fptr))();
  }
}
