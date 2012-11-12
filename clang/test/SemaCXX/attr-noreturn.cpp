// RUN: %clang_cc1 -fsyntax-only -verify %s

// Reachability tests have to come first because they get suppressed
// if any errors have occurred.
namespace test5 {
  struct A {
    __attribute__((noreturn)) void fail();
    void nofail();
  } a;

  int &test1() {
    a.nofail();
  } // expected-warning {{control reaches end of non-void function}}

  int &test2() {
    a.fail();
  }
}

// PR5620
void f0() __attribute__((__noreturn__));
void f1(void (*)()); 
void f2() { f1(f0); }

// Taking the address of a noreturn function
void test_f0a() {
  void (*fp)() = f0;
  void (*fp1)() __attribute__((noreturn)) = f0;
}

// Taking the address of an overloaded noreturn function 
void f0(int) __attribute__((__noreturn__));

void test_f0b() {
  void (*fp)() = f0;
  void (*fp1)() __attribute__((noreturn)) = f0;
}

// No-returned function pointers
typedef void (* noreturn_fp)() __attribute__((noreturn));

void f3(noreturn_fp); // expected-note{{candidate function}}

void test_f3() {
  f3(f0); // okay
  f3(f2); // expected-error{{no matching function for call}}
}


class xpto {
  int blah() __attribute__((noreturn));
};

int xpto::blah() {
  return 3; // expected-warning {{function 'blah' declared 'noreturn' should not return}}
}

// PR12948

namespace PR12948 {
  template<int>
  void foo() __attribute__((__noreturn__));

  template<int>
  void foo() {
    while (1) continue;
  }

  void bar() __attribute__((__noreturn__));

  void bar() {
    foo<0>();
  }


  void baz() __attribute__((__noreturn__));
  typedef void voidfn();
  voidfn baz;

  template<typename> void wibble()  __attribute__((__noreturn__));
  template<typename> voidfn wibble;
}
