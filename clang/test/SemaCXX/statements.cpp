// RUN: %clang_cc1 %s -fsyntax-only -pedantic -verify

void foo() { 
  return foo();
}

// PR6451 - C++ Jump checking
struct X {
  X();
};

void test2() {
  goto later;  // expected-error {{cannot jump}}
  X x;         // expected-note {{jump bypasses variable initialization}} 
later:
  ;
}

namespace PR6536 {
  struct A {};
  void a() { goto out; A x; out: return; }
}

void test3() {
    __asm__ ("":"+r" (test3)); // expected-error{{invalid lvalue in asm output}}
}

void test4();                // expected-note{{possible target for call}}
void test4(int) {            // expected-note{{possible target for call}}
  // expected-error@+1{{overloaded function could not be resolved}}
  __asm__ ("":"+r" (test4)); // expected-error{{invalid lvalue in asm output}}
}
void test5() {
  char buf[1];
  __asm__ ("":"+r" (buf));
}

struct MMX_t {};
void test6() { __asm__("" : "=m"(*(MMX_t *)0)); }

template <typename T>
T test7(T v) {
  return ({ // expected-warning{{use of GNU statement expression extension}}
    T a = v;
    a;
    ;
    ;
  });
}

void test8() {
  int a = test7(1);
  double b = test7(2.0);
}

namespace GH48405 {
void foo() {
  struct S {
    int i;
    int j = ({i;}); // expected-error {{invalid use of non-static data member 'i'}}
                    // expected-error@-1 {{cannot initialize a member subobject of type 'int' with an rvalue of type 'void'}}
		    // expected-warning@-2 {{use of GNU statement expression extension}}
  };
}
}
