// RUN: %clang_cc1 -fsyntax-only -verify %s
// PR clang/3175

void bar(int*);

class c {
  int var;
  static int svar;
  void foo() { 
    bar(&var); 
    bar(&svar);  
  }

  static void wibble() {
    bar(&var); // expected-error{{invalid use of member 'var' in static member function}}
    bar(&svar); 
  }
};

enum E {
  Enumerator
};

void test() {
  (void)&Enumerator; // expected-error{{address expression must be an lvalue or a function designator}}
}

template<int N>
void test2() {
  (void)&N; // expected-error{{address expression must be an lvalue or a function designator}}
}

// PR clang/3222
void xpto();
void (*xyz)(void) = &xpto;

struct PR11066 {
  static int foo(short);
  static int foo(float);
  void test();
};

void PR11066::test() {
  int (PR11066::*ptr)(int) = & &PR11066::foo; // expected-error{{address expression must be an lvalue or a function designator}}
}

