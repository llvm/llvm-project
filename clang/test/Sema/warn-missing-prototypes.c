// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -Wmissing-prototypes -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -Wmissing-prototypes -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

int f(); // expected-note{{this declaration is not a prototype; add parameter declarations to make it one}}
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:{{.*}}-[[@LINE-1]]:{{.*}}}:"{{.*}}"

int f(int x) { return x; } // expected-warning{{no previous prototype for function 'f'}}

static int g(int x) { return x; }

int h(int x) { return x; } // expected-warning{{no previous prototype for function 'h'}}
// expected-note@-1{{declare 'static' if the function is not intended to be used outside of this translation unit}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:1-[[@LINE-2]]:1}:"static "

static int g2();

int g2(int x) { return x; }

extern int g3(int x) { return x; } // expected-warning{{no previous prototype for function 'g3'}}
// expected-note@-1{{declare 'static' if the function is not intended to be used outside of this translation unit}}
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-2]]:{{.*}}-[[@LINE-2]]:{{.*}}}:"{{.*}}"

void test(void);

int h3(); // expected-note{{this declaration is not a prototype; add parameter declarations to make it one}}
// CHECK-NOT: fix-it:"{{.*}}":{[[@LINE-1]]:{{.*}}-[[@LINE-1]]:{{.*}}}:"{{.*}}"
int h4(int);
int h4();

void test(void) {
  int h2(int x);
  int h3(int x);
  int h4();
}

int h2(int x) { return x; } // expected-warning{{no previous prototype for function 'h2'}}
// expected-note@-1{{declare 'static' if the function is not intended to be used outside of this translation unit}}
int h3(int x) { return x; } // expected-warning{{no previous prototype for function 'h3'}}
int h4(int x) { return x; }

int f2(int);
int f2();

int f2(int x) { return x; }

// rdar://6759522
int main(void) { return 0; }

void not_a_prototype_test(); // expected-note{{this declaration is not a prototype; add 'void' to make it a prototype for a zero-parameter function}}
// CHECK: fix-it:"{{.*}}":{[[@LINE-1]]:27-[[@LINE-1]]:27}:"void"
void not_a_prototype_test() { } // expected-warning{{no previous prototype for function 'not_a_prototype_test'}}
