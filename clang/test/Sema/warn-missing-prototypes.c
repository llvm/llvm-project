// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -Wmissing-prototypes -Wno-deprecated-non-prototype -verify %s
// RUN: %clang_cc1 -fsyntax-only -Wdocumentation -Wmissing-prototypes -Wno-deprecated-non-prototype -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck %s

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

const int *get_const() { // expected-warning{{no previous prototype for function 'get_const'}}
  // expected-note@-1{{declare 'static' if the function is not intended to be used outside of this translation unit}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:1-[[@LINE-2]]:1}:"static "
  return (void *)0;
}

struct MyStruct {};

// FIXME: because qualifiers are ignored in the return type when forming the
// type from the declarator, we get the position incorrect for the fix-it hint.
// It suggests 'const static struct' instead of 'static const struct'. However,
// thanks to the awful rules of parsing in C, the effect is the same and the
// code is valid, if a bit funny looking.
const struct MyStruct get_struct() { // expected-warning{{no previous prototype for function 'get_struct'}}
  // expected-note@-1{{declare 'static' if the function is not intended to be used outside of this translation unit}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:7-[[@LINE-2]]:7}:"static "
  struct MyStruct ret;
  return ret;
}

// Turn off clang-format for white-space dependent testing.
// clang-format off
// Two spaces between cost and struct
const  struct MyStruct get_struct_2() {  // expected-warning{{no previous prototype for function 'get_struct_2'}}
  // expected-note@-1{{declare 'static' if the function is not intended to be used outside of this translation unit}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:8-[[@LINE-2]]:8}:"static "
  struct MyStruct ret;
  return ret;
}

// Two spaces bewteen const and struct
const  struct MyStruct* get_struct_ptr() {  // expected-warning{{no previous prototype for function 'get_struct_ptr'}}
  // expected-note@-1{{declare 'static' if the function is not intended to be used outside of this translation unit}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:1-[[@LINE-2]]:1}:"static "
  return (void*)0;
}

// Random comment before const.
/*some randome comment*/const  struct MyStruct* get_struct_3() {  // expected-warning{{no previous prototype for function 'get_struct_3'}}
  // expected-note@-1{{declare 'static' if the function is not intended to be used outside of this translation unit}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:25-[[@LINE-2]]:25}:"static "
  return (void*)0;
}

// Random comment after const.
const/*some randome comment*/ struct MyStruct* get_struct_4() {  // expected-warning{{no previous prototype for function 'get_struct_4'}}
  // expected-note@-1{{declare 'static' if the function is not intended to be used outside of this translation unit}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:1-[[@LINE-2]]:1}:"static "
  return (void*)0;
}
// clang-format on

#define MY_CONST const

// Since we can't easily understand what MY_CONST means while preparing the
// diagnostic, the fix-it suggests to add 'static' in a non-idiomatic place.
MY_CONST struct MyStruct *get_struct_nyi() { // expected-warning{{no previous prototype for function 'get_struct_nyi'}}
  // expected-note@-1{{declare 'static' if the function is not intended to be used outside of this translation unit}}
  // CHECK: fix-it:"{{.*}}":{[[@LINE-2]]:10-[[@LINE-2]]:10}:"static "
  return (void *)0;
}
