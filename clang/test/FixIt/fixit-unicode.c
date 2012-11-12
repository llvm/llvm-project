// RUN: %clang_cc1 -fsyntax-only %s 2>&1 | FileCheck -strict-whitespace %s
// RUN: %clang_cc1 -fsyntax-only -fdiagnostics-parseable-fixits %s 2>&1 | FileCheck -check-prefix=CHECK-MACHINE %s

struct Foo {
  int bar;
};

// PR13312
void test1() {
  struct Foo foo;
  (&foo)☃>bar = 42;
// CHECK: error: expected ';' after expression
// Make sure we emit the fixit right in front of the snowman.
// CHECK: {{^        \^}}
// CHECK: {{^        ;}}

// CHECK-MACHINE: fix-it:"{{.*}}fixit-unicode.c":{11:9-11:9}:";"
}


int printf(const char *, ...);
void test2() {
  printf("∆: %d", 1L);
// CHECK: warning: format specifies type 'int' but the argument has type 'long'
// Don't crash emitting a fixit after the delta.
// CHECK:  printf("
// CHECK: : %d", 1L);
// Unfortunately, we can't actually check the location of the printed fixit,
// because different systems will render the delta differently (either as a
// character, or as <U+2206>.) The fixit should line up with the %d regardless.

// CHECK-MACHINE: fix-it:"{{.*}}fixit-unicode.c":{23:16-23:18}:"%ld"
}
