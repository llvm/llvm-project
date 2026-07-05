// RUN: %clang_cc1 -verify -fsyntax-only -fobjc-arc -fblocks %s

#define CALLED_ONCE __attribute__((called_once))

void test1(int x CALLED_ONCE);    // expected-error{{'called_once' attribute only applies to function-like parameters}}
void test2(double x CALLED_ONCE); // expected-error{{'called_once' attribute only applies to function-like parameters}}

void test3(void (*foo)(void) CALLED_ONCE);   // no-error
void test4(int (^foo)(int) CALLED_ONCE); // no-error

void test5(void (*foo)(void) __attribute__((called_once(1))));
// expected-error@-1{{'called_once' attribute takes no arguments}}
void test6(void (*foo)(void) __attribute__((called_once("str1", "str2"))));
// expected-error@-1{{'called_once' attribute takes no arguments}}

CALLED_ONCE void test7(void); // expected-warning{{'called_once' attribute only applies to parameters}}
void test8(void) {
  void (*foo)(void) CALLED_ONCE; // expected-warning{{'called_once' attribute only applies to parameters}}
  foo();
}
