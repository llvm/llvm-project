// RUN: %clang_cc1 -fsyntax-only -verify %s

void a; // expected-error {{variable has incomplete type 'void'}}
extern int a[];

void test() {
  // Force constant evaluation of &a[6].
  // Before the fix this asserted in LValue::addUnsizedArray.
  (void)__builtin_constant_p(&a[6]);
}