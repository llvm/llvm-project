// Due to the fix having multiple edits we can't use
// '-fdiagnostics-parseable-fixits' to determine if fixes are correct. However,
// running fixit recompile with 'Werror' should fail if the fixes are invalid.

// RUN: %clang_cc1 %s -Werror=reorder-ctor -fixit-recompile -fixit-to-temporary
// RUN: %clang_cc1 %s -Wreorder-ctor -verify -verify-ignore-unexpected=note

struct Foo {
  int A, B, C;

  Foo() : A(1), B(3), C(2) {}
  Foo(int) : A(1), C(2), B(3) {}      // expected-warning {{field 'C' will be initialized after field 'B'}}
  Foo(unsigned) : C(2), B(3), A(1) {} // expected-warning {{initializer order does not match the declaration order}}
};

struct Bar : Foo {
  int D, E, F;

  Bar() : Foo(), D(1), E(2), F(3) {}
  Bar(int) : D(1), E(2), F(3), Foo(4) {}      // expected-warning {{field 'F' will be initialized after base 'Foo'}}
  Bar(unsigned) : F(3), E(2), D(1), Foo(4) {} // expected-warning {{initializer order does not match the declaration order}}
};
