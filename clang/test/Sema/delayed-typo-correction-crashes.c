// RUN: %clang_cc1 -fsyntax-only -fblocks -ffixed-point -verify %s

void GH137860_test(void) {
  struct S {
    char h;
  };
  _Atomic struct S s = { .h = UINT8_MIN }; // expected-error {{use of undeclared identifier 'UINT8_MIN'}}
  __c11_atomic_fetch_add(&s.h, UINT8_MIN); // expected-error {{use of undeclared identifier 'UINT8_MIN'}} \
                                              expected-error {{accessing a member of an atomic structure or union is undefined behavior}}
}

int (^GH69470) (int i, int j) = ^(int i, int j)
{ return i / j; }/ j; // expected-error {{use of undeclared identifier 'j'}}

void GH69874(void) {
  *a = (a_struct){0}; // expected-error {{use of undeclared identifier 'a'}} \
                         expected-error {{use of undeclared identifier 'a_struct'}}
}
