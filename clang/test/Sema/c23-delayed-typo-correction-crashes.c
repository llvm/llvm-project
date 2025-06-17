// RUN: %clang_cc1 -std=c23 -fsyntax-only -verify %s

void GH139913(...);
void GH139913_test() {
  GH139913(CONCAT(foo, )); // expected-error {{use of undeclared identifier 'CONCAT'}} \
                              expected-error {{use of undeclared identifier 'foo'}} \
                              expected-error {{expected expression}}
}

struct GH137867 {
 char value;
};
void GH137867_test() {
  _Atomic(struct GH137867) t;
  while (!atomic_load(&t.value)->value) // expected-error {{use of undeclared identifier 'atomic_load'}} \
                                           expected-error {{accessing a member of an atomic structure or union is undefined behavior}}
    ;
}
