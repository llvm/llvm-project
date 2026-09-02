// RUN: %clang_cc1 -triple x86_64-apple-macosx -fsyntax-only -verify -x objective-c++ %s
// REQUIRES: asserts

typedef struct {} Class;

void test() {
  Class c;
  c.className; // expected-error {{member reference base type 'Class' is not a structure or union}}
}
