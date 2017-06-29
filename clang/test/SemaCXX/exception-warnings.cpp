// RUN: %clang_cc1 -fcxx-exceptions -fexceptions -fsyntax-only -verify %s

struct B {};
struct D: B {};
void goodPlain() throw () {
  try {
    throw D();
  } catch (B) {}
}
void goodReference() throw () {
  try {
    throw D();
  } catch (B &) {}
}
void goodPointer() throw () {
  D d;
  try {
    throw &d;
  } catch (B *) {}
}
void badPlain() throw () { // expected-note {{non-throwing function declare here}}
  try {
    throw B(); // expected-warning {{'badPlain' has a non-throwing exception specification but can still throw, resulting in unexpected program termination}}
  } catch (D) {}
}
void badReference() throw () { // expected-note {{non-throwing function declare here}}
  try {
    throw B(); // expected-warning {{'badReference' has a non-throwing exception specification but can still throw, resulting in unexpected program termination}}
  } catch (D &) {}
}
void badPointer() throw () { // expected-note {{non-throwing function declare here}}
  B b;
  try {
    throw &b; // expected-warning {{'badPointer' has a non-throwing exception specification but can still throw, resulting in unexpected program termination}}
  } catch (D *) {}
}
