// RUN: %clang_cc1 -fsyntax-only -verify %s

struct Foo {};

void test(int a, Foo b, int *d, Foo *e, const Foo *f) {
  __builtin_clear_padding(a); // expected-error {{passing 'int' to parameter of incompatible type pointer: type mismatch at 1st parameter ('int' vs pointer)}}
  __builtin_clear_padding(b); // expected-error {{passing 'Foo' to parameter of incompatible type pointer: type mismatch at 1st parameter ('Foo' vs pointer)}}
  __builtin_clear_padding(d); // This should not error.
  __builtin_clear_padding(e); // This should not error.
  __builtin_clear_padding(f); // expected-error {{read-only variable is not assignable}}
}

struct Incomplete; // expected-note {{forward declaration of 'Incomplete'}}

void testIncomplete(void* v, Incomplete *i) {
  __builtin_clear_padding(v); // expected-error {{variable has incomplete type 'void'}}
  __builtin_clear_padding(i); // expected-error {{variable has incomplete type 'Incomplete'}}
}

void testNumArgs(int* i) {
  __builtin_clear_padding(); // expected-error {{too few arguments to function call, expected 1, have 0}}
  __builtin_clear_padding(i); // This should not error.
  __builtin_clear_padding(i, i); // expected-error {{too many arguments to function call, expected 1, have 2}}
  __builtin_clear_padding(i, i, i); // expected-error {{too many arguments to function call, expected 1, have 3}}
  __builtin_clear_padding(i, i, i, i); // expected-error {{too many arguments to function call, expected 1, have 4}}
}

struct NonTriviallyCopyable {
  NonTriviallyCopyable() {}
  NonTriviallyCopyable(const NonTriviallyCopyable&){}
};

struct DerivedNonTriviallyCopyable : NonTriviallyCopyable {};

void testNonTriviallyCopyable(NonTriviallyCopyable& ntc0, NonTriviallyCopyable ntc1, DerivedNonTriviallyCopyable& dntc0, DerivedNonTriviallyCopyable dntc1) {
  NonTriviallyCopyable ntc2;
  NonTriviallyCopyable& ntc3 = ntc0;
  DerivedNonTriviallyCopyable dntc2;
  DerivedNonTriviallyCopyable& dntc3 = dntc0;

  __builtin_clear_padding(&ntc0); // expected-error {{argument to __builtin_clear_padding must be a pointer to a trivially-copyable type ('NonTriviallyCopyable *' invalid)}}
  __builtin_clear_padding(&ntc1);
  __builtin_clear_padding(&ntc2);
  __builtin_clear_padding(&ntc3); // expected-error {{argument to __builtin_clear_padding must be a pointer to a trivially-copyable type ('NonTriviallyCopyable *' invalid)}}

  __builtin_clear_padding(&dntc0); // expected-error {{argument to __builtin_clear_padding must be a pointer to a trivially-copyable type ('DerivedNonTriviallyCopyable *' invalid)}}
  __builtin_clear_padding(&dntc1);
  __builtin_clear_padding(&dntc2);
  __builtin_clear_padding(&dntc3); // expected-error {{argument to __builtin_clear_padding must be a pointer to a trivially-copyable type ('DerivedNonTriviallyCopyable *' invalid)}}

  __builtin_clear_padding((NonTriviallyCopyable*)&ntc0); // expected-error {{argument to __builtin_clear_padding must be a pointer to a trivially-copyable type ('NonTriviallyCopyable *' invalid)}}
  __builtin_clear_padding((NonTriviallyCopyable*)&ntc1);
  __builtin_clear_padding((NonTriviallyCopyable*)&ntc2);
  __builtin_clear_padding((NonTriviallyCopyable*)&ntc3); // expected-error {{argument to __builtin_clear_padding must be a pointer to a trivially-copyable type ('NonTriviallyCopyable *' invalid)}}

  __builtin_clear_padding((DerivedNonTriviallyCopyable*)&dntc0); // expected-error {{argument to __builtin_clear_padding must be a pointer to a trivially-copyable type ('DerivedNonTriviallyCopyable *' invalid)}}
  __builtin_clear_padding((DerivedNonTriviallyCopyable*)&dntc1);
  __builtin_clear_padding((DerivedNonTriviallyCopyable*)&dntc2);
  __builtin_clear_padding((DerivedNonTriviallyCopyable*)&dntc3); // expected-error {{argument to __builtin_clear_padding must be a pointer to a trivially-copyable type ('DerivedNonTriviallyCopyable *' invalid)}}

  __builtin_clear_padding((NonTriviallyCopyable*)&dntc0); // expected-error {{argument to __builtin_clear_padding must be a pointer to a trivially-copyable type ('NonTriviallyCopyable *' invalid)}}
  __builtin_clear_padding((NonTriviallyCopyable*)&dntc1); // expected-error {{argument to __builtin_clear_padding must be a pointer to a trivially-copyable type ('NonTriviallyCopyable *' invalid)}}
  __builtin_clear_padding((NonTriviallyCopyable*)&dntc2); // expected-error {{argument to __builtin_clear_padding must be a pointer to a trivially-copyable type ('NonTriviallyCopyable *' invalid)}}
  __builtin_clear_padding((NonTriviallyCopyable*)&dntc3); // expected-error {{argument to __builtin_clear_padding must be a pointer to a trivially-copyable type ('NonTriviallyCopyable *' invalid)}}
}

struct Bar {
  Foo *foo;
};

void testMemberPointer(Foo* Bar::*mp) {
  __builtin_clear_padding(mp); // expected-error {{passing 'Foo *Bar::*' to parameter of incompatible type pointer: type mismatch at 1st parameter ('Foo *Bar::*' vs pointer)}}
}


void testFunctionPointer(void(*f)()) {
  __builtin_clear_padding(f); // expected-error {{argument to __builtin_clear_padding must be a pointer to a trivially-copyable type ('void (*)()' invalid)}}
}

struct WithVLA {
  int i;
  char c[];
};

struct WithVLA2 {
  int i2;
  WithVLA w;
};

struct WithVLA3 {
  WithVLA2 w2;
};

void testVLA(WithVLA* w1, WithVLA2* w2, WithVLA3* w3) {
  __builtin_clear_padding(w1); // expected-error {{'WithVLA' has flexible array member, which is unsupported by __builtin_clear_padding}}
  __builtin_clear_padding(w2); // expected-error {{'WithVLA2' has flexible array member, which is unsupported by __builtin_clear_padding}}
  __builtin_clear_padding(w3); // expected-error {{'WithVLA3' has flexible array member, which is unsupported by __builtin_clear_padding}}
}
