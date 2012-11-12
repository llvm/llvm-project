// RUN: %clang_cc1 -fsyntax-only -verify -triple x86_64-apple-darwin %s

// This file tests -Wconstant-conversion, a subcategory of -Wconversion
// which is on by default.

// rdar://problem/6792488
void test_6792488(void) {
  int x = 0x3ff0000000000000U; // expected-warning {{implicit conversion from 'unsigned long' to 'int' changes value from 4607182418800017408 to 0}}
}

void test_7809123(void) {
  struct { int i5 : 5; } a;

  a.i5 = 36; // expected-warning {{implicit truncation from 'int' to bitfield changes value from 36 to 4}}
}

void test() {
  struct { int bit : 1; } a;
  a.bit = 1; // shouldn't warn
}

enum Test2 { K_zero, K_one };
enum Test2 test2(enum Test2 *t) {
  *t = 20;
  return 10; // shouldn't warn
}

void test3() {
  struct A {
    unsigned int foo : 2;
    int bar : 2;
  };

  struct A a = { 0, 10 };            // expected-warning {{implicit truncation from 'int' to bitfield changes value from 10 to -2}}
  struct A b[] = { 0, 10, 0, 0 };    // expected-warning {{implicit truncation from 'int' to bitfield changes value from 10 to -2}}
  struct A c[] = {{10, 0}};          // expected-warning {{implicit truncation from 'int' to bitfield changes value from 10 to 2}}
  struct A d = (struct A) { 10, 0 }; // expected-warning {{implicit truncation from 'int' to bitfield changes value from 10 to 2}}
  struct A e = { .foo = 10 };        // expected-warning {{implicit truncation from 'int' to bitfield changes value from 10 to 2}}
}

void test4() {
  struct A {
    char c : 2;
  } a;

  a.c = 0x101; // expected-warning {{implicit truncation from 'int' to bitfield changes value from 257 to 1}}
}

void test5() {
  struct A {
    _Bool b : 1;
  } a;

  // Don't warn about this implicit conversion to bool, or at least
  // don't warn about it just because it's a bitfield.
  a.b = 100;
}

void test6() {
  // Test that unreachable code doesn't trigger the truncation warning.
  unsigned char x = 0 ? 65535 : 1; // no-warning
  unsigned char y = 1 ? 65535 : 1; // expected-warning {{changes value}}
}

void test7() {
	struct {
		unsigned int twoBits1:2;
		unsigned int twoBits2:2;
		unsigned int reserved:28;
	} f;

	f.twoBits1 = ~1; // expected-warning {{implicit truncation from 'int' to bitfield changes value from -2 to 2}}
	f.twoBits2 = ~2; // expected-warning {{implicit truncation from 'int' to bitfield changes value from -3 to 1}}
	f.twoBits1 &= ~1; // no-warning
	f.twoBits2 &= ~2; // no-warning
}

void test8() {
  enum E { A, B, C };
  struct { enum E x : 1; } f;
  f.x = C; // expected-warning {{implicit truncation from 'int' to bitfield changes value from 2 to 0}}
}
