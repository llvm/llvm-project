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

	f.twoBits1 = ~0; // no-warning
	f.twoBits1 = ~1; // no-warning
	f.twoBits2 = ~2; // expected-warning {{implicit truncation from 'int' to bitfield changes value from -3 to 1}}
	f.twoBits1 &= ~1; // no-warning
	f.twoBits2 &= ~2; // no-warning
}

void test8() {
  enum E { A, B, C };
  struct { enum E x : 1; } f;
  f.x = C; // expected-warning {{implicit truncation from 'int' to bitfield changes value from 2 to 0}}
}

void test9() {
  const char max_char = 0x7F;
  const short max_short = 0x7FFF;
  const int max_int = 0x7FFFFFFF;

  const short max_char_plus_one = (short)max_char + 1;
  const int max_short_plus_one = (int)max_short + 1;
  const long max_int_plus_one = (long)max_int + 1;

  char new_char = max_char_plus_one;  // expected-warning {{implicit conversion from 'const short' to 'char' changes value from 128 to -128}}
  short new_short = max_short_plus_one;  // expected-warning {{implicit conversion from 'const int' to 'short' changes value from 32768 to -32768}}
  int new_int = max_int_plus_one;  // expected-warning {{implicit conversion from 'const long' to 'int' changes value from 2147483648 to -2147483648}}

  char hex_char = 0x80;
  short hex_short = 0x8000;
  int hex_int = 0x80000000;

  char oct_char = 0200;
  short oct_short = 0100000;
  int oct_int = 020000000000;

  char bin_char = 0b10000000;
  short bin_short = 0b1000000000000000;
  int bin_int = 0b10000000000000000000000000000000;

#define CHAR_MACRO_HEX 0xff
  char macro_char_hex = CHAR_MACRO_HEX;
#define CHAR_MACRO_DEC 255
  char macro_char_dec = CHAR_MACRO_DEC;  // expected-warning {{implicit conversion from 'int' to 'char' changes value from 255 to -1}}

  char array_init[] = { 255, 127, 128, 129, 0 };
}

#define A 1

void test10() {
  struct S {
    unsigned a : 4;
  } s;
  s.a = -1;
  s.a = 15;
  s.a = -8;
  s.a = ~0;
  s.a = ~0U;
  s.a = ~(1<<A);

  s.a = -9;  // expected-warning{{implicit truncation from 'int' to bitfield changes value from -9 to 7}}
  s.a = 16;  // expected-warning{{implicit truncation from 'int' to bitfield changes value from 16 to 0}}
}
