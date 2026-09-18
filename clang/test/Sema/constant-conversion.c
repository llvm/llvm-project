// RUN: %clang_cc1 -fsyntax-only -ffreestanding -verify=expected,signed-plain-char,one-bit -triple x86_64-apple-darwin %s
// RUN: %clang_cc1 -fsyntax-only -ffreestanding -Wno-single-bit-bitfield-constant-conversion -verify=expected,signed-plain-char -triple x86_64-apple-darwin %s
// RUN: %clang_cc1 -fsyntax-only -ffreestanding -Wno-single-bit-bitfield-constant-conversion -verify -triple x86_64-apple-darwin -fno-signed-char %s

#include <stdbool.h>

// This file tests -Wconstant-conversion, a subcategory of -Wconversion
// which is on by default.

void test_6792488(void) {
  int x = 0x3ff0000000000000U; // expected-warning {{implicit conversion from 'unsigned long' to 'int' changes value from 4607182418800017408 to 0}}
}

void test_7809123(void) {
  struct { int i5 : 5; } a;

  a.i5 = 36; // expected-warning {{implicit truncation from 'int' to bit-field changes value from 36 to 4}}
}

void test(void) {
  struct S {
    int b : 1;  // The only valid values are 0 and -1.
  } s;

  s.b = -3;    // expected-warning {{implicit truncation from 'int' to bit-field changes value from -3 to -1}}
  s.b = -2;    // expected-warning {{implicit truncation from 'int' to bit-field changes value from -2 to 0}}
  s.b = -1;    // no-warning
  s.b = 0;     // no-warning
  s.b = 1;     // one-bit-warning {{implicit truncation from 'int' to a one-bit wide bit-field changes value from 1 to -1}}
  s.b = true;  // no-warning (we suppress it manually to reduce false positives)
  s.b = false; // no-warning
  s.b = 2;     // expected-warning {{implicit truncation from 'int' to bit-field changes value from 2 to 0}}
}

enum Test2 { K_zero, K_one };
enum Test2 test2(enum Test2 *t) {
  *t = 20;
  return 10; // shouldn't warn
}

void test3(void) {
  struct A {
    unsigned int foo : 2;
    int bar : 2;
  };

  struct A a = { 0, 10 };            // expected-warning {{implicit truncation from 'int' to bit-field changes value from 10 to -2}}
  struct A b[] = { 0, 10, 0, 0 };    // expected-warning {{implicit truncation from 'int' to bit-field changes value from 10 to -2}}
  struct A c[] = {{10, 0}};          // expected-warning {{implicit truncation from 'int' to bit-field changes value from 10 to 2}}
  struct A d = (struct A) { 10, 0 }; // expected-warning {{implicit truncation from 'int' to bit-field changes value from 10 to 2}}
  struct A e = { .foo = 10 };        // expected-warning {{implicit truncation from 'int' to bit-field changes value from 10 to 2}}
}

void test4(void) {
  struct A {
    char c : 2;
  } a;

  a.c = 0x101; // expected-warning {{implicit truncation from 'int' to bit-field changes value from 257 to 1}}
}

void test5(void) {
  struct A {
    _Bool b : 1;
  } a;

  // Don't warn about this implicit conversion to bool, or at least
  // don't warn about it just because it's a bit-field.
  a.b = 100;
}

// GH223923: Do not diagnose conversions in unselected operands of constant
// conditional expressions.
#define BASE64_INIT_1(v, ch_62, ch_63) \
  [v] = (v) >= 'A' && (v) <= 'Z' ? (v) - 'A' \
      : (v) >= 'a' && (v) <= 'z' ? (v) - 'a' + 26 \
      : (v) >= '0' && (v) <= '9' ? (v) - '0' + 52 \
      : (v) == (ch_62) ? 62 : (v) == (ch_63) ? 63 : -1

#define BASE64_INIT_2(v, ...) \
  BASE64_INIT_1(v, __VA_ARGS__), BASE64_INIT_1((v) + 1, __VA_ARGS__)
#define BASE64_INIT_4(v, ...) \
  BASE64_INIT_2(v, __VA_ARGS__), BASE64_INIT_2((v) + 2, __VA_ARGS__)
#define BASE64_INIT_8(v, ...) \
  BASE64_INIT_4(v, __VA_ARGS__), BASE64_INIT_4((v) + 4, __VA_ARGS__)
#define BASE64_INIT_16(v, ...) \
  BASE64_INIT_8(v, __VA_ARGS__), BASE64_INIT_8((v) + 8, __VA_ARGS__)
#define BASE64_INIT_32(v, ...) \
  BASE64_INIT_16(v, __VA_ARGS__), BASE64_INIT_16((v) + 16, __VA_ARGS__)

#define BASE64_REV_INIT(ch_62, ch_63) { \
  [0 ... 0x1f] = -1, \
  BASE64_INIT_32(0x20, ch_62, ch_63), \
  BASE64_INIT_32(0x40, ch_62, ch_63), \
  BASE64_INIT_32(0x60, ch_62, ch_63), \
  [0x80 ... 0xff] = -1 \
}

enum base64_variant {
  BASE64_STD,
  BASE64_URLSAFE,
  BASE64_IMAP,
};

static const signed char base64_rev_maps[][256] __attribute__((unused)) = {
  [BASE64_STD] = BASE64_REV_INIT('+', '/'),
  [BASE64_URLSAFE] = BASE64_REV_INIT('-', '_'),
  [BASE64_IMAP] = BASE64_REV_INIT('+', ',')
};

#undef BASE64_REV_INIT
#undef BASE64_INIT_32
#undef BASE64_INIT_16
#undef BASE64_INIT_8
#undef BASE64_INIT_4
#undef BASE64_INIT_2
#undef BASE64_INIT_1

static const signed char signed_char_array_live[] __attribute__((unused)) = {
  1 ? 128 : -1 // expected-warning {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
};

void GH223923(int condition) {
  signed char signed_char_dead = 0 ? 128 : -1;
  signed char signed_char_live = 1 ? 128 : -1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
  signed char signed_char_maybe = condition ? 128 : -1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}

  short short_dead = 0 ? 32768 : 1;
  short short_live = 1 ? 32768 : 1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'short' changes value from 32768 to -32768}}

  int int_dead = 0 ? 2147483648LL : 1;
  int int_live = 1 ? 2147483648LL : 1;
  // expected-warning@-1 {{implicit conversion from 'long long' to 'int' changes value from 2147483648 to -2147483648}}

  signed char truncation_dead = 0 ? 256 : 1;
  signed char truncation_live = 1 ? 256 : 1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 256 to 0}}

  signed char sink;
  int assignment_dead = 0 ? (sink = 128) : 1;
  int assignment_live = 1 ? (sink = 128) : 1;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}

  signed char binary_dead = 1 ?: 128;
  signed char binary_live = 0 ?: 128;
  // expected-warning@-1 {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
}

void test6(void) {
  // Test that unreachable code doesn't trigger the truncation warning.
  unsigned char x = 0 ? 65535 : 1; // no-warning
  unsigned char y = 1 ? 65535 : 1; // expected-warning {{changes value}}
}

void test7(void) {
	struct {
		unsigned int twoBits1:2;
		unsigned int twoBits2:2;
		unsigned int reserved:28;
	} f;

	f.twoBits1 = ~0; // no-warning
	f.twoBits1 = ~1; // no-warning
	f.twoBits2 = ~2; // expected-warning {{implicit truncation from 'int' to bit-field changes value from -3 to 1}}
	f.twoBits1 &= ~1; // no-warning
	f.twoBits2 &= ~2; // no-warning
}

void test8(void) {
  enum E { A, B, C };
  struct { enum E x : 1; } f;
  f.x = C; // expected-warning {{implicit truncation from 'int' to bit-field changes value from 2 to 0}}
}

void test9(void) {
  const char max_char = 0x7F;
  const short max_short = 0x7FFF;
  const int max_int = 0x7FFFFFFF;

  const short max_char_plus_one = (short)max_char + 1;
  const int max_short_plus_one = (int)max_short + 1;
  const long max_int_plus_one = (long)max_int + 1;

  char new_char = max_char_plus_one;  // signed-plain-char-warning {{implicit conversion from 'const short' to 'char' changes value from 128 to -128}}
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
  char macro_char_dec = CHAR_MACRO_DEC;  // signed-plain-char-warning {{implicit conversion from 'int' to 'char' changes value from 255 to -1}}

  char array_init[] = { 255 }; // signed-plain-char-warning {{implicit conversion from 'int' to 'char' changes value from 255 to -1}}
  unsigned char unsigned_array_init[] = { 255 };
  unsigned char unsigned_array_init_multi[] = { 255, 127, 128, 129, 0 };
  signed char signed_array_init[] = { 255 }; // expected-warning {{implicit conversion from 'int' to 'signed char' changes value from 255 to -1}}
  signed char signed_array_init_multi[] = {
    255, // expected-warning {{implicit conversion from 'int' to 'signed char' changes value from 255 to -1}}
    127,
    128, // expected-warning {{implicit conversion from 'int' to 'signed char' changes value from 128 to -128}}
    129, // expected-warning {{implicit conversion from 'int' to 'signed char' changes value from 129 to -127}}
    0
  };
}

#define A 1

void test10(void) {
  struct S {
    unsigned a : 4;
  } s;
  s.a = -1;
  s.a = 15;
  s.a = -8;
  s.a = ~0;
  s.a = ~0U;
  s.a = ~(1<<A);

  s.a = -9;  // expected-warning{{implicit truncation from 'int' to bit-field changes value from -9 to 7}}
  s.a = 16;  // expected-warning{{implicit truncation from 'int' to bit-field changes value from 16 to 0}}
}
