// Force x86-64 because some of our heuristics are actually based
// on integer sizes.

// RUN: %clang_cc1 -triple x86_64-apple-darwin -fsyntax-only -pedantic -verify -Wsign-compare %s

int test0(long a, unsigned long b) {
  enum EnumA {A};
  enum EnumB {B};
  enum EnumC {C = 0x10000};
  return
         // (a,b)
         (a == (unsigned long) b) +  // expected-warning {{comparison of integers of different signs}}
         (a == (unsigned int) b) +
         (a == (unsigned short) b) +
         (a == (unsigned char) b) +
         ((long) a == b) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a == b) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a == b) +  // expected-warning {{comparison of integers of different signs}}
         ((signed char) a == b) +  // expected-warning {{comparison of integers of different signs}}
         ((long) a == (unsigned long) b) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a == (unsigned int) b) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a == (unsigned short) b) +
         ((signed char) a == (unsigned char) b) +
         (a < (unsigned long) b) +  // expected-warning {{comparison of integers of different signs}}
         (a < (unsigned int) b) +
         (a < (unsigned short) b) +
         (a < (unsigned char) b) +
         ((long) a < b) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a < b) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a < b) +  // expected-warning {{comparison of integers of different signs}}
         ((signed char) a < b) +  // expected-warning {{comparison of integers of different signs}}
         ((long) a < (unsigned long) b) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a < (unsigned int) b) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a < (unsigned short) b) +
         ((signed char) a < (unsigned char) b) +

         // (A,b)
         (A == (unsigned long) b) +
         (A == (unsigned int) b) +
         (A == (unsigned short) b) +
         (A == (unsigned char) b) +
         ((long) A == b) +
         ((int) A == b) +
         ((short) A == b) +
         ((signed char) A == b) +
         ((long) A == (unsigned long) b) +
         ((int) A == (unsigned int) b) +
         ((short) A == (unsigned short) b) +
         ((signed char) A == (unsigned char) b) +
         (A < (unsigned long) b) +
         (A < (unsigned int) b) +
         (A < (unsigned short) b) +
         (A < (unsigned char) b) +
         ((long) A < b) +
         ((int) A < b) +
         ((short) A < b) +
         ((signed char) A < b) +
         ((long) A < (unsigned long) b) +
         ((int) A < (unsigned int) b) +
         ((short) A < (unsigned short) b) +
         ((signed char) A < (unsigned char) b) +

         // (a,B)
         (a == (unsigned long) B) +
         (a == (unsigned int) B) +
         (a == (unsigned short) B) +
         (a == (unsigned char) B) +
         ((long) a == B) +
         ((int) a == B) +
         ((short) a == B) +
         ((signed char) a == B) +
         ((long) a == (unsigned long) B) +
         ((int) a == (unsigned int) B) +
         ((short) a == (unsigned short) B) +
         ((signed char) a == (unsigned char) B) +
         (a < (unsigned long) B) +  // expected-warning {{comparison of integers of different signs}}
         (a < (unsigned int) B) +
         (a < (unsigned short) B) +
         (a < (unsigned char) B) +
         ((long) a < B) +
         ((int) a < B) +
         ((short) a < B) +
         ((signed char) a < B) +
         ((long) a < (unsigned long) B) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a < (unsigned int) B) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a < (unsigned short) B) +
         ((signed char) a < (unsigned char) B) +

         // (C,b)
         (C == (unsigned long) b) +
         (C == (unsigned int) b) +
         (C == (unsigned short) b) + // expected-warning {{comparison of constant 65536 with expression of type 'unsigned short' is always false}}
         (C == (unsigned char) b) +  // expected-warning {{comparison of constant 65536 with expression of type 'unsigned char' is always false}}
         ((long) C == b) +
         ((int) C == b) +
         ((short) C == b) +
         ((signed char) C == b) +
         ((long) C == (unsigned long) b) +
         ((int) C == (unsigned int) b) +
         ((short) C == (unsigned short) b) +
         ((signed char) C == (unsigned char) b) +
         (C < (unsigned long) b) +
         (C < (unsigned int) b) +
         (C < (unsigned short) b) + // expected-warning {{comparison of constant 65536 with expression of type 'unsigned short' is always false}}
         (C < (unsigned char) b) + // expected-warning {{comparison of constant 65536 with expression of type 'unsigned char' is always false}}
         ((long) C < b) +
         ((int) C < b) +
         ((short) C < b) +
         ((signed char) C < b) +
         ((long) C < (unsigned long) b) +
         ((int) C < (unsigned int) b) +
         ((short) C < (unsigned short) b) +
         ((signed char) C < (unsigned char) b) +

         // (a,C)
         (a == (unsigned long) C) +
         (a == (unsigned int) C) +
         (a == (unsigned short) C) +
         (a == (unsigned char) C) +
         ((long) a == C) +
         ((int) a == C) +
         ((short) a == C) + // expected-warning {{comparison of constant 65536 with expression of type 'short' is always false}}
         ((signed char) a == C) + // expected-warning {{comparison of constant 65536 with expression of type 'signed char' is always false}}
         ((long) a == (unsigned long) C) +
         ((int) a == (unsigned int) C) +
         ((short) a == (unsigned short) C) +
         ((signed char) a == (unsigned char) C) +
         (a < (unsigned long) C) +  // expected-warning {{comparison of integers of different signs}}
         (a < (unsigned int) C) +
         (a < (unsigned short) C) +
         (a < (unsigned char) C) +
         ((long) a < C) +
         ((int) a < C) +
         ((short) a < C) + // expected-warning {{comparison of constant 65536 with expression of type 'short' is always true}}
         ((signed char) a < C) + // expected-warning {{comparison of constant 65536 with expression of type 'signed char' is always true}}
         ((long) a < (unsigned long) C) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a < (unsigned int) C) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a < (unsigned short) C) +
         ((signed char) a < (unsigned char) C) +

         // (0x80000,b)
         (0x80000 == (unsigned long) b) +
         (0x80000 == (unsigned int) b) +
         (0x80000 == (unsigned short) b) + // expected-warning {{comparison of constant 524288 with expression of type 'unsigned short' is always false}}
         (0x80000 == (unsigned char) b) + // expected-warning {{comparison of constant 524288 with expression of type 'unsigned char' is always false}}
         ((long) 0x80000 == b) +
         ((int) 0x80000 == b) +
         ((short) 0x80000 == b) +
         ((signed char) 0x80000 == b) +
         ((long) 0x80000 == (unsigned long) b) +
         ((int) 0x80000 == (unsigned int) b) +
         ((short) 0x80000 == (unsigned short) b) +
         ((signed char) 0x80000 == (unsigned char) b) +
         (0x80000 < (unsigned long) b) +
         (0x80000 < (unsigned int) b) +
         (0x80000 < (unsigned short) b) + // expected-warning {{comparison of constant 524288 with expression of type 'unsigned short' is always false}}
         (0x80000 < (unsigned char) b) + // expected-warning {{comparison of constant 524288 with expression of type 'unsigned char' is always false}}
         ((long) 0x80000 < b) +
         ((int) 0x80000 < b) +
         ((short) 0x80000 < b) +
         ((signed char) 0x80000 < b) +
         ((long) 0x80000 < (unsigned long) b) +
         ((int) 0x80000 < (unsigned int) b) +
         ((short) 0x80000 < (unsigned short) b) +
         ((signed char) 0x80000 < (unsigned char) b) +

         // (a,0x80000)
         (a == (unsigned long) 0x80000) +
         (a == (unsigned int) 0x80000) +
         (a == (unsigned short) 0x80000) +
         (a == (unsigned char) 0x80000) +
         ((long) a == 0x80000) +
         ((int) a == 0x80000) +
         ((short) a == 0x80000) + // expected-warning {{comparison of constant 524288 with expression of type 'short' is always false}}
         ((signed char) a == 0x80000) + // expected-warning {{comparison of constant 524288 with expression of type 'signed char' is always false}}
         ((long) a == (unsigned long) 0x80000) +
         ((int) a == (unsigned int) 0x80000) +
         ((short) a == (unsigned short) 0x80000) +
         ((signed char) a == (unsigned char) 0x80000) +
         (a < (unsigned long) 0x80000) +  // expected-warning {{comparison of integers of different signs}}
         (a < (unsigned int) 0x80000) +
         (a < (unsigned short) 0x80000) +
         (a < (unsigned char) 0x80000) +
         ((long) a < 0x80000) +
         ((int) a < 0x80000) +
         ((short) a < 0x80000) + // expected-warning {{comparison of constant 524288 with expression of type 'short' is always true}}
         ((signed char) a < 0x80000) + // expected-warning {{comparison of constant 524288 with expression of type 'signed char' is always true}}
         ((long) a < (unsigned long) 0x80000) +  // expected-warning {{comparison of integers of different signs}}
         ((int) a < (unsigned int) 0x80000) +  // expected-warning {{comparison of integers of different signs}}
         ((short) a < (unsigned short) 0x80000) +
         ((signed char) a < (unsigned char) 0x80000) +

         10
    ;
}

int test1(int i) {
  enum en { zero };
  return i > zero;
}

enum E { e };
void test2(int i, void *vp) {
  if (test1 == vp) { } // expected-warning{{equality comparison between function pointer and void pointer}}
  if (test1 == e) { } // expected-error{{comparison between pointer and integer}}
  if (vp < 0) { }
  if (test1 < e) { } // expected-error{{comparison between pointer and integer}}
}

// PR7536
static const unsigned int kMax = 0;
int pr7536() {
  return (kMax > 0);
}

// -Wsign-compare should not warn when ?: operands have different signedness.
// This will be caught by -Wsign-conversion
void test3() {
  unsigned long a;
  signed long b;
  (void) (true ? a : b);
  (void) (true ? (unsigned int)a : (signed int)b);
  (void) (true ? b : a);
  (void) (true ? (unsigned char)b : (signed char)a);
}
