// RUN: %clang_analyze_cc1 -Wno-conversion -analyzer-checker=core,alpha.core.Conversion -verify %s

unsigned char U8;
signed char S8;

void assign(unsigned U, signed S) {
  if (S < -10)
    U8 = S; // expected-warning {{Loss of sign in implicit conversion}}
  if (U > 300)
    S8 = U; // expected-warning {{Loss of precision in implicit conversion}}
  if (S > 10)
    U8 = S;
  if (U < 200)
    S8 = U;
}

void init1() {
  long long A = 1LL << 60;
  short X = A; // expected-warning {{Loss of precision in implicit conversion}}
}

void relational(unsigned U, signed S) {
  if (S > 10) {
    if (U < S) {
    }
  }
  if (S < -10) {
    if (U < S) { // expected-warning {{Loss of sign in implicit conversion}}
    }
  }
}

void multiplication(unsigned U, signed S) {
  if (S > 5)
    S = U * S;
  if (S < -10)
    S = U * S; // expected-warning {{Loss of sign}}
}

void division(unsigned U, signed S) {
  if (S > 5)
    S = U / S;
  if (S < -10)
    S = U / S; // expected-warning {{Loss of sign}}
}

void dontwarn1(unsigned U, signed S) {
  U8 = S; // It might be known that S is always 0x00-0xff.
  S8 = U; // It might be known that U is always 0x00-0xff.

  U8 = -1;  // Explicit conversion.
  S8 = ~0U; // Explicit conversion.
  if (U > 300)
    U8 &= U; // No loss of precision since there is &=.
}

void dontwarn2(unsigned int U) {
  if (U <= 4294967295) {
  }
  if (U <= (2147483647 * 2U + 1U)) {
  }
}

void dontwarn3(int X) {
  S8 = X ? 'a' : 'b';
}

// don't warn for macros
#define DOSTUFF ({ unsigned X = 1000; U8 = X; })
void dontwarn4() {
  DOSTUFF;
}

// don't warn for calculations
// seen some fp. For instance:  c2 = (c2 >= 'A' && c2 <= 'Z') ? c2 - 'A' + 'a' : c2;
// there is a todo in the checker to handle calculations
void dontwarn5() {
  signed S = -32;
  U8 = S + 10;
}


// false positives..

int isascii(int c);
void falsePositive1() {
  char kb2[5];
  int X = 1000;
  if (isascii(X)) {
    // FIXME: should not warn here:
    kb2[0] = X; // expected-warning {{Loss of precision}}
  }
}


typedef struct FILE {} FILE; int getc(FILE *stream);
# define EOF (-1)
char reply_string[8192];
FILE *cin;
extern int dostuff (void);
int falsePositive2() {
  int c, n;
  int dig;
  char *cp = reply_string;
  int pflag = 0;
  int code;

  for (;;) {
    dig = n = code = 0;
    while ((c = getc(cin)) != '\n') {
      if (dig < 4 && dostuff())
        code = code * 10 + (c - '0');
      if (!pflag && code == 227)
        pflag = 1;
      if (n == 0)
        n = c;
      if (c == EOF)
        return(4);
      if (cp < &reply_string[sizeof(reply_string) - 1])
        // FIXME: should not warn here:
        *cp++ = c; // expected-warning {{Loss of precision}}
    }
  }
}

