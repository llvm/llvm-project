// RUN: %clang_analyze_cc1 %s \
// RUN:   -Wno-conversion -Wno-tautological-constant-compare \
// RUN:   -analyzer-checker=core,apiModeling,alpha.core.Conversion \
// RUN:   -verify

unsigned char U8;
signed char S8;

void assign(unsigned U, signed S) {
  if (S < -10)
    U8 = S; // expected-warning {{Loss of sign in implicit conversion}}
  if (U > 300)
    S8 = U; // expected-warning {{Loss of precision in implicit conversion}}
  if (S > 10)
    U8 = S; // no-warning
  if (U < 200)
    S8 = U; // no-warning
}

void addAssign() {
  unsigned long L = 1000;
  int I = -100;
  U8 += L; // expected-warning {{Loss of precision in implicit conversion}}
  L += I; // no-warning
}

void subAssign() {
  unsigned long L = 1000;
  int I = -100;
  U8 -= L; // expected-warning {{Loss of precision in implicit conversion}}
  L -= I; // no-warning
}

void mulAssign() {
  unsigned long L = 1000;
  int I = -1;
  U8 *= L; // expected-warning {{Loss of precision in implicit conversion}}
  L *= I;  // expected-warning {{Loss of sign in implicit conversion}}
  I = 10;
  L *= I; // no-warning
}

void divAssign() {
  unsigned long L = 1000;
  int I = -1;
  U8 /= L; // no-warning
  L /= I; // expected-warning {{Loss of sign in implicit conversion}}
}

void remAssign() {
  unsigned long L = 1000;
  int I = -1;
  U8 %= L; // no-warning
  L %= I; // expected-warning {{Loss of sign in implicit conversion}}
}

void andAssign() {
  unsigned long L = 1000;
  int I = -1;
  U8 &= L; // no-warning
  L &= I; // expected-warning {{Loss of sign in implicit conversion}}
}

void orAssign() {
  unsigned long L = 1000;
  int I = -1;
  U8 |= L; // expected-warning {{Loss of precision in implicit conversion}}
  L |= I;  // expected-warning {{Loss of sign in implicit conversion}}
}

void xorAssign() {
  unsigned long L = 1000;
  int I = -1;
  U8 ^= L; // expected-warning {{Loss of precision in implicit conversion}}
  L ^= I;  // expected-warning {{Loss of sign in implicit conversion}}
}

void init1() {
  long long A = 1LL << 60;
  short X = A; // expected-warning {{Loss of precision in implicit conversion}}
}

void relational(unsigned U, signed S) {
  if (S > 10) {
    if (U < S) { // no-warning
    }
  }
  if (S < -10) {
    if (U < S) { // expected-warning {{Loss of sign in implicit conversion}}
    }
  }
}

void multiplication(unsigned U, signed S) {
  if (S > 5)
    S = U * S; // no-warning
  if (S < -10)
    S = U * S; // expected-warning {{Loss of sign}}
}

void division(unsigned U, signed S) {
  if (S > 5)
    S = U / S; // no-warning
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

char dontwarn6(long long x) {
  long long y = 42;
  y += x;
  return y == 42;
}


// C library functions, handled via apiModeling.StdCLibraryFunctions

int isascii(int c);
void libraryFunction1() {
  char kb2[5];
  int X = 1000;
  if (isascii(X)) {
    kb2[0] = X; // no-warning
  }
}


typedef struct FILE {} FILE; int getc(FILE *stream);
# define EOF (-1)
char reply_string[8192];
FILE *cin;
extern int dostuff(void);
int libraryFunction2() {
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
        *cp++ = c; // no-warning
    }
  }
}

double floating_point(long long a, int b) {
  if (a > 1LL << 55) {
    double r = a; // expected-warning {{Loss of precision}}
    return r;
  } else if (b > 1 << 25) {
    float f = b; // expected-warning {{Loss of precision}}
    return f;
  }
  return 137;
}

double floating_point2() {
  int a = 1 << 24;
  long long b = 1LL << 53;
  float f = a; // no-warning
  double d = b; // no-warning
  return d - f;
}

int floating_point_3(unsigned long long a) {
  double b = a; // no-warning
  return 42;
}
