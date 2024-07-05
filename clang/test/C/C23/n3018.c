// RUN: %clang_cc1 -std=c23 -verify -triple x86_64 -pedantic -Wno-conversion -Wno-constant-conversion %s

/* WG14 N3018: Full
 * The constexpr specifier for object definitions
 */

#define ULLONG_MAX (__LONG_LONG_MAX__*2ULL+1ULL)
#define UINT_MAX  (__INT_MAX__  *2U +1U)

void Example0() {
  constexpr unsigned int minusOne    = -1;
  // expected-error@-1 {{constexpr initializer evaluates to -1 which is not exactly representable in type 'const unsigned int'}}
  constexpr unsigned int uint_max    = -1U;
  constexpr double onethird          = 1.0/3.0;
  constexpr double onethirdtrunc     = (double)(1.0/3.0);

  constexpr char string[] = { "\xFF", };
  constexpr unsigned char ucstring[] = { "\xFF", };
  // expected-error@-1 {{constexpr initializer evaluates to -1 which is not exactly representable in type 'const unsigned char'}}
  constexpr char string1[] = { -1, 0, };
  constexpr unsigned char ucstring1[] = { -1, 0, };
  // expected-error@-1 {{constexpr initializer evaluates to -1 which is not exactly representable in type 'const unsigned char'}}

  // TODO: Make sure these work correctly once char8_t and _Decimal are supported
  // constexpr char8_t u8string[] = { 255, 0, }; // ok
  // constexpr char8_t u8string[]       = { u8"\xFF", };     // ok
  // constexpr _Decimal32 small         = DEC64_TRUE_MIN * 0;// constraint violation
}

void Example1() {
  constexpr int K = 47;
  enum {
      A = K,
  };
  constexpr int L = K;
  static int b    = K + 1;
  int array[K];
  _Static_assert(K == 47);
}

constexpr int K = 47;
static const int b = K + 1;

void Example2() {
  constexpr int A          = 42LL;
  constexpr signed short B = ULLONG_MAX;
  // expected-error@-1 {{constexpr initializer evaluates to 18446744073709551615 which is not exactly representable in type 'const short'}}
  constexpr float C        = 47u;

  constexpr float D = 432000000;
  constexpr float E = 1.0 / 3.0;
  // expected-error@-1 {{constexpr initializer evaluates to 3.333333e-01 which is not exactly representable in type 'const float'}}
  constexpr float F = 1.0f / 3.0f;
}


void Example3() {
  constexpr static unsigned short array[] = {
      3000,
      300000,
      // expected-error@-1 {{constexpr initializer evaluates to 300000 which is not exactly representable in type 'const unsigned short'}}
      -1
      // expected-error@-1 {{constexpr initializer evaluates to -1 which is not exactly representable in type 'const unsigned short'}}
  };

  constexpr static unsigned short array1[] = {
      3000,
      3000,
      -1
       // expected-error@-1 {{constexpr initializer evaluates to -1 which is not exactly representable in type 'const unsigned short'}}
  };

  struct S {
      int x, y;
  };
  constexpr struct S s = {
      .x = __INT_MAX__,
      .y = UINT_MAX,
      // expected-error@-1 {{constexpr initializer evaluates to 4294967295 which is not exactly representable in type 'int'}}
  };
}

void Example4() {
  struct s { void *p; };
  constexpr struct s A = { nullptr };
  constexpr struct s B = A;
}
