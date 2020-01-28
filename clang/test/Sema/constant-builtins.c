// RUN: %clang_cc1 -fsyntax-only %s -verify -pedantic
// expected-no-diagnostics

// Math stuff

float        g0 = __builtin_huge_val();
double       g1 = __builtin_huge_valf();
long double  g2 = __builtin_huge_vall();
float        g3 = __builtin_inf();
double       g4 = __builtin_inff();
long double  g5 = __builtin_infl();

// GCC misc stuff

extern int f();

int h0 = __builtin_types_compatible_p(int,float);
//int h1 = __builtin_choose_expr(1, 10, f());
//int h2 = __builtin_expect(0, 0);
int h3 = __builtin_bswap16(0x1234) == 0x3412 ? 1 : f();
int h4 = __builtin_bswap32(0x1234) == 0x34120000 ? 1 : f();
int h5 = __builtin_bswap64(0x1234) == 0x3412000000000000 ? 1 : f();

short somefunc();

short t = __builtin_constant_p(5353) ? 42 : somefunc();


