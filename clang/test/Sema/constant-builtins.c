// RUN: %clang_cc1 -fsyntax-only %s -verify -pedantic -std=c99
// RUN: %clang_cc1 -fsyntax-only %s -verify -pedantic -std=c11
// RUN: %clang_cc1 -fsyntax-only %s -verify -pedantic -std=c17
// RUN: %clang_cc1 -fsyntax-only %s -verify -pedantic
#if __STDC_VERSION__ >= 201112L
// expected-no-diagnostics
#endif

// Math stuff

float        g0 = __builtin_huge_val();
double       g1 = __builtin_huge_valf();
long double  g2 = __builtin_huge_vall();
float        g3 = __builtin_inf();
double       g4 = __builtin_inff();
long double  g5 = __builtin_infl();

// GCC misc stuff

extern int f(void);

int h0 = __builtin_types_compatible_p(int,float);
//int h1 = __builtin_choose_expr(1, 10, f());
//int h2 = __builtin_expect(0, 0);
int h3 = __builtin_bswap16(0x1234) == 0x3412 ? 1 : f();
int h4 = __builtin_bswap32(0x1234) == 0x34120000 ? 1 : f();
int h5 = __builtin_bswap64(0x1234) == 0x3412000000000000 ? 1 : f();

short somefunc(void);

short t = __builtin_constant_p(5353) ? 42 : somefunc();

// The calls to _Static_assert and _Generic produce warnings if the compiler default standard is < c11
#if __STDC_VERSION__ < 201112L
// expected-warning@+9 {{'_Static_assert' is a C11 extension}}
// expected-warning@+9 {{'_Static_assert' is a C11 extension}}
// expected-warning@+9 {{'_Static_assert' is a C11 extension}}
// expected-warning@+9 {{'_Static_assert' is a C11 extension}} expected-warning@+9 {{'_Generic' is a C11 extension}}
// expected-warning@+9 {{'_Static_assert' is a C11 extension}} expected-warning@+9 {{'_Generic' is a C11 extension}}
// expected-warning@+9 {{'_Static_assert' is a C11 extension}} expected-warning@+9 {{'_Generic' is a C11 extension}}
#endif

// PR44684
_Static_assert((__builtin_clz)(1u) >= 15, "");
_Static_assert((__builtin_popcount)(1u) == 1, "");
_Static_assert((__builtin_ctz)(2u) == 1, "");
_Static_assert(_Generic(1u,unsigned:__builtin_clz)(1u) >= 15, "");
_Static_assert(_Generic(1u,unsigned:__builtin_popcount)(1u) == 1, "");
_Static_assert(_Generic(1u,unsigned:__builtin_ctz)(2u) == 1, "");

#if __STDC_VERSION__ < 201112L
// expected-warning@+3 {{'_Static_assert' is a C11 extension}}
#endif
__SIZE_TYPE__ strlen(const char*);
_Static_assert((__builtin_constant_p(1) ? (***&strlen)("foo") : 0) == 3, "");
