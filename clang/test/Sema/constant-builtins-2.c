// RUN: %clang_cc1 -fsyntax-only %s

// Math stuff

double       g0  = __builtin_huge_val();
float        g1  = __builtin_huge_valf();
long double  g2  = __builtin_huge_vall();

double       g3  = __builtin_inf();
float        g4  = __builtin_inff();
long double  g5  = __builtin_infl();

double       g6  = __builtin_nan("");
float        g7  = __builtin_nanf("");
long double  g8  = __builtin_nanl("");

// GCC constant folds these too (via native strtol):
//double       g6_1  = __builtin_nan("1");
//float        g7_1  = __builtin_nanf("1");
//long double  g8_1  = __builtin_nanl("1");

// APFloat doesn't have signalling NaN functions.
//double       g9  = __builtin_nans("");
//float        g10 = __builtin_nansf("");
//long double  g11 = __builtin_nansl("");

//int          g12 = __builtin_abs(-12);

double       g13 = __builtin_fabs(-12.);
double       g13_0 = __builtin_fabs(-0.);
double       g13_1 = __builtin_fabs(-__builtin_inf());
float        g14 = __builtin_fabsf(-12.f);
// GCC doesn't eat this one.
//long double  g15 = __builtin_fabsfl(-12.0L);

float        g16 = __builtin_copysign(1.0, -1.0);
double       g17 = __builtin_copysignf(1.0f, -1.0f);
long double  g18 = __builtin_copysignl(1.0L, -1.0L);

//double       g19 = __builtin_powi(2.0, 4);
//float        g20 = __builtin_powif(2.0f, 4);
//long double  g21 = __builtin_powil(2.0L, 4);

// GCC misc stuff

extern int f();

int h0 = __builtin_types_compatible_p(int, float);
//int h1 = __builtin_choose_expr(1, 10, f());
//int h2 = __builtin_expect(0, 0);
int h3 = __builtin_bswap16(0x1234) == 0x3412 ? 1 : f();
int h4 = __builtin_bswap32(0x1234) == 0x34120000 ? 1 : f();
int h5 = __builtin_bswap64(0x1234) == 0x3412000000000000 ? 1 : f();
extern long int bi0;
extern __typeof__(__builtin_expect(0, 0)) bi0;

// Strings
int array1[__builtin_strlen("ab\0cd")];
int array2[(sizeof(array1)/sizeof(int)) == 2? 1 : -1];
