// RUN: %clang_cc1 -fsyntax-only -verify %s

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

char classify_nan     [__builtin_fpclassify(+1, -1, -1, -1, -1, __builtin_nan(""))];
char classify_snan    [__builtin_fpclassify(+1, -1, -1, -1, -1, __builtin_nans(""))];
char classify_inf     [__builtin_fpclassify(-1, +1, -1, -1, -1, __builtin_inf())];
char classify_neg_inf [__builtin_fpclassify(-1, +1, -1, -1, -1, -__builtin_inf())];
char classify_normal  [__builtin_fpclassify(-1, -1, +1, -1, -1, 1.539)];
char classify_normal2 [__builtin_fpclassify(-1, -1, +1, -1, -1, 1e-307)];
char classify_denorm  [__builtin_fpclassify(-1, -1, -1, +1, -1, 1e-308)];
char classify_denorm2 [__builtin_fpclassify(-1, -1, -1, +1, -1, -1e-308)];
char classify_zero    [__builtin_fpclassify(-1, -1, -1, -1, +1, 0.0)];
char classify_neg_zero[__builtin_fpclassify(-1, -1, -1, -1, +1, -0.0)];

char isinf_sign_noninf1[__builtin_isinf_sign(-0.0) == 0 ? 1 : -1];
char isinf_sign_noninf2[__builtin_isinf_sign(1e307) == 0 ? 1 : -1];
char isinf_sign_noninf3[__builtin_isinf_sign(__builtin_nan("")) == 0 ? 1 : -1];
char isinf_sign_noninf4[__builtin_isinf_sign(-436.) == 0 ? 1 : -1];
char isinf_sign_inf    [__builtin_isinf_sign(__builtin_inf()) == 1 ? 1 : -1];
char isinf_sign_neg_inf[__builtin_isinf_sign(-__builtin_inf()) == -1 ? 1 : -1];

char isinf_inf_pos[__builtin_isinf(__builtin_inf()) ? 1 : -1];
char isinf_pos    [!__builtin_isinf(1.0) ? 1 : -1];
char isinf_normf  [!__builtin_isinf(1e-37f) ? 1 : -1];
char isinf_denormf[!__builtin_isinf(1e-38f) ? 1 : -1];
char isinf_norm   [!__builtin_isinf(1e-307) ? 1 : -1];
char isinf_denorm [!__builtin_isinf(1e-308) ? 1 : -1];
char isinf_zero   [!__builtin_isinf(0.0) ? 1 : -1];
char isinf_negzero[!__builtin_isinf(-0.0) ? 1 : -1];
char isinf_neg    [!__builtin_isinf(-1.0) ? 1 : -1];
char isinf_inf_neg[__builtin_isinf(-__builtin_inf()) ? 1 : -1];
char isinf_nan    [!__builtin_isinf(__builtin_nan("")) ? 1 : -1];
char isinf_snan   [!__builtin_isinf(__builtin_nans("")) ? 1 : -1];

char isfinite_inf_pos[!__builtin_isfinite(__builtin_inf()) ? 1 : -1];
char isfinite_pos    [__builtin_isfinite(1.0) ? 1 : -1];
char isfinite_normf  [__builtin_isfinite(1e-37f) ? 1 : -1];
char isfinite_denormf[__builtin_isfinite(1e-38f) ? 1 : -1];
char isfinite_norm   [__builtin_isfinite(1e-307) ? 1 : -1];
char isfinite_denorm [__builtin_isfinite(1e-308) ? 1 : -1];
char isfinite_zero   [__builtin_isfinite(0.0) ? 1 : -1];
char isfinite_negzero[__builtin_isfinite(-0.0) ? 1 : -1];
char isfinite_neg    [__builtin_isfinite(-1.0) ? 1 : -1];
char isfinite_inf_neg[!__builtin_isfinite(-__builtin_inf()) ? 1 : -1];
char isfinite_nan    [!__builtin_isfinite(__builtin_nan("")) ? 1 : -1];
char isfinite_snan   [!__builtin_isfinite(__builtin_nans("")) ? 1 : -1];

char isnan_inf_pos[!__builtin_isnan(__builtin_inf()) ? 1 : -1];
char isnan_pos    [!__builtin_isnan(1.0) ? 1 : -1];
char isnan_normf  [!__builtin_isnan(1e-37f) ? 1 : -1];
char isnan_denormf[!__builtin_isnan(1e-38f) ? 1 : -1];
char isnan_norm   [!__builtin_isnan(1e-307) ? 1 : -1];
char isnan_denorm [!__builtin_isnan(1e-308) ? 1 : -1];
char isnan_zero   [!__builtin_isnan(0.0) ? 1 : -1];
char isnan_negzero[!__builtin_isnan(-0.0) ? 1 : -1];
char isnan_neg    [!__builtin_isnan(-1.0) ? 1 : -1];
char isnan_inf_neg[!__builtin_isnan(-__builtin_inf()) ? 1 : -1];
char isnan_nan    [__builtin_isnan(__builtin_nan("")) ? 1 : -1];
char isnan_snan   [__builtin_isnan(__builtin_nans("")) ? 1 : -1];

char isnormal_inf_pos[!__builtin_isnormal(__builtin_inf()) ? 1 : -1];
char isnormal_pos    [__builtin_isnormal(1.0) ? 1 : -1];
char isnormal_normf  [__builtin_isnormal(1e-37f) ? 1 : -1];
char isnormal_denormf[!__builtin_isnormal(1e-38f) ? 1 : -1];
char isnormal_norm   [__builtin_isnormal(1e-307) ? 1 : -1];
char isnormal_denorm [!__builtin_isnormal(1e-308) ? 1 : -1];
char isnormal_zero   [!__builtin_isnormal(0.0) ? 1 : -1];
char isnormal_negzero[!__builtin_isnormal(-0.0) ? 1 : -1];
char isnormal_neg    [__builtin_isnormal(-1.0) ? 1 : -1];
char isnormal_inf_neg[!__builtin_isnormal(-__builtin_inf()) ? 1 : -1];
char isnormal_nan    [!__builtin_isnormal(__builtin_nan("")) ? 1 : -1];
char isnormal_snan   [!__builtin_isnormal(__builtin_nans("")) ? 1 : -1];

//double       g19 = __builtin_powi(2.0, 4);
//float        g20 = __builtin_powif(2.0f, 4);
//long double  g21 = __builtin_powil(2.0L, 4);

#define BITSIZE(x) (sizeof(x) * 8)
char clz1[__builtin_clz(1) == BITSIZE(int) - 1 ? 1 : -1];
char clz2[__builtin_clz(7) == BITSIZE(int) - 3 ? 1 : -1];
char clz3[__builtin_clz(1 << (BITSIZE(int) - 1)) == 0 ? 1 : -1];
int clz4 = __builtin_clz(0); // expected-error {{not a compile-time constant}}
char clz5[__builtin_clzl(0xFL) == BITSIZE(long) - 4 ? 1 : -1];
char clz6[__builtin_clzll(0xFFLL) == BITSIZE(long long) - 8 ? 1 : -1];
char clz7[__builtin_clzs(0x1) == BITSIZE(short) - 1 ? 1 : -1];
char clz8[__builtin_clzs(0xf) == BITSIZE(short) - 4 ? 1 : -1];
char clz9[__builtin_clzs(0xfff) == BITSIZE(short) - 12 ? 1 : -1];

char ctz1[__builtin_ctz(1) == 0 ? 1 : -1];
char ctz2[__builtin_ctz(8) == 3 ? 1 : -1];
char ctz3[__builtin_ctz(1 << (BITSIZE(int) - 1)) == BITSIZE(int) - 1 ? 1 : -1];
int ctz4 = __builtin_ctz(0); // expected-error {{not a compile-time constant}}
char ctz5[__builtin_ctzl(0x10L) == 4 ? 1 : -1];
char ctz6[__builtin_ctzll(0x100LL) == 8 ? 1 : -1];
char ctz7[__builtin_ctzs(1 << (BITSIZE(short) - 1)) == BITSIZE(short) - 1 ? 1 : -1];

char popcount1[__builtin_popcount(0) == 0 ? 1 : -1];
char popcount2[__builtin_popcount(0xF0F0) == 8 ? 1 : -1];
char popcount3[__builtin_popcount(~0) == BITSIZE(int) ? 1 : -1];
char popcount4[__builtin_popcount(~0L) == BITSIZE(int) ? 1 : -1];
char popcount5[__builtin_popcountl(0L) == 0 ? 1 : -1];
char popcount6[__builtin_popcountl(0xF0F0L) == 8 ? 1 : -1];
char popcount7[__builtin_popcountl(~0L) == BITSIZE(long) ? 1 : -1];
char popcount8[__builtin_popcountll(0LL) == 0 ? 1 : -1];
char popcount9[__builtin_popcountll(0xF0F0LL) == 8 ? 1 : -1];
char popcount10[__builtin_popcountll(~0LL) == BITSIZE(long long) ? 1 : -1];

char parity1[__builtin_parity(0) == 0 ? 1 : -1];
char parity2[__builtin_parity(0xb821) == 0 ? 1 : -1];
char parity3[__builtin_parity(0xb822) == 0 ? 1 : -1];
char parity4[__builtin_parity(0xb823) == 1 ? 1 : -1];
char parity5[__builtin_parity(0xb824) == 0 ? 1 : -1];
char parity6[__builtin_parity(0xb825) == 1 ? 1 : -1];
char parity7[__builtin_parity(0xb826) == 1 ? 1 : -1];
char parity8[__builtin_parity(~0) == 0 ? 1 : -1];
char parity9[__builtin_parityl(1L << (BITSIZE(long) - 1)) == 1 ? 1 : -1];
char parity10[__builtin_parityll(1LL << (BITSIZE(long long) - 1)) == 1 ? 1 : -1];

char ffs1[__builtin_ffs(0) == 0 ? 1 : -1];
char ffs2[__builtin_ffs(1) == 1 ? 1 : -1];
char ffs3[__builtin_ffs(0xfbe71) == 1 ? 1 : -1];
char ffs4[__builtin_ffs(0xfbe70) == 5 ? 1 : -1];
char ffs5[__builtin_ffs(1U << (BITSIZE(int) - 1)) == BITSIZE(int) ? 1 : -1];
char ffs6[__builtin_ffsl(0x10L) == 5 ? 1 : -1];
char ffs7[__builtin_ffsll(0x100LL) == 9 ? 1 : -1];
#undef BITSIZE

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
