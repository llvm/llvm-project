// RUN: %clang_cc1 -fsyntax-only -verify %s

// Math stuff

double       g0  = __builtin_huge_val();
float        g1  = __builtin_huge_valf();
long double  g2  = __builtin_huge_vall();
#if defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)
__float128   g2_2 = __builtin_huge_valf128();
#endif

double       g3  = __builtin_inf();
float        g4  = __builtin_inff();
long double  g5  = __builtin_infl();
#if defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)
__float128   g5_2 = __builtin_inff128();
#endif

double       g6  = __builtin_nan("");
float        g7  = __builtin_nanf("");
long double  g8  = __builtin_nanl("");
#if defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)
__float128   g8_2 = __builtin_nanf128("");
#endif

// GCC constant folds these too (via native strtol):
//double       g6_1  = __builtin_nan("1");
//float        g7_1  = __builtin_nanf("1");
//long double  g8_1  = __builtin_nanl("1");

double       g9  = __builtin_nans("");
float        g10 = __builtin_nansf("");
long double  g11 = __builtin_nansl("");
#if defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)
__float128   g11_2 = __builtin_nansf128("");
#endif

int g12 = __builtin_abs(-12);

double       g13 = __builtin_fabs(-12.);
double       g13_0 = __builtin_fabs(-0.);
double       g13_1 = __builtin_fabs(-__builtin_inf());
float        g14 = __builtin_fabsf(-12.f);
// GCC doesn't eat this one.
//long double  g15 = __builtin_fabsfl(-12.0L);
#if defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)
__float128   g15_2 = __builtin_fabsf128(-12.q);
#endif

float        g16 = __builtin_copysign(1.0, -1.0);
double       g17 = __builtin_copysignf(1.0f, -1.0f);
long double  g18 = __builtin_copysignl(1.0L, -1.0L);
#if defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)
__float128   g18_2 = __builtin_copysignf128(1.0q, -1.0q);
#endif

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

char iszero_inf_pos[!__builtin_iszero(__builtin_inf()) ? 1 : -1];
char iszero_pos    [!__builtin_iszero(1.0) ? 1 : -1];
char iszero_normf  [!__builtin_iszero(1e-37f) ? 1 : -1];
char iszero_denormf[!__builtin_iszero(1e-38f) ? 1 : -1];
char iszero_norm   [!__builtin_iszero(1e-307) ? 1 : -1];
char iszero_denorm [!__builtin_iszero(1e-308) ? 1 : -1];
char iszero_zero   [__builtin_iszero(0.0) ? 1 : -1];
char iszero_negzero[__builtin_iszero(-0.0) ? 1 : -1];
char iszero_neg    [!__builtin_iszero(-1.0) ? 1 : -1];
char iszero_inf_neg[!__builtin_iszero(-__builtin_inf()) ? 1 : -1];
char iszero_nan    [!__builtin_iszero(__builtin_nan("")) ? 1 : -1];
char iszero_snan   [!__builtin_iszero(__builtin_nans("")) ? 1 : -1];

char issubnormal_inf_pos[!__builtin_issubnormal(__builtin_inf()) ? 1 : -1];
char issubnormal_pos    [!__builtin_issubnormal(1.0) ? 1 : -1];
char issubnormal_normf  [!__builtin_issubnormal(1e-37f) ? 1 : -1];
char issubnormal_denormf[__builtin_issubnormal(1e-38f) ? 1 : -1];
char issubnormal_norm   [!__builtin_issubnormal(1e-307) ? 1 : -1];
char issubnormal_denorm [__builtin_issubnormal(1e-308) ? 1 : -1];
char issubnormal_zero   [!__builtin_issubnormal(0.0) ? 1 : -1];
char issubnormal_negzero[!__builtin_issubnormal(-0.0) ? 1 : -1];
char issubnormal_neg    [!__builtin_issubnormal(-1.0) ? 1 : -1];
char issubnormal_inf_neg[!__builtin_issubnormal(-__builtin_inf()) ? 1 : -1];
char issubnormal_nan    [!__builtin_issubnormal(__builtin_nan("")) ? 1 : -1];
char issubnormal_snan   [!__builtin_issubnormal(__builtin_nans("")) ? 1 : -1];

char issignaling_inf_pos[!__builtin_issignaling(__builtin_inf()) ? 1 : -1];
char issignaling_pos    [!__builtin_issignaling(1.0) ? 1 : -1];
char issignaling_normf  [!__builtin_issignaling(1e-37f) ? 1 : -1];
char issignaling_denormf[!__builtin_issignaling(1e-38f) ? 1 : -1];
char issignaling_norm   [!__builtin_issignaling(1e-307) ? 1 : -1];
char issignaling_denorm [!__builtin_issignaling(1e-308) ? 1 : -1];
char issignaling_zero   [!__builtin_issignaling(0.0) ? 1 : -1];
char issignaling_negzero[!__builtin_issignaling(-0.0) ? 1 : -1];
char issignaling_neg    [!__builtin_issignaling(-1.0) ? 1 : -1];
char issignaling_inf_neg[!__builtin_issignaling(-__builtin_inf()) ? 1 : -1];
char issignaling_nan    [!__builtin_issignaling(__builtin_nan("")) ? 1 : -1];
char issignaling_snan   [__builtin_issignaling(__builtin_nans("")) ? 1 : -1];

char isfpclass_inf_pos_0[__builtin_isfpclass(__builtin_inf(), 0x0200) ? 1 : -1]; // fcPosInf
char isfpclass_inf_pos_1[!__builtin_isfpclass(__builtin_inff(), 0x0004) ? 1 : -1]; // fcNegInf
char isfpclass_inf_pos_2[__builtin_isfpclass(__builtin_infl(), 0x0207) ? 1 : -1]; // fcSNan|fcQNan|fcNegInf|fcPosInf
char isfpclass_inf_pos_3[!__builtin_isfpclass(__builtin_inf(), 0x01F8) ? 1 : -1]; // fcFinite
char isfpclass_pos_0    [__builtin_isfpclass(1.0, 0x0100) ? 1 : -1]; // fcPosNormal
char isfpclass_pos_1    [!__builtin_isfpclass(1.0f, 0x0008) ? 1 : -1]; // fcNegNormal
char isfpclass_pos_2    [__builtin_isfpclass(1.0L, 0x01F8) ? 1 : -1]; // fcFinite
char isfpclass_pos_3    [!__builtin_isfpclass(1.0, 0x0003) ? 1 : -1]; // fcSNan|fcQNan
char isfpclass_pdenorm_0[__builtin_isfpclass(1.0e-40f, 0x0080) ? 1 : -1]; // fcPosSubnormal
char isfpclass_pdenorm_1[__builtin_isfpclass(1.0e-310, 0x01F8) ? 1 : -1]; // fcFinite
char isfpclass_pdenorm_2[!__builtin_isfpclass(1.0e-40f, 0x003C) ? 1 : -1]; // fcNegative
char isfpclass_pdenorm_3[!__builtin_isfpclass(1.0e-310, 0x0207) ? 1 : -1]; // ~fcFinite
char isfpclass_pzero_0  [__builtin_isfpclass(0.0f, 0x0060) ? 1 : -1]; // fcZero
char isfpclass_pzero_1  [__builtin_isfpclass(0.0, 0x01F8) ? 1 : -1]; // fcFinite
char isfpclass_pzero_2  [!__builtin_isfpclass(0.0L, 0x0020) ? 1 : -1]; // fcNegZero
char isfpclass_pzero_3  [!__builtin_isfpclass(0.0, 0x0003) ? 1 : -1]; // fcNan
char isfpclass_nzero_0  [__builtin_isfpclass(-0.0f, 0x0060) ? 1 : -1]; // fcZero
char isfpclass_nzero_1  [__builtin_isfpclass(-0.0, 0x01F8) ? 1 : -1]; // fcFinite
char isfpclass_nzero_2  [!__builtin_isfpclass(-0.0L, 0x0040) ? 1 : -1]; // fcPosZero
char isfpclass_nzero_3  [!__builtin_isfpclass(-0.0, 0x0003) ? 1 : -1]; // fcNan
char isfpclass_ndenorm_0[__builtin_isfpclass(-1.0e-40f, 0x0010) ? 1 : -1]; // fcNegSubnormal
char isfpclass_ndenorm_1[__builtin_isfpclass(-1.0e-310, 0x01F8) ? 1 : -1]; // fcFinite
char isfpclass_ndenorm_2[!__builtin_isfpclass(-1.0e-40f, 0x03C0) ? 1 : -1]; // fcPositive
char isfpclass_ndenorm_3[!__builtin_isfpclass(-1.0e-310, 0x0207) ? 1 : -1]; // ~fcFinite
char isfpclass_neg_0    [__builtin_isfpclass(-1.0, 0x0008) ? 1 : -1]; // fcNegNormal
char isfpclass_neg_1    [!__builtin_isfpclass(-1.0f, 0x00100) ? 1 : -1]; // fcPosNormal
char isfpclass_neg_2    [__builtin_isfpclass(-1.0L, 0x01F8) ? 1 : -1]; // fcFinite
char isfpclass_neg_3    [!__builtin_isfpclass(-1.0, 0x0003) ? 1 : -1]; // fcSNan|fcQNan
char isfpclass_inf_neg_0[__builtin_isfpclass(-__builtin_inf(), 0x0004) ? 1 : -1]; // fcNegInf
char isfpclass_inf_neg_1[!__builtin_isfpclass(-__builtin_inff(), 0x0200) ? 1 : -1]; // fcPosInf
char isfpclass_inf_neg_2[__builtin_isfpclass(-__builtin_infl(), 0x0207) ? 1 : -1]; // ~fcFinite
char isfpclass_inf_neg_3[!__builtin_isfpclass(-__builtin_inf(), 0x03C0) ? 1 : -1]; // fcPositive
char isfpclass_qnan_0   [__builtin_isfpclass(__builtin_nan(""), 0x0002) ? 1 : -1]; // fcQNan
char isfpclass_qnan_1   [!__builtin_isfpclass(__builtin_nanf(""), 0x0001) ? 1 : -1]; // fcSNan
char isfpclass_qnan_2   [__builtin_isfpclass(__builtin_nanl(""), 0x0207) ? 1 : -1]; // ~fcFinite
char isfpclass_qnan_3   [!__builtin_isfpclass(__builtin_nan(""), 0x01F8) ? 1 : -1]; // fcFinite
char isfpclass_snan_0   [__builtin_isfpclass(__builtin_nansf(""), 0x0001) ? 1 : -1]; // fcSNan
char isfpclass_snan_1   [!__builtin_isfpclass(__builtin_nans(""), 0x0002) ? 1 : -1]; // fcQNan
char isfpclass_snan_2   [__builtin_isfpclass(__builtin_nansl(""), 0x0207) ? 1 : -1]; // ~fcFinite
char isfpclass_snan_3   [!__builtin_isfpclass(__builtin_nans(""), 0x01F8) ? 1 : -1]; // fcFinite

__extension__ _Static_assert(
  !__builtin_signbit(1.0) && __builtin_signbit(-1.0) && !__builtin_signbit(0.0) && __builtin_signbit(-0.0) &&
  !__builtin_signbitf(1.0f) && __builtin_signbitf(-1.0f) && !__builtin_signbitf(0.0f) && __builtin_signbitf(-0.0f) &&
  !__builtin_signbitl(1.0L) && __builtin_signbitf(-1.0L) && !__builtin_signbitf(0.0L) && __builtin_signbitf(-0.0L) &&
  !__builtin_signbit(1.0f) && __builtin_signbit(-1.0f) && !__builtin_signbit(0.0f) && __builtin_signbit(-0.0f) &&
  !__builtin_signbit(1.0L) && __builtin_signbit(-1.0L) && !__builtin_signbit(0.0L) && __builtin_signbit(-0.0L) &&
#if defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)
  !__builtin_signbit(1.0q) && __builtin_signbit(-1.0q) && !__builtin_signbit(0.0q) && __builtin_signbit(-0.0q) &&
#endif
  1, ""
);

#define LESS(X, Y) \
  !__builtin_isgreater(X, Y) && __builtin_isgreater(Y, X) &&             \
  !__builtin_isgreaterequal(X, Y) && __builtin_isgreaterequal(Y, X) &&   \
  __builtin_isless(X, Y) && !__builtin_isless(Y, X) &&                   \
  __builtin_islessequal(X, Y) && !__builtin_islessequal(Y, X) &&         \
  __builtin_islessgreater(X, Y) && __builtin_islessgreater(Y, X) &&      \
  !__builtin_isunordered(X, Y) && !__builtin_isunordered(Y, X)
#define EQUAL(X, Y) \
  !__builtin_isgreater(X, Y) && !__builtin_isgreater(Y, X) &&            \
  __builtin_isgreaterequal(X, Y) && __builtin_isgreaterequal(Y, X) &&    \
  !__builtin_isless(X, Y) && !__builtin_isless(Y, X) &&                  \
  __builtin_islessequal(X, Y) && __builtin_islessequal(Y, X) &&          \
  !__builtin_islessgreater(X, Y) && !__builtin_islessgreater(Y, X) &&    \
  !__builtin_isunordered(X, Y) && !__builtin_isunordered(Y, X)
#define UNORDERED(X, Y) \
  !__builtin_isgreater(X, Y) && !__builtin_isgreater(Y, X) &&            \
  !__builtin_isgreaterequal(X, Y) && !__builtin_isgreaterequal(Y, X) &&  \
  !__builtin_isless(X, Y) && !__builtin_isless(Y, X) &&                  \
  !__builtin_islessequal(X, Y) && !__builtin_islessequal(Y, X) &&        \
  !__builtin_islessgreater(X, Y) && !__builtin_islessgreater(Y, X) &&    \
  __builtin_isunordered(X, Y) && __builtin_isunordered(Y, X)

__extension__ _Static_assert(
  LESS(0.0, 1.0) && EQUAL(1.0, 1.0) && EQUAL(0.0, -0.0) &&
  UNORDERED(__builtin_nan(""), 1.0) && UNORDERED(__builtin_nan(""), __builtin_inf()) && LESS(0.0, __builtin_inf()) &&
  LESS(0.0f, 1.0f) && EQUAL(1.0f, 1.0f) && EQUAL(0.0f, -0.0f) &&
  UNORDERED(__builtin_nanf(""), 1.0f) && UNORDERED(__builtin_nanf(""), __builtin_inff()) && LESS(0.0f, __builtin_inff()) &&
  LESS(0.0L, 1.0L) && EQUAL(1.0L, 1.0L) && EQUAL(0.0L, -0.0L) &&
  UNORDERED(__builtin_nanl(""), 1.0L) && UNORDERED(__builtin_nanl(""), __builtin_infl()) && LESS(0.0L, __builtin_infl()) &&
#if defined(__FLOAT128__) || defined(__SIZEOF_FLOAT128__)
  LESS(0.0q, 1.0q) && EQUAL(1.0q, 1.0q) && EQUAL(0.0q, -0.0q) &&
#endif
  1, ""
);

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
int clz10 = __builtin_clzg((unsigned char)0); // expected-error {{not a compile-time constant}}
char clz11[__builtin_clzg((unsigned char)0, 42) == 42 ? 1 : -1];
char clz12[__builtin_clzg((unsigned char)0x1) == BITSIZE(char) - 1 ? 1 : -1];
char clz13[__builtin_clzg((unsigned char)0x1, 42) == BITSIZE(char) - 1 ? 1 : -1];
char clz14[__builtin_clzg((unsigned char)0xf) == BITSIZE(char) - 4 ? 1 : -1];
char clz15[__builtin_clzg((unsigned char)0xf, 42) == BITSIZE(char) - 4 ? 1 : -1];
char clz16[__builtin_clzg((unsigned char)(1 << (BITSIZE(char) - 1))) == 0 ? 1 : -1];
char clz17[__builtin_clzg((unsigned char)(1 << (BITSIZE(char) - 1)), 42) == 0 ? 1 : -1];
int clz18 = __builtin_clzg((unsigned short)0); // expected-error {{not a compile-time constant}}
char clz19[__builtin_clzg((unsigned short)0, 42) == 42 ? 1 : -1];
char clz20[__builtin_clzg((unsigned short)0x1) == BITSIZE(short) - 1 ? 1 : -1];
char clz21[__builtin_clzg((unsigned short)0x1, 42) == BITSIZE(short) - 1 ? 1 : -1];
char clz22[__builtin_clzg((unsigned short)0xf) == BITSIZE(short) - 4 ? 1 : -1];
char clz23[__builtin_clzg((unsigned short)0xf, 42) == BITSIZE(short) - 4 ? 1 : -1];
char clz24[__builtin_clzg((unsigned short)(1 << (BITSIZE(short) - 1))) == 0 ? 1 : -1];
char clz25[__builtin_clzg((unsigned short)(1 << (BITSIZE(short) - 1)), 42) == 0 ? 1 : -1];
int clz26 = __builtin_clzg(0U); // expected-error {{not a compile-time constant}}
char clz27[__builtin_clzg(0U, 42) == 42 ? 1 : -1];
char clz28[__builtin_clzg(0x1U) == BITSIZE(int) - 1 ? 1 : -1];
char clz29[__builtin_clzg(0x1U, 42) == BITSIZE(int) - 1 ? 1 : -1];
char clz30[__builtin_clzg(0xfU) == BITSIZE(int) - 4 ? 1 : -1];
char clz31[__builtin_clzg(0xfU, 42) == BITSIZE(int) - 4 ? 1 : -1];
char clz32[__builtin_clzg(1U << (BITSIZE(int) - 1)) == 0 ? 1 : -1];
char clz33[__builtin_clzg(1U << (BITSIZE(int) - 1), 42) == 0 ? 1 : -1];
int clz34 = __builtin_clzg(0UL); // expected-error {{not a compile-time constant}}
char clz35[__builtin_clzg(0UL, 42) == 42 ? 1 : -1];
char clz36[__builtin_clzg(0x1UL) == BITSIZE(long) - 1 ? 1 : -1];
char clz37[__builtin_clzg(0x1UL, 42) == BITSIZE(long) - 1 ? 1 : -1];
char clz38[__builtin_clzg(0xfUL) == BITSIZE(long) - 4 ? 1 : -1];
char clz39[__builtin_clzg(0xfUL, 42) == BITSIZE(long) - 4 ? 1 : -1];
char clz40[__builtin_clzg(1UL << (BITSIZE(long) - 1)) == 0 ? 1 : -1];
char clz41[__builtin_clzg(1UL << (BITSIZE(long) - 1), 42) == 0 ? 1 : -1];
int clz42 = __builtin_clzg(0ULL); // expected-error {{not a compile-time constant}}
char clz43[__builtin_clzg(0ULL, 42) == 42 ? 1 : -1];
char clz44[__builtin_clzg(0x1ULL) == BITSIZE(long long) - 1 ? 1 : -1];
char clz45[__builtin_clzg(0x1ULL, 42) == BITSIZE(long long) - 1 ? 1 : -1];
char clz46[__builtin_clzg(0xfULL) == BITSIZE(long long) - 4 ? 1 : -1];
char clz47[__builtin_clzg(0xfULL, 42) == BITSIZE(long long) - 4 ? 1 : -1];
char clz48[__builtin_clzg(1ULL << (BITSIZE(long long) - 1)) == 0 ? 1 : -1];
char clz49[__builtin_clzg(1ULL << (BITSIZE(long long) - 1), 42) == 0 ? 1 : -1];
#ifdef __SIZEOF_INT128__
int clz50 = __builtin_clzg((unsigned __int128)0); // expected-error {{not a compile-time constant}}
char clz51[__builtin_clzg((unsigned __int128)0, 42) == 42 ? 1 : -1];
char clz52[__builtin_clzg((unsigned __int128)0x1) == BITSIZE(__int128) - 1 ? 1 : -1];
char clz53[__builtin_clzg((unsigned __int128)0x1, 42) == BITSIZE(__int128) - 1 ? 1 : -1];
char clz54[__builtin_clzg((unsigned __int128)0xf) == BITSIZE(__int128) - 4 ? 1 : -1];
char clz55[__builtin_clzg((unsigned __int128)0xf, 42) == BITSIZE(__int128) - 4 ? 1 : -1];
char clz56[__builtin_clzg((unsigned __int128)(1 << (BITSIZE(__int128) - 1))) == 0 ? 1 : -1]; // expected-error {{variable length array declaration not allowed at file scope}}
char clz57[__builtin_clzg((unsigned __int128)(1 << (BITSIZE(__int128) - 1)), 42) == 0 ? 1 : -1]; // expected-error {{variable length array declaration not allowed at file scope}}
#endif
int clz58 = __builtin_clzg((unsigned _BitInt(128))0); // expected-error {{not a compile-time constant}}
char clz59[__builtin_clzg((unsigned _BitInt(128))0, 42) == 42 ? 1 : -1];
char clz60[__builtin_clzg((unsigned _BitInt(128))0x1) == BITSIZE(_BitInt(128)) - 1 ? 1 : -1];
char clz61[__builtin_clzg((unsigned _BitInt(128))0x1, 42) == BITSIZE(_BitInt(128)) - 1 ? 1 : -1];
char clz62[__builtin_clzg((unsigned _BitInt(128))0xf) == BITSIZE(_BitInt(128)) - 4 ? 1 : -1];
char clz63[__builtin_clzg((unsigned _BitInt(128))0xf, 42) == BITSIZE(_BitInt(128)) - 4 ? 1 : -1];
char clz64[__builtin_clzg((unsigned _BitInt(128))(1 << (BITSIZE(_BitInt(128)) - 1))) == 0 ? 1 : -1]; // expected-error {{variable length array declaration not allowed at file scope}}
char clz65[__builtin_clzg((unsigned _BitInt(128))(1 << (BITSIZE(_BitInt(128)) - 1)), 42) == 0 ? 1 : -1]; // expected-error {{variable length array declaration not allowed at file scope}}

char ctz1[__builtin_ctz(1) == 0 ? 1 : -1];
char ctz2[__builtin_ctz(8) == 3 ? 1 : -1];
char ctz3[__builtin_ctz(1 << (BITSIZE(int) - 1)) == BITSIZE(int) - 1 ? 1 : -1];
int ctz4 = __builtin_ctz(0); // expected-error {{not a compile-time constant}}
char ctz5[__builtin_ctzl(0x10L) == 4 ? 1 : -1];
char ctz6[__builtin_ctzll(0x100LL) == 8 ? 1 : -1];
char ctz7[__builtin_ctzs(1 << (BITSIZE(short) - 1)) == BITSIZE(short) - 1 ? 1 : -1];
int ctz8 = __builtin_ctzg((unsigned char)0); // expected-error {{not a compile-time constant}}
char ctz9[__builtin_ctzg((unsigned char)0, 42) == 42 ? 1 : -1];
char ctz10[__builtin_ctzg((unsigned char)0x1) == 0 ? 1 : -1];
char ctz11[__builtin_ctzg((unsigned char)0x1, 42) == 0 ? 1 : -1];
char ctz12[__builtin_ctzg((unsigned char)0x10) == 4 ? 1 : -1];
char ctz13[__builtin_ctzg((unsigned char)0x10, 42) == 4 ? 1 : -1];
char ctz14[__builtin_ctzg((unsigned char)(1 << (BITSIZE(char) - 1))) == BITSIZE(char) - 1 ? 1 : -1];
char ctz15[__builtin_ctzg((unsigned char)(1 << (BITSIZE(char) - 1)), 42) == BITSIZE(char) - 1 ? 1 : -1];
int ctz16 = __builtin_ctzg((unsigned short)0); // expected-error {{not a compile-time constant}}
char ctz17[__builtin_ctzg((unsigned short)0, 42) == 42 ? 1 : -1];
char ctz18[__builtin_ctzg((unsigned short)0x1) == 0 ? 1 : -1];
char ctz19[__builtin_ctzg((unsigned short)0x1, 42) == 0 ? 1 : -1];
char ctz20[__builtin_ctzg((unsigned short)0x10) == 4 ? 1 : -1];
char ctz21[__builtin_ctzg((unsigned short)0x10, 42) == 4 ? 1 : -1];
char ctz22[__builtin_ctzg((unsigned short)(1 << (BITSIZE(short) - 1))) == BITSIZE(short) - 1 ? 1 : -1];
char ctz23[__builtin_ctzg((unsigned short)(1 << (BITSIZE(short) - 1)), 42) == BITSIZE(short) - 1 ? 1 : -1];
int ctz24 = __builtin_ctzg(0U); // expected-error {{not a compile-time constant}}
char ctz25[__builtin_ctzg(0U, 42) == 42 ? 1 : -1];
char ctz26[__builtin_ctzg(0x1U) == 0 ? 1 : -1];
char ctz27[__builtin_ctzg(0x1U, 42) == 0 ? 1 : -1];
char ctz28[__builtin_ctzg(0x10U) == 4 ? 1 : -1];
char ctz29[__builtin_ctzg(0x10U, 42) == 4 ? 1 : -1];
char ctz30[__builtin_ctzg(1U << (BITSIZE(int) - 1)) == BITSIZE(int) - 1 ? 1 : -1];
char ctz31[__builtin_ctzg(1U << (BITSIZE(int) - 1), 42) == BITSIZE(int) - 1 ? 1 : -1];
int ctz32 = __builtin_ctzg(0UL); // expected-error {{not a compile-time constant}}
char ctz33[__builtin_ctzg(0UL, 42) == 42 ? 1 : -1];
char ctz34[__builtin_ctzg(0x1UL) == 0 ? 1 : -1];
char ctz35[__builtin_ctzg(0x1UL, 42) == 0 ? 1 : -1];
char ctz36[__builtin_ctzg(0x10UL) == 4 ? 1 : -1];
char ctz37[__builtin_ctzg(0x10UL, 42) == 4 ? 1 : -1];
char ctz38[__builtin_ctzg(1UL << (BITSIZE(long) - 1)) == BITSIZE(long) - 1 ? 1 : -1];
char ctz39[__builtin_ctzg(1UL << (BITSIZE(long) - 1), 42) == BITSIZE(long) - 1 ? 1 : -1];
int ctz40 = __builtin_ctzg(0ULL); // expected-error {{not a compile-time constant}}
char ctz41[__builtin_ctzg(0ULL, 42) == 42 ? 1 : -1];
char ctz42[__builtin_ctzg(0x1ULL) == 0 ? 1 : -1];
char ctz43[__builtin_ctzg(0x1ULL, 42) == 0 ? 1 : -1];
char ctz44[__builtin_ctzg(0x10ULL) == 4 ? 1 : -1];
char ctz45[__builtin_ctzg(0x10ULL, 42) == 4 ? 1 : -1];
char ctz46[__builtin_ctzg(1ULL << (BITSIZE(long long) - 1)) == BITSIZE(long long) - 1 ? 1 : -1];
char ctz47[__builtin_ctzg(1ULL << (BITSIZE(long long) - 1), 42) == BITSIZE(long long) - 1 ? 1 : -1];
#ifdef __SIZEOF_INT128__
int ctz48 = __builtin_ctzg((unsigned __int128)0); // expected-error {{not a compile-time constant}}
char ctz49[__builtin_ctzg((unsigned __int128)0, 42) == 42 ? 1 : -1];
char ctz50[__builtin_ctzg((unsigned __int128)0x1) == 0 ? 1 : -1];
char ctz51[__builtin_ctzg((unsigned __int128)0x1, 42) == 0 ? 1 : -1];
char ctz52[__builtin_ctzg((unsigned __int128)0x10) == 4 ? 1 : -1];
char ctz53[__builtin_ctzg((unsigned __int128)0x10, 42) == 4 ? 1 : -1];
char ctz54[__builtin_ctzg((unsigned __int128)1 << (BITSIZE(__int128) - 1)) == BITSIZE(__int128) - 1 ? 1 : -1];
char ctz55[__builtin_ctzg((unsigned __int128)1 << (BITSIZE(__int128) - 1), 42) == BITSIZE(__int128) - 1 ? 1 : -1];
#endif
int ctz56 = __builtin_ctzg((unsigned _BitInt(128))0); // expected-error {{not a compile-time constant}}
char ctz57[__builtin_ctzg((unsigned _BitInt(128))0, 42) == 42 ? 1 : -1];
char ctz58[__builtin_ctzg((unsigned _BitInt(128))0x1) == 0 ? 1 : -1];
char ctz59[__builtin_ctzg((unsigned _BitInt(128))0x1, 42) == 0 ? 1 : -1];
char ctz60[__builtin_ctzg((unsigned _BitInt(128))0x10) == 4 ? 1 : -1];
char ctz61[__builtin_ctzg((unsigned _BitInt(128))0x10, 42) == 4 ? 1 : -1];
char ctz62[__builtin_ctzg((unsigned _BitInt(128))1 << (BITSIZE(_BitInt(128)) - 1)) == BITSIZE(_BitInt(128)) - 1 ? 1 : -1];
char ctz63[__builtin_ctzg((unsigned _BitInt(128))1 << (BITSIZE(_BitInt(128)) - 1), 42) == BITSIZE(_BitInt(128)) - 1 ? 1 : -1];

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
char popcount11[__builtin_popcountg(0U) == 0 ? 1 : -1];
char popcount12[__builtin_popcountg(0xF0F0U) == 8 ? 1 : -1];
char popcount13[__builtin_popcountg(~0U) == BITSIZE(int) ? 1 : -1];
char popcount14[__builtin_popcountg(~0UL) == BITSIZE(long) ? 1 : -1];
char popcount15[__builtin_popcountg(~0ULL) == BITSIZE(long long) ? 1 : -1];
#ifdef __SIZEOF_INT128__
char popcount16[__builtin_popcountg(~(unsigned __int128)0) == BITSIZE(__int128) ? 1 : -1];
#endif
char popcount17[__builtin_popcountg(~(unsigned _BitInt(128))0) == BITSIZE(_BitInt(128)) ? 1 : -1];

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

char bitreverse1[__builtin_bitreverse8(0x01) == 0x80 ? 1 : -1];
char bitreverse2[__builtin_bitreverse16(0x3C48) == 0x123C ? 1 : -1];
char bitreverse3[__builtin_bitreverse32(0x12345678) == 0x1E6A2C48 ? 1 : -1];
char bitreverse4[__builtin_bitreverse64(0x0123456789ABCDEFULL) == 0xF7B3D591E6A2C480 ? 1 : -1];

char rotateleft1[__builtin_rotateleft8(0x01, 5) == 0x20 ? 1 : -1];
char rotateleft2[__builtin_rotateleft16(0x3210, 11) == 0x8190 ? 1 : -1];
char rotateleft3[__builtin_rotateleft32(0x76543210, 22) == 0x841D950C ? 1 : -1];
char rotateleft4[__builtin_rotateleft64(0xFEDCBA9876543210ULL, 55) == 0x87F6E5D4C3B2A19ULL ? 1 : -1];

char rotateright1[__builtin_rotateright8(0x01, 5) == 0x08 ? 1 : -1];
char rotateright2[__builtin_rotateright16(0x3210, 11) == 0x4206 ? 1 : -1];
char rotateright3[__builtin_rotateright32(0x76543210, 22) == 0x50C841D9 ? 1 : -1];
char rotateright4[__builtin_rotateright64(0xFEDCBA9876543210ULL, 55) == 0xB97530ECA86421FDULL ? 1 : -1];

char ffs1[__builtin_ffs(0) == 0 ? 1 : -1];
char ffs2[__builtin_ffs(1) == 1 ? 1 : -1];
char ffs3[__builtin_ffs(0xfbe71) == 1 ? 1 : -1];
char ffs4[__builtin_ffs(0xfbe70) == 5 ? 1 : -1];
char ffs5[__builtin_ffs(1U << (BITSIZE(int) - 1)) == BITSIZE(int) ? 1 : -1];
char ffs6[__builtin_ffsl(0x10L) == 5 ? 1 : -1];
char ffs7[__builtin_ffsll(0x100LL) == 9 ? 1 : -1];

char clrsb1[__builtin_clrsb(0) == BITSIZE(int) - 1 ? 1 : -1];
char clrsb2[__builtin_clrsbl(0L) == BITSIZE(long) - 1 ? 1 : -1];
char clrsb3[__builtin_clrsbll(0LL) == BITSIZE(long long) - 1 ? 1 : -1];
char clrsb4[__builtin_clrsb(~0) == BITSIZE(int) - 1 ? 1 : -1];
char clrsb5[__builtin_clrsbl(~0L) == BITSIZE(long) - 1 ? 1 : -1];
char clrsb6[__builtin_clrsbll(~0LL) == BITSIZE(long long) - 1 ? 1 : -1];
char clrsb7[__builtin_clrsb(1) == BITSIZE(int) - 2 ? 1 : -1];
char clrsb8[__builtin_clrsb(~1) == BITSIZE(int) - 2 ? 1 : -1];
char clrsb9[__builtin_clrsb(1 << (BITSIZE(int) - 1)) == 0 ? 1 : -1];
char clrsb10[__builtin_clrsb(~(1 << (BITSIZE(int) - 1))) == 0 ? 1 : -1];
char clrsb11[__builtin_clrsb(0xf) == BITSIZE(int) - 5 ? 1 : -1];
char clrsb12[__builtin_clrsb(~0x1f) == BITSIZE(int) - 6 ? 1 : -1];

char abs1[__builtin_abs(-12)];
char abs2[__builtin_labs(-12L)];
char abs3[__builtin_llabs(-12LL)];
int abs4 = __builtin_abs(1 << (BITSIZE(int) - 1)); // expected-error {{not a compile-time constant}}
char abs5[__builtin_abs((1 << (BITSIZE(int) - 1)) + 1)];
long abs6 = __builtin_labs(1L << (BITSIZE(long) - 1)); // expected-error {{not a compile-time constant}}
long abs7 = __builtin_labs((1L << (BITSIZE(long) - 1)) + 1);
long long abs8 = __builtin_llabs(1LL << (BITSIZE(long long) - 1)); // expected-error {{not a compile-time constant}}
long long abs9 = __builtin_llabs((1LL << (BITSIZE(long long) - 1)) + 1);

#undef BITSIZE

// GCC misc stuff

extern int f(void);

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
