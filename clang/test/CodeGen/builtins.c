// RUN: %clang_cc1 -emit-llvm -o %t %s
// RUN: not grep __builtin %t
// RUN: %clang_cc1 -emit-llvm -triple armv7 -o %t %s
// RUN: not grep __builtin %t

// RUN: %clang_cc1 -emit-llvm -triple armv7-darwin-apple  -o - %s | FileCheck %s --check-prefixes=CHECK,LD64,LONG32
// RUN: %clang_cc1 -emit-llvm -triple arm64-darwin-apple  -o - %s | FileCheck %s --check-prefixes=CHECK,LD64,I128,LONG64,BITCOUNTG,BITCOUNTG-LE
// RUN: %clang_cc1 -emit-llvm -triple x86_64-darwin-apple -o - %s | FileCheck %s --check-prefixes=CHECK,LD80,I128,LONG64,BITCOUNTG,BITCOUNTG-LE
// RUN: %clang_cc1 -emit-llvm -triple x86_64-darwin-apple -o - %s -fexperimental-new-constant-interpreter | FileCheck --check-prefixes=CHECK,LD80,I128,LONG64,BITCOUNTG,BITCOUNTG-LE %s
// RUN: %clang_cc1 -emit-llvm -triple aarch64_be-apple-darwin -mlong-double-64 -o - %s | FileCheck %s --check-prefixes=CHECK,LD64,I128,LONG64,BITCOUNTG,BITCOUNTG-BE
// RUN: %clang_cc1 -emit-llvm -triple aarch64_be-apple-darwin -mlong-double-64 -O1 -o - %s | FileCheck %s --check-prefix=POPCOUNTG-BE-O1

int printf(const char *, ...);

void p(char *str, int x) {
  printf("%s: %d\n", str, x);
}
void q(char *str, double x) {
  printf("%s: %f\n", str, x);
}
void r(char *str, void *ptr) {
  printf("%s: %p\n", str, ptr);
}

int random(void);
int finite(double);

int main(void) {
  int N = random();
#define P(n,args) p(#n #args, __builtin_##n args)
#define Q(n,args) q(#n #args, __builtin_##n args)
#define R(n,args) r(#n #args, __builtin_##n args)
#define V(n,args) p(#n #args, (__builtin_##n args, 0))
  P(types_compatible_p, (int, float));
  P(choose_expr, (0, 10, 20));
  P(constant_p, (sizeof(10)));
  P(expect, (N == 12, 0));
  V(prefetch, (&N));
  V(prefetch, (&N, 1));
  V(prefetch, (&N, 1, 0));

  // Numeric Constants

  Q(huge_val, ());
  Q(huge_valf, ());
  Q(huge_vall, ());
  Q(inf, ());
  Q(inff, ());
  Q(infl, ());

  P(fpclassify, (0, 1, 2, 3, 4, 1.0));
  P(fpclassify, (0, 1, 2, 3, 4, 1.0f));
  P(fpclassify, (0, 1, 2, 3, 4, 1.0l));

  Q(nan, (""));
  Q(nanf, (""));
  Q(nanl, (""));
  Q(nans, (""));
  Q(nan, ("10"));
  Q(nanf, ("10"));
  Q(nanl, ("10"));
  Q(nans, ("10"));

  P(isgreater, (1., 2.));
  P(isgreaterequal, (1., 2.));
  P(isless, (1., 2.));
  P(islessequal, (1., 2.));
  P(islessgreater, (1., 2.));
  P(isunordered, (1., 2.));

  P(isinf, (1.));
  P(isinf_sign, (1.));
  P(isnan, (1.));
  P(isfinite, (1.));
  P(iszero, (1.));
  P(issubnormal, (1.));
  P(issignaling, (1.));
  P(isfpclass, (1., 1));

  Q(fmaximum_num, (1.0, 2.0));
  Q(fmaximum_numf, (1.0, 2.0));
  Q(fmaximum_numl, (1.0, 2.0));
  Q(fminimum_num, (1.0, 2.0));
  Q(fminimum_numf, (1.0, 2.0));
  Q(fminimum_numl, (1.0, 2.0));

  // Bitwise & Numeric Functions

  P(abs, (N));

  P(clz, (N));
  P(clzl, (N));
  P(clzll, (N));
  P(ctz, (N));
  P(ctzl, (N));
  P(ctzll, (N));
  P(ffs, (N));
  P(ffsl, (N));
  P(ffsll, (N));
  P(parity, (N));
  P(parityl, (N));
  P(parityll, (N));
  P(popcount, (N));
  P(popcountl, (N));
  P(popcountll, (N));
  Q(powi, (1.2f, N));
  Q(powif, (1.2f, N));
  Q(powil, (1.2f, N));

  // Lib functions
  int a, b, n = random(); // Avoid optimizing out.
  char s0[10], s1[] = "Hello";
  V(strcat, (s0, s1));
  V(strcmp, (s0, s1));
  V(strdup, (s0));
  V(strncat, (s0, s1, n));
  V(strndup, (s0, n));
  V(strchr, (s0, s1[0]));
  V(strrchr, (s0, s1[0]));
  V(strcpy, (s0, s1));
  V(strncpy, (s0, s1, n));
  V(sprintf, (s0, "%s", s1));
  V(snprintf, (s0, n, "%s", s1));

  // Object size checking
  V(__memset_chk, (s0, 0, sizeof s0, n));
  V(__memcpy_chk, (s0, s1, sizeof s0, n));
  V(__memmove_chk, (s0, s1, sizeof s0, n));
  V(__mempcpy_chk, (s0, s1, sizeof s0, n));
  V(__strncpy_chk, (s0, s1, sizeof s0, n));
  V(__strcpy_chk, (s0, s1, n));
  s0[0] = 0;
  V(__strcat_chk, (s0, s1, n));
  P(object_size, (s0, 0));
  P(object_size, (s0, 1));
  P(object_size, (s0, 2));
  P(object_size, (s0, 3));

  // Whatever
  P(bswapg, ((_Bool)N));
  P(bswapg, ((char)N));
  P(bswapg, ((short)N));
  P(bswapg, ((int)N));
  P(bswapg, ((unsigned long)N));
  P(bswapg, ((_BitInt(8))N));
  P(bswapg, ((_BitInt(16))N));
  P(bswapg, ((_BitInt(32))N));
  P(bswapg, ((_BitInt(64))N));
  P(bswapg, ((_BitInt(128))N));
  P(bswap16, (N));
  P(bswap32, (N));
  P(bswap64, (N));

  // CHECK: @llvm.bitreverse.i8
  // CHECK: @llvm.bitreverse.i16
  // CHECK: @llvm.bitreverse.i32
  // CHECK: @llvm.bitreverse.i64
  // CHECK: @llvm.bitreverse.i128
  P(bitreverseg, ((char)N));
  P(bitreverseg, ((short)N));
  P(bitreverseg, ((int)N));
  P(bitreverseg, ((unsigned long)N));
  P(bitreverseg, ((_BitInt(8))N));
  P(bitreverseg, ((_BitInt(16))N));
  P(bitreverseg, ((_BitInt(32))N));
  P(bitreverseg, ((_BitInt(64))N));
  P(bitreverseg, ((_BitInt(128))N));
  P(bitreverseg, (N));
  P(bitreverse8, (N));
  P(bitreverse16, (N));
  P(bitreverse32, (N));
  P(bitreverse64, (N));

  // FIXME
  // V(clear_cache, (&N, &N+1));
  V(trap, ());
  R(extract_return_addr, (&N));
  P(signbit, (1.0));

  R(launder, (&N));

  return 0;
}



void foo(void) {
 __builtin_strcat(0, 0);
}

// CHECK-LABEL: define{{.*}} void @bar(
void bar(void) {
  float f;
  double d;
  long double ld;

  f = __builtin_huge_valf();     // CHECK: float    +inf
  d = __builtin_huge_val();      // CHECK: double   +inf
  ld = __builtin_huge_vall();
      // While we can't manage the constants we use this test to give us LDTYPE
      // for the rest of the tests
      // LD80: [[LDTYPE:x86_fp80]] +inf
      // LD64: [[LDTYPE:double]] +inf
  f = __builtin_nanf("");        // CHECK: float    +qnan
  d = __builtin_nan("");         // CHECK: double   +qnan
  ld = __builtin_nanl("");       // CHECK: [[LDTYPE]] +qnan
  f = __builtin_nanf("0xAE98");  // CHECK: float    +nan(0xAE98)
  d = __builtin_nan("0xAE98");   // CHECK: double   +nan(0xAE98)
  ld = __builtin_nanl("0xAE98"); // CHECK: [[LDTYPE]] +nan(0xAE98)
  f = __builtin_nansf("");       // CHECK: float    +snan(0x200000)
  d = __builtin_nans("");        // CHECK: double   +snan(0x4000000000000)
  ld = __builtin_nansl("");
    // LD80: [[LDTYPE]] +snan(0x2000000000000000)
    // LD64: [[LDTYPE]] +snan(0x4000000000000)
  f = __builtin_nansf("0xAE98"); // CHECK: float    +snan(0xAE98)
  d = __builtin_nans("0xAE98");  // CHECK: double   +snan(0xAE98)
  ld = __builtin_nansl("0xAE98");// CHECK: [[LDTYPE]] +snan(0xAE98)

}
// CHECK: }

// CHECK-LABEL: define{{.*}} void @test_conditional_bzero
void test_conditional_bzero(void) {
  char dst[20];
  int _sz = 20, len = 20;
  return (_sz
          ? ((_sz >= len)
              ? __builtin_bzero(dst, len)
              : foo())
          : __builtin_bzero(dst, len));
  // CHECK: call void @llvm.memset
  // CHECK: call void @llvm.memset
  // CHECK-NOT: phi
}

// CHECK-LABEL: define{{.*}} void @test_conditional_bcopy
void test_conditional_bcopy(void) {
  char dst[20];
  char src[20];
  int _sz = 20, len = 20;
  return (_sz
          ? ((_sz >= len)
              ? __builtin_bcopy(src, dst, len)
              : foo())
          : __builtin_bcopy(src, dst, len));
  // CHECK: call void @llvm.memmove
  // CHECK: call void @llvm.memmove
  // CHECK-NOT: phi
}

// CHECK-LABEL: define{{.*}} void @test_float_builtins
void test_float_builtins(__fp16 *H, float F, double D, long double LD) {
  volatile int res;
  res = __builtin_isinf(*H);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f16(half {{.*}}, /* (inf) */ i32 516)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_isinf(F);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (inf) */ i32 516)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_isinf(D);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f64(double {{.*}}, /* (inf) */ i32 516)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_isinf(LD);
  // LD80: [[TMP:%.*]] = call i1 @llvm.is.fpclass.[[LDLLVMTY:f80]]([[LDTYPE]] {{.*}}, /* (inf) */ i32 516)
  // LD64: [[TMP:%.*]] = call i1 @llvm.is.fpclass.[[LDLLVMTY:f64]]([[LDTYPE]] {{.*}}, /* (inf) */ i32 516)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_isinf_sign(*H);
  // CHECK:  %[[ABS:.*]] = call half @llvm.fabs.f16(half %[[ARG:.*]])
  // CHECK:  %[[ISINF:.*]] = fcmp oeq half %[[ABS]], +inf
  // CHECK:  %[[BITCAST:.*]] = bitcast half %[[ARG]] to i16
  // CHECK:  %[[ISNEG:.*]] = icmp slt i16 %[[BITCAST]], 0
  // CHECK:  %[[SIGN:.*]] = select i1 %[[ISNEG]], i32 -1, i32 1
  // CHECK:  select i1 %[[ISINF]], i32 %[[SIGN]], i32 0

  res = __builtin_isinf_sign(F);
  // CHECK:  %[[ABS:.*]] = call float @llvm.fabs.f32(float %[[ARG:.*]])
  // CHECK:  %[[ISINF:.*]] = fcmp oeq float %[[ABS]], +inf
  // CHECK:  %[[BITCAST:.*]] = bitcast float %[[ARG]] to i32
  // CHECK:  %[[ISNEG:.*]] = icmp slt i32 %[[BITCAST]], 0
  // CHECK:  %[[SIGN:.*]] = select i1 %[[ISNEG]], i32 -1, i32 1
  // CHECK:  select i1 %[[ISINF]], i32 %[[SIGN]], i32 0

  res = __builtin_isinf_sign(D);
  // CHECK:  %[[ABS:.*]] = call double @llvm.fabs.f64(double %[[ARG:.*]])
  // CHECK:  %[[ISINF:.*]] = fcmp oeq double %[[ABS]], +inf
  // CHECK:  %[[BITCAST:.*]] = bitcast double %[[ARG]] to i64
  // CHECK:  %[[ISNEG:.*]] = icmp slt i64 %[[BITCAST]], 0
  // CHECK:  %[[SIGN:.*]] = select i1 %[[ISNEG]], i32 -1, i32 1
  // CHECK:  select i1 %[[ISINF]], i32 %[[SIGN]], i32 0

  res = __builtin_isinf_sign(LD);
  // CHECK:  %[[ABS:.*]] = call [[LDTYPE]] @llvm.fabs.[[LDLLVMTY]]([[LDTYPE]] %[[ARG:.*]])
  // CHECK:  %[[ISINF:.*]] = fcmp oeq [[LDTYPE]] %[[ABS]], +inf
  // LD80:   %[[BITCAST:.*]] = bitcast [[LDTYPE]] %[[ARG]] to [[LDINTTY:i80]]
  // LD64:   %[[BITCAST:.*]] = bitcast [[LDTYPE]] %[[ARG]] to [[LDINTTY:i64]]
  // CHECK:  %[[ISNEG:.*]] = icmp slt [[LDINTTY]] %[[BITCAST]], 0
  // CHECK:  %[[SIGN:.*]] = select i1 %[[ISNEG]], i32 -1, i32 1
  // CHECK:  select i1 %[[ISINF]], i32 %[[SIGN]], i32 0

  res = __builtin_isfinite(*H);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f16(half {{.*}}, /* (zero sub norm) */ i32 504)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_isfinite(F);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (zero sub norm) */ i32 504)
  // CHECK: zext i1 [[TMP]] to i32

  res = finite(D);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f64(double {{.*}}, /* (zero sub norm) */ i32 504)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_isnormal(*H);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f16(half {{.*}}, /* (norm) */ i32 264)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_isnormal(F);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (norm) */ i32 264)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_issubnormal(F);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (sub) */ i32 144)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_iszero(F);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (zero) */ i32 96)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_issignaling(F);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f32(float {{.*}}, /* (snan) */ i32 1)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_flt_rounds();
  // CHECK: call i32 @llvm.get.rounding(
}

// CHECK-LABEL: define{{.*}} void @test_float_builtin_ops
void test_float_builtin_ops(float F, double D, long double LD, int I) {
  volatile float resf;
  volatile double resd;
  volatile long double resld;
  volatile long int resli;
  volatile long long int reslli;

  resf = __builtin_fmodf(F,F);
  // CHECK: frem float

  resd = __builtin_fmod(D,D);
  // CHECK: frem double

  resld = __builtin_fmodl(LD,LD);
  // CHECK: frem [[LDTYPE]]

  resf = __builtin_fabsf(F);
  resd = __builtin_fabs(D);
  resld = __builtin_fabsl(LD);
  // CHECK: call float @llvm.fabs.f32(float
  // CHECK: call double @llvm.fabs.f64(double
  // CHECK: call [[LDTYPE]] @llvm.fabs.[[LDLLVMTY]]([[LDTYPE]]

  resf = __builtin_canonicalizef(F);
  resd = __builtin_canonicalize(D);
  resld = __builtin_canonicalizel(LD);
  // CHECK: call float @llvm.canonicalize.f32(float
  // CHECK: call double @llvm.canonicalize.f64(double
  // CHECK: call [[LDTYPE]] @llvm.canonicalize.[[LDLLVMTY]]([[LDTYPE]]

  resf = __builtin_fminf(F, F);
  // CHECK: call nsz float @llvm.minnum.f32

  resd = __builtin_fmin(D, D);
  // CHECK: call nsz double @llvm.minnum.f64

  resld = __builtin_fminl(LD, LD);
  // CHECK: call nsz [[LDTYPE]] @llvm.minnum.[[LDLLVMTY]]

  resf = __builtin_fmaxf(F, F);
  // CHECK: call nsz float @llvm.maxnum.f32

  resd = __builtin_fmax(D, D);
  // CHECK: call nsz double @llvm.maxnum.f64

  resld = __builtin_fmaxl(LD, LD);
  // CHECK: call nsz [[LDTYPE]] @llvm.maxnum.[[LDLLVMTY]]

  resf = __builtin_fminimum_numf(F, F);
  // CHECK: call float @llvm.minimumnum.f32

  resf = __builtin_fminimum_numf(I, I);
  // CHECK: sitofp i32 {{%[0-9]+}} to float
  // CHECK: sitofp i32 {{%[0-9]+}} to float
  // CHECK: call float @llvm.minimumnum.f32

  resf = __builtin_fminimum_numf(1.0, 2.0);
  // CHECK: store volatile float 1.000000e+00, ptr %resf

  resd = __builtin_fminimum_num(D, D);
  // CHECK: call double @llvm.minimumnum.f64

  resd = __builtin_fminimum_num(I, I);
  // CHECK: sitofp i32 {{%[0-9]+}} to double
  // CHECK: sitofp i32 {{%[0-9]+}} to double
  // CHECK: call double @llvm.minimumnum.f64

  resd = __builtin_fminimum_num(1.0, 2.0);
  // CHECK: store volatile double 1.000000e+00, ptr %resd

  //FIXME: __builtin_fminimum_numl is not supported well yet.
  resld = __builtin_fminimum_numl(1.0, 2.0);
  // LD80: store volatile x86_fp80 1.000000e+00, ptr %resld, align 16
  // LD64: store volatile double 1.000000e+00, ptr %resld

  resf = __builtin_fmaximum_numf(F, F);
  // CHECK: call float @llvm.maximumnum.f32

  resf = __builtin_fmaximum_numf(I, I);
  // CHECK: sitofp i32 {{%[0-9]+}} to float
  // CHECK: sitofp i32 {{%[0-9]+}} to float
  // CHECK: call float @llvm.maximumnum.f32

  resf = __builtin_fmaximum_numf(1.0, 2.0);
  // CHECK: store volatile float 2.000000e+00, ptr %resf

  resd = __builtin_fmaximum_num(D, D);
  // CHECK: call double @llvm.maximumnum.f64

  resd = __builtin_fmaximum_num(I, I);
  // CHECK: sitofp i32 {{%[0-9]+}} to double
  // CHECK: sitofp i32 {{%[0-9]+}} to double
  // CHECK: call double @llvm.maximumnum.f64

  resd = __builtin_fmaximum_num(1.0, 2.0);
  // CHECK: store volatile double 2.000000e+00, ptr %resd

  //FIXME: __builtin_fmaximum_numl is not supported well yet.
  resld = __builtin_fmaximum_numl(1.0, 2.0);
  // LD80: store volatile x86_fp80 2.000000e+00, ptr %resld, align 16
  // LD64: store volatile double 2.000000e+00, ptr %resld

  resf = __builtin_fabsf(F);
  // CHECK: call float @llvm.fabs.f32

  resd = __builtin_fabs(D);
  // CHECK: call double @llvm.fabs.f64

  resld = __builtin_fabsl(LD);
  // CHECK: call [[LDTYPE]] @llvm.fabs.[[LDLLVMTY]]

  resf = __builtin_copysignf(F, F);
  // CHECK: call float @llvm.copysign.f32

  resd = __builtin_copysign(D, D);
  // CHECK: call double @llvm.copysign.f64

  resld = __builtin_copysignl(LD, LD);
  // CHECK: call [[LDTYPE]] @llvm.copysign.[[LDLLVMTY]]

  resf = __builtin_ceilf(F);
  // CHECK: call float @llvm.ceil.f32

  resd = __builtin_ceil(D);
  // CHECK: call double @llvm.ceil.f64

  resld = __builtin_ceill(LD);
  // CHECK: call [[LDTYPE]] @llvm.ceil.[[LDLLVMTY]]

  resf = __builtin_floorf(F);
  // CHECK: call float @llvm.floor.f32

  resd = __builtin_floor(D);
  // CHECK: call double @llvm.floor.f64

  resld = __builtin_floorl(LD);
  // CHECK: call [[LDTYPE]] @llvm.floor.[[LDLLVMTY]]

  resf = __builtin_sqrtf(F);
  // CHECK: call float @llvm.sqrt.f32(

  resd = __builtin_sqrt(D);
  // CHECK: call double @llvm.sqrt.f64(

  resld = __builtin_sqrtl(LD);
  // CHECK: call [[LDTYPE]] @llvm.sqrt.[[LDLLVMTY]]

  resf = __builtin_truncf(F);
  // CHECK: call float @llvm.trunc.f32

  resd = __builtin_trunc(D);
  // CHECK: call double @llvm.trunc.f64

  resld = __builtin_truncl(LD);
  // CHECK: call [[LDTYPE]] @llvm.trunc.[[LDLLVMTY]]

  resf = __builtin_rintf(F);
  // CHECK: call float @llvm.rint.f32

  resd = __builtin_rint(D);
  // CHECK: call double @llvm.rint.f64

  resld = __builtin_rintl(LD);
  // CHECK: call [[LDTYPE]] @llvm.rint.[[LDLLVMTY]]

  resf = __builtin_nearbyintf(F);
  // CHECK: call float @llvm.nearbyint.f32

  resd = __builtin_nearbyint(D);
  // CHECK: call double @llvm.nearbyint.f64

  resld = __builtin_nearbyintl(LD);
  // CHECK: call [[LDTYPE]] @llvm.nearbyint.[[LDLLVMTY]]

  resf = __builtin_roundf(F);
  // CHECK: call float @llvm.round.f32

  resd = __builtin_round(D);
  // CHECK: call double @llvm.round.f64

  resld = __builtin_roundl(LD);
  // CHECK: call [[LDTYPE]] @llvm.round.[[LDLLVMTY]]

  resf = __builtin_roundevenf(F);
  // CHECK: call float @llvm.roundeven.f32

  resd = __builtin_roundeven(D);
  // CHECK: call double @llvm.roundeven.f64

  __asm__("foo_bar0:");
  // CHECK: foo_bar0
  resld = __builtin_roundevenl(LD);
  // CHECK: call [[LDTYPE]] @llvm.roundeven.[[LDLLVMTY]]
  
  __asm__("foo_bar1:");
// CHECK: foo_bar1
  resli = __builtin_lroundf (F);
  // LONG64: call [[LONGINTTY:i64]] @llvm.lround.[[LONGINTTY]].f32
  // LONG32: call [[LONGINTTY:i32]] @llvm.lround.[[LONGINTTY]].f32
// CHECK: foo_after
  __asm__("foo_after:");
  resli = __builtin_lround (D);
  // CHECK: call [[LONGINTTY]] @llvm.lround.[[LONGINTTY]].f64

  resli = __builtin_lroundl (LD);
  // CHECK: call [[LONGINTTY]] @llvm.lround.[[LONGINTTY]].[[LDLLVMTY]]

  resli = __builtin_lrintf (F);
  // CHECK: call [[LONGINTTY]] @llvm.lrint.[[LONGINTTY]].f32

  resli = __builtin_lrint (D);
  // CHECK: call [[LONGINTTY]] @llvm.lrint.[[LONGINTTY]].f64

  resli = __builtin_lrintl (LD);
  // CHECK: call [[LONGINTTY]] @llvm.lrint.[[LONGINTTY]].[[LDLLVMTY]]
}

// __builtin_longjmp isn't supported on all platforms, so only test it on X86.
#ifdef __x86_64__

// LD80-LABEL: define{{.*}} void @test_builtin_longjmp(ptr{{.*}}
void test_builtin_longjmp(void **buffer) {
  // LD80: [[LOAD:%[a-z0-9]+]] = load ptr, ptr
  // LD80-NEXT: call void @llvm.eh.sjlj.longjmp(ptr [[LOAD]])
  __builtin_longjmp(buffer, 1);
  // LD80-NEXT: unreachable
}

#endif

// CHECK-LABEL: define{{.*}} void @test_memory_builtins
void test_memory_builtins(int n) {
  // CHECK: call ptr @malloc
  void * p = __builtin_malloc(n);
  // CHECK: call void @free
  __builtin_free(p);
  // CHECK: call ptr @calloc
  p = __builtin_calloc(1, n);
  // CHECK: call ptr @realloc
  p = __builtin_realloc(p, n);
  // CHECK: call void @free
  __builtin_free(p);
}

// CHECK-LABEL: define{{.*}} i64 @test_builtin_readcyclecounter
long long test_builtin_readcyclecounter(void) {
  // CHECK: call i64 @llvm.readcyclecounter()
  return __builtin_readcyclecounter();
}

// CHECK-LABEL: define{{.*}} i64 @test_builtin_readsteadycounter
long long test_builtin_readsteadycounter(void) {
  // CHECK: call i64 @llvm.readsteadycounter()
  return __builtin_readsteadycounter();
}

/// __builtin_launder should be a NOP in C since there are no vtables.
// CHECK-LABEL: define{{.*}} void @test_builtin_launder
void test_builtin_launder(int *p) {
  // CHECK: [[TMP:%.*]] = load ptr,
  // CHECK-NOT: @llvm.launder
  // CHECK: store ptr [[TMP]],
  int *d = __builtin_launder(p);
}

#ifdef __SIZEOF_INT128__

// __warn_memset_zero_len should be NOP, see https://sourceware.org/bugzilla/show_bug.cgi?id=25399
// I128-LABEL: define{{.*}} void @test___warn_memset_zero_len
void test___warn_memset_zero_len(void) {
  // I128-NOT: @__warn_memset_zero_len
  __warn_memset_zero_len();
}

#endif

// POPCOUNTG-BE-O1-LABEL: define{{.*}} void @test_builtin_popcountg
// POPCOUNTG-BE-O1-NOT: @llvm.bitreverse
// POPCOUNTG-BE-O1: ret void
// BITCOUNTG-LABEL: define{{.*}} void @test_builtin_popcountg
void test_builtin_popcountg(unsigned char uc, unsigned short us,
                            unsigned int ui, unsigned long ul,
                            unsigned long long ull,
                            _Bool __attribute__((ext_vector_type(8))) vb8) {
  volatile int pop;
#if __aarch64__
  int x = 0;
  x = x * 2;
#endif
  //      BITCOUNTG: %{{[0-9]+}} = load i8, ptr %uc.addr, align 1
  // BITCOUNTG-NEXT: %{{[0-9]+}} = call i8 @llvm.ctpop.i8(i8 %{{[0-9]+}})
  // BITCOUNTG-NEXT: [[POP_UC:%.*]] = zext i8 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT: store volatile i32 [[POP_UC]], ptr %pop, align 4
  pop = __builtin_popcountg(uc);
  //      BITCOUNTG: %{{[0-9]+}} = load i16, ptr %us.addr, align 2
  // BITCOUNTG-NEXT: %{{[0-9]+}} = call i16 @llvm.ctpop.i16(i16 %{{[0-9]+}})
  // BITCOUNTG-NEXT: [[POP_US:%.*]] = zext i16 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT: store volatile i32 [[POP_US]], ptr %pop, align 4
  pop = __builtin_popcountg(us);
  //      BITCOUNTG: %{{[0-9]+}} = load i32, ptr %ui.addr, align 4
  // BITCOUNTG-NEXT: %{{[0-9]+}} = call i32 @llvm.ctpop.i32(i32 %{{[0-9]+}})
  // BITCOUNTG-NEXT: store volatile i32 %{{[0-9]+}}, ptr %pop, align 4
  pop = __builtin_popcountg(ui);
  // BITCOUNTG: %{{[0-9]+}} = load i64, ptr %ul.addr, align 8
  // BITCOUNTG-NEXT: %{{[0-9]+}} = call i64 @llvm.ctpop.i64(i64 %{{[0-9]+}})
  // BITCOUNTG-NEXT: [[POP_UL:%.*]] = trunc i64 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT: store volatile i32 [[POP_UL]], ptr %pop, align 4
  pop = __builtin_popcountg(ul);
  //      BITCOUNTG: %{{[0-9]+}} = load i64, ptr %ull.addr, align 8
  // BITCOUNTG-NEXT: %{{[0-9]+}} = call i64 @llvm.ctpop.i64(i64 %{{[0-9]+}})
  // BITCOUNTG-NEXT: [[POP_ULL:%.*]] = trunc i64 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT: store volatile i32 [[POP_ULL]], ptr %pop, align 4
  pop = __builtin_popcountg(ull);
  //         BITCOUNTG: [[POP_LOAD:%.*]] = load i8, ptr %vb8.addr, align 1
  //    BITCOUNTG-NEXT: [[POP_VEC:%.*]] = bitcast i8 [[POP_LOAD]] to <8 x i1>
  //    BITCOUNTG-NEXT: [[POP_BITS:%.*]] = bitcast <8 x i1> [[POP_VEC]] to i8
  // BITCOUNTG-LE-NEXT: call i8 @llvm.ctpop.i8(i8 [[POP_BITS]])
  // BITCOUNTG-BE-NEXT: [[POP_REVERSED:%.*]] = call i8 @llvm.bitreverse.i8(i8 [[POP_BITS]])
  // BITCOUNTG-BE-NEXT: call i8 @llvm.ctpop.i8(i8 [[POP_REVERSED]])
  //    BITCOUNTG-NEXT: [[POP_EXT:%.*]] = zext i8 %{{.*}} to i32
  //    BITCOUNTG-NEXT: store volatile i32 [[POP_EXT]], ptr %pop, align 4
  pop = __builtin_popcountg(vb8);
}

// BITCOUNTG-LABEL: define{{.*}} void @test_builtin_clzg
void test_builtin_clzg(unsigned char uc, unsigned short us, unsigned int ui,
                       unsigned long ul, unsigned long long ull,
                       signed char sc, short s, int i,
                       _Bool __attribute__((ext_vector_type(8))) vb8) {
  volatile int lz;
#if __aarch64__
  int x = 0;
  x = x * 2;
#endif
  //      BITCOUNTG:  %{{[0-9]+}} = load i8, ptr %uc.addr, align 1
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = call i8 @llvm.ctlz.i8(i8 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT:  [[CLZ_UC:%.*]] = zext i8 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT:  store volatile i32 [[CLZ_UC]], ptr %lz, align 4
  lz = __builtin_clzg(uc);
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = load i16, ptr %us.addr, align 2
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = call i16 @llvm.ctlz.i16(i16 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT:  [[CLZ_US:%.*]] = zext i16 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT:  store volatile i32 [[CLZ_US]], ptr %lz, align 4
  lz = __builtin_clzg(us);
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = load i32, ptr %ui.addr, align 4
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = call i32 @llvm.ctlz.i32(i32 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT:  store volatile i32 %{{[0-9]+}}, ptr %lz, align 4
  lz = __builtin_clzg(ui);
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = load i64, ptr %ul.addr, align 8
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = call i64 @llvm.ctlz.i64(i64 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT:  [[CLZ_UL:%.*]] = trunc i64 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT:  store volatile i32 [[CLZ_UL]], ptr %lz, align 4
  lz = __builtin_clzg(ul);
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = load i64, ptr %ull.addr, align 8
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = call i64 @llvm.ctlz.i64(i64 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT:  [[CLZ_ULL:%.*]] = trunc i64 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT:  store volatile i32 [[CLZ_ULL]], ptr %lz, align 4
  lz = __builtin_clzg(ull);
  //         BITCOUNTG: [[CLZ_LOAD:%.*]] = load i8, ptr %vb8.addr, align 1
  //    BITCOUNTG-NEXT: [[CLZ_VEC:%.*]] = bitcast i8 [[CLZ_LOAD]] to <8 x i1>
  //    BITCOUNTG-NEXT: [[CLZ_BITS:%.*]] = bitcast <8 x i1> [[CLZ_VEC]] to i8
  // BITCOUNTG-LE-NEXT: call i8 @llvm.ctlz.i8(i8 [[CLZ_BITS]], i1
  // BITCOUNTG-BE-NEXT: [[CLZ_REVERSED:%.*]] = call i8 @llvm.bitreverse.i8(i8 [[CLZ_BITS]])
  // BITCOUNTG-BE-NEXT: call i8 @llvm.ctlz.i8(i8 [[CLZ_REVERSED]], i1 false)
  //    BITCOUNTG-NEXT: [[CLZ_EXT:%.*]] = zext i8 %{{.*}} to i32
  //    BITCOUNTG-NEXT: store volatile i32 [[CLZ_EXT]], ptr %lz, align 4
  lz = __builtin_clzg(vb8);
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = load i8, ptr %uc.addr, align 1
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = call i8 @llvm.ctlz.i8(i8 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT:  [[CLZ_UC_FALLBACK_EXT:%.*]] = zext i8 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT:  [[CLZ_UC_FALLBACK_ISZERO:%.*]] = icmp eq i8 %{{[0-9]+}}, 0
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = load i8, ptr %sc.addr, align 1
  // BITCOUNTG-NEXT:  [[CLZ_UC_FALLBACK:%.*]] = sext i8 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT:  [[CLZ_UC_SELECT:%.*]] = select i1 [[CLZ_UC_FALLBACK_ISZERO]], i32 [[CLZ_UC_FALLBACK]], i32 [[CLZ_UC_FALLBACK_EXT]]
  // BITCOUNTG-NEXT:  store volatile i32 [[CLZ_UC_SELECT]], ptr %lz, align 4
  lz = __builtin_clzg(uc, sc);
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = load i16, ptr %us.addr, align 2
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = call i16 @llvm.ctlz.i16(i16 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT:  [[CLZ_US_FALLBACK_EXT:%.*]] = zext i16 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT:  [[CLZ_US_FALLBACK_ISZERO:%.*]] = icmp eq i16 %{{[0-9]+}}, 0
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = load i8, ptr %uc.addr, align 1
  // BITCOUNTG-NEXT:  [[CLZ_US_FALLBACK:%.*]] = zext i8 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT:  [[CLZ_US_SELECT:%.*]] = select i1 [[CLZ_US_FALLBACK_ISZERO]], i32 [[CLZ_US_FALLBACK]], i32 [[CLZ_US_FALLBACK_EXT]]
  // BITCOUNTG-NEXT:  store volatile i32 [[CLZ_US_SELECT]], ptr %lz, align 4
  lz = __builtin_clzg(us, uc);
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = load i32, ptr %ui.addr, align 4
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = call i32 @llvm.ctlz.i32(i32 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT:  [[CLZ_UI_FALLBACK_ISZERO:%.*]] = icmp eq i32 %{{[0-9]+}}, 0
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = load i16, ptr %s.addr, align 2
  // BITCOUNTG-NEXT:  [[CLZ_UI_FALLBACK:%.*]] = sext i16 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT:  [[CLZ_UI_SELECT:%.*]] = select i1 [[CLZ_UI_FALLBACK_ISZERO]], i32 [[CLZ_UI_FALLBACK]], i32 %{{[0-9]+}}
  // BITCOUNTG-NEXT:  store volatile i32 [[CLZ_UI_SELECT]], ptr %lz, align 4
  lz = __builtin_clzg(ui, s);
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = load i64, ptr %ul.addr, align 8
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = call i64 @llvm.ctlz.i64(i64 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT:  [[CLZ_UL_FALLBACK_EXT:%.*]] = trunc i64 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT:  [[CLZ_UL_FALLBACK_ISZERO:%.*]] = icmp eq i64 %{{[0-9]+}}, 0
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = load i16, ptr %us.addr, align 2
  // BITCOUNTG-NEXT:  [[CLZ_UL_FALLBACK:%.*]] = zext i16 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT:  [[CLZ_UL_SELECT:%.*]] = select i1 [[CLZ_UL_FALLBACK_ISZERO]], i32 [[CLZ_UL_FALLBACK]], i32 [[CLZ_UL_FALLBACK_EXT]]
  // BITCOUNTG-NEXT:  store volatile i32 [[CLZ_UL_SELECT]], ptr %lz, align 4
  lz = __builtin_clzg(ul, us);
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = load i64, ptr %ull.addr, align 8
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = call i64 @llvm.ctlz.i64(i64 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT:  [[CLZ_ULL_FALLBACK_EXT:%.*]] = trunc i64 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT:  [[CLZ_ULL_FALLBACK_ISZERO:%.*]] = icmp eq i64 %{{[0-9]+}}, 0
  // BITCOUNTG-NEXT:  %{{[0-9]+}} = load i32, ptr %i.addr, align 4
  // BITCOUNTG-NEXT:  [[CLZ_ULL_SELECT:%.*]] = select i1 [[CLZ_ULL_FALLBACK_ISZERO]], i32 %{{[0-9]+}}, i32 [[CLZ_ULL_FALLBACK_EXT]]
  // BITCOUNTG-NEXT:  store volatile i32 [[CLZ_ULL_SELECT]], ptr %lz, align 4
  lz = __builtin_clzg(ull, i);
  //         BITCOUNTG: [[CLZ_LOAD_WITH_FALLBACK:%.*]] = load i8, ptr %vb8.addr, align 1
  //    BITCOUNTG-NEXT: [[CLZ_VEC_WITH_FALLBACK:%.*]] = bitcast i8 [[CLZ_LOAD_WITH_FALLBACK]] to <8 x i1>
  //    BITCOUNTG-NEXT: [[CLZ_BITS_WITH_FALLBACK:%.*]] = bitcast <8 x i1> [[CLZ_VEC_WITH_FALLBACK]] to i8
  // BITCOUNTG-LE-NEXT: call i8 @llvm.ctlz.i8(i8 [[CLZ_BITS_WITH_FALLBACK]], i1 true)
  // BITCOUNTG-BE-NEXT: [[CLZ_REVERSED_WITH_FALLBACK:%.*]] = call i8 @llvm.bitreverse.i8(i8 [[CLZ_BITS_WITH_FALLBACK]])
  // BITCOUNTG-BE-NEXT: call i8 @llvm.ctlz.i8(i8 [[CLZ_REVERSED_WITH_FALLBACK]], i1 true)
  //    BITCOUNTG-NEXT: [[CLZ_EXT_WITH_FALLBACK:%.*]] = zext i8 %{{.*}} to i32
  // BITCOUNTG-LE-NEXT: icmp eq i8 [[CLZ_BITS_WITH_FALLBACK]], 0
  // BITCOUNTG-BE-NEXT: icmp eq i8 [[CLZ_REVERSED_WITH_FALLBACK]], 0
  //    BITCOUNTG-NEXT: [[CLZ_FALLBACK:%.*]] = load i32, ptr %i.addr, align 4
  //    BITCOUNTG-NEXT: [[CLZ_SELECT:%.*]] = select i1 %{{.*}}, i32 [[CLZ_FALLBACK]], i32 [[CLZ_EXT_WITH_FALLBACK]]
  //    BITCOUNTG-NEXT: store volatile i32 [[CLZ_SELECT]], ptr %lz, align 4
  lz = __builtin_clzg(vb8, i);
}

// BITCOUNTG-LABEL: define{{.*}} void @test_builtin_ctzg
void test_builtin_ctzg(unsigned char uc, unsigned short us, unsigned int ui,
                       unsigned long ul, unsigned long long ull,
                       signed char sc, short s, int i,
                       _Bool __attribute__((ext_vector_type(8))) vb8) {
  volatile int tz;
#if __aarch64__
  int x = 0;
  x = x * 2;
#endif
  //      BITCOUNTG: %{{[0-9]+}} = load i8, ptr %uc.addr, align 1
  // BITCOUNTG-NEXT: %{{[0-9]+}} = call i8 @llvm.cttz.i8(i8 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT: [[CTZ_UC:%.*]] = zext i8 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT: store volatile i32 [[CTZ_UC]], ptr %tz, align 4
  tz = __builtin_ctzg(uc);
  // BITCOUNTG-NEXT: %{{[0-9]+}} = load i16, ptr %us.addr, align 2
  // BITCOUNTG-NEXT: %{{[0-9]+}} = call i16 @llvm.cttz.i16(i16 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT: [[CTZ_US:%.*]] = zext i16 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT: store volatile i32 [[CTZ_US]], ptr %tz, align 4
  tz = __builtin_ctzg(us);
  // BITCOUNTG-NEXT: %{{[0-9]+}} = load i32, ptr %ui.addr, align 4
  // BITCOUNTG-NEXT: %{{[0-9]+}} = call i32 @llvm.cttz.i32(i32 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT: store volatile i32 %{{[0-9]+}}, ptr %tz, align 4
  tz = __builtin_ctzg(ui);
  // BITCOUNTG-NEXT: %{{[0-9]+}} = load i64, ptr %ul.addr, align 8
  // BITCOUNTG-NEXT: %{{[0-9]+}} = call i64 @llvm.cttz.i64(i64 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT: [[CTZ_UL:%.*]] = trunc i64 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT: store volatile i32 [[CTZ_UL]], ptr %tz, align 4
  tz = __builtin_ctzg(ul);
  // BITCOUNTG-NEXT: %{{[0-9]+}} = load i64, ptr %ull.addr, align 8
  // BITCOUNTG-NEXT: %{{[0-9]+}} = call i64 @llvm.cttz.i64(i64 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT: [[CTZ_ULL:%.*]] = trunc i64 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT: store volatile i32 [[CTZ_ULL]], ptr %tz, align 4
  tz = __builtin_ctzg(ull);
  //         BITCOUNTG: [[CTZ_LOAD:%.*]] = load i8, ptr %vb8.addr, align 1
  //    BITCOUNTG-NEXT: [[CTZ_VEC:%.*]] = bitcast i8 [[CTZ_LOAD]] to <8 x i1>
  //    BITCOUNTG-NEXT: [[CTZ_BITS:%.*]] = bitcast <8 x i1> [[CTZ_VEC]] to i8
  // BITCOUNTG-LE-NEXT: call i8 @llvm.cttz.i8(i8 [[CTZ_BITS]], i1
  // BITCOUNTG-BE-NEXT: [[CTZ_REVERSED:%.*]] = call i8 @llvm.bitreverse.i8(i8 [[CTZ_BITS]])
  // BITCOUNTG-BE-NEXT: call i8 @llvm.cttz.i8(i8 [[CTZ_REVERSED]], i1 false)
  //    BITCOUNTG-NEXT: [[CTZ_EXT:%.*]] = zext i8 %{{.*}} to i32
  //    BITCOUNTG-NEXT: store volatile i32 [[CTZ_EXT]], ptr %tz, align 4
  tz = __builtin_ctzg(vb8);
  // BITCOUNTG-NEXT: %{{[0-9]+}} = load i8, ptr %uc.addr, align 1
  // BITCOUNTG-NEXT: %{{[0-9]+}} = call i8 @llvm.cttz.i8(i8 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT: [[CTZ_UC_FALLBACK_EXT:%.*]] = zext i8 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT: [[CTZ_UC_FALLBACK_ISZERO:%.*]] = icmp eq i8 %{{[0-9]+}}, 0
  // BITCOUNTG-NEXT: %{{[0-9]+}} = load i8, ptr %sc.addr, align 1
  // BITCOUNTG-NEXT: [[CTZ_UC_FALLBACK:%.*]] = sext i8 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT: [[CTZ_UC_SELECT:%.*]] = select i1 [[CTZ_UC_FALLBACK_ISZERO]], i32 [[CTZ_UC_FALLBACK]], i32 [[CTZ_UC_FALLBACK_EXT]]
  // BITCOUNTG-NEXT: store volatile i32 [[CTZ_UC_SELECT]], ptr %tz, align 4
  tz = __builtin_ctzg(uc, sc);
  // BITCOUNTG-NEXT: %{{[0-9]+}} = load i16, ptr %us.addr, align 2
  // BITCOUNTG-NEXT: %{{[0-9]+}} = call i16 @llvm.cttz.i16(i16 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT: [[CTZ_US_FALLBACK_EXT:%.*]] = zext i16 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT: [[CTZ_US_FALLBACK_ISZERO:%.*]] = icmp eq i16 %{{[0-9]+}}, 0
  // BITCOUNTG-NEXT: %{{[0-9]+}} = load i8, ptr %uc.addr, align 1
  // BITCOUNTG-NEXT: [[CTZ_US_FALLBACK:%.*]] = zext i8 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT: [[CTZ_US_SELECT:%.*]] = select i1 [[CTZ_US_FALLBACK_ISZERO]], i32 [[CTZ_US_FALLBACK]], i32 [[CTZ_US_FALLBACK_EXT]]
  // BITCOUNTG-NEXT: store volatile i32 [[CTZ_US_SELECT]], ptr %tz, align 4
  tz = __builtin_ctzg(us, uc);
  // BITCOUNTG-NEXT: %{{[0-9]+}} = load i32, ptr %ui.addr, align 4
  // BITCOUNTG-NEXT: %{{[0-9]+}} = call i32 @llvm.cttz.i32(i32 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT: [[CTZ_UI_FALLBACK_ISZERO:%.*]] = icmp eq i32 %{{[0-9]+}}, 0
  // BITCOUNTG-NEXT: %{{[0-9]+}} = load i16, ptr %s.addr, align 2
  // BITCOUNTG-NEXT: [[CTZ_UI_FALLBACK:%.*]] = sext i16 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT: [[CTZ_UI_SELECT:%.*]] = select i1 [[CTZ_UI_FALLBACK_ISZERO]], i32 [[CTZ_UI_FALLBACK]], i32 %{{[0-9]+}}
  // BITCOUNTG-NEXT: store volatile i32 [[CTZ_UI_SELECT]], ptr %tz, align 4
  tz = __builtin_ctzg(ui, s);
  // BITCOUNTG-NEXT: %{{[0-9]+}} = load i64, ptr %ul.addr, align 8
  // BITCOUNTG-NEXT: %{{[0-9]+}} = call i64 @llvm.cttz.i64(i64 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT: [[CTZ_UL_FALLBACK_EXT:%.*]] = trunc i64 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT: [[CTZ_UL_FALLBACK_ISZERO:%.*]] = icmp eq i64 %{{[0-9]+}}, 0
  // BITCOUNTG-NEXT: %{{[0-9]+}} = load i16, ptr %us.addr, align 2
  // BITCOUNTG-NEXT: [[CTZ_UL_FALLBACK:%.*]] = zext i16 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT: [[CTZ_UL_SELECT:%.*]] = select i1 [[CTZ_UL_FALLBACK_ISZERO]], i32 [[CTZ_UL_FALLBACK]], i32 [[CTZ_UL_FALLBACK_EXT]]
  // BITCOUNTG-NEXT: store volatile i32 [[CTZ_UL_SELECT]], ptr %tz, align 4
  tz = __builtin_ctzg(ul, us);
  // BITCOUNTG-NEXT: %{{[0-9]+}} = load i64, ptr %ull.addr, align 8
  // BITCOUNTG-NEXT: %{{[0-9]+}} = call i64 @llvm.cttz.i64(i64 %{{[0-9]+}}, i1
  // BITCOUNTG-NEXT: [[CTZ_ULL_FALLBACK_EXT:%.*]] = trunc i64 %{{[0-9]+}} to i32
  // BITCOUNTG-NEXT: [[CTZ_ULL_FALLBACK_ISZERO:%.*]] = icmp eq i64 %{{[0-9]+}}, 0
  // BITCOUNTG-NEXT: %{{[0-9]+}} = load i32, ptr %i.addr, align 4
  // BITCOUNTG-NEXT: [[CTZ_ULL_SELECT:%.*]] = select i1 [[CTZ_ULL_FALLBACK_ISZERO]], i32 %{{[0-9]+}}, i32 [[CTZ_ULL_FALLBACK_EXT]]
  // BITCOUNTG-NEXT: store volatile i32 [[CTZ_ULL_SELECT]], ptr %tz, align 4
  tz = __builtin_ctzg(ull, i);
  //         BITCOUNTG: [[CTZ_LOAD_WITH_FALLBACK:%.*]] = load i8, ptr %vb8.addr, align 1
  //    BITCOUNTG-NEXT: [[CTZ_VEC_WITH_FALLBACK:%.*]] = bitcast i8 [[CTZ_LOAD_WITH_FALLBACK]] to <8 x i1>
  //    BITCOUNTG-NEXT: [[CTZ_BITS_WITH_FALLBACK:%.*]] = bitcast <8 x i1> [[CTZ_VEC_WITH_FALLBACK]] to i8
  // BITCOUNTG-LE-NEXT: call i8 @llvm.cttz.i8(i8 [[CTZ_BITS_WITH_FALLBACK]], i1 true)
  // BITCOUNTG-BE-NEXT: [[CTZ_REVERSED_WITH_FALLBACK:%.*]] = call i8 @llvm.bitreverse.i8(i8 [[CTZ_BITS_WITH_FALLBACK]])
  // BITCOUNTG-BE-NEXT: call i8 @llvm.cttz.i8(i8 [[CTZ_REVERSED_WITH_FALLBACK]], i1 true)
  //    BITCOUNTG-NEXT: [[CTZ_EXT_WITH_FALLBACK:%.*]] = zext i8 %{{.*}} to i32
  // BITCOUNTG-LE-NEXT: icmp eq i8 [[CTZ_BITS_WITH_FALLBACK]], 0
  // BITCOUNTG-BE-NEXT: icmp eq i8 [[CTZ_REVERSED_WITH_FALLBACK]], 0
  //    BITCOUNTG-NEXT: [[CTZ_FALLBACK:%.*]] = load i32, ptr %i.addr, align 4
  //    BITCOUNTG-NEXT: [[CTZ_SELECT:%.*]] = select i1 %{{.*}}, i32 [[CTZ_FALLBACK]], i32 [[CTZ_EXT_WITH_FALLBACK]]
  //    BITCOUNTG-NEXT: store volatile i32 [[CTZ_SELECT]], ptr %tz, align 4
  tz = __builtin_ctzg(vb8, i);
}

#ifdef __SIZEOF_INT128__

// I128-LABEL: define{{.*}} void @test_builtin_popcountg_i128
void test_builtin_popcountg_i128(unsigned __int128 ui128,
                                 unsigned _BitInt(128) ubi128) {
  volatile int pop;
  //      I128: %{{[0-9]+}} = load i128, ptr %ui128.addr, align 16
  // I128-NEXT: %{{[0-9]+}} = call i128 @llvm.ctpop.i128(i128 %{{[0-9]+}})
  // I128-NEXT: [[POP_UI128:%.*]] = trunc i128 %{{[0-9]+}} to i32
  // I128-NEXT: store volatile i32 [[POP_UI128]], ptr %pop, align 4
  pop = __builtin_popcountg(ui128);
  //      I128: %{{[0-9]+}} = load i128, ptr %ubi128.addr
  // I128-NEXT: %{{[0-9]+}} = call i128 @llvm.ctpop.i128(i128 %{{[0-9]+}})
  // I128-NEXT: [[POP_UBI128:%.*]] = trunc i128 %{{[0-9]+}} to i32
  // I128-NEXT: store volatile i32 [[POP_UBI128]], ptr %pop, align 4
  pop = __builtin_popcountg(ubi128);
}

// I128-LABEL: define{{.*}} void @test_builtin_clzg_i128
void test_builtin_clzg_i128(unsigned __int128 ui128,
                            unsigned _BitInt(128) ubi128, int i) {
  volatile int lz;
  //      I128: %{{[0-9]+}} = load i128, ptr %ui128.addr, align 16
  // I128-NEXT: %{{[0-9]+}} = call i128 @llvm.ctlz.i128(i128 %{{[0-9]+}}, i1
  // I128-NEXT: [[CLZ_UI128:%.*]] = trunc i128 %{{[0-9]+}} to i32
  // I128-NEXT: store volatile i32 [[CLZ_UI128]], ptr %lz, align 4
  lz = __builtin_clzg(ui128);
  // I128-NEXT: %{{[0-9]+}} = load i128, ptr %ubi128.addr
  // I128-NEXT: %{{[0-9]+}} = call i128 @llvm.ctlz.i128(i128 %{{[0-9]+}}, i1
  // I128-NEXT: [[CLZ_UBI128:%.*]] = trunc i128 %{{[0-9]+}} to i32
  // I128-NEXT: store volatile i32 [[CLZ_UBI128]], ptr %lz, align 4
  lz = __builtin_clzg(ubi128);
  // I128-NEXT: %{{[0-9]+}} = load i128, ptr %ui128.addr, align 16
  // I128-NEXT: %{{[0-9]+}} = call i128 @llvm.ctlz.i128(i128 %{{[0-9]+}}, i1
  // I128-NEXT: [[CLZ_UI128_FALLBACK_EXT:%.*]] = trunc i128 %{{[0-9]+}} to i32
  // I128-NEXT: [[CLZ_UI128_FALLBACK_ISZERO:%.*]] = icmp eq i128 %{{[0-9]+}}, 0
  // I128-NEXT: %{{[0-9]+}} = load i32, ptr %i.addr, align 4
  // I128-NEXT: [[CLZ_UI128_SELECT:%.*]] = select i1 [[CLZ_UI128_FALLBACK_ISZERO]], i32 %{{[0-9]+}}, i32 [[CLZ_UI128_FALLBACK_EXT]]
  // I128-NEXT: store volatile i32 [[CLZ_UI128_SELECT]], ptr %lz, align 4
  lz = __builtin_clzg(ui128, i);
  // I128-NEXT: %{{[0-9]+}} = load i128, ptr %ubi128.addr
  // I128-NEXT: %{{[0-9]+}} = call i128 @llvm.ctlz.i128(i128 %{{[0-9]+}}, i1
  // I128-NEXT: [[CLZ_UBI128_FALLBACK_EXT:%.*]] = trunc i128 %{{[0-9]+}} to i32
  // I128-NEXT: [[CLZ_UBI128_FALLBACK_ISZERO:%.*]] = icmp eq i128 %{{[0-9]+}}, 0
  // I128-NEXT: %{{[0-9]+}} = load i32, ptr %i.addr, align 4
  // I128-NEXT: [[CLZ_UBI128_SELECT:%.*]] = select i1 [[CLZ_UBI128_FALLBACK_ISZERO]], i32 %{{[0-9]+}}, i32 [[CLZ_UBI128_FALLBACK_EXT]]
  // I128-NEXT: store volatile i32 [[CLZ_UBI128_SELECT]], ptr %lz, align 4
  lz = __builtin_clzg(ubi128, i);
}

// I128-LABEL: define{{.*}} void @test_builtin_ctzg_i128
void test_builtin_ctzg_i128(unsigned __int128 ui128,
                            unsigned _BitInt(128) ubi128, int i) {
  volatile int tz;
  //      I128: %{{[0-9]+}} = load i128, ptr %ui128.addr, align 16
  // I128-NEXT: %{{[0-9]+}} = call i128 @llvm.cttz.i128(i128 %{{[0-9]+}}, i1
  // I128-NEXT: [[CTZ_UI128:%.*]] = trunc i128 %{{[0-9]+}} to i32
  // I128-NEXT: store volatile i32 [[CTZ_UI128]], ptr %tz, align 4
  tz = __builtin_ctzg(ui128);
  // I128-NEXT: %{{[0-9]+}} = load i128, ptr %ubi128.addr
  // I128-NEXT: %{{[0-9]+}} = call i128 @llvm.cttz.i128(i128 %{{[0-9]+}}, i1
  // I128-NEXT: [[CTZ_UBI128:%.*]] = trunc i128 %{{[0-9]+}} to i32
  // I128-NEXT: store volatile i32 [[CTZ_UBI128]], ptr %tz, align 4
  tz = __builtin_ctzg(ubi128);
  // I128-NEXT: %{{[0-9]+}} = load i128, ptr %ui128.addr, align 16
  // I128-NEXT: %{{[0-9]+}} = call i128 @llvm.cttz.i128(i128 %{{[0-9]+}}, i1
  // I128-NEXT: [[CTZ_UI128_FALLBACK_EXT:%.*]] = trunc i128 %{{[0-9]+}} to i32
  // I128-NEXT: [[CTZ_UI128_FALLBACK_ISZERO:%.*]] = icmp eq i128 %{{[0-9]+}}, 0
  // I128-NEXT: %{{[0-9]+}} = load i32, ptr %i.addr, align 4
  // I128-NEXT: [[CTZ_UI128_SELECT:%.*]] = select i1 [[CTZ_UI128_FALLBACK_ISZERO]], i32 %{{[0-9]+}}, i32 [[CTZ_UI128_FALLBACK_EXT]]
  // I128-NEXT: store volatile i32 [[CTZ_UI128_SELECT]], ptr %tz, align 4
  tz = __builtin_ctzg(ui128, i);
  // I128-NEXT: %{{[0-9]+}} = load i128, ptr %ubi128.addr
  // I128-NEXT: %{{[0-9]+}} = call i128 @llvm.cttz.i128(i128 %{{[0-9]+}}, i1
  // I128-NEXT: [[CTZ_UBI128_FALLBACK_EXT:%.*]] = trunc i128 %{{[0-9]+}} to i32
  // I128-NEXT: [[CTZ_UBI128_FALLBACK_ISZERO:%.*]] = icmp eq i128 %{{[0-9]+}}, 0
  // I128-NEXT: %{{[0-9]+}} = load i32, ptr %i.addr, align 4
  // I128-NEXT: [[CTZ_UBI128_SELECT:%.*]] = select i1 [[CTZ_UBI128_FALLBACK_ISZERO]], i32 %{{[0-9]+}}, i32 [[CTZ_UBI128_FALLBACK_EXT]]
  // I128-NEXT: store volatile i32 [[CTZ_UBI128_SELECT]], ptr %tz, align 4
  tz = __builtin_ctzg(ubi128, i);
}

#endif

#include <stdbool.h>
// CHECK-LABEL: define{{.*}} void @test_builtin_bswapg
void test_builtin_bswapg(unsigned char uc, unsigned short us, unsigned int ui,
                       unsigned long ul, unsigned long long ull, bool b,
#ifdef __SIZEOF_INT128__
                       unsigned __int128 ui128,
#endif
                       _BitInt(8) bi8,
                       _BitInt(16) bi16, _BitInt(32) bi32, 
                       _BitInt(64) bi64, _BitInt(128) bi128) {
#if __aarch64__
  int x = 0;
  x = x * 2;
#endif
  b = __builtin_bswapg(b);
  // CHECK: %{{.*}} = load i8, ptr %b.addr
  // CHECK: %{{.*}} = icmp ne i8 %{{.*}}, 0
  // CHECK: %{{.*}} = zext i1 %{{.*}} to i8
  // CHECK: store i8 %{{.*}}, ptr %b.addr
  uc = __builtin_bswapg(uc);
  // CHECK: %{{.*}} = load i8, ptr %uc.addr
  // CHECK: store i8 %{{.*}}, ptr %uc.addr
  us = __builtin_bswapg(us);
  // CHECK: call i16 @llvm.bswap.i16
  ui = __builtin_bswapg(ui);
  // CHECK: call i32 @llvm.bswap.i32
  ul = __builtin_bswapg(ul);
  // CHECK: call [[LONGINTTY]] @llvm.bswap.[[LONGINTTY]]
  ull = __builtin_bswapg(ull);
  // CHECK: call i64 @llvm.bswap.i64
#ifdef __SIZEOF_INT128__
  ui128 = __builtin_bswapg(ui128);
  // I128: call i128 @llvm.bswap.i128
#endif
  bi8 = __builtin_bswapg(bi8);
  // CHECK: [[BI8SWAP:%.*]] = load i8, ptr %bi8.addr, align 1
  // CHECK: store i8 [[BI8SWAP]], ptr %bi8.addr
  bi16 = __builtin_bswapg(bi16);
  // CHECK: call i16 @llvm.bswap.i16
  bi32 = __builtin_bswapg(bi32);
  // CHECK: call i32 @llvm.bswap.i32
  bi64 = __builtin_bswapg(bi64);
  // CHECK: call i64 @llvm.bswap.i64
  bi128 = __builtin_bswapg(bi128);
  // CHECK: call i128 @llvm.bswap.i128
}
// CHECK-LABEL: define{{.*}} void @test_builtin_bitreverseg
void test_builtin_bitreverseg(unsigned char uc, unsigned short us, unsigned int ui,
                       unsigned long ul, unsigned long long ull, bool b,
                       unsigned _BitInt(1) bi1,
#ifdef __SIZEOF_INT128__
                       unsigned __int128 ui128,
#endif
                       _BitInt(8) bi8,
                       _BitInt(16) bi16, _BitInt(32) bi32, 
                       _BitInt(64) bi64, _BitInt(128) bi128) {
#if __aarch64__
  int x = 0;
  x = x * 2;
#endif
  b = __builtin_bitreverseg(b);
  // CHECK: %{{.*}} = load i8, ptr %b.addr
  // CHECK: %{{.*}} = icmp ne i8 %{{.*}}, 0
  // CHECK: %{{.*}} = zext i1 %{{.*}} to i8
  // CHECK: store i8 %{{.*}}, ptr %b.addr
  bi1 = __builtin_bitreverseg(bi1);
  // CHECK: %{{.*}} = load i8, ptr %bi1.addr
  // CHECK: %{{.*}} = icmp ne i8 %{{.*}}, 0
  // CHECK: %{{.*}} = zext i1 %{{.*}} to i8
  // CHECK: store i8 %{{.*}}, ptr %bi1.addr
  uc = __builtin_bitreverseg(uc);
  // CHECK: %{{.*}} = load i8, ptr %uc.addr
  // CHECK: %{{.*}} = call i8 @llvm.bitreverse.i8(i8 %{{.*}})
  // CHECK: store i8 %{{.*}}, ptr %uc.addr
  us = __builtin_bitreverseg(us);
  // CHECK: call i16 @llvm.bitreverse.i16
  ui = __builtin_bitreverseg(ui);
  // CHECK: call i32 @llvm.bitreverse.i32
  ul = __builtin_bitreverseg(ul);
  // CHECK: call [[LONGINTTY]] @llvm.bitreverse.[[LONGINTTY]]
  ull = __builtin_bitreverseg(ull);
  // CHECK: call i64 @llvm.bitreverse.i64
#ifdef __SIZEOF_INT128__
  ui128 = __builtin_bitreverseg(ui128);
  // I128: call i128 @llvm.bitreverse.i128
#endif
  bi8 = __builtin_bitreverseg(bi8);
  // CHECK: %{{.*}} = load i8, ptr %bi8.addr, align 1
  // CHECK: %{{.*}} = call i8 @llvm.bitreverse.i8(i8 %{{.*}})
  // CHECK: store i8 %{{.*}}, ptr %bi8.addr
  bi16 = __builtin_bitreverseg(bi16);
  // CHECK: call i16 @llvm.bitreverse.i16
  bi32 = __builtin_bitreverseg(bi32);
  // CHECK: call i32 @llvm.bitreverse.i32
  bi64 = __builtin_bitreverseg(bi64);
  // CHECK: call i64 @llvm.bitreverse.i64
  bi128 = __builtin_bitreverseg(bi128);
  // CHECK: call i128 @llvm.bitreverse.i128
}
