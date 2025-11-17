// RUN: %clang_cc1 -emit-llvm -o %t %s
// RUN: not grep __builtin %t
// RUN: %clang_cc1 -emit-llvm -triple armv7 -o %t %s
// RUN: not grep __builtin %t

// RUN: %clang_cc1 -emit-llvm -triple armv7-darwin-apple  -o - %s | FileCheck %s --check-prefixes=CHECK,LD64,LONG32
// RUN: %clang_cc1 -emit-llvm -triple arm64-darwin-apple  -o - %s | FileCheck %s --check-prefixes=CHECK,LD64,I128,LONG64
// RUN: %clang_cc1 -emit-llvm -triple x86_64-darwin-apple -o - %s | FileCheck %s --check-prefixes=CHECK,LD80,I128,LONG64
// RUN: %clang_cc1 -emit-llvm -triple x86_64-darwin-apple -o - %s -fexperimental-new-constant-interpreter | FileCheck --check-prefixes=CHECK,LD80,I128,LONG64 %s

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

  // LLVM's hex representation of float constants is really unfortunate;
  // basically it does a float-to-double "conversion" and then prints the
  // hex form of that.  That gives us weird artifacts like exponents
  // that aren't numerically similar to the original exponent and
  // significand bit-patterns that are offset by three bits (because
  // the exponent was expanded from 8 bits to 11).
  //
  // 0xAE98 == 1010111010011000
  // 0x15D3 == 1010111010011

  f = __builtin_huge_valf();     // CHECK: float    0x7FF0000000000000
  d = __builtin_huge_val();      // CHECK: double   0x7FF0000000000000
  ld = __builtin_huge_vall();
      // While we can't manage the constants we use this test to give us LDTYPE
      // for the rest of the tests
      // LD80: [[LDTYPE:x86_fp80]] [[LDHUGE:0xK7FFF8000000000000000]]
      // LD64: [[LDTYPE:double]] [[LDHUGE:0x7FF0000000000000]]
  f = __builtin_nanf("");        // CHECK: float    0x7FF8000000000000
  d = __builtin_nan("");         // CHECK: double   0x7FF8000000000000
  ld = __builtin_nanl("");
      // LD80: [[LDTYPE]] 0xK7FFFC000000000000000
      // LD64: [[LDTYPE]] 0x7FF8000000000000
  f = __builtin_nanf("0xAE98");  // CHECK: float    0x7FF815D300000000
  d = __builtin_nan("0xAE98");   // CHECK: double   0x7FF800000000AE98
  ld = __builtin_nanl("0xAE98");
    // LD80: [[LDTYPE]] 0xK7FFFC00000000000AE98
    // LD64: [[LDTYPE]] 0x7FF800000000AE98
  f = __builtin_nansf("");       // CHECK: float    0x7FF4000000000000
  d = __builtin_nans("");        // CHECK: double   0x7FF4000000000000
  ld = __builtin_nansl("");
    // LD80: [[LDTYPE]] 0xK7FFFA000000000000000
    // LD64: [[LDTYPE]] 0x7FF4000000000000
  f = __builtin_nansf("0xAE98"); // CHECK: float    0x7FF015D300000000
  d = __builtin_nans("0xAE98");  // CHECK: double   0x7FF000000000AE98
  ld = __builtin_nansl("0xAE98");
    // LD80: [[LDTYPE]] 0xK7FFF800000000000AE98
    // LD64: [[LDTYPE]] 0x7FF000000000AE98

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
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f16(half {{.*}}, i32 516)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_isinf(F);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f32(float {{.*}}, i32 516)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_isinf(D);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f64(double {{.*}}, i32 516)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_isinf(LD);
  // LD80: [[TMP:%.*]] = call i1 @llvm.is.fpclass.[[LDLLVMTY:f80]]([[LDTYPE]] {{.*}}, i32 516)
  // LD64: [[TMP:%.*]] = call i1 @llvm.is.fpclass.[[LDLLVMTY:f64]]([[LDTYPE]] {{.*}}, i32 516)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_isinf_sign(*H);
  // CHECK:  %[[ABS:.*]] = call half @llvm.fabs.f16(half %[[ARG:.*]])
  // CHECK:  %[[ISINF:.*]] = fcmp oeq half %[[ABS]], 0xH7C00
  // CHECK:  %[[BITCAST:.*]] = bitcast half %[[ARG]] to i16
  // CHECK:  %[[ISNEG:.*]] = icmp slt i16 %[[BITCAST]], 0
  // CHECK:  %[[SIGN:.*]] = select i1 %[[ISNEG]], i32 -1, i32 1
  // CHECK:  select i1 %[[ISINF]], i32 %[[SIGN]], i32 0

  res = __builtin_isinf_sign(F);
  // CHECK:  %[[ABS:.*]] = call float @llvm.fabs.f32(float %[[ARG:.*]])
  // CHECK:  %[[ISINF:.*]] = fcmp oeq float %[[ABS]], 0x7FF0000000000000
  // CHECK:  %[[BITCAST:.*]] = bitcast float %[[ARG]] to i32
  // CHECK:  %[[ISNEG:.*]] = icmp slt i32 %[[BITCAST]], 0
  // CHECK:  %[[SIGN:.*]] = select i1 %[[ISNEG]], i32 -1, i32 1
  // CHECK:  select i1 %[[ISINF]], i32 %[[SIGN]], i32 0

  res = __builtin_isinf_sign(D);
  // CHECK:  %[[ABS:.*]] = call double @llvm.fabs.f64(double %[[ARG:.*]])
  // CHECK:  %[[ISINF:.*]] = fcmp oeq double %[[ABS]], 0x7FF0000000000000
  // CHECK:  %[[BITCAST:.*]] = bitcast double %[[ARG]] to i64
  // CHECK:  %[[ISNEG:.*]] = icmp slt i64 %[[BITCAST]], 0
  // CHECK:  %[[SIGN:.*]] = select i1 %[[ISNEG]], i32 -1, i32 1
  // CHECK:  select i1 %[[ISINF]], i32 %[[SIGN]], i32 0

  res = __builtin_isinf_sign(LD);
  // CHECK:  %[[ABS:.*]] = call [[LDTYPE]] @llvm.fabs.[[LDLLVMTY]]([[LDTYPE]] %[[ARG:.*]])
  // CHECK:  %[[ISINF:.*]] = fcmp oeq [[LDTYPE]] %[[ABS]], [[LDHUGE]]
  // LD80:   %[[BITCAST:.*]] = bitcast [[LDTYPE]] %[[ARG]] to [[LDINTTY:i80]]
  // LD64:   %[[BITCAST:.*]] = bitcast [[LDTYPE]] %[[ARG]] to [[LDINTTY:i64]]
  // CHECK:  %[[ISNEG:.*]] = icmp slt [[LDINTTY]] %[[BITCAST]], 0
  // CHECK:  %[[SIGN:.*]] = select i1 %[[ISNEG]], i32 -1, i32 1
  // CHECK:  select i1 %[[ISINF]], i32 %[[SIGN]], i32 0

  res = __builtin_isfinite(*H);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f16(half {{.*}}, i32 504)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_isfinite(F);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f32(float {{.*}}, i32 504)
  // CHECK: zext i1 [[TMP]] to i32

  res = finite(D);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f64(double {{.*}}, i32 504)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_isnormal(*H);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f16(half {{.*}}, i32 264)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_isnormal(F);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f32(float {{.*}}, i32 264)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_issubnormal(F);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f32(float {{.*}}, i32 144)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_iszero(F);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f32(float {{.*}}, i32 96)
  // CHECK: zext i1 [[TMP]] to i32

  res = __builtin_issignaling(F);
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f32(float {{.*}}, i32 1)
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
  // CHECK: call float @llvm.minnum.f32

  resd = __builtin_fmin(D, D);
  // CHECK: call double @llvm.minnum.f64

  resld = __builtin_fminl(LD, LD);
  // CHECK: call [[LDTYPE]] @llvm.minnum.[[LDLLVMTY]]

  resf = __builtin_fmaxf(F, F);
  // CHECK: call float @llvm.maxnum.f32

  resd = __builtin_fmax(D, D);
  // CHECK: call double @llvm.maxnum.f64

  resld = __builtin_fmaxl(LD, LD);
  // CHECK: call [[LDTYPE]] @llvm.maxnum.[[LDLLVMTY]]

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
  // LD80: store volatile x86_fp80 0xK3FFF8000000000000000, ptr %resld, align 16
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
  // LD80: store volatile x86_fp80 0xK40008000000000000000, ptr %resld, align 16
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

// I128-LABEL: define{{.*}} void @test_builtin_popcountg
void test_builtin_popcountg(unsigned char uc, unsigned short us,
                            unsigned int ui, unsigned long ul,
                            unsigned long long ull, unsigned __int128 ui128,
                            unsigned _BitInt(128) ubi128,
                            _Bool __attribute__((ext_vector_type(8))) vb8) {
  volatile int pop;
#if __aarch64__
  int x = 0;
  x = x * 2;
#endif
  //      I128: %2 = load i8, ptr %uc.addr, align 1
  // I128-NEXT: %3 = call i8 @llvm.ctpop.i8(i8 %2)
  // I128-NEXT: %cast = zext i8 %3 to i32
  // I128-NEXT: store volatile i32 %cast, ptr %pop, align 4
  pop = __builtin_popcountg(uc);
  //      I128: %4 = load i16, ptr %us.addr, align 2
  // I128-NEXT: %5 = call i16 @llvm.ctpop.i16(i16 %4)
  // I128-NEXT: %cast2 = zext i16 %5 to i32
  // I128-NEXT: store volatile i32 %cast2, ptr %pop, align 4
  pop = __builtin_popcountg(us);
  //      I128: %6 = load i32, ptr %ui.addr, align 4
  // I128-NEXT: %7 = call i32 @llvm.ctpop.i32(i32 %6)
  // I128-NEXT: store volatile i32 %7, ptr %pop, align 4
  pop = __builtin_popcountg(ui);
  // I128: %8 = load i64, ptr %ul.addr, align 8
  // I128-NEXT: %9 = call i64 @llvm.ctpop.i64(i64 %8)
  // I128-NEXT: %cast3 = trunc i64 %9 to i32
  // I128-NEXT: store volatile i32 %cast3, ptr %pop, align 4
  pop = __builtin_popcountg(ul);
  //      I128: %10 = load i64, ptr %ull.addr, align 8
  // I128-NEXT: %11 = call i64 @llvm.ctpop.i64(i64 %10)
  // I128-NEXT: %cast4 = trunc i64 %11 to i32
  // I128-NEXT: store volatile i32 %cast4, ptr %pop, align 4
  pop = __builtin_popcountg(ull);
  //      I128: %12 = load i128, ptr %ui128.addr, align 16
  // I128-NEXT: %13 = call i128 @llvm.ctpop.i128(i128 %12)
  // I128-NEXT: %cast5 = trunc i128 %13 to i32
  // I128-NEXT: store volatile i32 %cast5, ptr %pop, align 4
  pop = __builtin_popcountg(ui128);
  //      I128: %14 = load i128, ptr %ubi128.addr
  // I128-NEXT: %15 = call i128 @llvm.ctpop.i128(i128 %14)
  // I128-NEXT: %cast6 = trunc i128 %15 to i32
  // I128-NEXT: store volatile i32 %cast6, ptr %pop, align 4
  pop = __builtin_popcountg(ubi128);
  //      I128: %load_bits7 = load i8, ptr %vb8.addr, align 1
  // I128-NEXT: %16 = bitcast i8 %load_bits7 to <8 x i1>
  // I128-NEXT: %17 = bitcast <8 x i1> %16 to i8
  // I128-NEXT: %18 = call i8 @llvm.ctpop.i8(i8 %17)
  // I128-NEXT: %cast8 = zext i8 %18 to i32
  // I128-NEXT: store volatile i32 %cast8, ptr %pop, align 4
  pop = __builtin_popcountg(vb8);
}

// I128-LABEL: define{{.*}} void @test_builtin_clzg
void test_builtin_clzg(unsigned char uc, unsigned short us, unsigned int ui,
                       unsigned long ul, unsigned long long ull,
                       unsigned __int128 ui128, unsigned _BitInt(128) ubi128,
                       signed char sc, short s, int i,
                       _Bool __attribute__((ext_vector_type(8))) vb8) {
  volatile int lz;
#if __aarch64__
  int x = 0;
  x = x * 2;
#endif
  //      I128:  %2 = load i8, ptr %uc.addr, align 1
  // I128-NEXT:  %3 = call i8 @llvm.ctlz.i8(i8 %2, i1
  // I128-NEXT:  %cast = zext i8 %3 to i32
  // I128-NEXT:  store volatile i32 %cast, ptr %lz, align 4
  lz = __builtin_clzg(uc);
  // I128-NEXT:  %4 = load i16, ptr %us.addr, align 2
  // I128-NEXT:  %5 = call i16 @llvm.ctlz.i16(i16 %4, i1
  // I128-NEXT:  %cast2 = zext i16 %5 to i32
  // I128-NEXT:  store volatile i32 %cast2, ptr %lz, align 4
  lz = __builtin_clzg(us);
  // I128-NEXT:  %6 = load i32, ptr %ui.addr, align 4
  // I128-NEXT:  %7 = call i32 @llvm.ctlz.i32(i32 %6, i1
  // I128-NEXT:  store volatile i32 %7, ptr %lz, align 4
  lz = __builtin_clzg(ui);
  // I128-NEXT:  %8 = load i64, ptr %ul.addr, align 8
  // I128-NEXT:  %9 = call i64 @llvm.ctlz.i64(i64 %8, i1
  // I128-NEXT:  %cast3 = trunc i64 %9 to i32
  // I128-NEXT:  store volatile i32 %cast3, ptr %lz, align 4
  lz = __builtin_clzg(ul);
  // I128-NEXT:  %10 = load i64, ptr %ull.addr, align 8
  // I128-NEXT:  %11 = call i64 @llvm.ctlz.i64(i64 %10, i1
  // I128-NEXT:  %cast4 = trunc i64 %11 to i32
  // I128-NEXT:  store volatile i32 %cast4, ptr %lz, align 4
  lz = __builtin_clzg(ull);
  // I128-NEXT:  %12 = load i128, ptr %ui128.addr, align 16
  // I128-NEXT:  %13 = call i128 @llvm.ctlz.i128(i128 %12, i1
  // I128-NEXT:  %cast5 = trunc i128 %13 to i32
  // I128-NEXT:  store volatile i32 %cast5, ptr %lz, align 4
  lz = __builtin_clzg(ui128);
  // I128-NEXT:  %14 = load i128, ptr %ubi128.addr
  // I128-NEXT:  %15 = call i128 @llvm.ctlz.i128(i128 %14, i1
  // I128-NEXT:  %cast6 = trunc i128 %15 to i32
  // I128-NEXT:  store volatile i32 %cast6, ptr %lz, align 4
  lz = __builtin_clzg(ubi128);
  // I128-NEXT:  %load_bits7 = load i8, ptr %vb8.addr, align 1
  // I128-NEXT:  %16 = bitcast i8 %load_bits7 to <8 x i1>
  // I128-NEXT:  %17 = bitcast <8 x i1> %16 to i8
  // I128-NEXT:  %18 = call i8 @llvm.ctlz.i8(i8 %17, i1
  // I128-NEXT:  %cast8 = zext i8 %18 to i32
  // I128-NEXT:  store volatile i32 %cast8, ptr %lz, align 4
  lz = __builtin_clzg(vb8);
  // I128-NEXT:  %19 = load i8, ptr %uc.addr, align 1
  // I128-NEXT:  %20 = call i8 @llvm.ctlz.i8(i8 %19, i1
  // I128-NEXT:  %cast9 = zext i8 %20 to i32
  // I128-NEXT:  %iszero = icmp eq i8 %19, 0
  // I128-NEXT:  %21 = load i8, ptr %sc.addr, align 1
  // I128-NEXT:  %conv = sext i8 %21 to i32
  // I128-NEXT:  %clzg = select i1 %iszero, i32 %conv, i32 %cast9
  // I128-NEXT:  store volatile i32 %clzg, ptr %lz, align 4
  lz = __builtin_clzg(uc, sc);
  // I128-NEXT:  %22 = load i16, ptr %us.addr, align 2
  // I128-NEXT:  %23 = call i16 @llvm.ctlz.i16(i16 %22, i1
  // I128-NEXT:  %cast10 = zext i16 %23 to i32
  // I128-NEXT:  %iszero11 = icmp eq i16 %22, 0
  // I128-NEXT:  %24 = load i8, ptr %uc.addr, align 1
  // I128-NEXT:  %conv12 = zext i8 %24 to i32
  // I128-NEXT:  %clzg13 = select i1 %iszero11, i32 %conv12, i32 %cast10
  // I128-NEXT:  store volatile i32 %clzg13, ptr %lz, align 4
  lz = __builtin_clzg(us, uc);
  // I128-NEXT:  %25 = load i32, ptr %ui.addr, align 4
  // I128-NEXT:  %26 = call i32 @llvm.ctlz.i32(i32 %25, i1
  // I128-NEXT:  %iszero14 = icmp eq i32 %25, 0
  // I128-NEXT:  %27 = load i16, ptr %s.addr, align 2
  // I128-NEXT:  %conv15 = sext i16 %27 to i32
  // I128-NEXT:  %clzg16 = select i1 %iszero14, i32 %conv15, i32 %26
  // I128-NEXT:  store volatile i32 %clzg16, ptr %lz, align 4
  lz = __builtin_clzg(ui, s);
  // I128-NEXT:  %28 = load i64, ptr %ul.addr, align 8
  // I128-NEXT:  %29 = call i64 @llvm.ctlz.i64(i64 %28, i1
  // I128-NEXT:  %cast17 = trunc i64 %29 to i32
  // I128-NEXT:  %iszero18 = icmp eq i64 %28, 0
  // I128-NEXT:  %30 = load i16, ptr %us.addr, align 2
  // I128-NEXT:  %conv19 = zext i16 %30 to i32
  // I128-NEXT:  %clzg20 = select i1 %iszero18, i32 %conv19, i32 %cast17
  // I128-NEXT:  store volatile i32 %clzg20, ptr %lz, align 4
  lz = __builtin_clzg(ul, us);
  // I128-NEXT:  %31 = load i64, ptr %ull.addr, align 8
  // I128-NEXT:  %32 = call i64 @llvm.ctlz.i64(i64 %31, i1
  // I128-NEXT:  %cast21 = trunc i64 %32 to i32
  // I128-NEXT:  %iszero22 = icmp eq i64 %31, 0
  // I128-NEXT:  %33 = load i32, ptr %i.addr, align 4
  // I128-NEXT:  %clzg23 = select i1 %iszero22, i32 %33, i32 %cast21
  // I128-NEXT:  store volatile i32 %clzg23, ptr %lz, align 4
  lz = __builtin_clzg(ull, i);
  // I128-NEXT:  %34 = load i128, ptr %ui128.addr, align 16
  // I128-NEXT:  %35 = call i128 @llvm.ctlz.i128(i128 %34, i1
  // I128-NEXT:  %cast24 = trunc i128 %35 to i32
  // I128-NEXT:  %iszero25 = icmp eq i128 %34, 0
  // I128-NEXT:  %36 = load i32, ptr %i.addr, align 4
  // I128-NEXT:  %clzg26 = select i1 %iszero25, i32 %36, i32 %cast24
  // I128-NEXT:  store volatile i32 %clzg26, ptr %lz, align 4
  lz = __builtin_clzg(ui128, i);
  // I128-NEXT:  %37 = load i128, ptr %ubi128.addr
  // I128-NEXT:  %38 = call i128 @llvm.ctlz.i128(i128 %37, i1
  // I128-NEXT:  %cast27 = trunc i128 %38 to i32
  // I128-NEXT:  %iszero28 = icmp eq i128 %37, 0
  // I128-NEXT:  %39 = load i32, ptr %i.addr, align 4
  // I128-NEXT:  %clzg29 = select i1 %iszero28, i32 %39, i32 %cast27
  // I128-NEXT:  store volatile i32 %clzg29, ptr %lz, align 4
  lz = __builtin_clzg(ubi128, i);
  // I128-NEXT:  %load_bits30 = load i8, ptr %vb8.addr, align 1
  // I128-NEXT:  %40 = bitcast i8 %load_bits30 to <8 x i1>
  // I128-NEXT:  %41 = bitcast <8 x i1> %40 to i8
  // I128-NEXT:  %42 = call i8 @llvm.ctlz.i8(i8 %41, i1
  // I128-NEXT:  %cast31 = zext i8 %42 to i32
  // I128-NEXT:  %iszero32 = icmp eq i8 %41, 0
  // I128-NEXT:  %43 = load i32, ptr %i.addr, align 4
  // I128-NEXT:  %clzg33 = select i1 %iszero32, i32 %43, i32 %cast31
  // I128-NEXT:  store volatile i32 %clzg33, ptr %lz, align 4
  lz = __builtin_clzg(vb8, i);
}

// I128-LABEL: define{{.*}} void @test_builtin_ctzg
void test_builtin_ctzg(unsigned char uc, unsigned short us, unsigned int ui,
                       unsigned long ul, unsigned long long ull,
                       unsigned __int128 ui128, unsigned _BitInt(128) ubi128,
                       signed char sc, short s, int i,
                       _Bool __attribute__((ext_vector_type(8))) vb8) {
  volatile int tz;
#if __aarch64__
  int x = 0;
  x = x * 2;
#endif
  //      I128: %2 = load i8, ptr %uc.addr, align 1
  // I128-NEXT: %3 = call i8 @llvm.cttz.i8(i8 %2, i1
  // I128-NEXT: %cast = zext i8 %3 to i32
  // I128-NEXT: store volatile i32 %cast, ptr %tz, align 4
  tz = __builtin_ctzg(uc);
  // I128-NEXT: %4 = load i16, ptr %us.addr, align 2
  // I128-NEXT: %5 = call i16 @llvm.cttz.i16(i16 %4, i1
  // I128-NEXT: %cast2 = zext i16 %5 to i32
  // I128-NEXT: store volatile i32 %cast2, ptr %tz, align 4
  tz = __builtin_ctzg(us);
  // I128-NEXT: %6 = load i32, ptr %ui.addr, align 4
  // I128-NEXT: %7 = call i32 @llvm.cttz.i32(i32 %6, i1
  // I128-NEXT: store volatile i32 %7, ptr %tz, align 4
  tz = __builtin_ctzg(ui);
  // I128-NEXT: %8 = load i64, ptr %ul.addr, align 8
  // I128-NEXT: %9 = call i64 @llvm.cttz.i64(i64 %8, i1
  // I128-NEXT: %cast3 = trunc i64 %9 to i32
  // I128-NEXT: store volatile i32 %cast3, ptr %tz, align 4
  tz = __builtin_ctzg(ul);
  // I128-NEXT: %10 = load i64, ptr %ull.addr, align 8
  // I128-NEXT: %11 = call i64 @llvm.cttz.i64(i64 %10, i1
  // I128-NEXT: %cast4 = trunc i64 %11 to i32
  // I128-NEXT: store volatile i32 %cast4, ptr %tz, align 4
  tz = __builtin_ctzg(ull);
  // I128-NEXT: %12 = load i128, ptr %ui128.addr, align 16
  // I128-NEXT: %13 = call i128 @llvm.cttz.i128(i128 %12, i1
  // I128-NEXT: %cast5 = trunc i128 %13 to i32
  // I128-NEXT: store volatile i32 %cast5, ptr %tz, align 4
  tz = __builtin_ctzg(ui128);
  // I128-NEXT: %14 = load i128, ptr %ubi128.addr
  // I128-NEXT: %15 = call i128 @llvm.cttz.i128(i128 %14, i1
  // I128-NEXT: %cast6 = trunc i128 %15 to i32
  // I128-NEXT: store volatile i32 %cast6, ptr %tz, align 4
  tz = __builtin_ctzg(ubi128);
  // I128-NEXT: %load_bits7 = load i8, ptr %vb8.addr, align 1
  // I128-NEXT: %16 = bitcast i8 %load_bits7 to <8 x i1>
  // I128-NEXT: %17 = bitcast <8 x i1> %16 to i8
  // I128-NEXT: %18 = call i8 @llvm.cttz.i8(i8 %17, i1
  // I128-NEXT: %cast8 = zext i8 %18 to i32
  // I128-NEXT: store volatile i32 %cast8, ptr %tz, align 4
  tz = __builtin_ctzg(vb8);
  // I128-NEXT: %19 = load i8, ptr %uc.addr, align 1
  // I128-NEXT: %20 = call i8 @llvm.cttz.i8(i8 %19, i1
  // I128-NEXT: %cast9 = zext i8 %20 to i32
  // I128-NEXT: %iszero = icmp eq i8 %19, 0
  // I128-NEXT: %21 = load i8, ptr %sc.addr, align 1
  // I128-NEXT: %conv = sext i8 %21 to i32
  // I128-NEXT: %ctzg = select i1 %iszero, i32 %conv, i32 %cast9
  // I128-NEXT: store volatile i32 %ctzg, ptr %tz, align 4
  tz = __builtin_ctzg(uc, sc);
  // I128-NEXT: %22 = load i16, ptr %us.addr, align 2
  // I128-NEXT: %23 = call i16 @llvm.cttz.i16(i16 %22, i1
  // I128-NEXT: %cast10 = zext i16 %23 to i32
  // I128-NEXT: %iszero11 = icmp eq i16 %22, 0
  // I128-NEXT: %24 = load i8, ptr %uc.addr, align 1
  // I128-NEXT: %conv12 = zext i8 %24 to i32
  // I128-NEXT: %ctzg13 = select i1 %iszero11, i32 %conv12, i32 %cast10
  // I128-NEXT: store volatile i32 %ctzg13, ptr %tz, align 4
  tz = __builtin_ctzg(us, uc);
  // I128-NEXT: %25 = load i32, ptr %ui.addr, align 4
  // I128-NEXT: %26 = call i32 @llvm.cttz.i32(i32 %25, i1
  // I128-NEXT: %iszero14 = icmp eq i32 %25, 0
  // I128-NEXT: %27 = load i16, ptr %s.addr, align 2
  // I128-NEXT: %conv15 = sext i16 %27 to i32
  // I128-NEXT: %ctzg16 = select i1 %iszero14, i32 %conv15, i32 %26
  // I128-NEXT: store volatile i32 %ctzg16, ptr %tz, align 4
  tz = __builtin_ctzg(ui, s);
  // I128-NEXT: %28 = load i64, ptr %ul.addr, align 8
  // I128-NEXT: %29 = call i64 @llvm.cttz.i64(i64 %28, i1
  // I128-NEXT: %cast17 = trunc i64 %29 to i32
  // I128-NEXT: %iszero18 = icmp eq i64 %28, 0
  // I128-NEXT: %30 = load i16, ptr %us.addr, align 2
  // I128-NEXT: %conv19 = zext i16 %30 to i32
  // I128-NEXT: %ctzg20 = select i1 %iszero18, i32 %conv19, i32 %cast17
  // I128-NEXT: store volatile i32 %ctzg20, ptr %tz, align 4
  tz = __builtin_ctzg(ul, us);
  // I128-NEXT: %31 = load i64, ptr %ull.addr, align 8
  // I128-NEXT: %32 = call i64 @llvm.cttz.i64(i64 %31, i1
  // I128-NEXT: %cast21 = trunc i64 %32 to i32
  // I128-NEXT: %iszero22 = icmp eq i64 %31, 0
  // I128-NEXT: %33 = load i32, ptr %i.addr, align 4
  // I128-NEXT: %ctzg23 = select i1 %iszero22, i32 %33, i32 %cast21
  // I128-NEXT: store volatile i32 %ctzg23, ptr %tz, align 4
  tz = __builtin_ctzg(ull, i);
  // I128-NEXT: %34 = load i128, ptr %ui128.addr, align 16
  // I128-NEXT: %35 = call i128 @llvm.cttz.i128(i128 %34, i1
  // I128-NEXT: %cast24 = trunc i128 %35 to i32
  // I128-NEXT: %iszero25 = icmp eq i128 %34, 0
  // I128-NEXT: %36 = load i32, ptr %i.addr, align 4
  // I128-NEXT: %ctzg26 = select i1 %iszero25, i32 %36, i32 %cast24
  // I128-NEXT: store volatile i32 %ctzg26, ptr %tz, align 4
  tz = __builtin_ctzg(ui128, i);
  // I128-NEXT: %37 = load i128, ptr %ubi128.addr
  // I128-NEXT: %38 = call i128 @llvm.cttz.i128(i128 %37, i1
  // I128-NEXT: %cast27 = trunc i128 %38 to i32
  // I128-NEXT: %iszero28 = icmp eq i128 %37, 0
  // I128-NEXT: %39 = load i32, ptr %i.addr, align 4
  // I128-NEXT: %ctzg29 = select i1 %iszero28, i32 %39, i32 %cast27
  // I128-NEXT: store volatile i32 %ctzg29, ptr %tz, align 4
  tz = __builtin_ctzg(ubi128, i);
  // I128-NEXT: %load_bits30 = load i8, ptr %vb8.addr, align 1
  // I128-NEXT: %40 = bitcast i8 %load_bits30 to <8 x i1>
  // I128-NEXT: %41 = bitcast <8 x i1> %40 to i8
  // I128-NEXT: %42 = call i8 @llvm.cttz.i8(i8 %41, i1
  // I128-NEXT: %cast31 = zext i8 %42 to i32
  // I128-NEXT: %iszero32 = icmp eq i8 %41, 0
  // I128-NEXT: %43 = load i32, ptr %i.addr, align 4
  // I128-NEXT: %ctzg33 = select i1 %iszero32, i32 %43, i32 %cast31
  // I128-NEXT: store volatile i32 %ctzg33, ptr %tz, align 4
  tz = __builtin_ctzg(vb8, i);
}

#endif

// CHECK-LABEL: define{{.*}} void @test_builtin_bswapg
void test_builtin_bswapg(unsigned char uc, unsigned short us, unsigned int ui,
                       unsigned long ul, unsigned long long ull,
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
  uc = __builtin_bswapg(uc);
  // CHECK: %1 = load i8, ptr %uc.addr
  // CHECK: store i8 %1, ptr %uc.addr
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
