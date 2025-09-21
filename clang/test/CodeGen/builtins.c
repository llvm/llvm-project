// RUN: %clang_cc1 -emit-llvm -o %t %s
// RUN: not grep __builtin %t
// RUN: %clang_cc1 -emit-llvm -triple x86_64-darwin-apple -o - %s | FileCheck %s
// RUN: %clang_cc1 -emit-llvm -triple x86_64-darwin-apple -o - %s -fexperimental-new-constant-interpreter | FileCheck %s

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
  ld = __builtin_huge_vall();    // CHECK: x86_fp80 0xK7FFF8000000000000000
  f = __builtin_nanf("");        // CHECK: float    0x7FF8000000000000
  d = __builtin_nan("");         // CHECK: double   0x7FF8000000000000
  ld = __builtin_nanl("");       // CHECK: x86_fp80 0xK7FFFC000000000000000
  f = __builtin_nanf("0xAE98");  // CHECK: float    0x7FF815D300000000
  d = __builtin_nan("0xAE98");   // CHECK: double   0x7FF800000000AE98
  ld = __builtin_nanl("0xAE98"); // CHECK: x86_fp80 0xK7FFFC00000000000AE98
  f = __builtin_nansf("");       // CHECK: float    0x7FF4000000000000
  d = __builtin_nans("");        // CHECK: double   0x7FF4000000000000
  ld = __builtin_nansl("");      // CHECK: x86_fp80 0xK7FFFA000000000000000
  f = __builtin_nansf("0xAE98"); // CHECK: float    0x7FF015D300000000
  d = __builtin_nans("0xAE98");  // CHECK: double   0x7FF000000000AE98
  ld = __builtin_nansl("0xAE98");// CHECK: x86_fp80 0xK7FFF800000000000AE98

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
  // CHECK: [[TMP:%.*]] = call i1 @llvm.is.fpclass.f80(x86_fp80 {{.*}}, i32 516)
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
  // CHECK:  %[[ABS:.*]] = call x86_fp80 @llvm.fabs.f80(x86_fp80 %[[ARG:.*]])
  // CHECK:  %[[ISINF:.*]] = fcmp oeq x86_fp80 %[[ABS]], 0xK7FFF8000000000000000
  // CHECK:  %[[BITCAST:.*]] = bitcast x86_fp80 %[[ARG]] to i80
  // CHECK:  %[[ISNEG:.*]] = icmp slt i80 %[[BITCAST]], 0
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
  // CHECK: frem x86_fp80

  resf = __builtin_fabsf(F);
  resd = __builtin_fabs(D);
  resld = __builtin_fabsl(LD);
  // CHECK: call float @llvm.fabs.f32(float
  // CHECK: call double @llvm.fabs.f64(double
  // CHECK: call x86_fp80 @llvm.fabs.f80(x86_fp80

  resf = __builtin_canonicalizef(F);
  resd = __builtin_canonicalize(D);
  resld = __builtin_canonicalizel(LD);
  // CHECK: call float @llvm.canonicalize.f32(float
  // CHECK: call double @llvm.canonicalize.f64(double
  // CHECK: call x86_fp80 @llvm.canonicalize.f80(x86_fp80

  resf = __builtin_fminf(F, F);
  // CHECK: call float @llvm.minnum.f32

  resd = __builtin_fmin(D, D);
  // CHECK: call double @llvm.minnum.f64

  resld = __builtin_fminl(LD, LD);
  // CHECK: call x86_fp80 @llvm.minnum.f80

  resf = __builtin_fmaxf(F, F);
  // CHECK: call float @llvm.maxnum.f32

  resd = __builtin_fmax(D, D);
  // CHECK: call double @llvm.maxnum.f64

  resld = __builtin_fmaxl(LD, LD);
  // CHECK: call x86_fp80 @llvm.maxnum.f80

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
  // CHECK: store volatile x86_fp80 0xK3FFF8000000000000000, ptr %resld, align 16

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
  // CHECK: store volatile x86_fp80 0xK40008000000000000000, ptr %resld, align 16

  resf = __builtin_fabsf(F);
  // CHECK: call float @llvm.fabs.f32

  resd = __builtin_fabs(D);
  // CHECK: call double @llvm.fabs.f64

  resld = __builtin_fabsl(LD);
  // CHECK: call x86_fp80 @llvm.fabs.f80

  resf = __builtin_copysignf(F, F);
  // CHECK: call float @llvm.copysign.f32

  resd = __builtin_copysign(D, D);
  // CHECK: call double @llvm.copysign.f64

  resld = __builtin_copysignl(LD, LD);
  // CHECK: call x86_fp80 @llvm.copysign.f80


  resf = __builtin_ceilf(F);
  // CHECK: call float @llvm.ceil.f32

  resd = __builtin_ceil(D);
  // CHECK: call double @llvm.ceil.f64

  resld = __builtin_ceill(LD);
  // CHECK: call x86_fp80 @llvm.ceil.f80

  resf = __builtin_floorf(F);
  // CHECK: call float @llvm.floor.f32

  resd = __builtin_floor(D);
  // CHECK: call double @llvm.floor.f64

  resld = __builtin_floorl(LD);
  // CHECK: call x86_fp80 @llvm.floor.f80

  resf = __builtin_sqrtf(F);
  // CHECK: call float @llvm.sqrt.f32(

  resd = __builtin_sqrt(D);
  // CHECK: call double @llvm.sqrt.f64(

  resld = __builtin_sqrtl(LD);
  // CHECK: call x86_fp80 @llvm.sqrt.f80

  resf = __builtin_truncf(F);
  // CHECK: call float @llvm.trunc.f32

  resd = __builtin_trunc(D);
  // CHECK: call double @llvm.trunc.f64

  resld = __builtin_truncl(LD);
  // CHECK: call x86_fp80 @llvm.trunc.f80

  resf = __builtin_rintf(F);
  // CHECK: call float @llvm.rint.f32

  resd = __builtin_rint(D);
  // CHECK: call double @llvm.rint.f64

  resld = __builtin_rintl(LD);
  // CHECK: call x86_fp80 @llvm.rint.f80

  resf = __builtin_nearbyintf(F);
  // CHECK: call float @llvm.nearbyint.f32

  resd = __builtin_nearbyint(D);
  // CHECK: call double @llvm.nearbyint.f64

  resld = __builtin_nearbyintl(LD);
  // CHECK: call x86_fp80 @llvm.nearbyint.f80

  resf = __builtin_roundf(F);
  // CHECK: call float @llvm.round.f32

  resd = __builtin_round(D);
  // CHECK: call double @llvm.round.f64

  resld = __builtin_roundl(LD);
  // CHECK: call x86_fp80 @llvm.round.f80

  resf = __builtin_roundevenf(F);
  // CHECK: call float @llvm.roundeven.f32

  resd = __builtin_roundeven(D);
  // CHECK: call double @llvm.roundeven.f64

  resld = __builtin_roundevenl(LD);
  // CHECK: call x86_fp80 @llvm.roundeven.f80
  
  resli = __builtin_lroundf (F);
  // CHECK: call i64 @llvm.lround.i64.f32

  resli = __builtin_lround (D);
  // CHECK: call i64 @llvm.lround.i64.f64

  resli = __builtin_lroundl (LD);
  // CHECK: call i64 @llvm.lround.i64.f80

  resli = __builtin_lrintf (F);
  // CHECK: call i64 @llvm.lrint.i64.f32

  resli = __builtin_lrint (D);
  // CHECK: call i64 @llvm.lrint.i64.f64

  resli = __builtin_lrintl (LD);
  // CHECK: call i64 @llvm.lrint.i64.f80
}

// __builtin_longjmp isn't supported on all platforms, so only test it on X86.
#ifdef __x86_64__

// CHECK-LABEL: define{{.*}} void @test_builtin_longjmp(ptr{{.*}}
void test_builtin_longjmp(void **buffer) {
  // CHECK: [[LOAD:%[a-z0-9]+]] = load ptr, ptr
  // CHECK-NEXT: call void @llvm.eh.sjlj.longjmp(ptr [[LOAD]])
  __builtin_longjmp(buffer, 1);
  // CHECK-NEXT: unreachable
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

// __warn_memset_zero_len should be NOP, see https://sourceware.org/bugzilla/show_bug.cgi?id=25399
// CHECK-LABEL: define{{.*}} void @test___warn_memset_zero_len
void test___warn_memset_zero_len(void) {
  // CHECK-NOT: @__warn_memset_zero_len
  __warn_memset_zero_len();
}

// Behavior of __builtin_os_log differs between platforms, so only test on X86
#ifdef __x86_64__

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log
// CHECK: (ptr noundef %[[BUF:.*]], i32 noundef %[[I:.*]], ptr noundef %[[DATA:.*]])
void test_builtin_os_log(void *buf, int i, const char *data) {
  volatile int len;
  // CHECK: %[[BUF_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[I_ADDR:.*]] = alloca i32, align 4
  // CHECK: %[[DATA_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[LEN:.*]] = alloca i32, align 4
  // CHECK: store ptr %[[BUF]], ptr %[[BUF_ADDR]], align 8
  // CHECK: store i32 %[[I]], ptr %[[I_ADDR]], align 4
  // CHECK: store ptr %[[DATA]], ptr %[[DATA_ADDR]], align 8

  // CHECK: store volatile i32 34, ptr %[[LEN]]
  len = __builtin_os_log_format_buffer_size("%d %{public}s %{private}.16P", i, data, data);

  // CHECK: %[[V1:.*]] = load ptr, ptr %[[BUF_ADDR]]
  // CHECK: %[[V2:.*]] = load i32, ptr %[[I_ADDR]]
  // CHECK: %[[V3:.*]] = load ptr, ptr %[[DATA_ADDR]]
  // CHECK: %[[V4:.*]] = ptrtoint ptr %[[V3]] to i64
  // CHECK: %[[V5:.*]] = load ptr, ptr %[[DATA_ADDR]]
  // CHECK: %[[V6:.*]] = ptrtoint ptr %[[V5]] to i64
  // CHECK: call void @__os_log_helper_1_3_4_4_0_8_34_4_17_8_49(ptr noundef %[[V1]], i32 noundef %[[V2]], i64 noundef %[[V4]], i32 noundef 16, i64 noundef %[[V6]])
  __builtin_os_log_format(buf, "%d %{public}s %{private}.16P", i, data, data);

  // privacy annotations aren't recognized when they are preceded or followed
  // by non-whitespace characters.

  // CHECK: call void @__os_log_helper_1_2_1_8_32(
  __builtin_os_log_format(buf, "%{xyz public}s", data);

  // CHECK: call void @__os_log_helper_1_2_1_8_32(
  __builtin_os_log_format(buf, "%{ public xyz}s", data);

  // CHECK: call void @__os_log_helper_1_2_1_8_32(
  __builtin_os_log_format(buf, "%{ public1}s", data);

  // Privacy annotations do not have to be in the first comma-delimited string.

  // CHECK: call void @__os_log_helper_1_2_1_8_34(
  __builtin_os_log_format(buf, "%{ xyz, public }s", "abc");

  // CHECK: call void @__os_log_helper_1_3_1_8_33(
  __builtin_os_log_format(buf, "%{ xyz, private }s", "abc");

  // CHECK: call void @__os_log_helper_1_3_1_8_37(
  __builtin_os_log_format(buf, "%{ xyz, sensitive }s", "abc");

  // The strictest privacy annotation in the string wins.

  // CHECK: call void @__os_log_helper_1_3_1_8_33(
  __builtin_os_log_format(buf, "%{ private, public, private, public}s", "abc");

  // CHECK: call void @__os_log_helper_1_3_1_8_37(
  __builtin_os_log_format(buf, "%{ private, sensitive, private, public}s",
                          "abc");

  // CHECK: store volatile i32 22, ptr %[[LEN]], align 4
  len = __builtin_os_log_format_buffer_size("%{mask.xyz}s", "abc");

  // CHECK: call void @__os_log_helper_1_2_2_8_112_8_34(ptr noundef {{.*}}, i64 noundef 8026488
  __builtin_os_log_format(buf, "%{mask.xyz, public}s", "abc");

  // CHECK: call void @__os_log_helper_1_3_2_8_112_4_1(ptr noundef {{.*}}, i64 noundef 8026488
  __builtin_os_log_format(buf, "%{ mask.xyz, private }d", 11);

  // Mask type is silently ignored.
  // CHECK: call void @__os_log_helper_1_2_1_8_32(
  __builtin_os_log_format(buf, "%{ mask. xyz }s", "abc");

  // CHECK: call void @__os_log_helper_1_2_1_8_32(
  __builtin_os_log_format(buf, "%{ mask.xy z }s", "abc");
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_3_4_4_0_8_34_4_17_8_49
// CHECK: (ptr noundef %[[BUFFER:.*]], i32 noundef %[[ARG0:.*]], i64 noundef %[[ARG1:.*]], i32 noundef %[[ARG2:.*]], i64 noundef %[[ARG3:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[ARG0_ADDR:.*]] = alloca i32, align 4
// CHECK: %[[ARG1_ADDR:.*]] = alloca i64, align 8
// CHECK: %[[ARG2_ADDR:.*]] = alloca i32, align 4
// CHECK: %[[ARG3_ADDR:.*]] = alloca i64, align 8
// CHECK: store ptr %[[BUFFER]], ptr %[[BUFFER_ADDR]], align 8
// CHECK: store i32 %[[ARG0]], ptr %[[ARG0_ADDR]], align 4
// CHECK: store i64 %[[ARG1]], ptr %[[ARG1_ADDR]], align 8
// CHECK: store i32 %[[ARG2]], ptr %[[ARG2_ADDR]], align 4
// CHECK: store i64 %[[ARG3]], ptr %[[ARG3_ADDR]], align 8
// CHECK: %[[BUF:.*]] = load ptr, ptr %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, ptr %[[BUF]], i64 0
// CHECK: store i8 3, ptr %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, ptr %[[BUF]], i64 1
// CHECK: store i8 4, ptr %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, ptr %[[BUF]], i64 2
// CHECK: store i8 0, ptr %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, ptr %[[BUF]], i64 3
// CHECK: store i8 4, ptr %[[ARGSIZE]], align 1
// CHECK: %[[ARGDATA:.*]] = getelementptr i8, ptr %[[BUF]], i64 4
// CHECK: %[[V0:.*]] = load i32, ptr %[[ARG0_ADDR]], align 4
// CHECK: store i32 %[[V0]], ptr %[[ARGDATA]], align 1
// CHECK: %[[ARGDESCRIPTOR1:.*]] = getelementptr i8, ptr %[[BUF]], i64 8
// CHECK: store i8 34, ptr %[[ARGDESCRIPTOR1]], align 1
// CHECK: %[[ARGSIZE2:.*]] = getelementptr i8, ptr %[[BUF]], i64 9
// CHECK: store i8 8, ptr %[[ARGSIZE2]], align 1
// CHECK: %[[ARGDATA3:.*]] = getelementptr i8, ptr %[[BUF]], i64 10
// CHECK: %[[V1:.*]] = load i64, ptr %[[ARG1_ADDR]], align 8
// CHECK: store i64 %[[V1]], ptr %[[ARGDATA3]], align 1
// CHECK: %[[ARGDESCRIPTOR5:.*]] = getelementptr i8, ptr %[[BUF]], i64 18
// CHECK: store i8 17, ptr %[[ARGDESCRIPTOR5]], align 1
// CHECK: %[[ARGSIZE6:.*]] = getelementptr i8, ptr %[[BUF]], i64 19
// CHECK: store i8 4, ptr %[[ARGSIZE6]], align 1
// CHECK: %[[ARGDATA7:.*]] = getelementptr i8, ptr %[[BUF]], i64 20
// CHECK: %[[V2:.*]] = load i32, ptr %[[ARG2_ADDR]], align 4
// CHECK: store i32 %[[V2]], ptr %[[ARGDATA7]], align 1
// CHECK: %[[ARGDESCRIPTOR9:.*]] = getelementptr i8, ptr %[[BUF]], i64 24
// CHECK: store i8 49, ptr %[[ARGDESCRIPTOR9]], align 1
// CHECK: %[[ARGSIZE10:.*]] = getelementptr i8, ptr %[[BUF]], i64 25
// CHECK: store i8 8, ptr %[[ARGSIZE10]], align 1
// CHECK: %[[ARGDATA11:.*]] = getelementptr i8, ptr %[[BUF]], i64 26
// CHECK: %[[V3:.*]] = load i64, ptr %[[ARG3_ADDR]], align 8
// CHECK: store i64 %[[V3]], ptr %[[ARGDATA11]], align 1

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_wide
// CHECK: (ptr noundef %[[BUF:.*]], ptr noundef %[[DATA:.*]], ptr noundef %[[STR:.*]])
typedef int wchar_t;
void test_builtin_os_log_wide(void *buf, const char *data, wchar_t *str) {
  volatile int len;

  // CHECK: %[[BUF_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[DATA_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[STR_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[LEN:.*]] = alloca i32, align 4
  // CHECK: store ptr %[[BUF]], ptr %[[BUF_ADDR]], align 8
  // CHECK: store ptr %[[DATA]], ptr %[[DATA_ADDR]], align 8
  // CHECK: store ptr %[[STR]], ptr %[[STR_ADDR]], align 8

  // CHECK: store volatile i32 12, ptr %[[LEN]], align 4
  len = __builtin_os_log_format_buffer_size("%S", str);

  // CHECK: %[[V1:.*]] = load ptr, ptr %[[BUF_ADDR]], align 8
  // CHECK: %[[V2:.*]] = load ptr, ptr %[[STR_ADDR]], align 8
  // CHECK: %[[V3:.*]] = ptrtoint ptr %[[V2]] to i64
  // CHECK: call void @__os_log_helper_1_2_1_8_80(ptr noundef %[[V1]], i64 noundef %[[V3]])

  __builtin_os_log_format(buf, "%S", str);
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_2_1_8_80
// CHECK: (ptr noundef %[[BUFFER:.*]], i64 noundef %[[ARG0:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[ARG0_ADDR:.*]] = alloca i64, align 8
// CHECK: store ptr %[[BUFFER]], ptr %[[BUFFER_ADDR]], align 8
// CHECK: store i64 %[[ARG0]], ptr %[[ARG0_ADDR]], align 8
// CHECK: %[[BUF:.*]] = load ptr, ptr %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, ptr %[[BUF]], i64 0
// CHECK: store i8 2, ptr %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, ptr %[[BUF]], i64 1
// CHECK: store i8 1, ptr %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, ptr %[[BUF]], i64 2
// CHECK: store i8 80, ptr %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, ptr %[[BUF]], i64 3
// CHECK: store i8 8, ptr %[[ARGSIZE]], align 1
// CHECK: %[[ARGDATA:.*]] = getelementptr i8, ptr %[[BUF]], i64 4
// CHECK: %[[V0:.*]] = load i64, ptr %[[ARG0_ADDR]], align 8
// CHECK: store i64 %[[V0]], ptr %[[ARGDATA]], align 1

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_precision_width
// CHECK: (ptr noundef %[[BUF:.*]], ptr noundef %[[DATA:.*]], i32 noundef %[[PRECISION:.*]], i32 noundef %[[WIDTH:.*]])
void test_builtin_os_log_precision_width(void *buf, const char *data,
                                         int precision, int width) {
  volatile int len;
  // CHECK: %[[BUF_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[DATA_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[PRECISION_ADDR:.*]] = alloca i32, align 4
  // CHECK: %[[WIDTH_ADDR:.*]] = alloca i32, align 4
  // CHECK: %[[LEN:.*]] = alloca i32, align 4
  // CHECK: store ptr %[[BUF]], ptr %[[BUF_ADDR]], align 8
  // CHECK: store ptr %[[DATA]], ptr %[[DATA_ADDR]], align 8
  // CHECK: store i32 %[[PRECISION]], ptr %[[PRECISION_ADDR]], align 4
  // CHECK: store i32 %[[WIDTH]], ptr %[[WIDTH_ADDR]], align 4

  // CHECK: store volatile i32 24, ptr %[[LEN]], align 4
  len = __builtin_os_log_format_buffer_size("Hello %*.*s World", precision, width, data);

  // CHECK: %[[V1:.*]] = load ptr, ptr %[[BUF_ADDR]], align 8
  // CHECK: %[[V2:.*]] = load i32, ptr %[[PRECISION_ADDR]], align 4
  // CHECK: %[[V3:.*]] = load i32, ptr %[[WIDTH_ADDR]], align 4
  // CHECK: %[[V4:.*]] = load ptr, ptr %[[DATA_ADDR]], align 8
  // CHECK: %[[V5:.*]] = ptrtoint ptr %[[V4]] to i64
  // CHECK: call void @__os_log_helper_1_2_3_4_0_4_16_8_32(ptr noundef %[[V1]], i32 noundef %[[V2]], i32 noundef %[[V3]], i64 noundef %[[V5]])
  __builtin_os_log_format(buf, "Hello %*.*s World", precision, width, data);
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_2_3_4_0_4_16_8_32
// CHECK: (ptr noundef %[[BUFFER:.*]], i32 noundef %[[ARG0:.*]], i32 noundef %[[ARG1:.*]], i64 noundef %[[ARG2:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[ARG0_ADDR:.*]] = alloca i32, align 4
// CHECK: %[[ARG1_ADDR:.*]] = alloca i32, align 4
// CHECK: %[[ARG2_ADDR:.*]] = alloca i64, align 8
// CHECK: store ptr %[[BUFFER]], ptr %[[BUFFER_ADDR]], align 8
// CHECK: store i32 %[[ARG0]], ptr %[[ARG0_ADDR]], align 4
// CHECK: store i32 %[[ARG1]], ptr %[[ARG1_ADDR]], align 4
// CHECK: store i64 %[[ARG2]], ptr %[[ARG2_ADDR]], align 8
// CHECK: %[[BUF:.*]] = load ptr, ptr %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, ptr %[[BUF]], i64 0
// CHECK: store i8 2, ptr %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, ptr %[[BUF]], i64 1
// CHECK: store i8 3, ptr %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, ptr %[[BUF]], i64 2
// CHECK: store i8 0, ptr %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, ptr %[[BUF]], i64 3
// CHECK: store i8 4, ptr %[[ARGSIZE]], align 1
// CHECK: %[[ARGDATA:.*]] = getelementptr i8, ptr %[[BUF]], i64 4
// CHECK: %[[V0:.*]] = load i32, ptr %[[ARG0_ADDR]], align 4
// CHECK: store i32 %[[V0]], ptr %[[ARGDATA]], align 1
// CHECK: %[[ARGDESCRIPTOR1:.*]] = getelementptr i8, ptr %[[BUF]], i64 8
// CHECK: store i8 16, ptr %[[ARGDESCRIPTOR1]], align 1
// CHECK: %[[ARGSIZE2:.*]] = getelementptr i8, ptr %[[BUF]], i64 9
// CHECK: store i8 4, ptr %[[ARGSIZE2]], align 1
// CHECK: %[[ARGDATA3:.*]] = getelementptr i8, ptr %[[BUF]], i64 10
// CHECK: %[[V1:.*]] = load i32, ptr %[[ARG1_ADDR]], align 4
// CHECK: store i32 %[[V1]], ptr %[[ARGDATA3]], align 1
// CHECK: %[[ARGDESCRIPTOR5:.*]] = getelementptr i8, ptr %[[BUF]], i64 14
// CHECK: store i8 32, ptr %[[ARGDESCRIPTOR5]], align 1
// CHECK: %[[ARGSIZE6:.*]] = getelementptr i8, ptr %[[BUF]], i64 15
// CHECK: store i8 8, ptr %[[ARGSIZE6]], align 1
// CHECK: %[[ARGDATA7:.*]] = getelementptr i8, ptr %[[BUF]], i64 16
// CHECK: %[[V2:.*]] = load i64, ptr %[[ARG2_ADDR]], align 8
// CHECK: store i64 %[[V2]], ptr %[[ARGDATA7]], align 1

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_invalid
// CHECK: (ptr noundef %[[BUF:.*]], i32 noundef %[[DATA:.*]])
void test_builtin_os_log_invalid(void *buf, int data) {
  volatile int len;
  // CHECK: %[[BUF_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[DATA_ADDR:.*]] = alloca i32, align 4
  // CHECK: %[[LEN:.*]] = alloca i32, align 4
  // CHECK: store ptr %[[BUF]], ptr %[[BUF_ADDR]], align 8
  // CHECK: store i32 %[[DATA]], ptr %[[DATA_ADDR]], align 4

  // CHECK: store volatile i32 8, ptr %[[LEN]], align 4
  len = __builtin_os_log_format_buffer_size("invalid specifier %: %d even a trailing one%", data);

  // CHECK: %[[V1:.*]] = load ptr, ptr %[[BUF_ADDR]], align 8
  // CHECK: %[[V2:.*]] = load i32, ptr %[[DATA_ADDR]], align 4
  // CHECK: call void @__os_log_helper_1_0_1_4_0(ptr noundef %[[V1]], i32 noundef %[[V2]])

  __builtin_os_log_format(buf, "invalid specifier %: %d even a trailing one%", data);
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_0_1_4_0
// CHECK: (ptr noundef %[[BUFFER:.*]], i32 noundef %[[ARG0:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[ARG0_ADDR:.*]] = alloca i32, align 4
// CHECK: store ptr %[[BUFFER]], ptr %[[BUFFER_ADDR]], align 8
// CHECK: store i32 %[[ARG0]], ptr %[[ARG0_ADDR]], align 4
// CHECK: %[[BUF:.*]] = load ptr, ptr %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, ptr %[[BUF]], i64 0
// CHECK: store i8 0, ptr %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, ptr %[[BUF]], i64 1
// CHECK: store i8 1, ptr %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, ptr %[[BUF]], i64 2
// CHECK: store i8 0, ptr %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, ptr %[[BUF]], i64 3
// CHECK: store i8 4, ptr %[[ARGSIZE]], align 1
// CHECK: %[[ARGDATA:.*]] = getelementptr i8, ptr %[[BUF]], i64 4
// CHECK: %[[V0:.*]] = load i32, ptr %[[ARG0_ADDR]], align 4
// CHECK: store i32 %[[V0]], ptr %[[ARGDATA]], align 1

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_percent
// CHECK: (ptr noundef %[[BUF:.*]], ptr noundef %[[DATA1:.*]], ptr noundef %[[DATA2:.*]])
// Check that the %% which does not consume any argument is correctly handled
void test_builtin_os_log_percent(void *buf, const char *data1, const char *data2) {
  volatile int len;
  // CHECK: %[[BUF_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[DATA1_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[DATA2_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[LEN:.*]] = alloca i32, align 4
  // CHECK: store ptr %[[BUF]], ptr %[[BUF_ADDR]], align 8
  // CHECK: store ptr %[[DATA1]], ptr %[[DATA1_ADDR]], align 8
  // CHECK: store ptr %[[DATA2]], ptr %[[DATA2_ADDR]], align 8
  // CHECK: store volatile i32 22, ptr %[[LEN]], align 4

  len = __builtin_os_log_format_buffer_size("%s %% %s", data1, data2);

  // CHECK: %[[V1:.*]] = load ptr, ptr %[[BUF_ADDR]], align 8
  // CHECK: %[[V2:.*]] = load ptr, ptr %[[DATA1_ADDR]], align 8
  // CHECK: %[[V3:.*]] = ptrtoint ptr %[[V2]] to i64
  // CHECK: %[[V4:.*]] = load ptr, ptr %[[DATA2_ADDR]], align 8
  // CHECK: %[[V5:.*]] = ptrtoint ptr %[[V4]] to i64
  // CHECK: call void @__os_log_helper_1_2_2_8_32_8_32(ptr noundef %[[V1]], i64 noundef %[[V3]], i64 noundef %[[V5]])

  __builtin_os_log_format(buf, "%s %% %s", data1, data2);
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_2_2_8_32_8_32
// CHECK: (ptr noundef %[[BUFFER:.*]], i64 noundef %[[ARG0:.*]], i64 noundef %[[ARG1:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[ARG0_ADDR:.*]] = alloca i64, align 8
// CHECK: %[[ARG1_ADDR:.*]] = alloca i64, align 8
// CHECK: store ptr %[[BUFFER]], ptr %[[BUFFER_ADDR]], align 8
// CHECK: store i64 %[[ARG0]], ptr %[[ARG0_ADDR]], align 8
// CHECK: store i64 %[[ARG1]], ptr %[[ARG1_ADDR]], align 8
// CHECK: %[[BUF:.*]] = load ptr, ptr %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, ptr %[[BUF]], i64 0
// CHECK: store i8 2, ptr %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, ptr %[[BUF]], i64 1
// CHECK: store i8 2, ptr %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, ptr %[[BUF]], i64 2
// CHECK: store i8 32, ptr %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, ptr %[[BUF]], i64 3
// CHECK: store i8 8, ptr %[[ARGSIZE]], align 1
// CHECK: %[[ARGDATA:.*]] = getelementptr i8, ptr %[[BUF]], i64 4
// CHECK: %[[V0:.*]] = load i64, ptr %[[ARG0_ADDR]], align 8
// CHECK: store i64 %[[V0]], ptr %[[ARGDATA]], align 1
// CHECK: %[[ARGDESCRIPTOR1:.*]] = getelementptr i8, ptr %[[BUF]], i64 12
// CHECK: store i8 32, ptr %[[ARGDESCRIPTOR1]], align 1
// CHECK: %[[ARGSIZE2:.*]] = getelementptr i8, ptr %[[BUF]], i64 13
// CHECK: store i8 8, ptr %[[ARGSIZE2]], align 1
// CHECK: %[[ARGDATA3:.*]] = getelementptr i8, ptr %[[BUF]], i64 14
// CHECK: %[[V1:.*]] = load i64, ptr %[[ARG1_ADDR]], align 8
// CHECK: store i64 %[[V1]], ptr %[[ARGDATA3]], align 1

// Check that the following two functions call the same helper function.

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_merge_helper0
// CHECK: call void @__os_log_helper_1_0_2_4_0_8_0(
void test_builtin_os_log_merge_helper0(void *buf, int i, double d) {
  __builtin_os_log_format(buf, "%d %f", i, d);
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_0_2_4_0_8_0(

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_merge_helper1
// CHECK: call void @__os_log_helper_1_0_2_4_0_8_0(
void test_builtin_os_log_merge_helper1(void *buf, unsigned u, long long ll) {
  __builtin_os_log_format(buf, "%u %lld", u, ll);
}

// Check that this function doesn't write past the end of array 'buf'.

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_errno
void test_builtin_os_log_errno(void) {
  // CHECK-NOT: @stacksave
  // CHECK: %[[BUF:.*]] = alloca [4 x i8], align 1
  // CHECK: %[[DECAY:.*]] = getelementptr inbounds [4 x i8], ptr %[[BUF]], i64 0, i64 0
  // CHECK: call void @__os_log_helper_1_2_1_0_96(ptr noundef %[[DECAY]])
  // CHECK-NOT: @stackrestore

  char buf[__builtin_os_log_format_buffer_size("%m")];
  __builtin_os_log_format(buf, "%m");
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_2_1_0_96
// CHECK: (ptr noundef %[[BUFFER:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca ptr, align 8
// CHECK: store ptr %[[BUFFER]], ptr %[[BUFFER_ADDR]], align 8
// CHECK: %[[BUF:.*]] = load ptr, ptr %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, ptr %[[BUF]], i64 0
// CHECK: store i8 2, ptr %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, ptr %[[BUF]], i64 1
// CHECK: store i8 1, ptr %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, ptr %[[BUF]], i64 2
// CHECK: store i8 96, ptr %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, ptr %[[BUF]], i64 3
// CHECK: store i8 0, ptr %[[ARGSIZE]], align 1
// CHECK-NEXT: ret void

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_long_double
// CHECK: (ptr noundef %[[BUF:.*]], x86_fp80 noundef %[[LD:.*]])
void test_builtin_os_log_long_double(void *buf, long double ld) {
  // CHECK: %[[BUF_ADDR:.*]] = alloca ptr, align 8
  // CHECK: %[[LD_ADDR:.*]] = alloca x86_fp80, align 16
  // CHECK: store ptr %[[BUF]], ptr %[[BUF_ADDR]], align 8
  // CHECK: store x86_fp80 %[[LD]], ptr %[[LD_ADDR]], align 16
  // CHECK: %[[V0:.*]] = load ptr, ptr %[[BUF_ADDR]], align 8
  // CHECK: %[[V1:.*]] = load x86_fp80, ptr %[[LD_ADDR]], align 16
  // CHECK: %[[V2:.*]] = bitcast x86_fp80 %[[V1]] to i80
  // CHECK: %[[V3:.*]] = zext i80 %[[V2]] to i128
  // CHECK: call void @__os_log_helper_1_0_1_16_0(ptr noundef %[[V0]], i128 noundef %[[V3]])

  __builtin_os_log_format(buf, "%Lf", ld);
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_0_1_16_0
// CHECK: (ptr noundef %[[BUFFER:.*]], i128 noundef %[[ARG0:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca ptr, align 8
// CHECK: %[[ARG0_ADDR:.*]] = alloca i128, align 16
// CHECK: store ptr %[[BUFFER]], ptr %[[BUFFER_ADDR]], align 8
// CHECK: store i128 %[[ARG0]], ptr %[[ARG0_ADDR]], align 16
// CHECK: %[[BUF:.*]] = load ptr, ptr %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, ptr %[[BUF]], i64 0
// CHECK: store i8 0, ptr %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, ptr %[[BUF]], i64 1
// CHECK: store i8 1, ptr %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, ptr %[[BUF]], i64 2
// CHECK: store i8 0, ptr %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, ptr %[[BUF]], i64 3
// CHECK: store i8 16, ptr %[[ARGSIZE]], align 1
// CHECK: %[[ARGDATA:.*]] = getelementptr i8, ptr %[[BUF]], i64 4
// CHECK: %[[V3:.*]] = load i128, ptr %[[ARG0_ADDR]], align 16
// CHECK: store i128 %[[V3]], ptr %[[ARGDATA]], align 1

// CHECK-LABEL: define{{.*}} void @test_builtin_popcountg
void test_builtin_popcountg(unsigned char uc, unsigned short us,
                            unsigned int ui, unsigned long ul,
                            unsigned long long ull, unsigned __int128 ui128,
                            unsigned _BitInt(128) ubi128,
                            _Bool __attribute__((ext_vector_type(8))) vb8) {
  volatile int pop;
  //      CHECK: %2 = load i8, ptr %uc.addr, align 1
  // CHECK-NEXT: %3 = call i8 @llvm.ctpop.i8(i8 %2)
  // CHECK-NEXT: %cast = zext i8 %3 to i32
  // CHECK-NEXT: store volatile i32 %cast, ptr %pop, align 4
  pop = __builtin_popcountg(uc);
  //      CHECK: %4 = load i16, ptr %us.addr, align 2
  // CHECK-NEXT: %5 = call i16 @llvm.ctpop.i16(i16 %4)
  // CHECK-NEXT: %cast2 = zext i16 %5 to i32
  // CHECK-NEXT: store volatile i32 %cast2, ptr %pop, align 4
  pop = __builtin_popcountg(us);
  //      CHECK: %6 = load i32, ptr %ui.addr, align 4
  // CHECK-NEXT: %7 = call i32 @llvm.ctpop.i32(i32 %6)
  // CHECK-NEXT: store volatile i32 %7, ptr %pop, align 4
  pop = __builtin_popcountg(ui);
  // CHECK: %8 = load i64, ptr %ul.addr, align 8
  // CHECK-NEXT: %9 = call i64 @llvm.ctpop.i64(i64 %8)
  // CHECK-NEXT: %cast3 = trunc i64 %9 to i32
  // CHECK-NEXT: store volatile i32 %cast3, ptr %pop, align 4
  pop = __builtin_popcountg(ul);
  //      CHECK: %10 = load i64, ptr %ull.addr, align 8
  // CHECK-NEXT: %11 = call i64 @llvm.ctpop.i64(i64 %10)
  // CHECK-NEXT: %cast4 = trunc i64 %11 to i32
  // CHECK-NEXT: store volatile i32 %cast4, ptr %pop, align 4
  pop = __builtin_popcountg(ull);
  //      CHECK: %12 = load i128, ptr %ui128.addr, align 16
  // CHECK-NEXT: %13 = call i128 @llvm.ctpop.i128(i128 %12)
  // CHECK-NEXT: %cast5 = trunc i128 %13 to i32
  // CHECK-NEXT: store volatile i32 %cast5, ptr %pop, align 4
  pop = __builtin_popcountg(ui128);
  //      CHECK: %14 = load i128, ptr %ubi128.addr, align 8
  // CHECK-NEXT: %15 = call i128 @llvm.ctpop.i128(i128 %14)
  // CHECK-NEXT: %cast6 = trunc i128 %15 to i32
  // CHECK-NEXT: store volatile i32 %cast6, ptr %pop, align 4
  pop = __builtin_popcountg(ubi128);
  //      CHECK: %load_bits7 = load i8, ptr %vb8.addr, align 1
  // CHECK-NEXT: %16 = bitcast i8 %load_bits7 to <8 x i1>
  // CHECK-NEXT: %17 = bitcast <8 x i1> %16 to i8
  // CHECK-NEXT: %18 = call i8 @llvm.ctpop.i8(i8 %17)
  // CHECK-NEXT: %cast8 = zext i8 %18 to i32
  // CHECK-NEXT: store volatile i32 %cast8, ptr %pop, align 4
  pop = __builtin_popcountg(vb8);
}

// CHECK-LABEL: define{{.*}} void @test_builtin_clzg
void test_builtin_clzg(unsigned char uc, unsigned short us, unsigned int ui,
                       unsigned long ul, unsigned long long ull,
                       unsigned __int128 ui128, unsigned _BitInt(128) ubi128,
                       signed char sc, short s, int i,
                       _Bool __attribute__((ext_vector_type(8))) vb8) {
  volatile int lz;
  //      CHECK:  %2 = load i8, ptr %uc.addr, align 1
  // CHECK-NEXT:  %3 = call i8 @llvm.ctlz.i8(i8 %2, i1 true)
  // CHECK-NEXT:  %cast = zext i8 %3 to i32
  // CHECK-NEXT:  store volatile i32 %cast, ptr %lz, align 4
  lz = __builtin_clzg(uc);
  // CHECK-NEXT:  %4 = load i16, ptr %us.addr, align 2
  // CHECK-NEXT:  %5 = call i16 @llvm.ctlz.i16(i16 %4, i1 true)
  // CHECK-NEXT:  %cast2 = zext i16 %5 to i32
  // CHECK-NEXT:  store volatile i32 %cast2, ptr %lz, align 4
  lz = __builtin_clzg(us);
  // CHECK-NEXT:  %6 = load i32, ptr %ui.addr, align 4
  // CHECK-NEXT:  %7 = call i32 @llvm.ctlz.i32(i32 %6, i1 true)
  // CHECK-NEXT:  store volatile i32 %7, ptr %lz, align 4
  lz = __builtin_clzg(ui);
  // CHECK-NEXT:  %8 = load i64, ptr %ul.addr, align 8
  // CHECK-NEXT:  %9 = call i64 @llvm.ctlz.i64(i64 %8, i1 true)
  // CHECK-NEXT:  %cast3 = trunc i64 %9 to i32
  // CHECK-NEXT:  store volatile i32 %cast3, ptr %lz, align 4
  lz = __builtin_clzg(ul);
  // CHECK-NEXT:  %10 = load i64, ptr %ull.addr, align 8
  // CHECK-NEXT:  %11 = call i64 @llvm.ctlz.i64(i64 %10, i1 true)
  // CHECK-NEXT:  %cast4 = trunc i64 %11 to i32
  // CHECK-NEXT:  store volatile i32 %cast4, ptr %lz, align 4
  lz = __builtin_clzg(ull);
  // CHECK-NEXT:  %12 = load i128, ptr %ui128.addr, align 16
  // CHECK-NEXT:  %13 = call i128 @llvm.ctlz.i128(i128 %12, i1 true)
  // CHECK-NEXT:  %cast5 = trunc i128 %13 to i32
  // CHECK-NEXT:  store volatile i32 %cast5, ptr %lz, align 4
  lz = __builtin_clzg(ui128);
  // CHECK-NEXT:  %14 = load i128, ptr %ubi128.addr, align 8
  // CHECK-NEXT:  %15 = call i128 @llvm.ctlz.i128(i128 %14, i1 true)
  // CHECK-NEXT:  %cast6 = trunc i128 %15 to i32
  // CHECK-NEXT:  store volatile i32 %cast6, ptr %lz, align 4
  lz = __builtin_clzg(ubi128);
  // CHECK-NEXT:  %load_bits7 = load i8, ptr %vb8.addr, align 1
  // CHECK-NEXT:  %16 = bitcast i8 %load_bits7 to <8 x i1>
  // CHECK-NEXT:  %17 = bitcast <8 x i1> %16 to i8
  // CHECK-NEXT:  %18 = call i8 @llvm.ctlz.i8(i8 %17, i1 true)
  // CHECK-NEXT:  %cast8 = zext i8 %18 to i32
  // CHECK-NEXT:  store volatile i32 %cast8, ptr %lz, align 4
  lz = __builtin_clzg(vb8);
  // CHECK-NEXT:  %19 = load i8, ptr %uc.addr, align 1
  // CHECK-NEXT:  %20 = call i8 @llvm.ctlz.i8(i8 %19, i1 true)
  // CHECK-NEXT:  %cast9 = zext i8 %20 to i32
  // CHECK-NEXT:  %iszero = icmp eq i8 %19, 0
  // CHECK-NEXT:  %21 = load i8, ptr %sc.addr, align 1
  // CHECK-NEXT:  %conv = sext i8 %21 to i32
  // CHECK-NEXT:  %clzg = select i1 %iszero, i32 %conv, i32 %cast9
  // CHECK-NEXT:  store volatile i32 %clzg, ptr %lz, align 4
  lz = __builtin_clzg(uc, sc);
  // CHECK-NEXT:  %22 = load i16, ptr %us.addr, align 2
  // CHECK-NEXT:  %23 = call i16 @llvm.ctlz.i16(i16 %22, i1 true)
  // CHECK-NEXT:  %cast10 = zext i16 %23 to i32
  // CHECK-NEXT:  %iszero11 = icmp eq i16 %22, 0
  // CHECK-NEXT:  %24 = load i8, ptr %uc.addr, align 1
  // CHECK-NEXT:  %conv12 = zext i8 %24 to i32
  // CHECK-NEXT:  %clzg13 = select i1 %iszero11, i32 %conv12, i32 %cast10
  // CHECK-NEXT:  store volatile i32 %clzg13, ptr %lz, align 4
  lz = __builtin_clzg(us, uc);
  // CHECK-NEXT:  %25 = load i32, ptr %ui.addr, align 4
  // CHECK-NEXT:  %26 = call i32 @llvm.ctlz.i32(i32 %25, i1 true)
  // CHECK-NEXT:  %iszero14 = icmp eq i32 %25, 0
  // CHECK-NEXT:  %27 = load i16, ptr %s.addr, align 2
  // CHECK-NEXT:  %conv15 = sext i16 %27 to i32
  // CHECK-NEXT:  %clzg16 = select i1 %iszero14, i32 %conv15, i32 %26
  // CHECK-NEXT:  store volatile i32 %clzg16, ptr %lz, align 4
  lz = __builtin_clzg(ui, s);
  // CHECK-NEXT:  %28 = load i64, ptr %ul.addr, align 8
  // CHECK-NEXT:  %29 = call i64 @llvm.ctlz.i64(i64 %28, i1 true)
  // CHECK-NEXT:  %cast17 = trunc i64 %29 to i32
  // CHECK-NEXT:  %iszero18 = icmp eq i64 %28, 0
  // CHECK-NEXT:  %30 = load i16, ptr %us.addr, align 2
  // CHECK-NEXT:  %conv19 = zext i16 %30 to i32
  // CHECK-NEXT:  %clzg20 = select i1 %iszero18, i32 %conv19, i32 %cast17
  // CHECK-NEXT:  store volatile i32 %clzg20, ptr %lz, align 4
  lz = __builtin_clzg(ul, us);
  // CHECK-NEXT:  %31 = load i64, ptr %ull.addr, align 8
  // CHECK-NEXT:  %32 = call i64 @llvm.ctlz.i64(i64 %31, i1 true)
  // CHECK-NEXT:  %cast21 = trunc i64 %32 to i32
  // CHECK-NEXT:  %iszero22 = icmp eq i64 %31, 0
  // CHECK-NEXT:  %33 = load i32, ptr %i.addr, align 4
  // CHECK-NEXT:  %clzg23 = select i1 %iszero22, i32 %33, i32 %cast21
  // CHECK-NEXT:  store volatile i32 %clzg23, ptr %lz, align 4
  lz = __builtin_clzg(ull, i);
  // CHECK-NEXT:  %34 = load i128, ptr %ui128.addr, align 16
  // CHECK-NEXT:  %35 = call i128 @llvm.ctlz.i128(i128 %34, i1 true)
  // CHECK-NEXT:  %cast24 = trunc i128 %35 to i32
  // CHECK-NEXT:  %iszero25 = icmp eq i128 %34, 0
  // CHECK-NEXT:  %36 = load i32, ptr %i.addr, align 4
  // CHECK-NEXT:  %clzg26 = select i1 %iszero25, i32 %36, i32 %cast24
  // CHECK-NEXT:  store volatile i32 %clzg26, ptr %lz, align 4
  lz = __builtin_clzg(ui128, i);
  // CHECK-NEXT:  %37 = load i128, ptr %ubi128.addr, align 8
  // CHECK-NEXT:  %38 = call i128 @llvm.ctlz.i128(i128 %37, i1 true)
  // CHECK-NEXT:  %cast27 = trunc i128 %38 to i32
  // CHECK-NEXT:  %iszero28 = icmp eq i128 %37, 0
  // CHECK-NEXT:  %39 = load i32, ptr %i.addr, align 4
  // CHECK-NEXT:  %clzg29 = select i1 %iszero28, i32 %39, i32 %cast27
  // CHECK-NEXT:  store volatile i32 %clzg29, ptr %lz, align 4
  lz = __builtin_clzg(ubi128, i);
  // CHECK-NEXT:  %load_bits30 = load i8, ptr %vb8.addr, align 1
  // CHECK-NEXT:  %40 = bitcast i8 %load_bits30 to <8 x i1>
  // CHECK-NEXT:  %41 = bitcast <8 x i1> %40 to i8
  // CHECK-NEXT:  %42 = call i8 @llvm.ctlz.i8(i8 %41, i1 true)
  // CHECK-NEXT:  %cast31 = zext i8 %42 to i32
  // CHECK-NEXT:  %iszero32 = icmp eq i8 %41, 0
  // CHECK-NEXT:  %43 = load i32, ptr %i.addr, align 4
  // CHECK-NEXT:  %clzg33 = select i1 %iszero32, i32 %43, i32 %cast31
  // CHECK-NEXT:  store volatile i32 %clzg33, ptr %lz, align 4
  lz = __builtin_clzg(vb8, i);
}

// CHECK-LABEL: define{{.*}} void @test_builtin_ctzg
void test_builtin_ctzg(unsigned char uc, unsigned short us, unsigned int ui,
                       unsigned long ul, unsigned long long ull,
                       unsigned __int128 ui128, unsigned _BitInt(128) ubi128,
                       signed char sc, short s, int i,
                       _Bool __attribute__((ext_vector_type(8))) vb8) {
  volatile int tz;
  //      CHECK: %2 = load i8, ptr %uc.addr, align 1
  // CHECK-NEXT: %3 = call i8 @llvm.cttz.i8(i8 %2, i1 true)
  // CHECK-NEXT: %cast = zext i8 %3 to i32
  // CHECK-NEXT: store volatile i32 %cast, ptr %tz, align 4
  tz = __builtin_ctzg(uc);
  // CHECK-NEXT: %4 = load i16, ptr %us.addr, align 2
  // CHECK-NEXT: %5 = call i16 @llvm.cttz.i16(i16 %4, i1 true)
  // CHECK-NEXT: %cast2 = zext i16 %5 to i32
  // CHECK-NEXT: store volatile i32 %cast2, ptr %tz, align 4
  tz = __builtin_ctzg(us);
  // CHECK-NEXT: %6 = load i32, ptr %ui.addr, align 4
  // CHECK-NEXT: %7 = call i32 @llvm.cttz.i32(i32 %6, i1 true)
  // CHECK-NEXT: store volatile i32 %7, ptr %tz, align 4
  tz = __builtin_ctzg(ui);
  // CHECK-NEXT: %8 = load i64, ptr %ul.addr, align 8
  // CHECK-NEXT: %9 = call i64 @llvm.cttz.i64(i64 %8, i1 true)
  // CHECK-NEXT: %cast3 = trunc i64 %9 to i32
  // CHECK-NEXT: store volatile i32 %cast3, ptr %tz, align 4
  tz = __builtin_ctzg(ul);
  // CHECK-NEXT: %10 = load i64, ptr %ull.addr, align 8
  // CHECK-NEXT: %11 = call i64 @llvm.cttz.i64(i64 %10, i1 true)
  // CHECK-NEXT: %cast4 = trunc i64 %11 to i32
  // CHECK-NEXT: store volatile i32 %cast4, ptr %tz, align 4
  tz = __builtin_ctzg(ull);
  // CHECK-NEXT: %12 = load i128, ptr %ui128.addr, align 16
  // CHECK-NEXT: %13 = call i128 @llvm.cttz.i128(i128 %12, i1 true)
  // CHECK-NEXT: %cast5 = trunc i128 %13 to i32
  // CHECK-NEXT: store volatile i32 %cast5, ptr %tz, align 4
  tz = __builtin_ctzg(ui128);
  // CHECK-NEXT: %14 = load i128, ptr %ubi128.addr, align 8
  // CHECK-NEXT: %15 = call i128 @llvm.cttz.i128(i128 %14, i1 true)
  // CHECK-NEXT: %cast6 = trunc i128 %15 to i32
  // CHECK-NEXT: store volatile i32 %cast6, ptr %tz, align 4
  tz = __builtin_ctzg(ubi128);
  // CHECK-NEXT: %load_bits7 = load i8, ptr %vb8.addr, align 1
  // CHECK-NEXT: %16 = bitcast i8 %load_bits7 to <8 x i1>
  // CHECK-NEXT: %17 = bitcast <8 x i1> %16 to i8
  // CHECK-NEXT: %18 = call i8 @llvm.cttz.i8(i8 %17, i1 true)
  // CHECK-NEXT: %cast8 = zext i8 %18 to i32
  // CHECK-NEXT: store volatile i32 %cast8, ptr %tz, align 4
  tz = __builtin_ctzg(vb8);
  // CHECK-NEXT: %19 = load i8, ptr %uc.addr, align 1
  // CHECK-NEXT: %20 = call i8 @llvm.cttz.i8(i8 %19, i1 true)
  // CHECK-NEXT: %cast9 = zext i8 %20 to i32
  // CHECK-NEXT: %iszero = icmp eq i8 %19, 0
  // CHECK-NEXT: %21 = load i8, ptr %sc.addr, align 1
  // CHECK-NEXT: %conv = sext i8 %21 to i32
  // CHECK-NEXT: %ctzg = select i1 %iszero, i32 %conv, i32 %cast9
  // CHECK-NEXT: store volatile i32 %ctzg, ptr %tz, align 4
  tz = __builtin_ctzg(uc, sc);
  // CHECK-NEXT: %22 = load i16, ptr %us.addr, align 2
  // CHECK-NEXT: %23 = call i16 @llvm.cttz.i16(i16 %22, i1 true)
  // CHECK-NEXT: %cast10 = zext i16 %23 to i32
  // CHECK-NEXT: %iszero11 = icmp eq i16 %22, 0
  // CHECK-NEXT: %24 = load i8, ptr %uc.addr, align 1
  // CHECK-NEXT: %conv12 = zext i8 %24 to i32
  // CHECK-NEXT: %ctzg13 = select i1 %iszero11, i32 %conv12, i32 %cast10
  // CHECK-NEXT: store volatile i32 %ctzg13, ptr %tz, align 4
  tz = __builtin_ctzg(us, uc);
  // CHECK-NEXT: %25 = load i32, ptr %ui.addr, align 4
  // CHECK-NEXT: %26 = call i32 @llvm.cttz.i32(i32 %25, i1 true)
  // CHECK-NEXT: %iszero14 = icmp eq i32 %25, 0
  // CHECK-NEXT: %27 = load i16, ptr %s.addr, align 2
  // CHECK-NEXT: %conv15 = sext i16 %27 to i32
  // CHECK-NEXT: %ctzg16 = select i1 %iszero14, i32 %conv15, i32 %26
  // CHECK-NEXT: store volatile i32 %ctzg16, ptr %tz, align 4
  tz = __builtin_ctzg(ui, s);
  // CHECK-NEXT: %28 = load i64, ptr %ul.addr, align 8
  // CHECK-NEXT: %29 = call i64 @llvm.cttz.i64(i64 %28, i1 true)
  // CHECK-NEXT: %cast17 = trunc i64 %29 to i32
  // CHECK-NEXT: %iszero18 = icmp eq i64 %28, 0
  // CHECK-NEXT: %30 = load i16, ptr %us.addr, align 2
  // CHECK-NEXT: %conv19 = zext i16 %30 to i32
  // CHECK-NEXT: %ctzg20 = select i1 %iszero18, i32 %conv19, i32 %cast17
  // CHECK-NEXT: store volatile i32 %ctzg20, ptr %tz, align 4
  tz = __builtin_ctzg(ul, us);
  // CHECK-NEXT: %31 = load i64, ptr %ull.addr, align 8
  // CHECK-NEXT: %32 = call i64 @llvm.cttz.i64(i64 %31, i1 true)
  // CHECK-NEXT: %cast21 = trunc i64 %32 to i32
  // CHECK-NEXT: %iszero22 = icmp eq i64 %31, 0
  // CHECK-NEXT: %33 = load i32, ptr %i.addr, align 4
  // CHECK-NEXT: %ctzg23 = select i1 %iszero22, i32 %33, i32 %cast21
  // CHECK-NEXT: store volatile i32 %ctzg23, ptr %tz, align 4
  tz = __builtin_ctzg(ull, i);
  // CHECK-NEXT: %34 = load i128, ptr %ui128.addr, align 16
  // CHECK-NEXT: %35 = call i128 @llvm.cttz.i128(i128 %34, i1 true)
  // CHECK-NEXT: %cast24 = trunc i128 %35 to i32
  // CHECK-NEXT: %iszero25 = icmp eq i128 %34, 0
  // CHECK-NEXT: %36 = load i32, ptr %i.addr, align 4
  // CHECK-NEXT: %ctzg26 = select i1 %iszero25, i32 %36, i32 %cast24
  // CHECK-NEXT: store volatile i32 %ctzg26, ptr %tz, align 4
  tz = __builtin_ctzg(ui128, i);
  // CHECK-NEXT: %37 = load i128, ptr %ubi128.addr, align 8
  // CHECK-NEXT: %38 = call i128 @llvm.cttz.i128(i128 %37, i1 true)
  // CHECK-NEXT: %cast27 = trunc i128 %38 to i32
  // CHECK-NEXT: %iszero28 = icmp eq i128 %37, 0
  // CHECK-NEXT: %39 = load i32, ptr %i.addr, align 4
  // CHECK-NEXT: %ctzg29 = select i1 %iszero28, i32 %39, i32 %cast27
  // CHECK-NEXT: store volatile i32 %ctzg29, ptr %tz, align 4
  tz = __builtin_ctzg(ubi128, i);
  // CHECK-NEXT: %load_bits30 = load i8, ptr %vb8.addr, align 1
  // CHECK-NEXT: %40 = bitcast i8 %load_bits30 to <8 x i1>
  // CHECK-NEXT: %41 = bitcast <8 x i1> %40 to i8
  // CHECK-NEXT: %42 = call i8 @llvm.cttz.i8(i8 %41, i1 true)
  // CHECK-NEXT: %cast31 = zext i8 %42 to i32
  // CHECK-NEXT: %iszero32 = icmp eq i8 %41, 0
  // CHECK-NEXT: %43 = load i32, ptr %i.addr, align 4
  // CHECK-NEXT: %ctzg33 = select i1 %iszero32, i32 %43, i32 %cast31
  // CHECK-NEXT: store volatile i32 %ctzg33, ptr %tz, align 4
  tz = __builtin_ctzg(vb8, i);
}

#endif
