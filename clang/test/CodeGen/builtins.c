// RUN: %clang_cc1 -emit-llvm -o %t %s
// RUN: not grep __builtin %t
// RUN: %clang_cc1 %s -emit-llvm -o - -triple x86_64-darwin-apple | FileCheck %s

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

int main() {
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



void foo() {
 __builtin_strcat(0, 0);
}

// CHECK-LABEL: define{{.*}} void @bar(
void bar() {
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
void test_conditional_bzero() {
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

// CHECK-LABEL: define{{.*}} void @test_float_builtins
void test_float_builtins(__fp16 *H, float F, double D, long double LD) {
  volatile int res;
  res = __builtin_isinf(*H);
  // CHECK:  call half @llvm.fabs.f16(half
  // CHECK:  fcmp oeq half {{.*}}, 0xH7C00

  res = __builtin_isinf(F);
  // CHECK:  call float @llvm.fabs.f32(float
  // CHECK:  fcmp oeq float {{.*}}, 0x7FF0000000000000

  res = __builtin_isinf(D);
  // CHECK:  call double @llvm.fabs.f64(double
  // CHECK:  fcmp oeq double {{.*}}, 0x7FF0000000000000

  res = __builtin_isinf(LD);
  // CHECK:  call x86_fp80 @llvm.fabs.f80(x86_fp80
  // CHECK:  fcmp oeq x86_fp80 {{.*}}, 0xK7FFF8000000000000000

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
  // CHECK: call half @llvm.fabs.f16(half
  // CHECK: fcmp one half {{.*}}, 0xH7C00

  res = __builtin_isfinite(F);
  // CHECK: call float @llvm.fabs.f32(float
  // CHECK: fcmp one float {{.*}}, 0x7FF0000000000000

  res = finite(D);
  // CHECK: call double @llvm.fabs.f64(double
  // CHECK: fcmp one double {{.*}}, 0x7FF0000000000000

  res = __builtin_isnormal(*H);
  // CHECK: fcmp oeq half
  // CHECK: call half @llvm.fabs.f16(half
  // CHECK: fcmp ult half {{.*}}, 0xH7C00
  // CHECK: fcmp uge half {{.*}}, 0xH0400
  // CHECK: and i1
  // CHECK: and i1

  res = __builtin_isnormal(F);
  // CHECK: fcmp oeq float
  // CHECK: call float @llvm.fabs.f32(float
  // CHECK: fcmp ult float {{.*}}, 0x7FF0000000000000
  // CHECK: fcmp uge float {{.*}}, 0x3810000000000000
  // CHECK: and i1
  // CHECK: and i1

  res = __builtin_flt_rounds();
  // CHECK: call i32 @llvm.flt.rounds(
}

// CHECK-LABEL: define{{.*}} void @test_float_builtin_ops
void test_float_builtin_ops(float F, double D, long double LD) {
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

// CHECK-LABEL: define{{.*}} void @test_builtin_longjmp
void test_builtin_longjmp(void **buffer) {
  // CHECK: [[BITCAST:%.*]] = bitcast
  // CHECK-NEXT: call void @llvm.eh.sjlj.longjmp(i8* [[BITCAST]])
  __builtin_longjmp(buffer, 1);
  // CHECK-NEXT: unreachable
}

#endif

// CHECK-LABEL: define{{.*}} void @test_memory_builtins
void test_memory_builtins(int n) {
  // CHECK: call i8* @malloc
  void * p = __builtin_malloc(n);
  // CHECK: call void @free
  __builtin_free(p);
  // CHECK: call i8* @calloc
  p = __builtin_calloc(1, n);
  // CHECK: call i8* @realloc
  p = __builtin_realloc(p, n);
  // CHECK: call void @free
  __builtin_free(p);
}

// CHECK-LABEL: define{{.*}} i64 @test_builtin_readcyclecounter
long long test_builtin_readcyclecounter() {
  // CHECK: call i64 @llvm.readcyclecounter()
  return __builtin_readcyclecounter();
}

/// __builtin_launder should be a NOP in C since there are no vtables.
// CHECK-LABEL: define{{.*}} void @test_builtin_launder
void test_builtin_launder(int *p) {
  // CHECK: [[TMP:%.*]] = load i32*,
  // CHECK-NOT: @llvm.launder
  // CHECK: store i32* [[TMP]],
  int *d = __builtin_launder(p);
}

// __warn_memset_zero_len should be NOP, see https://sourceware.org/bugzilla/show_bug.cgi?id=25399
// CHECK-LABEL: define{{.*}} void @test___warn_memset_zero_len
void test___warn_memset_zero_len() {
  // CHECK-NOT: @__warn_memset_zero_len
  __warn_memset_zero_len();
}

// Behavior of __builtin_os_log differs between platforms, so only test on X86
#ifdef __x86_64__

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log
// CHECK: (i8* %[[BUF:.*]], i32 %[[I:.*]], i8* %[[DATA:.*]])
void test_builtin_os_log(void *buf, int i, const char *data) {
  volatile int len;
  // CHECK: %[[BUF_ADDR:.*]] = alloca i8*, align 8
  // CHECK: %[[I_ADDR:.*]] = alloca i32, align 4
  // CHECK: %[[DATA_ADDR:.*]] = alloca i8*, align 8
  // CHECK: %[[LEN:.*]] = alloca i32, align 4
  // CHECK: store i8* %[[BUF]], i8** %[[BUF_ADDR]], align 8
  // CHECK: store i32 %[[I]], i32* %[[I_ADDR]], align 4
  // CHECK: store i8* %[[DATA]], i8** %[[DATA_ADDR]], align 8

  // CHECK: store volatile i32 34, i32* %[[LEN]]
  len = __builtin_os_log_format_buffer_size("%d %{public}s %{private}.16P", i, data, data);

  // CHECK: %[[V1:.*]] = load i8*, i8** %[[BUF_ADDR]]
  // CHECK: %[[V2:.*]] = load i32, i32* %[[I_ADDR]]
  // CHECK: %[[V3:.*]] = load i8*, i8** %[[DATA_ADDR]]
  // CHECK: %[[V4:.*]] = ptrtoint i8* %[[V3]] to i64
  // CHECK: %[[V5:.*]] = load i8*, i8** %[[DATA_ADDR]]
  // CHECK: %[[V6:.*]] = ptrtoint i8* %[[V5]] to i64
  // CHECK: call void @__os_log_helper_1_3_4_4_0_8_34_4_17_8_49(i8* %[[V1]], i32 %[[V2]], i64 %[[V4]], i32 16, i64 %[[V6]])
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

  // CHECK: store volatile i32 22, i32* %[[LEN]], align 4
  len = __builtin_os_log_format_buffer_size("%{mask.xyz}s", "abc");

  // CHECK: call void @__os_log_helper_1_2_2_8_112_8_34(i8* {{.*}}, i64 8026488
  __builtin_os_log_format(buf, "%{mask.xyz, public}s", "abc");

  // CHECK: call void @__os_log_helper_1_3_2_8_112_4_1(i8* {{.*}}, i64 8026488
  __builtin_os_log_format(buf, "%{ mask.xyz, private }d", 11);

  // Mask type is silently ignored.
  // CHECK: call void @__os_log_helper_1_2_1_8_32(
  __builtin_os_log_format(buf, "%{ mask. xyz }s", "abc");

  // CHECK: call void @__os_log_helper_1_2_1_8_32(
  __builtin_os_log_format(buf, "%{ mask.xy z }s", "abc");
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_3_4_4_0_8_34_4_17_8_49
// CHECK: (i8* %[[BUFFER:.*]], i32 %[[ARG0:.*]], i64 %[[ARG1:.*]], i32 %[[ARG2:.*]], i64 %[[ARG3:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca i8*, align 8
// CHECK: %[[ARG0_ADDR:.*]] = alloca i32, align 4
// CHECK: %[[ARG1_ADDR:.*]] = alloca i64, align 8
// CHECK: %[[ARG2_ADDR:.*]] = alloca i32, align 4
// CHECK: %[[ARG3_ADDR:.*]] = alloca i64, align 8
// CHECK: store i8* %[[BUFFER]], i8** %[[BUFFER_ADDR]], align 8
// CHECK: store i32 %[[ARG0]], i32* %[[ARG0_ADDR]], align 4
// CHECK: store i64 %[[ARG1]], i64* %[[ARG1_ADDR]], align 8
// CHECK: store i32 %[[ARG2]], i32* %[[ARG2_ADDR]], align 4
// CHECK: store i64 %[[ARG3]], i64* %[[ARG3_ADDR]], align 8
// CHECK: %[[BUF:.*]] = load i8*, i8** %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, i8* %[[BUF]], i64 0
// CHECK: store i8 3, i8* %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, i8* %[[BUF]], i64 1
// CHECK: store i8 4, i8* %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, i8* %[[BUF]], i64 2
// CHECK: store i8 0, i8* %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, i8* %[[BUF]], i64 3
// CHECK: store i8 4, i8* %[[ARGSIZE]], align 1
// CHECK: %[[ARGDATA:.*]] = getelementptr i8, i8* %[[BUF]], i64 4
// CHECK: %[[ARGDATACAST:.*]] = bitcast i8* %[[ARGDATA]] to i32*
// CHECK: %[[V0:.*]] = load i32, i32* %[[ARG0_ADDR]], align 4
// CHECK: store i32 %[[V0]], i32* %[[ARGDATACAST]], align 1
// CHECK: %[[ARGDESCRIPTOR1:.*]] = getelementptr i8, i8* %[[BUF]], i64 8
// CHECK: store i8 34, i8* %[[ARGDESCRIPTOR1]], align 1
// CHECK: %[[ARGSIZE2:.*]] = getelementptr i8, i8* %[[BUF]], i64 9
// CHECK: store i8 8, i8* %[[ARGSIZE2]], align 1
// CHECK: %[[ARGDATA3:.*]] = getelementptr i8, i8* %[[BUF]], i64 10
// CHECK: %[[ARGDATACAST4:.*]] = bitcast i8* %[[ARGDATA3]] to i64*
// CHECK: %[[V1:.*]] = load i64, i64* %[[ARG1_ADDR]], align 8
// CHECK: store i64 %[[V1]], i64* %[[ARGDATACAST4]], align 1
// CHECK: %[[ARGDESCRIPTOR5:.*]] = getelementptr i8, i8* %[[BUF]], i64 18
// CHECK: store i8 17, i8* %[[ARGDESCRIPTOR5]], align 1
// CHECK: %[[ARGSIZE6:.*]] = getelementptr i8, i8* %[[BUF]], i64 19
// CHECK: store i8 4, i8* %[[ARGSIZE6]], align 1
// CHECK: %[[ARGDATA7:.*]] = getelementptr i8, i8* %[[BUF]], i64 20
// CHECK: %[[ARGDATACAST8:.*]] = bitcast i8* %[[ARGDATA7]] to i32*
// CHECK: %[[V2:.*]] = load i32, i32* %[[ARG2_ADDR]], align 4
// CHECK: store i32 %[[V2]], i32* %[[ARGDATACAST8]], align 1
// CHECK: %[[ARGDESCRIPTOR9:.*]] = getelementptr i8, i8* %[[BUF]], i64 24
// CHECK: store i8 49, i8* %[[ARGDESCRIPTOR9]], align 1
// CHECK: %[[ARGSIZE10:.*]] = getelementptr i8, i8* %[[BUF]], i64 25
// CHECK: store i8 8, i8* %[[ARGSIZE10]], align 1
// CHECK: %[[ARGDATA11:.*]] = getelementptr i8, i8* %[[BUF]], i64 26
// CHECK: %[[ARGDATACAST12:.*]] = bitcast i8* %[[ARGDATA11]] to i64*
// CHECK: %[[V3:.*]] = load i64, i64* %[[ARG3_ADDR]], align 8
// CHECK: store i64 %[[V3]], i64* %[[ARGDATACAST12]], align 1

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_wide
// CHECK: (i8* %[[BUF:.*]], i8* %[[DATA:.*]], i32* %[[STR:.*]])
typedef int wchar_t;
void test_builtin_os_log_wide(void *buf, const char *data, wchar_t *str) {
  volatile int len;

  // CHECK: %[[BUF_ADDR:.*]] = alloca i8*, align 8
  // CHECK: %[[DATA_ADDR:.*]] = alloca i8*, align 8
  // CHECK: %[[STR_ADDR:.*]] = alloca i32*, align 8
  // CHECK: %[[LEN:.*]] = alloca i32, align 4
  // CHECK: store i8* %[[BUF]], i8** %[[BUF_ADDR]], align 8
  // CHECK: store i8* %[[DATA]], i8** %[[DATA_ADDR]], align 8
  // CHECK: store i32* %[[STR]], i32** %[[STR_ADDR]], align 8

  // CHECK: store volatile i32 12, i32* %[[LEN]], align 4
  len = __builtin_os_log_format_buffer_size("%S", str);

  // CHECK: %[[V1:.*]] = load i8*, i8** %[[BUF_ADDR]], align 8
  // CHECK: %[[V2:.*]] = load i32*, i32** %[[STR_ADDR]], align 8
  // CHECK: %[[V3:.*]] = ptrtoint i32* %[[V2]] to i64
  // CHECK: call void @__os_log_helper_1_2_1_8_80(i8* %[[V1]], i64 %[[V3]])

  __builtin_os_log_format(buf, "%S", str);
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_2_1_8_80
// CHECK: (i8* %[[BUFFER:.*]], i64 %[[ARG0:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca i8*, align 8
// CHECK: %[[ARG0_ADDR:.*]] = alloca i64, align 8
// CHECK: store i8* %[[BUFFER]], i8** %[[BUFFER_ADDR]], align 8
// CHECK: store i64 %[[ARG0]], i64* %[[ARG0_ADDR]], align 8
// CHECK: %[[BUF:.*]] = load i8*, i8** %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, i8* %[[BUF]], i64 0
// CHECK: store i8 2, i8* %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, i8* %[[BUF]], i64 1
// CHECK: store i8 1, i8* %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, i8* %[[BUF]], i64 2
// CHECK: store i8 80, i8* %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, i8* %[[BUF]], i64 3
// CHECK: store i8 8, i8* %[[ARGSIZE]], align 1
// CHECK: %[[ARGDATA:.*]] = getelementptr i8, i8* %[[BUF]], i64 4
// CHECK: %[[ARGDATACAST:.*]] = bitcast i8* %[[ARGDATA]] to i64*
// CHECK: %[[V0:.*]] = load i64, i64* %[[ARG0_ADDR]], align 8
// CHECK: store i64 %[[V0]], i64* %[[ARGDATACAST]], align 1

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_precision_width
// CHECK: (i8* %[[BUF:.*]], i8* %[[DATA:.*]], i32 %[[PRECISION:.*]], i32 %[[WIDTH:.*]])
void test_builtin_os_log_precision_width(void *buf, const char *data,
                                         int precision, int width) {
  volatile int len;
  // CHECK: %[[BUF_ADDR:.*]] = alloca i8*, align 8
  // CHECK: %[[DATA_ADDR:.*]] = alloca i8*, align 8
  // CHECK: %[[PRECISION_ADDR:.*]] = alloca i32, align 4
  // CHECK: %[[WIDTH_ADDR:.*]] = alloca i32, align 4
  // CHECK: %[[LEN:.*]] = alloca i32, align 4
  // CHECK: store i8* %[[BUF]], i8** %[[BUF_ADDR]], align 8
  // CHECK: store i8* %[[DATA]], i8** %[[DATA_ADDR]], align 8
  // CHECK: store i32 %[[PRECISION]], i32* %[[PRECISION_ADDR]], align 4
  // CHECK: store i32 %[[WIDTH]], i32* %[[WIDTH_ADDR]], align 4

  // CHECK: store volatile i32 24, i32* %[[LEN]], align 4
  len = __builtin_os_log_format_buffer_size("Hello %*.*s World", precision, width, data);

  // CHECK: %[[V1:.*]] = load i8*, i8** %[[BUF_ADDR]], align 8
  // CHECK: %[[V2:.*]] = load i32, i32* %[[PRECISION_ADDR]], align 4
  // CHECK: %[[V3:.*]] = load i32, i32* %[[WIDTH_ADDR]], align 4
  // CHECK: %[[V4:.*]] = load i8*, i8** %[[DATA_ADDR]], align 8
  // CHECK: %[[V5:.*]] = ptrtoint i8* %[[V4]] to i64
  // CHECK: call void @__os_log_helper_1_2_3_4_0_4_16_8_32(i8* %[[V1]], i32 %[[V2]], i32 %[[V3]], i64 %[[V5]])
  __builtin_os_log_format(buf, "Hello %*.*s World", precision, width, data);
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_2_3_4_0_4_16_8_32
// CHECK: (i8* %[[BUFFER:.*]], i32 %[[ARG0:.*]], i32 %[[ARG1:.*]], i64 %[[ARG2:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca i8*, align 8
// CHECK: %[[ARG0_ADDR:.*]] = alloca i32, align 4
// CHECK: %[[ARG1_ADDR:.*]] = alloca i32, align 4
// CHECK: %[[ARG2_ADDR:.*]] = alloca i64, align 8
// CHECK: store i8* %[[BUFFER]], i8** %[[BUFFER_ADDR]], align 8
// CHECK: store i32 %[[ARG0]], i32* %[[ARG0_ADDR]], align 4
// CHECK: store i32 %[[ARG1]], i32* %[[ARG1_ADDR]], align 4
// CHECK: store i64 %[[ARG2]], i64* %[[ARG2_ADDR]], align 8
// CHECK: %[[BUF:.*]] = load i8*, i8** %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, i8* %[[BUF]], i64 0
// CHECK: store i8 2, i8* %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, i8* %[[BUF]], i64 1
// CHECK: store i8 3, i8* %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, i8* %[[BUF]], i64 2
// CHECK: store i8 0, i8* %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, i8* %[[BUF]], i64 3
// CHECK: store i8 4, i8* %[[ARGSIZE]], align 1
// CHECK: %[[ARGDATA:.*]] = getelementptr i8, i8* %[[BUF]], i64 4
// CHECK: %[[ARGDATACAST:.*]] = bitcast i8* %[[ARGDATA]] to i32*
// CHECK: %[[V0:.*]] = load i32, i32* %[[ARG0_ADDR]], align 4
// CHECK: store i32 %[[V0]], i32* %[[ARGDATACAST]], align 1
// CHECK: %[[ARGDESCRIPTOR1:.*]] = getelementptr i8, i8* %[[BUF]], i64 8
// CHECK: store i8 16, i8* %[[ARGDESCRIPTOR1]], align 1
// CHECK: %[[ARGSIZE2:.*]] = getelementptr i8, i8* %[[BUF]], i64 9
// CHECK: store i8 4, i8* %[[ARGSIZE2]], align 1
// CHECK: %[[ARGDATA3:.*]] = getelementptr i8, i8* %[[BUF]], i64 10
// CHECK: %[[ARGDATACAST4:.*]] = bitcast i8* %[[ARGDATA3]] to i32*
// CHECK: %[[V1:.*]] = load i32, i32* %[[ARG1_ADDR]], align 4
// CHECK: store i32 %[[V1]], i32* %[[ARGDATACAST4]], align 1
// CHECK: %[[ARGDESCRIPTOR5:.*]] = getelementptr i8, i8* %[[BUF]], i64 14
// CHECK: store i8 32, i8* %[[ARGDESCRIPTOR5]], align 1
// CHECK: %[[ARGSIZE6:.*]] = getelementptr i8, i8* %[[BUF]], i64 15
// CHECK: store i8 8, i8* %[[ARGSIZE6]], align 1
// CHECK: %[[ARGDATA7:.*]] = getelementptr i8, i8* %[[BUF]], i64 16
// CHECK: %[[ARGDATACAST8:.*]] = bitcast i8* %[[ARGDATA7]] to i64*
// CHECK: %[[V2:.*]] = load i64, i64* %[[ARG2_ADDR]], align 8
// CHECK: store i64 %[[V2]], i64* %[[ARGDATACAST8]], align 1

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_invalid
// CHECK: (i8* %[[BUF:.*]], i32 %[[DATA:.*]])
void test_builtin_os_log_invalid(void *buf, int data) {
  volatile int len;
  // CHECK: %[[BUF_ADDR:.*]] = alloca i8*, align 8
  // CHECK: %[[DATA_ADDR:.*]] = alloca i32, align 4
  // CHECK: %[[LEN:.*]] = alloca i32, align 4
  // CHECK: store i8* %[[BUF]], i8** %[[BUF_ADDR]], align 8
  // CHECK: store i32 %[[DATA]], i32* %[[DATA_ADDR]], align 4

  // CHECK: store volatile i32 8, i32* %[[LEN]], align 4
  len = __builtin_os_log_format_buffer_size("invalid specifier %: %d even a trailing one%", data);

  // CHECK: %[[V1:.*]] = load i8*, i8** %[[BUF_ADDR]], align 8
  // CHECK: %[[V2:.*]] = load i32, i32* %[[DATA_ADDR]], align 4
  // CHECK: call void @__os_log_helper_1_0_1_4_0(i8* %[[V1]], i32 %[[V2]])

  __builtin_os_log_format(buf, "invalid specifier %: %d even a trailing one%", data);
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_0_1_4_0
// CHECK: (i8* %[[BUFFER:.*]], i32 %[[ARG0:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca i8*, align 8
// CHECK: %[[ARG0_ADDR:.*]] = alloca i32, align 4
// CHECK: store i8* %[[BUFFER]], i8** %[[BUFFER_ADDR]], align 8
// CHECK: store i32 %[[ARG0]], i32* %[[ARG0_ADDR]], align 4
// CHECK: %[[BUF:.*]] = load i8*, i8** %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, i8* %[[BUF]], i64 0
// CHECK: store i8 0, i8* %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, i8* %[[BUF]], i64 1
// CHECK: store i8 1, i8* %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, i8* %[[BUF]], i64 2
// CHECK: store i8 0, i8* %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, i8* %[[BUF]], i64 3
// CHECK: store i8 4, i8* %[[ARGSIZE]], align 1
// CHECK: %[[ARGDATA:.*]] = getelementptr i8, i8* %[[BUF]], i64 4
// CHECK: %[[ARGDATACAST:.*]] = bitcast i8* %[[ARGDATA]] to i32*
// CHECK: %[[V0:.*]] = load i32, i32* %[[ARG0_ADDR]], align 4
// CHECK: store i32 %[[V0]], i32* %[[ARGDATACAST]], align 1

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_percent
// CHECK: (i8* %[[BUF:.*]], i8* %[[DATA1:.*]], i8* %[[DATA2:.*]])
// Check that the %% which does not consume any argument is correctly handled
void test_builtin_os_log_percent(void *buf, const char *data1, const char *data2) {
  volatile int len;
  // CHECK: %[[BUF_ADDR:.*]] = alloca i8*, align 8
  // CHECK: %[[DATA1_ADDR:.*]] = alloca i8*, align 8
  // CHECK: %[[DATA2_ADDR:.*]] = alloca i8*, align 8
  // CHECK: %[[LEN:.*]] = alloca i32, align 4
  // CHECK: store i8* %[[BUF]], i8** %[[BUF_ADDR]], align 8
  // CHECK: store i8* %[[DATA1]], i8** %[[DATA1_ADDR]], align 8
  // CHECK: store i8* %[[DATA2]], i8** %[[DATA2_ADDR]], align 8
  // CHECK: store volatile i32 22, i32* %[[LEN]], align 4

  len = __builtin_os_log_format_buffer_size("%s %% %s", data1, data2);

  // CHECK: %[[V1:.*]] = load i8*, i8** %[[BUF_ADDR]], align 8
  // CHECK: %[[V2:.*]] = load i8*, i8** %[[DATA1_ADDR]], align 8
  // CHECK: %[[V3:.*]] = ptrtoint i8* %[[V2]] to i64
  // CHECK: %[[V4:.*]] = load i8*, i8** %[[DATA2_ADDR]], align 8
  // CHECK: %[[V5:.*]] = ptrtoint i8* %[[V4]] to i64
  // CHECK: call void @__os_log_helper_1_2_2_8_32_8_32(i8* %[[V1]], i64 %[[V3]], i64 %[[V5]])

  __builtin_os_log_format(buf, "%s %% %s", data1, data2);
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_2_2_8_32_8_32
// CHECK: (i8* %[[BUFFER:.*]], i64 %[[ARG0:.*]], i64 %[[ARG1:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca i8*, align 8
// CHECK: %[[ARG0_ADDR:.*]] = alloca i64, align 8
// CHECK: %[[ARG1_ADDR:.*]] = alloca i64, align 8
// CHECK: store i8* %[[BUFFER]], i8** %[[BUFFER_ADDR]], align 8
// CHECK: store i64 %[[ARG0]], i64* %[[ARG0_ADDR]], align 8
// CHECK: store i64 %[[ARG1]], i64* %[[ARG1_ADDR]], align 8
// CHECK: %[[BUF:.*]] = load i8*, i8** %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, i8* %[[BUF]], i64 0
// CHECK: store i8 2, i8* %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, i8* %[[BUF]], i64 1
// CHECK: store i8 2, i8* %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, i8* %[[BUF]], i64 2
// CHECK: store i8 32, i8* %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, i8* %[[BUF]], i64 3
// CHECK: store i8 8, i8* %[[ARGSIZE]], align 1
// CHECK: %[[ARGDATA:.*]] = getelementptr i8, i8* %[[BUF]], i64 4
// CHECK: %[[ARGDATACAST:.*]] = bitcast i8* %[[ARGDATA]] to i64*
// CHECK: %[[V0:.*]] = load i64, i64* %[[ARG0_ADDR]], align 8
// CHECK: store i64 %[[V0]], i64* %[[ARGDATACAST]], align 1
// CHECK: %[[ARGDESCRIPTOR1:.*]] = getelementptr i8, i8* %[[BUF]], i64 12
// CHECK: store i8 32, i8* %[[ARGDESCRIPTOR1]], align 1
// CHECK: %[[ARGSIZE2:.*]] = getelementptr i8, i8* %[[BUF]], i64 13
// CHECK: store i8 8, i8* %[[ARGSIZE2]], align 1
// CHECK: %[[ARGDATA3:.*]] = getelementptr i8, i8* %[[BUF]], i64 14
// CHECK: %[[ARGDATACAST4:.*]] = bitcast i8* %[[ARGDATA3]] to i64*
// CHECK: %[[V1:.*]] = load i64, i64* %[[ARG1_ADDR]], align 8
// CHECK: store i64 %[[V1]], i64* %[[ARGDATACAST4]], align 1

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
void test_builtin_os_log_errno() {
  // CHECK-NOT: @stacksave
  // CHECK: %[[BUF:.*]] = alloca [4 x i8], align 1
  // CHECK: %[[DECAY:.*]] = getelementptr inbounds [4 x i8], [4 x i8]* %[[BUF]], i64 0, i64 0
  // CHECK: call void @__os_log_helper_1_2_1_0_96(i8* %[[DECAY]])
  // CHECK-NOT: @stackrestore

  char buf[__builtin_os_log_format_buffer_size("%m")];
  __builtin_os_log_format(buf, "%m");
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_2_1_0_96
// CHECK: (i8* %[[BUFFER:.*]])

// CHECK: %[[BUFFER_ADDR:.*]] = alloca i8*, align 8
// CHECK: store i8* %[[BUFFER]], i8** %[[BUFFER_ADDR]], align 8
// CHECK: %[[BUF:.*]] = load i8*, i8** %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, i8* %[[BUF]], i64 0
// CHECK: store i8 2, i8* %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, i8* %[[BUF]], i64 1
// CHECK: store i8 1, i8* %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, i8* %[[BUF]], i64 2
// CHECK: store i8 96, i8* %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, i8* %[[BUF]], i64 3
// CHECK: store i8 0, i8* %[[ARGSIZE]], align 1
// CHECK-NEXT: ret void

// CHECK-LABEL: define{{.*}} void @test_builtin_os_log_long_double
// CHECK: (i8* %[[BUF:.*]], x86_fp80 %[[LD:.*]])
void test_builtin_os_log_long_double(void *buf, long double ld) {
  // CHECK: %[[BUF_ADDR:.*]] = alloca i8*, align 8
  // CHECK: %[[LD_ADDR:.*]] = alloca x86_fp80, align 16
  // CHECK: %[[COERCE:.*]] = alloca i128, align 16
  // CHECK: store i8* %[[BUF]], i8** %[[BUF_ADDR]], align 8
  // CHECK: store x86_fp80 %[[LD]], x86_fp80* %[[LD_ADDR]], align 16
  // CHECK: %[[V0:.*]] = load i8*, i8** %[[BUF_ADDR]], align 8
  // CHECK: %[[V1:.*]] = load x86_fp80, x86_fp80* %[[LD_ADDR]], align 16
  // CHECK: %[[V2:.*]] = bitcast x86_fp80 %[[V1]] to i80
  // CHECK: %[[V3:.*]] = zext i80 %[[V2]] to i128
  // CHECK: store i128 %[[V3]], i128* %[[COERCE]], align 16
  // CHECK: %[[V4:.*]] = bitcast i128* %[[COERCE]] to { i64, i64 }*
  // CHECK: %[[V5:.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* %[[V4]], i32 0, i32 0
  // CHECK: %[[V6:.*]] = load i64, i64* %[[V5]], align 16
  // CHECK: %[[V7:.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* %[[V4]], i32 0, i32 1
  // CHECK: %[[V8:.*]] = load i64, i64* %[[V7]], align 8
  // CHECK: call void @__os_log_helper_1_0_1_16_0(i8* %[[V0]], i64 %[[V6]], i64 %[[V8]])

  __builtin_os_log_format(buf, "%Lf", ld);
}

// CHECK-LABEL: define linkonce_odr hidden void @__os_log_helper_1_0_1_16_0
// CHECK: (i8* %[[BUFFER:.*]], i64 %[[ARG0_COERCE0:.*]], i64 %[[ARG0_COERCE1:.*]])

// CHECK: %[[ARG0:.*]] = alloca i128, align 16
// CHECK: %[[BUFFER_ADDR:.*]] = alloca i8*, align 8
// CHECK: %[[ARG0_ADDR:.*]] = alloca i128, align 16
// CHECK: %[[V0:.*]] = bitcast i128* %[[ARG0]] to { i64, i64 }*
// CHECK: %[[V1:.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* %[[V0]], i32 0, i32 0
// CHECK: store i64 %[[ARG0_COERCE0]], i64* %[[V1]], align 16
// CHECK: %[[V2:.*]] = getelementptr inbounds { i64, i64 }, { i64, i64 }* %[[V0]], i32 0, i32 1
// CHECK: store i64 %[[ARG0_COERCE1]], i64* %[[V2]], align 8
// CHECK: %[[ARG01:.*]] = load i128, i128* %[[ARG0]], align 16
// CHECK: store i8* %[[BUFFER]], i8** %[[BUFFER_ADDR]], align 8
// CHECK: store i128 %[[ARG01]], i128* %[[ARG0_ADDR]], align 16
// CHECK: %[[BUF:.*]] = load i8*, i8** %[[BUFFER_ADDR]], align 8
// CHECK: %[[SUMMARY:.*]] = getelementptr i8, i8* %[[BUF]], i64 0
// CHECK: store i8 0, i8* %[[SUMMARY]], align 1
// CHECK: %[[NUMARGS:.*]] = getelementptr i8, i8* %[[BUF]], i64 1
// CHECK: store i8 1, i8* %[[NUMARGS]], align 1
// CHECK: %[[ARGDESCRIPTOR:.*]] = getelementptr i8, i8* %[[BUF]], i64 2
// CHECK: store i8 0, i8* %[[ARGDESCRIPTOR]], align 1
// CHECK: %[[ARGSIZE:.*]] = getelementptr i8, i8* %[[BUF]], i64 3
// CHECK: store i8 16, i8* %[[ARGSIZE]], align 1
// CHECK: %[[ARGDATA:.*]] = getelementptr i8, i8* %[[BUF]], i64 4
// CHECK: %[[ARGDATACAST:.*]] = bitcast i8* %[[ARGDATA]] to i128*
// CHECK: %[[V3:.*]] = load i128, i128* %[[ARG0_ADDR]], align 16
// CHECK: store i128 %[[V3]], i128* %[[ARGDATACAST]], align 1

#endif
