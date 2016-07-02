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
  V(strncat, (s0, s1, n));
  V(strchr, (s0, s1[0]));
  V(strrchr, (s0, s1[0]));
  V(strcpy, (s0, s1));
  V(strncpy, (s0, s1, n));
  
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

  return 0;
}



void foo() {
 __builtin_strcat(0, 0);
}

// CHECK-LABEL: define void @bar(
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


// CHECK-LABEL: define void @test_float_builtins
void test_float_builtins(float F, double D, long double LD) {
  volatile int res;
  res = __builtin_isinf(F);
  // CHECK:  call float @llvm.fabs.f32(float
  // CHECK:  fcmp oeq float {{.*}}, 0x7FF0000000000000

  res = __builtin_isinf(D);
  // CHECK:  call double @llvm.fabs.f64(double
  // CHECK:  fcmp oeq double {{.*}}, 0x7FF0000000000000

  res = __builtin_isinf(LD);
  // CHECK:  call x86_fp80 @llvm.fabs.f80(x86_fp80
  // CHECK:  fcmp oeq x86_fp80 {{.*}}, 0xK7FFF8000000000000000

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

  res = __builtin_isfinite(F);
  // CHECK: call float @llvm.fabs.f32(float
  // CHECK: fcmp one float {{.*}}, 0x7FF0000000000000

  res = __builtin_isnormal(F);
  // CHECK: fcmp oeq float
  // CHECK: call float @llvm.fabs.f32(float
  // CHECK: fcmp ult float {{.*}}, 0x7FF0000000000000
  // CHECK: fcmp uge float {{.*}}, 0x3810000000000000
  // CHECK: and i1
  // CHECK: and i1
}

// CHECK-LABEL: define void @test_float_builtin_ops
void test_float_builtin_ops(float F, double D, long double LD) {
  volatile float resf;
  volatile double resd;
  volatile long double resld;

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

}

// __builtin_longjmp isn't supported on all platforms, so only test it on X86.
#ifdef __x86_64__
// CHECK-LABEL: define void @test_builtin_longjmp
void test_builtin_longjmp(void **buffer) {
  // CHECK: [[BITCAST:%.*]] = bitcast
  // CHECK-NEXT: call void @llvm.eh.sjlj.longjmp(i8* [[BITCAST]])
  __builtin_longjmp(buffer, 1);
  // CHECK-NEXT: unreachable
}
#endif

// CHECK-LABEL: define i64 @test_builtin_readcyclecounter
long long test_builtin_readcyclecounter() {
  // CHECK: call i64 @llvm.readcyclecounter()
  return __builtin_readcyclecounter();
}

// Behavior of __builtin_os_log differs between platforms, so only test on X86
#ifdef __x86_64__
// CHECK-LABEL: define void @test_builtin_os_log
// CHECK: (i8* [[BUF:%.*]], i32 [[I:%.*]], i8* [[DATA:%.*]])
void test_builtin_os_log(void *buf, int i, const char *data) {
  volatile int len;
  // CHECK: store i8* [[BUF]], i8** [[BUF_ADDR:%.*]], align 8
  // CHECK: store i32 [[I]], i32* [[I_ADDR:%.*]], align 4
  // CHECK: store i8* [[DATA]], i8** [[DATA_ADDR:%.*]], align 8

  // CHECK: store volatile i32 34
  len = __builtin_os_log_format_buffer_size("%d %{public}s %{private}.16P", i, data, data);

  // CHECK: [[BUF2:%.*]] = load i8*, i8** [[BUF_ADDR]]
  // CHECK: [[SUMMARY:%.*]] = getelementptr i8, i8* [[BUF2]], i64 0
  // CHECK: store i8 3, i8* [[SUMMARY]]
  // CHECK: [[NUM_ARGS:%.*]] = getelementptr i8, i8* [[BUF2]], i64 1
  // CHECK: store i8 4, i8* [[NUM_ARGS]]
  //
  // CHECK: [[ARG1_DESC:%.*]] = getelementptr i8, i8* [[BUF2]], i64 2
  // CHECK: store i8 0, i8* [[ARG1_DESC]]
  // CHECK: [[ARG1_SIZE:%.*]] = getelementptr i8, i8* [[BUF2]], i64 3
  // CHECK: store i8 4, i8* [[ARG1_SIZE]]
  // CHECK: [[ARG1:%.*]] = getelementptr i8, i8* [[BUF2]], i64 4
  // CHECK: [[ARG1_INT:%.*]] = bitcast i8* [[ARG1]] to i32*
  // CHECK: [[I2:%.*]] = load i32, i32* [[I_ADDR]]
  // CHECK: store i32 [[I2]], i32* [[ARG1_INT]]

  // CHECK: [[ARG2_DESC:%.*]] = getelementptr i8, i8* [[BUF2]], i64 8
  // CHECK: store i8 34, i8* [[ARG2_DESC]]
  // CHECK: [[ARG2_SIZE:%.*]] = getelementptr i8, i8* [[BUF2]], i64 9
  // CHECK: store i8 8, i8* [[ARG2_SIZE]]
  // CHECK: [[ARG2:%.*]] = getelementptr i8, i8* [[BUF2]], i64 10
  // CHECK: [[ARG2_PTR:%.*]] = bitcast i8* [[ARG2]] to i8**
  // CHECK: [[DATA2:%.*]] = load i8*, i8** [[DATA_ADDR]]
  // CHECK: store i8* [[DATA2]], i8** [[ARG2_PTR]]

  // CHECK: [[ARG3_DESC:%.*]] = getelementptr i8, i8* [[BUF2]], i64 18
  // CHECK: store i8 17, i8* [[ARG3_DESC]]
  // CHECK: [[ARG3_SIZE:%.*]] = getelementptr i8, i8* [[BUF2]], i64 19
  // CHECK: store i8 4, i8* [[ARG3_SIZE]]
  // CHECK: [[ARG3:%.*]] = getelementptr i8, i8* [[BUF2]], i64 20
  // CHECK: [[ARG3_INT:%.*]] = bitcast i8* [[ARG3]] to i32*
  // CHECK: store i32 16, i32* [[ARG3_INT]]

  // CHECK: [[ARG4_DESC:%.*]] = getelementptr i8, i8* [[BUF2]], i64 24
  // CHECK: store i8 49, i8* [[ARG4_DESC]]
  // CHECK: [[ARG4_SIZE:%.*]] = getelementptr i8, i8* [[BUF2]], i64 25
  // CHECK: store i8 8, i8* [[ARG4_SIZE]]
  // CHECK: [[ARG4:%.*]] = getelementptr i8, i8* [[BUF2]], i64 26
  // CHECK: [[ARG4_PTR:%.*]] = bitcast i8* [[ARG4]] to i8**
  // CHECK: [[DATA3:%.*]] = load i8*, i8** [[DATA_ADDR]]
  // CHECK: store i8* [[DATA3]], i8** [[ARG4_PTR]]

  __builtin_os_log_format(buf, "%d %{public}s %{private}.16P", i, data, data);
}

#endif
