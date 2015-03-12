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
  // FIXME:
  //  P(isinf_sign, (1.0));

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
  
  res = __builtin_isfinite(F);
  // CHECK: fcmp oeq float 
  // CHECK: call float @llvm.fabs.f32(float
  // CHECK: fcmp une float {{.*}}, 0x7FF0000000000000
  // CHECK: and i1 

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
