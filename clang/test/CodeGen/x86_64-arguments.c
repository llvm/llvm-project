// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s| FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -o - %s -target-feature +avx | FileCheck %s -check-prefix=AVX
#include <stdarg.h>

// CHECK: define signext i8 @f0()
char f0(void) {
  return 0;
}

// CHECK: define signext i16 @f1()
short f1(void) {
  return 0;
}

// CHECK: define i32 @f2()
int f2(void) {
  return 0;
}

// CHECK: define float @f3()
float f3(void) {
  return 0;
}

// CHECK: define double @f4()
double f4(void) {
  return 0;
}

// CHECK: define x86_fp80 @f5()
long double f5(void) {
  return 0;
}

// CHECK: define void @f6(i8 signext %a0, i16 signext %a1, i32 %a2, i64 %a3, i8* %a4)
void f6(char a0, short a1, int a2, long long a3, void *a4) {
}

// CHECK: define void @f7(i32 %a0)
typedef enum { A, B, C } e7;
void f7(e7 a0) {
}

// Test merging/passing of upper eightbyte with X87 class.
//
// CHECK: define void @f8_1(%union.u8* noalias sret %agg.result)
// CHECK: define void @f8_2(%union.u8* byval align 16 %a0)
union u8 {
  long double a;
  int b;
};
union u8 f8_1() { while (1) {} }
void f8_2(union u8 a0) {}

// CHECK: define i64 @f9()
struct s9 { int a; int b; int : 0; } f9(void) { while (1) {} }

// CHECK: define void @f10(i64 %a0.coerce)
struct s10 { int a; int b; int : 0; };
void f10(struct s10 a0) {}

// CHECK: define void @f11(%union.anon* noalias sret %agg.result)
union { long double a; float b; } f11() { while (1) {} }

// CHECK: define i32 @f12_0()
// CHECK: define void @f12_1(i32 %a0.coerce)
struct s12 { int a __attribute__((aligned(16))); };
struct s12 f12_0(void) { while (1) {} }
void f12_1(struct s12 a0) {}

// Check that sret parameter is accounted for when checking available integer
// registers.
// CHECK: define void @f13(%struct.s13_0* noalias sret %agg.result, i32 %a, i32 %b, i32 %c, i32 %d, {{.*}}* byval align 8 %e, i32 %f)

struct s13_0 { long long f0[3]; };
struct s13_1 { long long f0[2]; };
struct s13_0 f13(int a, int b, int c, int d,
                 struct s13_1 e, int f) { while (1) {} }

// CHECK: define void @f14({{.*}}, i8 signext %X)
void f14(int a, int b, int c, int d, int e, int f, char X) {}

// CHECK: define void @f15({{.*}}, i8* %X)
void f15(int a, int b, int c, int d, int e, int f, void *X) {}

// CHECK: define void @f16({{.*}}, float %X)
void f16(float a, float b, float c, float d, float e, float f, float g, float h,
         float X) {}

// CHECK: define void @f17({{.*}}, x86_fp80 %X)
void f17(float a, float b, float c, float d, float e, float f, float g, float h,
         long double X) {}

// Check for valid coercion.  The struct should be passed/returned as i32, not
// as i64 for better code quality.
// rdar://8135035
// CHECK: define void @f18(i32 %a, i32 %f18_arg1.coerce) 
struct f18_s0 { int f0; };
void f18(int a, struct f18_s0 f18_arg1) { while (1) {} }

// Check byval alignment.

// CHECK: define void @f19(%struct.s19* byval align 16 %x)
struct s19 {
  long double a;
};
void f19(struct s19 x) {}

// CHECK: define void @f20(%struct.s20* byval align 32 %x)
struct __attribute__((aligned(32))) s20 {
  int x;
  int y;
};
void f20(struct s20 x) {}

struct StringRef {
  long x;
  const char *Ptr;
};

// rdar://7375902
// CHECK: define i8* @f21(i64 %S.coerce0, i8* %S.coerce1) 
const char *f21(struct StringRef S) { return S.x+S.Ptr; }

// PR7567
typedef __attribute__ ((aligned(16))) struct f22s { unsigned long long x[2]; } L;
void f22(L x, L y) { }
// CHECK: @f22
// CHECK: %x = alloca{{.*}}, align 16
// CHECK: %y = alloca{{.*}}, align 16



// PR7714
struct f23S {
  short f0;
  unsigned f1;
  int f2;
};


void f23(int A, struct f23S B) {
  // CHECK: define void @f23(i32 %A, i64 %B.coerce0, i32 %B.coerce1)
}

struct f24s { long a; int b; };

struct f23S f24(struct f23S *X, struct f24s *P2) {
  return *X;
  
  // CHECK: define { i64, i32 } @f24(%struct.f23S* %X, %struct.f24s* %P2)
}

// rdar://8248065
typedef float v4f32 __attribute__((__vector_size__(16)));
v4f32 f25(v4f32 X) {
  // CHECK: define <4 x float> @f25(<4 x float> %X)
  // CHECK-NOT: alloca
  // CHECK: alloca <4 x float>
  // CHECK-NOT: alloca
  // CHECK: store <4 x float> %X, <4 x float>*
  // CHECK-NOT: store
  // CHECK: ret <4 x float>
  return X+X;
}

struct foo26 {
  int *X;
  float *Y;
};

struct foo26 f26(struct foo26 *P) {
  // CHECK: define { i32*, float* } @f26(%struct.foo26* %P)
  return *P;
}


struct v4f32wrapper {
  v4f32 v;
};

struct v4f32wrapper f27(struct v4f32wrapper X) {
  // CHECK: define <4 x float> @f27(<4 x float> %X.coerce)
  return X;
}

// rdar://5711709
struct f28c {
  double x;
  int y;
};
void f28(struct f28c C) {
  // CHECK: define void @f28(double %C.coerce0, i32 %C.coerce1)
}

struct f29a {
  struct c {
    double x;
    int y;
  } x[1];
};

void f29a(struct f29a A) {
  // CHECK: define void @f29a(double %A.coerce0, i32 %A.coerce1)
}

// rdar://8249586
struct S0 { char f0[8]; char f2; char f3; char f4; };
void f30(struct S0 p_4) {
  // CHECK: define void @f30(i64 %p_4.coerce0, i24 %p_4.coerce1)
}

// Pass the third element as a float when followed by tail padding.
// rdar://8251384
struct f31foo { float a, b, c; };
float f31(struct f31foo X) {
  // CHECK: define float @f31(<2 x float> %X.coerce0, float %X.coerce1)
  return X.c;
}

_Complex float f32(_Complex float A, _Complex float B) {
  // rdar://6379669
  // CHECK: define <2 x float> @f32(<2 x float> %A.coerce, <2 x float> %B.coerce)
  return A+B;
}


// rdar://8357396
struct f33s { long x; float c,d; };

void f33(va_list X) {
  va_arg(X, struct f33s);
}

typedef unsigned long long v1i64 __attribute__((__vector_size__(8)));

// rdar://8359248
// CHECK: define i64 @f34(i64 %arg.coerce)
v1i64 f34(v1i64 arg) { return arg; }


// rdar://8358475
// CHECK: define i64 @f35(i64 %arg.coerce)
typedef unsigned long v1i64_2 __attribute__((__vector_size__(8)));
v1i64_2 f35(v1i64_2 arg) { return arg+arg; }

// rdar://9122143
// CHECK: declare void @func(%struct._str* byval align 16)
typedef struct _str {
  union {
    long double a;
    long c;
  };
} str;

void func(str s);
str ss;
void f9122143()
{
  func(ss);
}

// CHECK: define double @f36(double %arg.coerce)
typedef unsigned v2i32 __attribute((__vector_size__(8)));
v2i32 f36(v2i32 arg) { return arg; }

// AVX: declare void @f38(<8 x float>)
// AVX: declare void @f37(<8 x float>)
// CHECK: declare void @f38(%struct.s256* byval align 32)
// CHECK: declare void @f37(<8 x float>* byval align 32)
typedef float __m256 __attribute__ ((__vector_size__ (32)));
typedef struct {
  __m256 m;
} s256;

s256 x38;
__m256 x37;

void f38(s256 x);
void f37(__m256 x);
void f39() { f38(x38); f37(x37); }

// The two next tests make sure that the struct below is passed
// in the same way regardless of avx being used

// CHECK: declare void @func40(%struct.t128* byval align 16)
typedef float __m128 __attribute__ ((__vector_size__ (16)));
typedef struct t128 {
  __m128 m;
  __m128 n;
} two128;

extern void func40(two128 s);
void func41(two128 s) {
  func40(s);
}

// CHECK: declare void @func42(%struct.t128_2* byval align 16)
typedef struct xxx {
  __m128 array[2];
} Atwo128;
typedef struct t128_2 {
  Atwo128 x;
} SA;

extern void func42(SA s);
void func43(SA s) {
  func42(s);
}

// CHECK: define i32 @f44
// CHECK: ptrtoint
// CHECK-NEXT: and {{.*}}, -32
// CHECK-NEXT: inttoptr
typedef int T44 __attribute((vector_size(32)));
struct s44 { T44 x; int y; };
int f44(int i, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, i);
  struct s44 s = __builtin_va_arg(ap, struct s44);
  __builtin_va_end(ap);
  return s.y;
}

// Text that vec3 returns the correct LLVM IR type.
// AVX: define i32 @foo(<3 x i64> %X)
typedef long long3 __attribute((ext_vector_type(3)));
int foo(long3 X)
{
  return 0;
}

// Make sure we don't use a varargs convention for a function without a
// prototype where AVX types are involved.
// AVX: @test45
// AVX: call i32 bitcast (i32 (...)* @f45 to i32 (<8 x float>)*)
int f45();
__m256 x45;
void test45() { f45(x45); }

// Make sure we use byval to pass 64-bit vectors in memory; the LLVM call
// lowering can't handle this case correctly because it runs after legalization.
// CHECK: @test46
// CHECK: call void @f46({{.*}}<2 x float>* byval align 8 {{.*}}, <2 x float>* byval align 8 {{.*}})
typedef float v46 __attribute((vector_size(8)));
void f46(v46,v46,v46,v46,v46,v46,v46,v46,v46,v46);
void test46() { v46 x = {1,2}; f46(x,x,x,x,x,x,x,x,x,x); }

// Check that we pass the struct below without using byval, which helps out
// codegen.
//
// CHECK: @test47
// CHECK: call void @f47(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
struct s47 { unsigned a; };
void f47(int,int,int,int,int,int,struct s47);
void test47(int a, struct s47 b) { f47(a, a, a, a, a, a, b); }
