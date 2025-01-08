// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -debug-info-kind=limited -Wno-strict-prototypes -o - %s | \
// RUN:   FileCheck %s -check-prefix=CHECK -check-prefix=SSE -check-prefix=NO-AVX512
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -debug-info-kind=limited -Wno-strict-prototypes -o - %s -target-feature +avx | \
// RUN:   FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=NO-AVX512
// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm -debug-info-kind=limited -Wno-strict-prototypes -o - %s -target-feature +avx512f | \
// RUN:   FileCheck %s -check-prefix=CHECK -check-prefix=AVX -check-prefix=AVX512
#include <stdarg.h>

// CHECK-LABEL: define{{.*}} signext i8 @f0()
char f0(void) {
  return 0;
}

// CHECK-LABEL: define{{.*}} signext i16 @f1()
short f1(void) {
  return 0;
}

// CHECK-LABEL: define{{.*}} i32 @f2()
int f2(void) {
  return 0;
}

// CHECK-LABEL: define{{.*}} float @f3()
float f3(void) {
  return 0;
}

// CHECK-LABEL: define{{.*}} double @f4()
double f4(void) {
  return 0;
}

// CHECK-LABEL: define{{.*}} x86_fp80 @f5()
long double f5(void) {
  return 0;
}

// CHECK-LABEL: define{{.*}} void @f6(i8 noundef signext %a0, i16 noundef signext %a1, i32 noundef %a2, i64 noundef %a3, ptr noundef %a4)
void f6(char a0, short a1, int a2, long long a3, void *a4) {
}

// CHECK-LABEL: define{{.*}} void @f7(i32 noundef %a0)
typedef enum { A, B, C } e7;
void f7(e7 a0) {
}

// Test merging/passing of upper eightbyte with X87 class.
//
// CHECK-LABEL: define{{.*}} void @f8_1(ptr dead_on_unwind noalias writable sret(%union.u8) align 16 %agg.result)
// CHECK-LABEL: define{{.*}} void @f8_2(ptr noundef byval(%union.u8) align 16 %a0)
union u8 {
  long double a;
  int b;
};
union u8 f8_1(void) { while (1) {} }
void f8_2(union u8 a0) {}

// CHECK-LABEL: define{{.*}} i64 @f9()
struct s9 { int a; int b; int : 0; } f9(void) { while (1) {} }

// CHECK-LABEL: define{{.*}} void @f10(i64 %a0.coerce)
struct s10 { int a; int b; int : 0; };
void f10(struct s10 a0) {}

// CHECK-LABEL: define{{.*}} void @f11(ptr dead_on_unwind noalias writable sret(%union.anon) align 16 %agg.result)
union { long double a; float b; } f11(void) { while (1) {} }

// CHECK-LABEL: define{{.*}} i32 @f12_0()
// CHECK-LABEL: define{{.*}} void @f12_1(i32 %a0.coerce)
struct s12 { int a __attribute__((aligned(16))); };
struct s12 f12_0(void) { while (1) {} }
void f12_1(struct s12 a0) {}

// Check that sret parameter is accounted for when checking available integer
// registers.
// CHECK: define{{.*}} void @f13(ptr dead_on_unwind noalias writable sret(%struct.s13_0) align 8 %agg.result, i32 noundef %a, i32 noundef %b, i32 noundef %c, i32 noundef %d, ptr noundef byval({{.*}}) align 8 %e, i32 noundef %f)

struct s13_0 { long long f0[3]; };
struct s13_1 { long long f0[2]; };
struct s13_0 f13(int a, int b, int c, int d,
                 struct s13_1 e, int f) { while (1) {} }

// CHECK: define{{.*}} void @f14({{.*}}, i8 noundef signext %X)
void f14(int a, int b, int c, int d, int e, int f, char X) {}

// CHECK: define{{.*}} void @f15({{.*}}, ptr noundef %X)
void f15(int a, int b, int c, int d, int e, int f, void *X) {}

// CHECK: define{{.*}} void @f16({{.*}}, float noundef %X)
void f16(float a, float b, float c, float d, float e, float f, float g, float h,
         float X) {}

// CHECK: define{{.*}} void @f17({{.*}}, x86_fp80 noundef %X)
void f17(float a, float b, float c, float d, float e, float f, float g, float h,
         long double X) {}

// Check for valid coercion.  The struct should be passed/returned as i32, not
// as i64 for better code quality.
// CHECK-LABEL: define{{.*}} void @f18(i32 noundef %a, i32 %f18_arg1.coerce)
struct f18_s0 { int f0; };
void f18(int a, struct f18_s0 f18_arg1) { while (1) {} }

// Check byval alignment.

// CHECK-LABEL: define{{.*}} void @f19(ptr noundef byval(%struct.s19) align 16 %x)
struct s19 {
  long double a;
};
void f19(struct s19 x) {}

// CHECK-LABEL: define{{.*}} void @f20(ptr noundef byval(%struct.s20) align 32 %x)
struct __attribute__((aligned(32))) s20 {
  int x;
  int y;
};
void f20(struct s20 x) {}

struct StringRef {
  long x;
  const char *Ptr;
};

// CHECK-LABEL: define{{.*}} ptr @f21(i64 %S.coerce0, ptr %S.coerce1)
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
  // CHECK-LABEL: define{{.*}} void @f23(i32 noundef %A, i64 %B.coerce0, i32 %B.coerce1)
}

struct f24s { long a; int b; };

struct f23S f24(struct f23S *X, struct f24s *P2) {
  return *X;

  // CHECK: define{{.*}} { i64, i32 } @f24(ptr noundef %X, ptr noundef %P2)
}

typedef float v4f32 __attribute__((__vector_size__(16)));
v4f32 f25(v4f32 X) {
  // CHECK-LABEL: define{{.*}} <4 x float> @f25(<4 x float> noundef %X)
  // CHECK-NOT: alloca
  // CHECK: alloca <4 x float>
  // CHECK-NOT: alloca
  // CHECK: store <4 x float> %X, ptr
  // CHECK-NOT: store
  // CHECK: ret <4 x float>
  return X+X;
}

struct foo26 {
  int *X;
  float *Y;
};

struct foo26 f26(struct foo26 *P) {
  // CHECK: define{{.*}} { ptr, ptr } @f26(ptr noundef %P)
  return *P;
}


struct v4f32wrapper {
  v4f32 v;
};

struct v4f32wrapper f27(struct v4f32wrapper X) {
  // CHECK-LABEL: define{{.*}} <4 x float> @f27(<4 x float> %X.coerce)
  return X;
}

// PR22563 - We should unwrap simple structs and arrays to pass
// and return them in the appropriate vector registers if possible.

typedef float v8f32 __attribute__((__vector_size__(32)));
struct v8f32wrapper {
  v8f32 v;
};

struct v8f32wrapper f27a(struct v8f32wrapper X) {
  // AVX-LABEL: define{{.*}} <8 x float> @f27a(<8 x float> %X.coerce)
  return X;
}

struct v8f32wrapper_wrapper {
  v8f32 v[1];
};

struct v8f32wrapper_wrapper f27b(struct v8f32wrapper_wrapper X) {
  // AVX-LABEL: define{{.*}} <8 x float> @f27b(<8 x float> %X.coerce)
  return X;
}

struct f28c {
  double x;
  int y;
};
void f28(struct f28c C) {
  // CHECK-LABEL: define{{.*}} void @f28(double %C.coerce0, i32 %C.coerce1)
}

struct f29a {
  struct c {
    double x;
    int y;
  } x[1];
};

void f29a(struct f29a A) {
  // CHECK-LABEL: define{{.*}} void @f29a(double %A.coerce0, i32 %A.coerce1)
}

struct S0 { char f0[8]; char f2; char f3; char f4; };
void f30(struct S0 p_4) {
  // CHECK-LABEL: define{{.*}} void @f30(i64 %p_4.coerce0, i24 %p_4.coerce1)
}

// Pass the third element as a float when followed by tail padding.
struct f31foo { float a, b, c; };
float f31(struct f31foo X) {
  // CHECK-LABEL: define{{.*}} float @f31(<2 x float> %X.coerce0, float %X.coerce1)
  return X.c;
}

_Complex float f32(_Complex float A, _Complex float B) {
  // CHECK-LABEL: define{{.*}} <2 x float> @f32(<2 x float> noundef %A.coerce, <2 x float> noundef %B.coerce)
  return A+B;
}

struct f33s { long x; float c,d; };

void f33(va_list X) {
  va_arg(X, struct f33s);
}

typedef unsigned long long v1i64 __attribute__((__vector_size__(8)));

// CHECK-LABEL: define{{.*}} double @f34(double noundef %arg.coerce)
v1i64 f34(v1i64 arg) { return arg; }

// CHECK-LABEL: define{{.*}} double @f35(double noundef %arg.coerce)
typedef unsigned long v1i64_2 __attribute__((__vector_size__(8)));
v1i64_2 f35(v1i64_2 arg) { return arg+arg; }

// CHECK: declare void @func(ptr noundef byval(%struct._str) align 16)
typedef struct _str {
  union {
    long double a;
    long c;
  };
} str;

void func(str s);
str ss;
void f9122143(void)
{
  func(ss);
}

// CHECK-LABEL: define{{.*}} double @f36(double noundef %arg.coerce)
typedef unsigned v2i32 __attribute((__vector_size__(8)));
v2i32 f36(v2i32 arg) { return arg; }

// AVX: declare void @f38(<8 x float>)
// AVX: declare void @f37(<8 x float> noundef)
// SSE: declare void @f38(ptr noundef byval(%struct.s256) align 32)
// SSE: declare void @f37(ptr noundef byval(<8 x float>) align 32)
typedef float __m256 __attribute__ ((__vector_size__ (32)));
typedef struct {
  __m256 m;
} s256;

s256 x38;
__m256 x37;

void f38(s256 x);
void f37(__m256 x);
void f39(void) { f38(x38); f37(x37); }

// The two next tests make sure that the struct below is passed
// in the same way regardless of avx being used

// CHECK: declare void @func40(ptr noundef byval(%struct.t128) align 16)
typedef float __m128 __attribute__ ((__vector_size__ (16)));
typedef struct t128 {
  __m128 m;
  __m128 n;
} two128;

extern void func40(two128 s);
void func41(two128 s) {
  func40(s);
}

// CHECK: declare void @func42(ptr noundef byval(%struct.t128_2) align 16)
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

// CHECK-LABEL: define{{.*}} i32 @f44
// CHECK: getelementptr inbounds i8, ptr %{{.+}}, i32 31
// CHECK-NEXT: call ptr @llvm.ptrmask.p0.i64(ptr %{{[0-9]+}}, i64 -32)
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
// AVX-LABEL: define{{.*}} i32 @foo(<3 x i64> noundef %X)
typedef long long3 __attribute((ext_vector_type(3)));
int foo(long3 X)
{
  return 0;
}

// Make sure we don't use a varargs convention for a function without a
// prototype where AVX types are involved.
// AVX: @test45
// AVX: call i32 @f45
int f45();
__m256 x45;
void test45(void) { f45(x45); }

// Make sure we use byval to pass 64-bit vectors in memory; the LLVM call
// lowering can't handle this case correctly because it runs after legalization.
// CHECK: @test46
// CHECK: call void @f46({{.*}}ptr noundef byval(<2 x float>) align 8 {{.*}}, ptr noundef byval(<2 x float>) align 8 {{.*}})
typedef float v46 __attribute((vector_size(8)));
void f46(v46,v46,v46,v46,v46,v46,v46,v46,v46,v46);
void test46(void) { v46 x = {1,2}; f46(x,x,x,x,x,x,x,x,x,x); }

// Check that we pass the struct below without using byval, which helps out
// codegen.
//
// CHECK: @test47
// CHECK: call void @f47(i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}}, i32 {{.*}})
struct s47 { unsigned a; };
void f47(int,int,int,int,int,int,struct s47);
void test47(int a, struct s47 b) { f47(a, a, a, a, a, a, b); }

// In the following example, there are holes in T4 at the 3rd byte and the 4th
// byte, however, T2 does not have those holes. T4 is chosen to be the
// representing type for union T1, but we can't use load or store of T4 since
// it will skip the 3rd byte and the 4th byte.
// In general, Since we don't accurately represent the data fields of a union,
// do not use load or store of the representing llvm type for the union.
typedef _Complex int T2;
typedef _Complex char T5;
typedef _Complex int T7;
typedef struct T4 { T5 field0; T7 field1; } T4;
typedef union T1 { T2 field0; T4 field1; } T1;
extern T1 T1_retval;
T1 test48(void) {
// CHECK: @test48
// CHECK: memcpy
// CHECK: memcpy
  return T1_retval;
}

void test49_helper(double, ...);
void test49(double d, double e) {
  test49_helper(d, e);
}
// CHECK-LABEL:    define{{.*}} void @test49(
// CHECK:      [[T0:%.*]] = load double, ptr
// CHECK-NEXT: [[T1:%.*]] = load double, ptr
// CHECK-NEXT: call void (double, ...) @test49_helper(double noundef [[T0]], double noundef [[T1]])

void test50_helper();
void test50(double d, double e) {
  test50_helper(d, e);
}
// CHECK-LABEL:    define{{.*}} void @test50(
// CHECK:      [[T0:%.*]] = load double, ptr
// CHECK-NEXT: [[T1:%.*]] = load double, ptr
// CHECK-NEXT: call void (double, double, ...) @test50_helper(double noundef [[T0]], double noundef [[T1]])

struct test51_s { __uint128_t intval; };
void test51(struct test51_s *s, __builtin_va_list argList) {
    *s = __builtin_va_arg(argList, struct test51_s);
}

// CHECK-LABEL: define{{.*}} void @test51
// CHECK: [[TMP_ADDR:%.*]] = alloca [[STRUCT_TEST51:%.*]], align 16
// CHECK: br i1
// CHECK: [[REG_SAVE_AREA_PTR:%.*]] = getelementptr inbounds {{.*}}, i32 0, i32 3
// CHECK-NEXT: [[REG_SAVE_AREA:%.*]] = load ptr, ptr [[REG_SAVE_AREA_PTR]]
// CHECK-NEXT: [[VALUE_ADDR:%.*]] = getelementptr i8, ptr [[REG_SAVE_AREA]], i32 {{.*}}
// CHECK-NEXT: call void @llvm.memcpy.p0.p0.i64(ptr align 16 [[TMP_ADDR]], ptr align 8 [[VALUE_ADDR]], i64 16, i1 false)
// CHECK-NEXT: add i32 {{.*}}, 16
// CHECK-NEXT: store i32 {{.*}}, ptr {{.*}}
// CHECK-NEXT: br label

void test52_helper(int, ...);
__m256 x52;
void test52(void) {
  test52_helper(0, x52, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0i);
}
// AVX: @test52_helper(i32 noundef 0, <8 x float> noundef {{%[a-zA-Z0-9]+}}, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef {{%[a-zA-Z0-9]+}}, double noundef {{%[a-zA-Z0-9]+}})

void test53(__m256 *m, __builtin_va_list argList) {
  *m = __builtin_va_arg(argList, __m256);
}
// AVX-LABEL: define{{.*}} void @test53
// AVX-NOT: br i1
// AVX: ret void

void test54_helper(__m256, ...);
__m256 x54;
void test54(void) {
  test54_helper(x54, x54, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0i);
  test54_helper(x54, x54, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0i);
}
// AVX: @test54_helper(<8 x float> noundef {{%[a-zA-Z0-9]+}}, <8 x float> noundef {{%[a-zA-Z0-9]+}}, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef {{%[a-zA-Z0-9]+}}, double noundef {{%[a-zA-Z0-9]+}})
// AVX: @test54_helper(<8 x float> noundef {{%[a-zA-Z0-9]+}}, <8 x float> noundef {{%[a-zA-Z0-9]+}}, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, ptr noundef byval({ double, double }) align 8 {{%[^)]+}})

typedef float __m512 __attribute__ ((__vector_size__ (64)));
typedef struct {
  __m512 m;
} s512;

s512 x55;
__m512 x56;

// On AVX512, aggregates which contain a __m512 type are classified as SSE/SSEUP
// as per https://github.com/hjl-tools/x86-psABI/commit/30f9c9 3.2.3p2 Rule 1
//
// AVX512: declare void @f55(<16 x float>)
// NO-AVX512: declare void @f55(ptr noundef byval(%struct.s512) align 64)
void f55(s512 x);

// __m512 has type SSE/SSEUP on AVX512.
//
// AVX512: declare void @f56(<16 x float> noundef)
// NO-AVX512: declare void @f56(ptr noundef byval(<16 x float>) align 64)
void f56(__m512 x);
void f57(void) { f55(x55); f56(x56); }

// Like for __m128 on AVX, check that the struct below is passed
// in the same way regardless of AVX512 being used.
//
// CHECK: declare void @f58(ptr noundef byval(%struct.t256) align 32)
typedef struct t256 {
  __m256 m;
  __m256 n;
} two256;

extern void f58(two256 s);
void f59(two256 s) {
  f58(s);
}

// CHECK: declare void @f60(ptr noundef byval(%struct.sat256) align 32)
typedef struct at256 {
  __m256 array[2];
} Atwo256;
typedef struct sat256 {
  Atwo256 x;
} SAtwo256;

extern void f60(SAtwo256 s);
void f61(SAtwo256 s) {
  f60(s);
}

// AVX512: @f62_helper(i32 noundef 0, <16 x float> noundef {{%[a-zA-Z0-9]+}}, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef {{%[a-zA-Z0-9]+}}, double noundef {{%[a-zA-Z0-9]+}})
void f62_helper(int, ...);
__m512 x62;
void f62(void) {
  f62_helper(0, x62, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0i);
}

// Like for __m256 on AVX, we always pass __m512 in memory, and don't
// need to use the register save area.
//
// AVX512-LABEL: define{{.*}} void @f63
// AVX512-NOT: br i1
// AVX512: ret void
void f63(__m512 *m, __builtin_va_list argList) {
  *m = __builtin_va_arg(argList, __m512);
}

// AVX512: @f64_helper(<16 x float> noundef {{%[a-zA-Z0-9]+}}, <16 x float> noundef {{%[a-zA-Z0-9]+}}, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef {{%[a-zA-Z0-9]+}}, double noundef {{%[a-zA-Z0-9]+}})
// AVX512: @f64_helper(<16 x float> noundef {{%[a-zA-Z0-9]+}}, <16 x float> noundef {{%[a-zA-Z0-9]+}}, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, double noundef 1.000000e+00, ptr noundef byval({ double, double }) align 8 {{%[^)]+}})
void f64_helper(__m512, ...);
__m512 x64;
void f64(void) {
  f64_helper(x64, x64, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0i);
  f64_helper(x64, x64, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0i);
}

struct t65 {
  __m256 m;
  int : 0;
};
// SSE-LABEL: @f65(ptr noundef byval(%struct.t65) align 32 %{{[^,)]+}})
// AVX: @f65(<8 x float> %{{[^,)]+}})
void f65(struct t65 a0) {
}

typedef float t66 __attribute__((__vector_size__(128), __aligned__(128)));

// AVX512: @f66(ptr noundef byval(<32 x float>) align 128 %0)
void f66(t66 a0) {
}

typedef long long t67 __attribute__((aligned (4)));
struct s67 {
  int a;
  t67 b;
};
// CHECK-LABEL: define{{.*}} void @f67(ptr noundef byval(%struct.s67) align 8 %x)
void f67(struct s67 x) {
}

typedef double t68 __attribute__((aligned (4)));
struct s68 {
  int a;
  t68 b;
};
// CHECK-LABEL: define{{.*}} void @f68(ptr noundef byval(%struct.s68) align 8 %x)
void f68(struct s68 x) {
}

/// The synthesized __va_list_tag does not have file/line fields.
// CHECK:      = distinct !DICompositeType(tag: DW_TAG_structure_type, name: "__va_list_tag",
// CHECK-NOT:  file:
// CHECK-NOT:  line:
// CHECK-SAME: size:
