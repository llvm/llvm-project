// RUN: clang-cc -triple x86_64-unknown-unknown -emit-llvm -o %t %s &&
// RUN: grep 'define signext i8 @f0()' %t &&
// RUN: grep 'define signext i16 @f1()' %t &&
// RUN: grep 'define i32 @f2()' %t &&
// RUN: grep 'define float @f3()' %t &&
// RUN: grep 'define double @f4()' %t &&
// RUN: grep 'define x86_fp80 @f5()' %t &&
// RUN: grep 'define void @f6(i8 signext %a0, i16 signext %a1, i32 %a2, i64 %a3, i8\* %a4)' %t &&
// RUN: grep 'define void @f7(i32 %a0)' %t &&
// RUN: grep '.0 = type { i64, double }' %t &&
// RUN: grep 'define .0 @f8_1()' %t &&
// RUN: grep 'define void @f8_2(.0)' %t &&

char f0(void) {
  return 0;
}

short f1(void) {
  return 0;
}

int f2(void) {
  return 0;
}

float f3(void) {
  return 0;
}

double f4(void) {
  return 0;
}

long double f5(void) {
  return 0;
}

void f6(char a0, short a1, int a2, long long a3, void *a4) {
}

typedef enum { A, B, C } E;

void f7(E a0) {
}

// Test merging/passing of upper eightbyte with X87 class.
union u8 {
  long double a;
  int b;
};
union u8 f8_1() { while (1) {} }
void f8_2(union u8 a0) {}

// RUN: grep 'define i64 @f9()' %t &&
struct s9 { int a; int b; int : 0; } f9(void) { while (1) {} }

// RUN: grep 'define void @f10(i64)' %t &&
struct s10 { int a; int b; int : 0; };
void f10(struct s10 a0) {}

// RUN: grep 'define void @f11(.union.anon. noalias sret .agg.result)' %t &&
union { long double a; float b; } f11() { while (1) {} }

// RUN: grep 'define i64 @f12_0()' %t &&
// RUN: grep 'define void @f12_1(i64)' %t &&
struct s12 { int a __attribute__((aligned(16))); };
struct s12 f12_0(void) { while (1) {} }
void f12_1(struct s12 a0) {}

// Check that sret parameter is accounted for when checking available integer
// registers.
// RUN: grep 'define void @f13(.struct.s13_0. noalias sret .agg.result, i32 .a, i32 .b, i32 .c, i32 .d, .struct.s13_1. byval .e, i32 .f)' %t &&

struct s13_0 { long long f0[3]; };
struct s13_1 { long long f0[2]; };
struct s13_0 f13(int a, int b, int c, int d, 
                 struct s13_1 e, int f) { while (1) {} }

// RUN: grep 'define void @f14(.*, i8 signext .X)' %t &&
void f14(int a, int b, int c, int d, int e, int f, 
         char X) {}
// RUN: grep 'define void @f15(.*, i8\* .X)' %t &&
void f15(int a, int b, int c, int d, int e, int f, 
         void *X) {}
// RUN: grep 'define void @f16(.*, float .X)' %t &&
void f16(float a, float b, float c, float d, float e, float f, float g, float h,
         float X) {}
// RUN: grep 'define void @f17(.*, x86_fp80 .X)' %t &&
void f17(float a, float b, float c, float d, float e, float f, float g, float h,
         long double X) {}

// Check for valid coercion.
// RUN: grep '.. = bitcast i64. .* to .struct.f18_s0.' %t &&
// RUN: grep '.. = load .struct.f18_s0. .., align 1' %t &&
// RUN: grep 'store .struct.f18_s0 .., .struct.f18_s0. .f18_arg1' %t &&
struct f18_s0 { int f0; };
void f18(int a, struct f18_s0 f18_arg1) { while (1) {} }

// RUN: true
