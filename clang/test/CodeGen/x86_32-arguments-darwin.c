// RUN: %clang_cc1 -w -fblocks -triple i386-apple-darwin9 -target-cpu yonah -emit-llvm -o - %s | FileCheck %s

// CHECK-LABEL: define signext i8 @f0()
char f0(void) {
  return 0;
}

// CHECK-LABEL: define signext i16 @f1()
short f1(void) {
  return 0;
}

// CHECK-LABEL: define i32 @f2()
int f2(void) {
  return 0;
}

// CHECK-LABEL: define float @f3()
float f3(void) {
  return 0;
}

// CHECK-LABEL: define double @f4()
double f4(void) {
  return 0;
}

// CHECK-LABEL: define x86_fp80 @f5()
long double f5(void) {
  return 0;
}

// CHECK-LABEL: define void @f6(i8 signext %a0, i16 signext %a1, i32 %a2, i64 %a3, i8* %a4)
void f6(char a0, short a1, int a2, long long a3, void *a4) {}

// CHECK-LABEL: define void @f7(i32 %a0)
typedef enum { A, B, C } e7;
void f7(e7 a0) {}

// CHECK-LABEL: define i64 @f8_1()
// CHECK-LABEL: define void @f8_2(i32 %a0.0, i32 %a0.1)
struct s8 {
  int a;
  int b;
};
struct s8 f8_1(void) { while (1) {} }
void f8_2(struct s8 a0) {}

// This should be passed just as s8.

// CHECK-LABEL: define i64 @f9_1()

// FIXME: llvm-gcc expands this, this may have some value for the
// backend in terms of optimization but doesn't change the ABI.
// CHECK-LABEL: define void @f9_2(%struct.s9* byval(%struct.s9) align 4 %a0)
struct s9 {
  int a : 17;
  int b;
};
struct s9 f9_1(void) { while (1) {} }
void f9_2(struct s9 a0) {}

// Return of small structures and unions

// CHECK: float @f10()
struct s10 {
  union { };
  float f;
} f10(void) { while (1) {} }

// Small vectors and 1 x {i64,double} are returned in registers

// CHECK: i32 @f11()
// CHECK: void @f12(<2 x i32>* noalias sret %agg.result)
// CHECK: i64 @f13()
// CHECK: i64 @f14()
// CHECK: <2 x i64> @f15()
// CHECK: <2 x i64> @f16()
typedef short T11 __attribute__ ((vector_size (4)));
T11 f11(void) { while (1) {} }
typedef int T12 __attribute__ ((vector_size (8)));
T12 f12(void) { while (1) {} }
typedef long long T13 __attribute__ ((vector_size (8)));
T13 f13(void) { while (1) {} }
typedef double T14 __attribute__ ((vector_size (8)));
T14 f14(void) { while (1) {} }
typedef long long T15 __attribute__ ((vector_size (16)));
T15 f15(void) { while (1) {} }
typedef double T16 __attribute__ ((vector_size (16)));
T16 f16(void) { while (1) {} }

// And when the single element in a struct (but not for 64 and
// 128-bits).

// CHECK: i32 @f17()
// CHECK: void @f18(%{{.*}}* noalias sret %agg.result)
// CHECK: void @f19(%{{.*}}* noalias sret %agg.result)
// CHECK: void @f20(%{{.*}}* noalias sret %agg.result)
// CHECK: void @f21(%{{.*}}* noalias sret %agg.result)
// CHECK: void @f22(%{{.*}}* noalias sret %agg.result)
struct { T11 a; } f17(void) { while (1) {} }
struct { T12 a; } f18(void) { while (1) {} }
struct { T13 a; } f19(void) { while (1) {} }
struct { T14 a; } f20(void) { while (1) {} }
struct { T15 a; } f21(void) { while (1) {} }
struct { T16 a; } f22(void) { while (1) {} }

// Single element structures are handled specially

// CHECK: float @f23()
// CHECK: float @f24()
// CHECK: float @f25()
struct { float a; } f23(void) { while (1) {} }
struct { float a[1]; } f24(void) { while (1) {} }
struct { struct {} a; struct { float a[1]; } b; } f25(void) { while (1) {} }

// Small structures are handled recursively
// CHECK: i32 @f26()
// CHECK: void @f27(%struct.s27* noalias sret %agg.result)
struct s26 { struct { char a, b; } a; struct { char a, b; } b; } f26(void) { while (1) {} }
struct s27 { struct { char a, b, c; } a; struct { char a; } b; } f27(void) { while (1) {} }

// CHECK: void @f28(%struct.s28* noalias sret %agg.result)
struct s28 { int a; int b[]; } f28(void) { while (1) {} }

// CHECK-LABEL: define i16 @f29()
struct s29 { struct { } a[1]; char b; char c; } f29(void) { while (1) {} }

// CHECK-LABEL: define i16 @f30()
struct s30 { char a; char b : 4; } f30(void) { while (1) {} }

// CHECK-LABEL: define float @f31()
struct s31 { char : 0; float b; char : 0; } f31(void) { while (1) {} }

// CHECK-LABEL: define i32 @f32()
struct s32 { char a; unsigned : 0; } f32(void) { while (1) {} }

// CHECK-LABEL: define float @f33()
struct s33 { float a; long long : 0; } f33(void) { while (1) {} }

// CHECK-LABEL: define float @f34()
struct s34 { struct { int : 0; } a; float b; } f34(void) { while (1) {} }

// CHECK-LABEL: define i16 @f35()
struct s35 { struct { int : 0; } a; char b; char c; } f35(void) { while (1) {} }

// CHECK-LABEL: define i16 @f36()
struct s36 { struct { int : 0; } a[2][10]; char b; char c; } f36(void) { while (1) {} }

// CHECK-LABEL: define float @f37()
struct s37 { float c[1][1]; } f37(void) { while (1) {} }

// CHECK-LABEL: define void @f38(%struct.s38* noalias sret %agg.result)
struct s38 { char a[3]; short b; } f38(void) { while (1) {} }

// CHECK-LABEL: define void @f39(%struct.s39* byval(%struct.s39) align 16 %x)
typedef int v39 __attribute((vector_size(16)));
struct s39 { v39 x; };
void f39(struct s39 x) {}

// <rdar://problem/7247671>
// CHECK-LABEL: define i32 @f40()
enum e40 { ec0 = 0 };
enum e40 f40(void) { }

// CHECK-LABEL: define void ()* @f41()
typedef void (^vvbp)(void);
vvbp f41(void) { }

// CHECK-LABEL: define i32 @f42()
struct s42 { enum e40 f0; } f42(void) {  }

// CHECK-LABEL: define i64 @f43()
struct s43 { enum e40 f0; int f1; } f43(void) {  }

// CHECK-LABEL: define void ()* @f44()
struct s44 { vvbp f0; } f44(void) {  }

// CHECK-LABEL: define i64 @f45()
struct s45 { vvbp f0; int f1; } f45(void) {  }

// CHECK-LABEL: define void @f46(i32 %a0)
void f46(enum e40 a0) { }

// CHECK-LABEL: define void @f47(void ()* %a1)
void f47(vvbp a1) { }

// CHECK-LABEL: define void @f48(i32 %a0.0)
struct s48 { enum e40 f0; };
void f48(struct s48 a0) { }

// CHECK-LABEL: define void @f49(i32 %a0.0, i32 %a0.1)
struct s49 { enum e40 f0; int f1; };
void f49(struct s49 a0) { }

// CHECK-LABEL: define void @f50(void ()* %a0.0)
struct s50 { vvbp f0; };
void f50(struct s50 a0) { }

// CHECK-LABEL: define void @f51(void ()* %a0.0, i32 %a0.1)
struct s51 { vvbp f0; int f1; };
void f51(struct s51 a0) { }

// CHECK-LABEL: define void @f52(%struct.s52* byval(%struct.s52) align 4)
struct s52 {
  long double a;
};
void f52(struct s52 x) {}

// CHECK-LABEL: define void @f53(%struct.s53* byval(%struct.s53) align 4)
struct __attribute__((aligned(32))) s53 {
  int x;
  int y;
};
void f53(struct s53 x) {}

typedef unsigned short v2i16 __attribute__((__vector_size__(4)));

// CHECK-LABEL: define i32 @f54(i32 %arg.coerce)
// rdar://8359483
v2i16 f54(v2i16 arg) { return arg+arg; }


typedef int v4i32 __attribute__((__vector_size__(16)));

// CHECK-LABEL: define <2 x i64> @f55(<4 x i32> %arg)
// PR8029
v4i32 f55(v4i32 arg) { return arg+arg; }

// CHECK-LABEL: define void @f56(
// CHECK: i8 signext %a0, %struct.s56_0* byval(%struct.s56_0) align 4 %a1,
// CHECK: i64 %a2.coerce, %struct.s56_1* byval(%struct.s56_1) align 4,
// CHECK: i64 %a4.coerce, %struct.s56_2* byval(%struct.s56_2) align 4,
// CHECK: <4 x i32> %a6, %struct.s56_3* byval(%struct.s56_3) align 16 %a7,
// CHECK: <2 x double> %a8, %struct.s56_4* byval(%struct.s56_4) align 16 %a9,
// CHECK: <8 x i32> %a10, %struct.s56_5* byval(%struct.s56_5) align 4,
// CHECK: <4 x double> %a12, %struct.s56_6* byval(%struct.s56_6) align 4)

// CHECK:   call void (i32, ...) @f56_0(i32 1,
// CHECK: i32 %{{[^ ]*}}, %struct.s56_0* byval(%struct.s56_0) align 4 %{{[^ ]*}},
// CHECK: i64 %{{[^ ]*}}, %struct.s56_1* byval(%struct.s56_1) align 4 %{{[^ ]*}},
// CHECK: i64 %{{[^ ]*}}, %struct.s56_2* byval(%struct.s56_2) align 4 %{{[^ ]*}},
// CHECK: <4 x i32> %{{[^ ]*}}, %struct.s56_3* byval(%struct.s56_3) align 16 %{{[^ ]*}},
// CHECK: <2 x double> %{{[^ ]*}}, %struct.s56_4* byval(%struct.s56_4) align 16 %{{[^ ]*}},
// CHECK: <8 x i32> {{[^ ]*}}, %struct.s56_5* byval(%struct.s56_5) align 4 %{{[^ ]*}},
// CHECK: <4 x double> {{[^ ]*}}, %struct.s56_6* byval(%struct.s56_6) align 4 %{{[^ ]*}})
// CHECK: }
//
// <rdar://problem/7964854> [i386] clang misaligns long double in structures
// when passed byval
// <rdar://problem/8431367> clang misaligns parameters on stack
typedef int __attribute__((vector_size (8))) t56_v2i;
typedef double __attribute__((vector_size (8))) t56_v1d;
typedef int __attribute__((vector_size (16))) t56_v4i;
typedef double __attribute__((vector_size (16))) t56_v2d;
typedef int __attribute__((vector_size (32))) t56_v8i;
typedef double __attribute__((vector_size (32))) t56_v4d;

struct s56_0 { char a; };
struct s56_1 { t56_v2i a; };
struct s56_2 { t56_v1d a; };
struct s56_3 { t56_v4i a; };
struct s56_4 { t56_v2d a; };
struct s56_5 { t56_v8i a; };
struct s56_6 { t56_v4d a; };

void f56(char a0, struct s56_0 a1, 
         t56_v2i a2, struct s56_1 a3, 
         t56_v1d a4, struct s56_2 a5, 
         t56_v4i a6, struct s56_3 a7, 
         t56_v2d a8, struct s56_4 a9, 
         t56_v8i a10, struct s56_5 a11, 
         t56_v4d a12, struct s56_6 a13) {
  extern void f56_0(int x, ...);
  f56_0(1, a0, a1, a2, a3, a4, a5, a6, a7, a8, a9,
        a10, a11, a12, a13);
}

// CHECK-LABEL: define void @f57(i32 %x.0, i32 %x.1)
// CHECK: call void @f57(
struct s57 { _Complex int x; };
void f57(struct s57 x) {} void f57a(void) { f57((struct s57){1}); }

// CHECK-LABEL: define void @f58()
union u58 {};
void f58(union u58 x) {}

// CHECK-LABEL: define i64 @f59()
struct s59 { float x __attribute((aligned(8))); };
struct s59 f59() { while (1) {} }

// CHECK-LABEL: define void @f60(%struct.s60* byval(%struct.s60) align 4, i32 %y)
struct s60 { int x __attribute((aligned(8))); };
void f60(struct s60 x, int y) {}

// CHECK-LABEL: define void @f61(i32 %x, %struct.s61* byval(%struct.s61) align 16 %y)
typedef int T61 __attribute((vector_size(16)));
struct s61 { T61 x; int y; };
void f61(int x, struct s61 y) {}

// CHECK-LABEL: define void @f62(i32 %x, %struct.s62* byval(%struct.s62) align 4)
typedef int T62 __attribute((vector_size(16)));
struct s62 { T62 x; int y; } __attribute((packed, aligned(8)));
void f62(int x, struct s62 y) {}

// CHECK-LABEL: define i32 @f63
// CHECK: ptrtoint
// CHECK: and {{.*}}, -16
// CHECK: inttoptr
typedef int T63 __attribute((vector_size(16)));
struct s63 { T63 x; int y; };
int f63(int i, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, i);
  struct s63 s = __builtin_va_arg(ap, struct s63);
  __builtin_va_end(ap);
  return s.y;
}

// CHECK-LABEL: define void @f64(%struct.s64* byval(%struct.s64) align 4 %x)
struct s64 { signed char a[0]; signed char b[]; };
void f64(struct s64 x) {}

// CHECK-LABEL: define float @f65()
struct s65 { signed char a[0]; float b; };
struct s65 f65() { return (struct s65){{},2}; }

// CHECK-LABEL: define <2 x i64> @f66
// CHECK: ptrtoint
// CHECK: and {{.*}}, -16
// CHECK: inttoptr
typedef int T66 __attribute((vector_size(16)));
T66 f66(int i, ...) {
  __builtin_va_list ap;
  __builtin_va_start(ap, i);
  T66 v = __builtin_va_arg(ap, T66);
  __builtin_va_end(ap);
  return v;
}

// PR14453
struct s67 { _Complex unsigned short int a; };
void f67(struct s67 x) {}
// CHECK-LABEL: define void @f67(%struct.s67* byval(%struct.s67) align 4 %x)
