// RUN: %clang_cc1 -triple arm64-apple-ios7 -target-feature +neon -target-abi darwinpcs -ffreestanding -emit-llvm -w -o - %s | FileCheck %s

// CHECK: define{{.*}} signext i8 @f0()
char f0(void) {
  return 0;
}

// Struct as return type. Aggregates <= 16 bytes are passed directly and round
// up to multiple of 8 bytes.
// CHECK: define{{.*}} i64 @f1()
struct s1 { char f0; };
struct s1 f1(void) {}

// CHECK: define{{.*}} i64 @f2()
struct s2 { short f0; };
struct s2 f2(void) {}

// CHECK: define{{.*}} i64 @f3()
struct s3 { int f0; };
struct s3 f3(void) {}

// CHECK: define{{.*}} i64 @f4()
struct s4 { struct s4_0 { int f0; } f0; };
struct s4 f4(void) {}

// CHECK: define{{.*}} i64 @f5()
struct s5 { struct { } f0; int f1; };
struct s5 f5(void) {}

// CHECK: define{{.*}} i64 @f6()
struct s6 { int f0[1]; };
struct s6 f6(void) {}

// CHECK: define{{.*}} void @f7()
struct s7 { struct { int : 0; } f0; };
struct s7 f7(void) {}

// CHECK: define{{.*}} void @f8()
struct s8 { struct { int : 0; } f0[1]; };
struct s8 f8(void) {}

// CHECK: define{{.*}} i64 @f9()
struct s9 { int f0; int : 0; };
struct s9 f9(void) {}

// CHECK: define{{.*}} i64 @f10()
struct s10 { int f0; int : 0; int : 0; };
struct s10 f10(void) {}

// CHECK: define{{.*}} i64 @f11()
struct s11 { int : 0; int f0; };
struct s11 f11(void) {}

// CHECK: define{{.*}} i64 @f12()
union u12 { char f0; short f1; int f2; };
union u12 f12(void) {}

// Homogeneous Aggregate as return type will be passed directly.
// CHECK: define{{.*}} %struct.s13 @f13()
struct s13 { float f0; };
struct s13 f13(void) {}
// CHECK: define{{.*}} %union.u14 @f14()
union u14 { float f0; };
union u14 f14(void) {}

// CHECK: define{{.*}} void @f15()
void f15(struct s7 a0) {}

// CHECK: define{{.*}} void @f16()
void f16(struct s8 a0) {}

// CHECK: define{{.*}} i64 @f17()
struct s17 { short f0 : 13; char f1 : 4; };
struct s17 f17(void) {}

// CHECK: define{{.*}} i64 @f18()
struct s18 { short f0; char f1 : 4; };
struct s18 f18(void) {}

// CHECK: define{{.*}} i64 @f19()
struct s19 { int f0; struct s8 f1; };
struct s19 f19(void) {}

// CHECK: define{{.*}} i64 @f20()
struct s20 { struct s8 f1; int f0; };
struct s20 f20(void) {}

// CHECK: define{{.*}} i64 @f21()
struct s21 { struct {} f1; int f0 : 4; };
struct s21 f21(void) {}

// CHECK: define{{.*}} i64 @f22()
// CHECK: define{{.*}} i64 @f23()
// CHECK: define{{.*}} i64 @f24()
// CHECK: define{{.*}} [2 x i64] @f25()
// CHECK: define{{.*}} { float, float } @f26()
// CHECK: define{{.*}} { double, double } @f27()
_Complex char       f22(void) {}
_Complex short      f23(void) {}
_Complex int        f24(void) {}
_Complex long long  f25(void) {}
_Complex float      f26(void) {}
_Complex double     f27(void) {}

// CHECK: define{{.*}} i64 @f28()
struct s28 { _Complex char f0; };
struct s28 f28() {}

// CHECK: define{{.*}} i64 @f29()
struct s29 { _Complex short f0; };
struct s29 f29() {}

// CHECK: define{{.*}} i64 @f30()
struct s30 { _Complex int f0; };
struct s30 f30() {}

struct s31 { char x; };
void f31(struct s31 s) { }
// CHECK: define{{.*}} void @f31(i64 %s.coerce)
// CHECK: %s = alloca %struct.s31, align 1
// CHECK: trunc i64 %s.coerce to i8
// CHECK: store i8 %{{.*}},

struct s32 { double x; };
void f32(struct s32 s) { }
// CHECK: @f32([1 x double] %{{.*}})

// A composite type larger than 16 bytes should be passed indirectly.
struct s33 { char buf[32*32]; };
void f33(struct s33 s) { }
// CHECK: define{{.*}} void @f33(%struct.s33* %s)

struct s34 { char c; };
void f34(struct s34 s);
void g34(struct s34 *s) { f34(*s); }
// CHECK: @g34(%struct.s34* %s)
// CHECK: %[[a:.*]] = load i8, i8* %{{.*}}
// CHECK: zext i8 %[[a]] to i64
// CHECK: call void @f34(i64 %{{.*}})

/*
 * Check that va_arg accesses stack according to ABI alignment
 */
long long t1(int i, ...) {
    // CHECK: t1
    __builtin_va_list ap;
    __builtin_va_start(ap, i);
    // CHECK-NOT: add i32 %{{.*}} 7
    // CHECK-NOT: and i32 %{{.*}} -8
    long long ll = __builtin_va_arg(ap, long long);
    __builtin_va_end(ap);
    return ll;
}
double t2(int i, ...) {
    // CHECK: t2
    __builtin_va_list ap;
    __builtin_va_start(ap, i);
    // CHECK-NOT: add i32 %{{.*}} 7
    // CHECK-NOT: and i32 %{{.*}} -8
    double ll = __builtin_va_arg(ap, double);
    __builtin_va_end(ap);
    return ll;
}

#include <arm_neon.h>

// Homogeneous Vector Aggregate as return type and argument type.
// CHECK: define{{.*}} %struct.int8x16x2_t @f0_0(<16 x i8> %{{.*}}, <16 x i8> %{{.*}})
int8x16x2_t f0_0(int8x16_t a0, int8x16_t a1) {
  return vzipq_s8(a0, a1);
}

// Test direct vector passing.
typedef float T_float32x2 __attribute__ ((__vector_size__ (8)));
typedef float T_float32x4 __attribute__ ((__vector_size__ (16)));
typedef float T_float32x8 __attribute__ ((__vector_size__ (32)));
typedef float T_float32x16 __attribute__ ((__vector_size__ (64)));

// CHECK: define{{.*}} <2 x float> @f1_0(<2 x float> %{{.*}})
T_float32x2 f1_0(T_float32x2 a0) { return a0; }
// CHECK: define{{.*}} <4 x float> @f1_1(<4 x float> %{{.*}})
T_float32x4 f1_1(T_float32x4 a0) { return a0; }
// Vector with length bigger than 16-byte is illegal and is passed indirectly.
// CHECK: define{{.*}} void @f1_2(<8 x float>* noalias sret(<8 x float>) align 16 %{{.*}}, <8 x float>* %0)
T_float32x8 f1_2(T_float32x8 a0) { return a0; }
// CHECK: define{{.*}} void @f1_3(<16 x float>* noalias sret(<16 x float>) align 16 %{{.*}}, <16 x float>* %0)
T_float32x16 f1_3(T_float32x16 a0) { return a0; }

// Testing alignment with aggregates: HFA, aggregates with size <= 16 bytes and
// aggregates with size > 16 bytes.
struct s35
{
   float v[4]; //Testing HFA.
} __attribute__((aligned(16)));
typedef struct s35 s35_with_align;

typedef __attribute__((neon_vector_type(4))) float float32x4_t;
float32x4_t f35(int i, s35_with_align s1, s35_with_align s2) {
// CHECK: define{{.*}} <4 x float> @f35(i32 %i, [4 x float] %s1.coerce, [4 x float] %s2.coerce)
// CHECK: %s1 = alloca %struct.s35, align 16
// CHECK: %s2 = alloca %struct.s35, align 16
// CHECK: %[[a:.*]] = bitcast %struct.s35* %s1 to <4 x float>*
// CHECK: load <4 x float>, <4 x float>* %[[a]], align 16
// CHECK: %[[b:.*]] = bitcast %struct.s35* %s2 to <4 x float>*
// CHECK: load <4 x float>, <4 x float>* %[[b]], align 16
  float32x4_t v = vaddq_f32(*(float32x4_t *)&s1,
                            *(float32x4_t *)&s2);
  return v;
}

struct s36
{
   int v[4]; //Testing 16-byte aggregate.
} __attribute__((aligned(16)));
typedef struct s36 s36_with_align;

typedef __attribute__((neon_vector_type(4))) int int32x4_t;
int32x4_t f36(int i, s36_with_align s1, s36_with_align s2) {
// CHECK: define{{.*}} <4 x i32> @f36(i32 %i, i128 %s1.coerce, i128 %s2.coerce)
// CHECK: %s1 = alloca %struct.s36, align 16
// CHECK: %s2 = alloca %struct.s36, align 16
// CHECK: store i128 %s1.coerce, i128* %{{.*}}, align 16
// CHECK: store i128 %s2.coerce, i128* %{{.*}}, align 16
// CHECK: %[[a:.*]] = bitcast %struct.s36* %s1 to <4 x i32>*
// CHECK: load <4 x i32>, <4 x i32>* %[[a]], align 16
// CHECK: %[[b:.*]] = bitcast %struct.s36* %s2 to <4 x i32>*
// CHECK: load <4 x i32>, <4 x i32>* %[[b]], align 16
  int32x4_t v = vaddq_s32(*(int32x4_t *)&s1,
                          *(int32x4_t *)&s2);
  return v;
}

struct s37
{
   int v[18]; //Testing large aggregate.
} __attribute__((aligned(16)));
typedef struct s37 s37_with_align;

int32x4_t f37(int i, s37_with_align s1, s37_with_align s2) {
// CHECK: define{{.*}} <4 x i32> @f37(i32 %i, %struct.s37* %s1, %struct.s37* %s2)
// CHECK: %[[a:.*]] = bitcast %struct.s37* %s1 to <4 x i32>*
// CHECK: load <4 x i32>, <4 x i32>* %[[a]], align 16
// CHECK: %[[b:.*]] = bitcast %struct.s37* %s2 to <4 x i32>*
// CHECK: load <4 x i32>, <4 x i32>* %[[b]], align 16
  int32x4_t v = vaddq_s32(*(int32x4_t *)&s1,
                          *(int32x4_t *)&s2);
  return v;
}
s37_with_align g37;
int32x4_t caller37() {
// CHECK: caller37
// CHECK: %[[a:.*]] = alloca %struct.s37, align 16
// CHECK: %[[b:.*]] = alloca %struct.s37, align 16
// CHECK: call void @llvm.memcpy
// CHECK: call void @llvm.memcpy
// CHECK: call <4 x i32> @f37(i32 3, %struct.s37* %[[a]], %struct.s37* %[[b]])
  return f37(3, g37, g37);
}

// rdar://problem/12648441
// Test passing structs with size < 8, < 16 and > 16
// with alignment of 16 and without

// structs with size <= 8 bytes, without alignment attribute
// passed as i64 regardless of the align attribute
struct s38
{
  int i;
  short s;
};
typedef struct s38 s38_no_align;
// passing structs in registers
__attribute__ ((noinline))
int f38(int i, s38_no_align s1, s38_no_align s2) {
// CHECK: define{{.*}} i32 @f38(i32 %i, i64 %s1.coerce, i64 %s2.coerce)
// CHECK: %s1 = alloca %struct.s38, align 4
// CHECK: %s2 = alloca %struct.s38, align 4
// CHECK: store i64 %s1.coerce, i64* %{{.*}}, align 4
// CHECK: store i64 %s2.coerce, i64* %{{.*}}, align 4
// CHECK: getelementptr inbounds %struct.s38, %struct.s38* %s1, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s38, %struct.s38* %s2, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s38, %struct.s38* %s1, i32 0, i32 1
// CHECK: getelementptr inbounds %struct.s38, %struct.s38* %s2, i32 0, i32 1
  return s1.i + s2.i + i + s1.s + s2.s;
}
s38_no_align g38;
s38_no_align g38_2;
int caller38() {
// CHECK: define{{.*}} i32 @caller38()
// CHECK: %[[a:.*]] = load i64, i64* bitcast (%struct.s38* @g38 to i64*), align 4
// CHECK: %[[b:.*]] = load i64, i64* bitcast (%struct.s38* @g38_2 to i64*), align 4
// CHECK: call i32 @f38(i32 3, i64 %[[a]], i64 %[[b]])
  return f38(3, g38, g38_2);
}
// passing structs on stack
__attribute__ ((noinline))
int f38_stack(int i, int i2, int i3, int i4, int i5, int i6, int i7, int i8,
              int i9, s38_no_align s1, s38_no_align s2) {
// CHECK: define{{.*}} i32 @f38_stack(i32 %i, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6, i32 %i7, i32 %i8, i32 %i9, i64 %s1.coerce, i64 %s2.coerce)
// CHECK: %s1 = alloca %struct.s38, align 4
// CHECK: %s2 = alloca %struct.s38, align 4
// CHECK: store i64 %s1.coerce, i64* %{{.*}}, align 4
// CHECK: store i64 %s2.coerce, i64* %{{.*}}, align 4
// CHECK: getelementptr inbounds %struct.s38, %struct.s38* %s1, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s38, %struct.s38* %s2, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s38, %struct.s38* %s1, i32 0, i32 1
// CHECK: getelementptr inbounds %struct.s38, %struct.s38* %s2, i32 0, i32 1
  return s1.i + s2.i + i + i2 + i3 + i4 + i5 + i6 + i7 + i8 + i9 + s1.s + s2.s;
}
int caller38_stack() {
// CHECK: define{{.*}} i32 @caller38_stack()
// CHECK: %[[a:.*]] = load i64, i64* bitcast (%struct.s38* @g38 to i64*), align 4
// CHECK: %[[b:.*]] = load i64, i64* bitcast (%struct.s38* @g38_2 to i64*), align 4
// CHECK: call i32 @f38_stack(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i64 %[[a]], i64 %[[b]])
  return f38_stack(1, 2, 3, 4, 5, 6, 7, 8, 9, g38, g38_2);
}

// structs with size <= 8 bytes, with alignment attribute
struct s39
{
  int i;
  short s;
} __attribute__((aligned(16)));
typedef struct s39 s39_with_align;
// passing aligned structs in registers
__attribute__ ((noinline))
int f39(int i, s39_with_align s1, s39_with_align s2) {
// CHECK: define{{.*}} i32 @f39(i32 %i, i128 %s1.coerce, i128 %s2.coerce)
// CHECK: %s1 = alloca %struct.s39, align 16
// CHECK: %s2 = alloca %struct.s39, align 16
// CHECK: store i128 %s1.coerce, i128* %{{.*}}, align 16
// CHECK: store i128 %s2.coerce, i128* %{{.*}}, align 16
// CHECK: getelementptr inbounds %struct.s39, %struct.s39* %s1, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s39, %struct.s39* %s2, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s39, %struct.s39* %s1, i32 0, i32 1
// CHECK: getelementptr inbounds %struct.s39, %struct.s39* %s2, i32 0, i32 1
  return s1.i + s2.i + i + s1.s + s2.s;
}
s39_with_align g39;
s39_with_align g39_2;
int caller39() {
// CHECK: define{{.*}} i32 @caller39()
// CHECK: %[[a:.*]] = load i128, i128* bitcast (%struct.s39* @g39 to i128*), align 16
// CHECK: %[[b:.*]] = load i128, i128* bitcast (%struct.s39* @g39_2 to i128*), align 16
// CHECK: call i32 @f39(i32 3, i128 %[[a]], i128 %[[b]])
  return f39(3, g39, g39_2);
}
// passing aligned structs on stack
__attribute__ ((noinline))
int f39_stack(int i, int i2, int i3, int i4, int i5, int i6, int i7, int i8,
              int i9, s39_with_align s1, s39_with_align s2) {
// CHECK: define{{.*}} i32 @f39_stack(i32 %i, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6, i32 %i7, i32 %i8, i32 %i9, i128 %s1.coerce, i128 %s2.coerce)
// CHECK: %s1 = alloca %struct.s39, align 16
// CHECK: %s2 = alloca %struct.s39, align 16
// CHECK: store i128 %s1.coerce, i128* %{{.*}}, align 16
// CHECK: store i128 %s2.coerce, i128* %{{.*}}, align 16
// CHECK: getelementptr inbounds %struct.s39, %struct.s39* %s1, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s39, %struct.s39* %s2, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s39, %struct.s39* %s1, i32 0, i32 1
// CHECK: getelementptr inbounds %struct.s39, %struct.s39* %s2, i32 0, i32 1
  return s1.i + s2.i + i + i2 + i3 + i4 + i5 + i6 + i7 + i8 + i9 + s1.s + s2.s;
}
int caller39_stack() {
// CHECK: define{{.*}} i32 @caller39_stack()
// CHECK: %[[a:.*]] = load i128, i128* bitcast (%struct.s39* @g39 to i128*), align 16
// CHECK: %[[b:.*]] = load i128, i128* bitcast (%struct.s39* @g39_2 to i128*), align 16
// CHECK: call i32 @f39_stack(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i128 %[[a]], i128 %[[b]])
  return f39_stack(1, 2, 3, 4, 5, 6, 7, 8, 9, g39, g39_2);
}

// structs with size <= 16 bytes, without alignment attribute
struct s40
{
  int i;
  short s;
  int i2;
  short s2;
};
typedef struct s40 s40_no_align;
// passing structs in registers
__attribute__ ((noinline))
int f40(int i, s40_no_align s1, s40_no_align s2) {
// CHECK: define{{.*}} i32 @f40(i32 %i, [2 x i64] %s1.coerce, [2 x i64] %s2.coerce)
// CHECK: %s1 = alloca %struct.s40, align 4
// CHECK: %s2 = alloca %struct.s40, align 4
// CHECK: store [2 x i64] %s1.coerce, [2 x i64]* %{{.*}}, align 4
// CHECK: store [2 x i64] %s2.coerce, [2 x i64]* %{{.*}}, align 4
// CHECK: getelementptr inbounds %struct.s40, %struct.s40* %s1, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s40, %struct.s40* %s2, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s40, %struct.s40* %s1, i32 0, i32 1
// CHECK: getelementptr inbounds %struct.s40, %struct.s40* %s2, i32 0, i32 1
  return s1.i + s2.i + i + s1.s + s2.s;
}
s40_no_align g40;
s40_no_align g40_2;
int caller40() {
// CHECK: define{{.*}} i32 @caller40()
// CHECK: %[[a:.*]] = load [2 x i64], [2 x i64]* bitcast (%struct.s40* @g40 to [2 x i64]*), align 4
// CHECK: %[[b:.*]] = load [2 x i64], [2 x i64]* bitcast (%struct.s40* @g40_2 to [2 x i64]*), align 4
// CHECK: call i32 @f40(i32 3, [2 x i64] %[[a]], [2 x i64] %[[b]])
  return f40(3, g40, g40_2);
}
// passing structs on stack
__attribute__ ((noinline))
int f40_stack(int i, int i2, int i3, int i4, int i5, int i6, int i7, int i8,
              int i9, s40_no_align s1, s40_no_align s2) {
// CHECK: define{{.*}} i32 @f40_stack(i32 %i, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6, i32 %i7, i32 %i8, i32 %i9, [2 x i64] %s1.coerce, [2 x i64] %s2.coerce)
// CHECK: %s1 = alloca %struct.s40, align 4
// CHECK: %s2 = alloca %struct.s40, align 4
// CHECK: store [2 x i64] %s1.coerce, [2 x i64]* %{{.*}}, align 4
// CHECK: store [2 x i64] %s2.coerce, [2 x i64]* %{{.*}}, align 4
// CHECK: getelementptr inbounds %struct.s40, %struct.s40* %s1, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s40, %struct.s40* %s2, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s40, %struct.s40* %s1, i32 0, i32 1
// CHECK: getelementptr inbounds %struct.s40, %struct.s40* %s2, i32 0, i32 1
  return s1.i + s2.i + i + i2 + i3 + i4 + i5 + i6 + i7 + i8 + i9 + s1.s + s2.s;
}
int caller40_stack() {
// CHECK: define{{.*}} i32 @caller40_stack()
// CHECK: %[[a:.*]] = load [2 x i64], [2 x i64]* bitcast (%struct.s40* @g40 to [2 x i64]*), align 4
// CHECK: %[[b:.*]] = load [2 x i64], [2 x i64]* bitcast (%struct.s40* @g40_2 to [2 x i64]*), align 4
// CHECK: call i32 @f40_stack(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, [2 x i64] %[[a]], [2 x i64] %[[b]])
  return f40_stack(1, 2, 3, 4, 5, 6, 7, 8, 9, g40, g40_2);
}

// structs with size <= 16 bytes, with alignment attribute
struct s41
{
  int i;
  short s;
  int i2;
  short s2;
} __attribute__((aligned(16)));
typedef struct s41 s41_with_align;
// passing aligned structs in registers
__attribute__ ((noinline))
int f41(int i, s41_with_align s1, s41_with_align s2) {
// CHECK: define{{.*}} i32 @f41(i32 %i, i128 %s1.coerce, i128 %s2.coerce)
// CHECK: %s1 = alloca %struct.s41, align 16
// CHECK: %s2 = alloca %struct.s41, align 16
// CHECK: store i128 %s1.coerce, i128* %{{.*}}, align 16
// CHECK: store i128 %s2.coerce, i128* %{{.*}}, align 16
// CHECK: getelementptr inbounds %struct.s41, %struct.s41* %s1, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s41, %struct.s41* %s2, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s41, %struct.s41* %s1, i32 0, i32 1
// CHECK: getelementptr inbounds %struct.s41, %struct.s41* %s2, i32 0, i32 1
  return s1.i + s2.i + i + s1.s + s2.s;
}
s41_with_align g41;
s41_with_align g41_2;
int caller41() {
// CHECK: define{{.*}} i32 @caller41()
// CHECK: %[[a:.*]] = load i128, i128* bitcast (%struct.s41* @g41 to i128*), align 16
// CHECK: %[[b:.*]] = load i128, i128* bitcast (%struct.s41* @g41_2 to i128*), align 16
// CHECK: call i32 @f41(i32 3, i128 %[[a]], i128 %[[b]])
  return f41(3, g41, g41_2);
}
// passing aligned structs on stack
__attribute__ ((noinline))
int f41_stack(int i, int i2, int i3, int i4, int i5, int i6, int i7, int i8,
              int i9, s41_with_align s1, s41_with_align s2) {
// CHECK: define{{.*}} i32 @f41_stack(i32 %i, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6, i32 %i7, i32 %i8, i32 %i9, i128 %s1.coerce, i128 %s2.coerce)
// CHECK: %s1 = alloca %struct.s41, align 16
// CHECK: %s2 = alloca %struct.s41, align 16
// CHECK: store i128 %s1.coerce, i128* %{{.*}}, align 16
// CHECK: store i128 %s2.coerce, i128* %{{.*}}, align 16
// CHECK: getelementptr inbounds %struct.s41, %struct.s41* %s1, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s41, %struct.s41* %s2, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s41, %struct.s41* %s1, i32 0, i32 1
// CHECK: getelementptr inbounds %struct.s41, %struct.s41* %s2, i32 0, i32 1
  return s1.i + s2.i + i + i2 + i3 + i4 + i5 + i6 + i7 + i8 + i9 + s1.s + s2.s;
}
int caller41_stack() {
// CHECK: define{{.*}} i32 @caller41_stack()
// CHECK: %[[a:.*]] = load i128, i128* bitcast (%struct.s41* @g41 to i128*), align 16
// CHECK: %[[b:.*]] = load i128, i128* bitcast (%struct.s41* @g41_2 to i128*), align 16
// CHECK: call i32 @f41_stack(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i128 %[[a]], i128 %[[b]])
  return f41_stack(1, 2, 3, 4, 5, 6, 7, 8, 9, g41, g41_2);
}

// structs with size > 16 bytes, without alignment attribute
struct s42
{
  int i;
  short s;
  int i2;
  short s2;
  int i3;
  short s3;
};
typedef struct s42 s42_no_align;
// passing structs in registers
__attribute__ ((noinline))
int f42(int i, s42_no_align s1, s42_no_align s2) {
// CHECK: define{{.*}} i32 @f42(i32 %i, %struct.s42* %s1, %struct.s42* %s2)
// CHECK: getelementptr inbounds %struct.s42, %struct.s42* %s1, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s42, %struct.s42* %s2, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s42, %struct.s42* %s1, i32 0, i32 1
// CHECK: getelementptr inbounds %struct.s42, %struct.s42* %s2, i32 0, i32 1
  return s1.i + s2.i + i + s1.s + s2.s;
}
s42_no_align g42;
s42_no_align g42_2;
int caller42() {
// CHECK: define{{.*}} i32 @caller42()
// CHECK: %[[a:.*]] = alloca %struct.s42, align 4
// CHECK: %[[b:.*]] = alloca %struct.s42, align 4
// CHECK: %[[c:.*]] = bitcast %struct.s42* %[[a]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
// CHECK: %[[d:.*]] = bitcast %struct.s42* %[[b]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
// CHECK: call i32 @f42(i32 3, %struct.s42* %[[a]], %struct.s42* %[[b]])
  return f42(3, g42, g42_2);
}
// passing structs on stack
__attribute__ ((noinline))
int f42_stack(int i, int i2, int i3, int i4, int i5, int i6, int i7, int i8,
              int i9, s42_no_align s1, s42_no_align s2) {
// CHECK: define{{.*}} i32 @f42_stack(i32 %i, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6, i32 %i7, i32 %i8, i32 %i9, %struct.s42* %s1, %struct.s42* %s2)
// CHECK: getelementptr inbounds %struct.s42, %struct.s42* %s1, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s42, %struct.s42* %s2, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s42, %struct.s42* %s1, i32 0, i32 1
// CHECK: getelementptr inbounds %struct.s42, %struct.s42* %s2, i32 0, i32 1
  return s1.i + s2.i + i + i2 + i3 + i4 + i5 + i6 + i7 + i8 + i9 + s1.s + s2.s;
}
int caller42_stack() {
// CHECK: define{{.*}} i32 @caller42_stack()
// CHECK: %[[a:.*]] = alloca %struct.s42, align 4
// CHECK: %[[b:.*]] = alloca %struct.s42, align 4
// CHECK: %[[c:.*]] = bitcast %struct.s42* %[[a]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
// CHECK: %[[d:.*]] = bitcast %struct.s42* %[[b]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
// CHECK: call i32 @f42_stack(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, %struct.s42* %[[a]], %struct.s42* %[[b]])
  return f42_stack(1, 2, 3, 4, 5, 6, 7, 8, 9, g42, g42_2);
}

// structs with size > 16 bytes, with alignment attribute
struct s43
{
  int i;
  short s;
  int i2;
  short s2;
  int i3;
  short s3;
} __attribute__((aligned(16)));
typedef struct s43 s43_with_align;
// passing aligned structs in registers
__attribute__ ((noinline))
int f43(int i, s43_with_align s1, s43_with_align s2) {
// CHECK: define{{.*}} i32 @f43(i32 %i, %struct.s43* %s1, %struct.s43* %s2)
// CHECK: getelementptr inbounds %struct.s43, %struct.s43* %s1, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s43, %struct.s43* %s2, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s43, %struct.s43* %s1, i32 0, i32 1
// CHECK: getelementptr inbounds %struct.s43, %struct.s43* %s2, i32 0, i32 1
  return s1.i + s2.i + i + s1.s + s2.s;
}
s43_with_align g43;
s43_with_align g43_2;
int caller43() {
// CHECK: define{{.*}} i32 @caller43()
// CHECK: %[[a:.*]] = alloca %struct.s43, align 16
// CHECK: %[[b:.*]] = alloca %struct.s43, align 16
// CHECK: %[[c:.*]] = bitcast %struct.s43* %[[a]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
// CHECK: %[[d:.*]] = bitcast %struct.s43* %[[b]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
// CHECK: call i32 @f43(i32 3, %struct.s43* %[[a]], %struct.s43* %[[b]])
  return f43(3, g43, g43_2);
}
// passing aligned structs on stack
__attribute__ ((noinline))
int f43_stack(int i, int i2, int i3, int i4, int i5, int i6, int i7, int i8,
              int i9, s43_with_align s1, s43_with_align s2) {
// CHECK: define{{.*}} i32 @f43_stack(i32 %i, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6, i32 %i7, i32 %i8, i32 %i9, %struct.s43* %s1, %struct.s43* %s2)
// CHECK: getelementptr inbounds %struct.s43, %struct.s43* %s1, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s43, %struct.s43* %s2, i32 0, i32 0
// CHECK: getelementptr inbounds %struct.s43, %struct.s43* %s1, i32 0, i32 1
// CHECK: getelementptr inbounds %struct.s43, %struct.s43* %s2, i32 0, i32 1
  return s1.i + s2.i + i + i2 + i3 + i4 + i5 + i6 + i7 + i8 + i9 + s1.s + s2.s;
}
int caller43_stack() {
// CHECK: define{{.*}} i32 @caller43_stack()
// CHECK: %[[a:.*]] = alloca %struct.s43, align 16
// CHECK: %[[b:.*]] = alloca %struct.s43, align 16
// CHECK: %[[c:.*]] = bitcast %struct.s43* %[[a]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
// CHECK: %[[d:.*]] = bitcast %struct.s43* %[[b]] to i8*
// CHECK: call void @llvm.memcpy.p0i8.p0i8.i64
// CHECK: call i32 @f43_stack(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, %struct.s43* %[[a]], %struct.s43* %[[b]])
  return f43_stack(1, 2, 3, 4, 5, 6, 7, 8, 9, g43, g43_2);
}

// rdar://13668927
// We should not split argument s1 between registers and stack.
__attribute__ ((noinline))
int f40_split(int i, int i2, int i3, int i4, int i5, int i6, int i7,
              s40_no_align s1, s40_no_align s2) {
// CHECK: define{{.*}} i32 @f40_split(i32 %i, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6, i32 %i7, [2 x i64] %s1.coerce, [2 x i64] %s2.coerce)
  return s1.i + s2.i + i + i2 + i3 + i4 + i5 + i6 + i7 + s1.s + s2.s;
}
int caller40_split() {
// CHECK: define{{.*}} i32 @caller40_split()
// CHECK: call i32 @f40_split(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, [2 x i64] %{{.*}} [2 x i64] %{{.*}})
  return f40_split(1, 2, 3, 4, 5, 6, 7, g40, g40_2);
}

__attribute__ ((noinline))
int f41_split(int i, int i2, int i3, int i4, int i5, int i6, int i7,
              s41_with_align s1, s41_with_align s2) {
// CHECK: define{{.*}} i32 @f41_split(i32 %i, i32 %i2, i32 %i3, i32 %i4, i32 %i5, i32 %i6, i32 %i7, i128 %s1.coerce, i128 %s2.coerce)
  return s1.i + s2.i + i + i2 + i3 + i4 + i5 + i6 + i7 + s1.s + s2.s;
}
int caller41_split() {
// CHECK: define{{.*}} i32 @caller41_split()
// CHECK: call i32 @f41_split(i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i128 %{{.*}}, i128 %{{.*}})
  return f41_split(1, 2, 3, 4, 5, 6, 7, g41, g41_2);
}

// Handle homogeneous aggregates properly in variadic functions.
struct HFA {
  float a, b, c, d;
};

float test_hfa(int n, ...) {
// CHECK-LABEL: define{{.*}} float @test_hfa(i32 %n, ...)
// CHECK: [[THELIST:%.*]] = alloca i8*
// CHECK: [[CURLIST:%.*]] = load i8*, i8** [[THELIST]]

  // HFA is not indirect, so occupies its full 16 bytes on the stack.
// CHECK: [[NEXTLIST:%.*]] = getelementptr inbounds i8, i8* [[CURLIST]], i64 16
// CHECK: store i8* [[NEXTLIST]], i8** [[THELIST]]

// CHECK: bitcast i8* [[CURLIST]] to %struct.HFA*
  __builtin_va_list thelist;
  __builtin_va_start(thelist, n);
  struct HFA h = __builtin_va_arg(thelist, struct HFA);
  return h.d;
}

float test_hfa_call(struct HFA *a) {
// CHECK-LABEL: define{{.*}} float @test_hfa_call(%struct.HFA* %a)
// CHECK: call float (i32, ...) @test_hfa(i32 1, [4 x float] {{.*}})
  test_hfa(1, *a);
}

struct TooBigHFA {
  float a, b, c, d, e;
};

float test_toobig_hfa(int n, ...) {
// CHECK-LABEL: define{{.*}} float @test_toobig_hfa(i32 %n, ...)
// CHECK: [[THELIST:%.*]] = alloca i8*
// CHECK: [[CURLIST:%.*]] = load i8*, i8** [[THELIST]]

  // TooBigHFA is not actually an HFA, so gets passed indirectly. Only 8 bytes
  // of stack consumed.
// CHECK: [[NEXTLIST:%.*]] = getelementptr inbounds i8, i8* [[CURLIST]], i64 8
// CHECK: store i8* [[NEXTLIST]], i8** [[THELIST]]

// CHECK: [[HFAPTRPTR:%.*]] = bitcast i8* [[CURLIST]] to %struct.TooBigHFA**
// CHECK: [[HFAPTR:%.*]] = load %struct.TooBigHFA*, %struct.TooBigHFA** [[HFAPTRPTR]]
  __builtin_va_list thelist;
  __builtin_va_start(thelist, n);
  struct TooBigHFA h = __builtin_va_arg(thelist, struct TooBigHFA);
  return h.d;
}

struct HVA {
  int32x4_t a, b;
};

int32x4_t test_hva(int n, ...) {
// CHECK-LABEL: define{{.*}} <4 x i32> @test_hva(i32 %n, ...)
// CHECK: [[THELIST:%.*]] = alloca i8*
// CHECK: [[CURLIST:%.*]] = load i8*, i8** [[THELIST]]

  // HVA is not indirect, so occupies its full 16 bytes on the stack. but it
  // must be properly aligned.
// CHECK: [[ALIGN0:%.*]] = ptrtoint i8* [[CURLIST]] to i64
// CHECK: [[ALIGN1:%.*]] = add i64 [[ALIGN0]], 15
// CHECK: [[ALIGN2:%.*]] = and i64 [[ALIGN1]], -16
// CHECK: [[ALIGNED_LIST:%.*]] = inttoptr i64 [[ALIGN2]] to i8*

// CHECK: [[NEXTLIST:%.*]] = getelementptr inbounds i8, i8* [[ALIGNED_LIST]], i64 32
// CHECK: store i8* [[NEXTLIST]], i8** [[THELIST]]

// CHECK: bitcast i8* [[ALIGNED_LIST]] to %struct.HVA*
  __builtin_va_list thelist;
  __builtin_va_start(thelist, n);
  struct HVA h = __builtin_va_arg(thelist, struct HVA);
  return h.b;
}

struct TooBigHVA {
  int32x4_t a, b, c, d, e;
};

int32x4_t test_toobig_hva(int n, ...) {
// CHECK-LABEL: define{{.*}} <4 x i32> @test_toobig_hva(i32 %n, ...)
// CHECK: [[THELIST:%.*]] = alloca i8*
// CHECK: [[CURLIST:%.*]] = load i8*, i8** [[THELIST]]

  // TooBigHVA is not actually an HVA, so gets passed indirectly. Only 8 bytes
  // of stack consumed.
// CHECK: [[NEXTLIST:%.*]] = getelementptr inbounds i8, i8* [[CURLIST]], i64 8
// CHECK: store i8* [[NEXTLIST]], i8** [[THELIST]]

// CHECK: [[HVAPTRPTR:%.*]] = bitcast i8* [[CURLIST]] to %struct.TooBigHVA**
// CHECK: [[HVAPTR:%.*]] = load %struct.TooBigHVA*, %struct.TooBigHVA** [[HVAPTRPTR]]
  __builtin_va_list thelist;
  __builtin_va_start(thelist, n);
  struct TooBigHVA h = __builtin_va_arg(thelist, struct TooBigHVA);
  return h.d;
}

typedef __attribute__((__ext_vector_type__(3))) float float32x3_t;
typedef struct { float32x3_t arr[4]; } HFAv3;

float32x3_t test_hva_v3(int n, ...) {
// CHECK-LABEL: define{{.*}} <3 x float> @test_hva_v3(i32 %n, ...)
// CHECK: [[THELIST:%.*]] = alloca i8*
// CHECK: [[CURLIST:%.*]] = load i8*, i8** [[THELIST]]

  // HVA is not indirect, so occupies its full 16 bytes on the stack. but it
  // must be properly aligned.
// CHECK: [[ALIGN0:%.*]] = ptrtoint i8* [[CURLIST]] to i64
// CHECK: [[ALIGN1:%.*]] = add i64 [[ALIGN0]], 15
// CHECK: [[ALIGN2:%.*]] = and i64 [[ALIGN1]], -16
// CHECK: [[ALIGNED_LIST:%.*]] = inttoptr i64 [[ALIGN2]] to i8*

// CHECK: [[NEXTLIST:%.*]] = getelementptr inbounds i8, i8* [[ALIGNED_LIST]], i64 64
// CHECK: store i8* [[NEXTLIST]], i8** [[THELIST]]

// CHECK: bitcast i8* [[ALIGNED_LIST]] to %struct.HFAv3*
  __builtin_va_list l;
  __builtin_va_start(l, n);
  HFAv3 r = __builtin_va_arg(l, HFAv3);
  return r.arr[2];
}

float32x3_t test_hva_v3_call(HFAv3 *a) {
// CHECK-LABEL: define{{.*}} <3 x float> @test_hva_v3_call(%struct.HFAv3* %a)
// CHECK: call <3 x float> (i32, ...) @test_hva_v3(i32 1, [4 x <4 x float>] {{.*}})
  return test_hva_v3(1, *a);
}
