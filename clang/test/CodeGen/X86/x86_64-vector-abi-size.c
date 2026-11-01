// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,NOAVX,NOAVX512
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fexperimental-abi-lowering %s -o - | FileCheck %s --check-prefixes=CHECK,NOAVX,NOAVX512
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,AVX,NOAVX512
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx -emit-llvm -fexperimental-abi-lowering %s -o - | FileCheck %s --check-prefixes=CHECK,AVX,NOAVX512
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512f -emit-llvm %s -o - | FileCheck %s --check-prefixes=CHECK,AVX,AVX512
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -target-feature +avx512f -emit-llvm -fexperimental-abi-lowering %s -o - | FileCheck %s --check-prefixes=CHECK,AVX,AVX512

typedef long double ld1 __attribute__((vector_size(16)));
typedef long double ld2 __attribute__((vector_size(32)));
typedef long double ld4 __attribute__((vector_size(64)));
typedef long double ld3 __attribute__((ext_vector_type(3)));
typedef char c3 __attribute__((ext_vector_type(3)));
typedef short s3 __attribute__((ext_vector_type(3)));
typedef float f3 __attribute__((ext_vector_type(3)));
typedef double d3 __attribute__((ext_vector_type(3)));
typedef _Bool b4 __attribute__((ext_vector_type(4)));
typedef _Bool b17 __attribute__((ext_vector_type(17)));
typedef _BitInt(4) bi4x2 __attribute__((ext_vector_type(2)));
typedef _BitInt(128) bi128x3 __attribute__((ext_vector_type(3)));
typedef _BitInt(4) bi4x16 __attribute__((ext_vector_type(16)));
typedef _BitInt(128) bi128x1 __attribute__((ext_vector_type(1)));
typedef __int128 i128x1 __attribute__((ext_vector_type(1)));

// A one-element x87 vector takes 16 bytes, so a struct or union wrapping it,
// directly or in a one-element array, is exactly as large as the vector.
struct S1 { ld1 v; };
struct S1 s1(struct S1 s) { return s; }
// CHECK: define{{.*}} <1 x x86_fp80> @s1(<1 x x86_fp80> %{{.*}})

struct A1 { ld1 a[1]; };
struct A1 a1(struct A1 a) { return a; }
// CHECK: define{{.*}} <1 x x86_fp80> @a1(<1 x x86_fp80> %{{.*}})

union U1 { ld1 v; };
union U1 u1(union U1 u) { return u; }
// CHECK: define{{.*}} <1 x x86_fp80> @u1(<1 x x86_fp80> %{{.*}})

// An unnamed argument is passed the same way.
void var(int n, ...);
void call_var(struct S1 s) { var(1, s); }
// CHECK: call void (i32, ...) @var(i32 noundef 1, <1 x x86_fp80> %{{.*}})

// Two x87 elements take 32 bytes, which AVX passes directly.
struct S2 { ld2 v; };
struct S2 s2(struct S2 s) { return s; }
// NOAVX: define{{.*}} void @s2(ptr dead_on_unwind noalias writable sret(%struct.S2) align 32 %{{.*}}, ptr noundef byval(%struct.S2) align 32 %{{.*}})
// AVX: define{{.*}} <2 x x86_fp80> @s2(<2 x x86_fp80> %{{.*}})

struct A2 { ld2 a[1]; };
struct A2 a2(struct A2 a) { return a; }
// NOAVX: define{{.*}} void @a2(ptr dead_on_unwind noalias writable sret(%struct.A2) align 32 %{{.*}}, ptr noundef byval(%struct.A2) align 32 %{{.*}})
// AVX: define{{.*}} <2 x x86_fp80> @a2(<2 x x86_fp80> %{{.*}})

// Four take 64 bytes, which AVX-512 passes directly.
struct S4 { ld4 v; };
struct S4 s4(struct S4 s) { return s; }
// NOAVX512: define{{.*}} void @s4(ptr dead_on_unwind noalias writable sret(%struct.S4) align 64 %{{.*}}, ptr noundef byval(%struct.S4) align 64 %{{.*}})
// AVX512: define{{.*}} <4 x x86_fp80> @s4(<4 x x86_fp80> %{{.*}})

// Three are rounded up to 64 bytes, so only AVX-512 passes them directly.
void v3(ld3 v) {}
// NOAVX512: define{{.*}} void @v3(ptr noundef byval(<3 x x86_fp80>) align 64 %{{.*}})
// AVX512: define{{.*}} void @v3(<3 x x86_fp80> noundef %{{.*}})

// The vector spans all 16 bytes of the union, so both eightbytes count as data.
struct __attribute__((aligned(16))) LI { long l; int i; };
union ULI { struct LI s; ld1 v; };
union ULI uli(union ULI u) { return u; }
// CHECK: define{{.*}} { i64, i64 } @uli(i64 %{{.*}}, i64 %{{.*}})

// Three chars take 4 bytes.
c3 vc3(c3 v) { return v; }
// CHECK: define{{.*}} i32 @vc3(i32 %{{.*}})

struct SC3 { c3 v; };
struct SC3 sc3(struct SC3 s) { return s; }
// CHECK: define{{.*}} i32 @sc3(i32 %{{.*}})

// On the stack a three-char vector is coerced to i32 too.
void stk(long a, long b, long c, long d, long e, long f, c3 v) {}
// CHECK: define{{.*}} void @stk(i64 noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, i32 %{{.*}})

// Three shorts take 8 bytes.
s3 vs3(s3 v) { return v; }
// CHECK: define{{.*}} double @vs3(double %{{.*}})

// The third vector starts at byte 8 and reaches past the short.
union UA { struct { long l; short s; } ls; c3 a[3]; };
union UA ua(union UA u) { return u; }
// CHECK: define{{.*}} { i64, i64 } @ua(i64 %{{.*}}, i64 %{{.*}})

// Three floats take 16 bytes.
struct SF3 { f3 v; };
struct SF3 sf3(struct SF3 s) { return s; }
// CHECK: define{{.*}} <3 x float> @sf3(<3 x float> %{{.*}})

// Three doubles take 32 bytes.
struct SD3 { d3 v; };
struct SD3 sd3(struct SD3 s) { return s; }
// NOAVX: define{{.*}} void @sd3(ptr dead_on_unwind noalias writable sret(%struct.SD3) align 32 %{{.*}}, ptr noundef byval(%struct.SD3) align 32 %{{.*}})
// AVX: define{{.*}} <3 x double> @sd3(<3 x double> %{{.*}})

// Three 128-bit _BitInt elements round up to 64 bytes, which AVX-512 passes as
// an <8 x i64> vector.
struct SBI3 { bi128x3 v; };
struct SBI3 sbi3(struct SBI3 s) { return s; }
// NOAVX512: define{{.*}} void @sbi3(ptr dead_on_unwind noalias writable sret(%struct.SBI3) align 64 %{{.*}}, ptr noundef byval(%struct.SBI3) align 64 %{{.*}})
// AVX512: define{{.*}} <8 x i64> @sbi3(<8 x i64> %{{.*}})

// Each two-element vector of 4-bit _BitInt takes 2 bytes, so the fifth reaches
// the second eightbyte.
struct AB { bi4x2 a[5]; };
struct AB ab(struct AB s) { return s; }
// CHECK: define{{.*}} { i64, i16 } @ab(i64 %{{.*}}, i16 %{{.*}})

// Padding past the vector keeps these from being passed as the vector alone.
union __attribute__((aligned(32))) UA32 { ld1 v; };
union UA32 ua32(union UA32 u) { return u; }
// NOAVX: define{{.*}} void @ua32(ptr dead_on_unwind noalias writable sret(%union.UA32) align 32 %{{.*}}, ptr noundef byval(%union.UA32) align 32 %{{.*}})
// AVX: define{{.*}} <4 x double> @ua32(<4 x double> %{{.*}})

struct __attribute__((aligned(32))) SF32 { f3 v; };
struct SF32 sf32(struct SF32 s) { return s; }
// CHECK: define{{.*}} void @sf32(ptr dead_on_unwind noalias writable sret(%struct.SF32) align 32 %{{.*}}, ptr noundef byval(%struct.SF32) align 32 %{{.*}})

// Four bools take one byte.
b4 vb4(b4 v) { return v; }
// CHECK: define{{.*}} i8 @vb4(i8 noundef %{{.*}})

struct SB4 { b4 v; };
struct SB4 sb4(struct SB4 s) { return s; }
// CHECK: define{{.*}} i8 @sb4(i8 %{{.*}})

// Seventeen bools take 17 bits, which round up to 4 bytes.
b17 vb17(b17 v) { return v; }
// CHECK: define{{.*}} i32 @vb17(i32 %{{.*}})

// On the stack four bools are coerced to i8.
void stkb(long a, long b, long c, long d, long e, long f, b4 v) {}
// CHECK: define{{.*}} void @stkb(i64 noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, i8 noundef %{{.*}})

// Each 4-bit _BitInt element takes a byte.
bi4x2 vbi4x2(bi4x2 v) { return v; }
// CHECK: define{{.*}} i16 @vbi4x2(i16 %{{.*}})

// Sixteen 4-bit _BitInt elements take 16 bytes, so once the SSE registers run
// out the vector is still passed directly.
void exhaust(double a, double b, double c, double d, double e, double f,
             double g, double h, bi4x16 v) {}
// CHECK: define{{.*}} void @exhaust(double noundef %{{.*}}, double noundef %{{.*}}, double noundef %{{.*}}, double noundef %{{.*}}, double noundef %{{.*}}, double noundef %{{.*}}, double noundef %{{.*}}, double noundef %{{.*}}, <16 x i4> noundef %{{.*}})

// Once the SSE registers run out, a vector of _BitInt(128) is still passed
// directly, while a vector of __int128 is passed in memory.
void ex128(double a, double b, double c, double d, double e, double f,
           double g, double h, bi128x1 v) {}
// CHECK: define{{.*}} void @ex128(double noundef %{{.*}}, double noundef %{{.*}}, double noundef %{{.*}}, double noundef %{{.*}}, double noundef %{{.*}}, double noundef %{{.*}}, double noundef %{{.*}}, double noundef %{{.*}}, <1 x i128> noundef %{{.*}})

void exi128(double a, double b, double c, double d, double e, double f,
            double g, double h, i128x1 v) {}
// CHECK: define{{.*}} void @exi128(double noundef %{{.*}}, double noundef %{{.*}}, double noundef %{{.*}}, double noundef %{{.*}}, double noundef %{{.*}}, double noundef %{{.*}}, double noundef %{{.*}}, double noundef %{{.*}}, ptr noundef byval(<1 x i128>) align 16 %{{.*}})
