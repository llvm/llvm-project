// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm -fexperimental-abi-lowering %s -o - | FileCheck %s

typedef _Bool b3 __attribute__((ext_vector_type(3)));
typedef _Bool b4 __attribute__((ext_vector_type(4)));
typedef _Bool b8 __attribute__((ext_vector_type(8)));
typedef _Bool b12 __attribute__((ext_vector_type(12)));
typedef _Bool b16 __attribute__((ext_vector_type(16)));
typedef _Bool b32 __attribute__((ext_vector_type(32)));
typedef unsigned _BitInt(1) ub1x1 __attribute__((ext_vector_type(1)));
typedef unsigned _BitInt(1) ub1x4 __attribute__((ext_vector_type(4)));
typedef char c1 __attribute__((ext_vector_type(1)));
typedef float f1 __attribute__((ext_vector_type(1)));

// Each _BitInt(17) takes 4 bytes, so the last two fill the high eightbyte.
struct BI17x4 { _BitInt(17) a[4]; };
struct BI17x4 bi17x4(struct BI17x4 s) { return s; }
// CHECK: define dso_local { i64, i64 } @bi17x4(i64 %{{.*}}, i64 %{{.*}})

struct IntBI17x3 { int x; _BitInt(17) a[3]; };
struct IntBI17x3 intbi17x3(struct IntBI17x3 s) { return s; }
// CHECK: define dso_local { i64, i64 } @intbi17x3(i64 %{{.*}}, i64 %{{.*}})

// The one-byte element is followed only by padding.
struct LongUBI3x1 { long l; unsigned _BitInt(3) a[1]; };
struct LongUBI3x1 longubi3x1(struct LongUBI3x1 s) { return s; }
// CHECK: define dso_local { i64, i8 } @longubi3x1(i64 %{{.*}}, i8 %{{.*}})

struct LongBool1 { long l; _Bool b[1]; };
struct LongBool1 longbool1(struct LongBool1 s) { return s; }
// CHECK: define dso_local { i64, i8 } @longbool1(i64 %{{.*}}, i8 %{{.*}})

// Each _BitInt element is padded out to its alignment.
struct BI3x16 { _BitInt(3) a[16]; };
struct BI3x16 bi3x16(struct BI3x16 s) { return s; }
// CHECK: define dso_local { i64, i64 } @bi3x16(i64 %{{.*}}, i64 %{{.*}})

struct BI9x5 { _BitInt(9) a[5]; };
struct BI9x5 bi9x5(struct BI9x5 s) { return s; }
// CHECK: define dso_local { i64, i16 } @bi9x5(i64 %{{.*}}, i16 %{{.*}})

struct BI33x2 { _BitInt(33) a[2]; };
struct BI33x2 bi33x2(struct BI33x2 s) { return s; }
// CHECK: define dso_local { i64, i64 } @bi33x2(i64 %{{.*}}, i64 %{{.*}})

struct CharBI7x8 { char c; _BitInt(7) a[8]; };
struct CharBI7x8 charbi7x8(struct CharBI7x8 s) { return s; }
// CHECK: define dso_local { i64, i8 } @charbi7x8(i64 %{{.*}}, i8 %{{.*}})

// Each bool takes a byte.
struct Bool16 { _Bool b[16]; };
struct Bool16 bool16(struct Bool16 s) { return s; }
// CHECK: define dso_local { i64, i64 } @bool16(i64 %{{.*}}, i64 %{{.*}})

struct Bool9 { _Bool b[9]; };
struct Bool9 bool9(struct Bool9 s) { return s; }
// CHECK: define dso_local { i64, i8 } @bool9(i64 %{{.*}}, i8 %{{.*}})

// The third element starts at byte 2, so the union's data reaches past the
// short.
union U3S { unsigned _BitInt(3) a[3]; short s; };
union U3S u3s(union U3S u) { return u; }
// CHECK: define dso_local i32 @u3s(i32 %{{.*}})

union UBool3S { _Bool a[3]; short s; };
union UBool3S ubool3s(union UBool3S u) { return u; }
// CHECK: define dso_local i32 @ubool3s(i32 %{{.*}})

// The ninth element shares the high eightbyte with the float.
struct UBI3x9Float { unsigned _BitInt(3) a[9]; float f; };
struct UBI3x9Float ubi3x9float(struct UBI3x9Float s) { return s; }
// CHECK: define dso_local { i64, i64 } @ubi3x9float(i64 %{{.*}}, i64 %{{.*}})

struct Bool9Float { _Bool b[9]; float f; };
struct Bool9Float bool9float(struct Bool9Float s) { return s; }
// CHECK: define dso_local { i64, i64 } @bool9float(i64 %{{.*}}, i64 %{{.*}})

// A bool or narrow _BitInt bit-field counts at the size of its type, a whole
// byte, so it reaches the second byte.
struct __attribute__((aligned(2))) BoolBitField {
  unsigned char x : 3;
  _Bool b : 1;
};
struct BoolBitField boolbitfield(struct BoolBitField s) { return s; }
// CHECK: define dso_local i16 @boolbitfield(i16 %{{.*}})

struct __attribute__((aligned(2))) BitIntBitField {
  unsigned char x : 3;
  _BitInt(5) y : 2;
};
struct BitIntBitField bitintbitfield(struct BitIntBitField s) { return s; }
// CHECK: define dso_local i16 @bitintbitfield(i16 %{{.*}})

// The long double covers all 16 bytes, so the high eightbyte is a whole i64 in
// either member order.
struct __attribute__((aligned(16))) LongShort { long a; short b; };
union LDFirst { long double ld; struct LongShort s; };
union LDFirst ldfirst(union LDFirst u) { return u; }
// CHECK: define dso_local { i64, i64 } @ldfirst(i64 %{{.*}}, i64 %{{.*}})

union LSFirst { struct LongShort s; long double ld; };
union LSFirst lsfirst(union LSFirst u) { return u; }
// CHECK: define dso_local { i64, i64 } @lsfirst(i64 %{{.*}}, i64 %{{.*}})

// In registers a _BitInt is coerced to the eightbytes covering its storage.
_BitInt(40) bi40(_BitInt(40) x) { return x; }
// CHECK: define dso_local i64 @bi40(i64 noundef %{{.*}})

_BitInt(100) bi100(_BitInt(100) x) { return x; }
// CHECK: define dso_local { i64, i64 } @bi100(i64 noundef %{{.*}}, i64 noundef %{{.*}})

// On the stack a _BitInt is coerced to the integer covering its storage.
void stkbi17(long a, long b, long c, long d, long e, long f, _BitInt(17) x) {}
// CHECK: define dso_local void @stkbi17(i64 noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, i32 noundef %{{.*}})

void stkubi3(long a, long b, long c, long d, long e, long f,
             unsigned _BitInt(3) x) {}
// CHECK: define dso_local void @stkubi3(i64 noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, i64 noundef %{{.*}}, i8 noundef %{{.*}})

// An aligned typedef raises a scalar's alignment but not its size.
typedef float af __attribute__((aligned(8)));
struct AlignedFloat { af x; float y; };
struct AlignedFloat alignedfloat(struct AlignedFloat s) { return s; }
// CHECK: define dso_local <2 x float> @alignedfloat(<2 x float> %{{.*}})

typedef _BitInt(17) abi17 __attribute__((aligned(8)));
struct AlignedBI17 { abi17 x; };
struct AlignedBI17 alignedbi17(struct AlignedBI17 s) { return s; }
// CHECK: define dso_local i32 @alignedbi17(i32 %{{.*}})

// A bool vector is stored as an integer with one bit per element, at least a
// byte wide.
struct LongB3 { long l; b3 v; };
struct LongB3 longb3(struct LongB3 s) { return s; }
// CHECK: define dso_local { i64, i8 } @longb3(i64 %{{.*}}, i8 %{{.*}})

struct LongB4 { long l; b4 v; };
struct LongB4 longb4(struct LongB4 s) { return s; }
// CHECK: define dso_local { i64, i8 } @longb4(i64 %{{.*}}, i8 %{{.*}})

struct LongB8 { long l; b8 v; };
struct LongB8 longb8(struct LongB8 s) { return s; }
// CHECK: define dso_local { i64, i8 } @longb8(i64 %{{.*}}, i8 %{{.*}})

struct LongB16 { long l; b16 v; };
struct LongB16 longb16(struct LongB16 s) { return s; }
// CHECK: define dso_local { i64, i16 } @longb16(i64 %{{.*}}, i16 %{{.*}})

struct LongB32 { long l; b32 v; };
struct LongB32 longb32(struct LongB32 s) { return s; }
// CHECK: define dso_local { i64, i32 } @longb32(i64 %{{.*}}, i32 %{{.*}})

struct B4x1Long { b4 a[1]; long l; };
struct B4x1Long b4x1long(struct B4x1Long s) { return s; }
// CHECK: define dso_local { i8, i64 } @b4x1long(i8 %{{.*}}, i64 %{{.*}})

struct __attribute__((aligned(4))) B4Al4 { b4 v; };
struct B4Al4 b4al4(struct B4Al4 s) { return s; }
// CHECK: define dso_local i8 @b4al4(i8 %{{.*}})

union __attribute__((aligned(4))) UB4Al4 { b4 v; };
union UB4Al4 ub4al4(union UB4Al4 u) { return u; }
// CHECK: define dso_local i8 @ub4al4(i8 %{{.*}})

struct LongInB4 { long l; struct { b4 v; } in; };
struct LongInB4 longinb4(struct LongInB4 s) { return s; }
// CHECK: define dso_local { i64, i8 } @longinb4(i64 %{{.*}}, i8 %{{.*}})

struct __attribute__((aligned(8))) B16Al8 { b16 v; };
struct B16Al8 b16al8(struct B16Al8 s) { return s; }
// CHECK: define dso_local i16 @b16al8(i16 %{{.*}})

// Twelve bools make an i12, which is not an i8, i16 or i32, so the eightbyte
// stays an i64.
struct LongB12 { long l; b12 v; };
struct LongB12 longb12(struct LongB12 s) { return s; }
// CHECK: define dso_local { i64, i64 } @longb12(i64 %{{.*}}, i64 %{{.*}})

// A one-bit _BitInt element is not a bool, so it takes a byte.
ub1x4 vub1x4(ub1x4 v) { return v; }
// CHECK: define dso_local i32 @vub1x4(i32 %{{.*}})

// A one-bit _BitInt, char or float vector is not a bool vector, so the
// eightbyte holding it stays an i64.
struct LongUB1x4 { long l; ub1x4 v; };
struct LongUB1x4 longub1x4(struct LongUB1x4 s) { return s; }
// CHECK: define dso_local { i64, i64 } @longub1x4(i64 %{{.*}}, i64 %{{.*}})

struct LongUB1x1 { long l; ub1x1 v; };
struct LongUB1x1 longub1x1(struct LongUB1x1 s) { return s; }
// CHECK: define dso_local { i64, i64 } @longub1x1(i64 %{{.*}}, i64 %{{.*}})

struct LongC1 { long l; c1 v; };
struct LongC1 longc1(struct LongC1 s) { return s; }
// CHECK: define dso_local { i64, i64 } @longc1(i64 %{{.*}}, i64 %{{.*}})

struct LongF1 { long l; f1 v; };
struct LongF1 longf1(struct LongF1 s) { return s; }
// CHECK: define dso_local { i64, i64 } @longf1(i64 %{{.*}}, i64 %{{.*}})
