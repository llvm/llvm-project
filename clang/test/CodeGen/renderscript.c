// RUN: %clang_cc1 %s -triple=renderscript32-none-linux-gnueabi -emit-llvm -o - -Werror | FileCheck %s -check-prefix=CHECK-RS32
// RUN: %clang_cc1 %s -triple=renderscript64-none-linux-android -emit-llvm -o - -Werror | FileCheck %s -check-prefix=CHECK-RS64
// RUN: %clang_cc1 %s -triple=armv7-none-linux-gnueabi -emit-llvm -o - -Werror | FileCheck %s -check-prefix=CHECK-ARM

// Ensure that the bitcode has the correct triple
// CHECK-RS32: target triple = "armv7-none-linux-gnueabi"
// CHECK-RS64: target triple = "aarch64-none-linux-android"
// CHECK-ARM: target triple = "armv7-none-linux-gnueabi"

// Ensure that long data type has 8-byte size and alignment in RenderScript
#ifdef __RENDERSCRIPT__
#define LONG_WIDTH_AND_ALIGN 8
#else
#define LONG_WIDTH_AND_ALIGN 4
#endif

_Static_assert(sizeof(long) == LONG_WIDTH_AND_ALIGN, "sizeof long is wrong");
_Static_assert(_Alignof(long) == LONG_WIDTH_AND_ALIGN, "sizeof long is wrong");

// CHECK-RS32: i64 @test_long(i64 %v)
// CHECK-RS64: i64 @test_long(i64 %v)
// CHECK-ARM: i32 @test_long(i32 %v)
long test_long(long v) {
  return v + 1;
}

// =============================================================================
// Test coercion of aggregate argument or return value into integer arrays
// =============================================================================

// =============================================================================
// aggregate parameter <= 4 bytes: coerced to [a x iNN] for both 32-bit and
// 64-bit RenderScript
// ==============================================================================

typedef struct {char c1, c2, c3; } sChar3;
typedef struct {short s; char c;} sShortChar;

// CHECK-RS32: void @argChar3([3 x i8] %s.coerce)
// CHECK-RS64: void @argChar3([3 x i8] %s.coerce)
void argChar3(sChar3 s) {}

// CHECK-RS32: void @argShortChar([2 x i16] %s.coerce)
// CHECK-RS64: void @argShortChar([2 x i16] %s.coerce)
void argShortChar(sShortChar s) {}

// =============================================================================
// aggregate return value <= 4 bytes: coerced to [a x iNN] for both 32-bit and
// 64-bit RenderScript
// =============================================================================

// CHECK-RS32: [3 x i8] @retChar3()
// CHECK-RS64: [3 x i8] @retChar3()
sChar3 retChar3() { sChar3 r; return r; }

// CHECK-RS32: [2 x i16] @retShortChar()
// CHECK-RS64: [2 x i16] @retShortChar()
sShortChar retShortChar() { sShortChar r; return r; }

// =============================================================================
// aggregate parameter <= 16 bytes: coerced to [a x iNN] for both 32-bit and
// 64-bit RenderScript
// =============================================================================

typedef struct {short s1; char c; short s2; } sShortCharShort;
typedef struct {int i; short s; char c; } sIntShortChar;
typedef struct {long l; int i; } sLongInt;

// CHECK-RS32: void @argShortCharShort([3 x i16] %s.coerce)
// CHECK-RS64: void @argShortCharShort([3 x i16] %s.coerce)
void argShortCharShort(sShortCharShort s) {}

// CHECK-RS32: void @argIntShortChar([2 x i32] %s.coerce)
// CHECK-RS64: void @argIntShortChar([2 x i32] %s.coerce)
void argIntShortChar(sIntShortChar s) {}

// CHECK-RS32: void @argLongInt([2 x i64] %s.coerce)
// CHECK-RS64: void @argLongInt([2 x i64] %s.coerce)
void argLongInt(sLongInt s) {}

// =============================================================================
// aggregate return value <= 16 bytes: returned on stack for 32-bit RenderScript
// and coerced to [a x iNN] for 64-bit RenderScript
// =============================================================================

// CHECK-RS32: void @retShortCharShort(%struct.sShortCharShort* noalias sret(%struct.sShortCharShort) align 2 %agg.result)
// CHECK-RS64: [3 x i16] @retShortCharShort()
sShortCharShort retShortCharShort() { sShortCharShort r; return r; }

// CHECK-RS32: void @retIntShortChar(%struct.sIntShortChar* noalias sret(%struct.sIntShortChar) align 4 %agg.result)
// CHECK-RS64: [2 x i32] @retIntShortChar()
sIntShortChar retIntShortChar() { sIntShortChar r; return r; }

// CHECK-RS32: void @retLongInt(%struct.sLongInt* noalias sret(%struct.sLongInt) align 8 %agg.result)
// CHECK-RS64: [2 x i64] @retLongInt()
sLongInt retLongInt() { sLongInt r; return r; }

// =============================================================================
// aggregate parameter <= 64 bytes: coerced to [a x iNN] for 32-bit RenderScript
// and passed on the stack for 64-bit RenderScript
// =============================================================================

typedef struct {int i1, i2, i3, i4, i5; } sInt5;
typedef struct {long l1, l2; char c; } sLong2Char;

// CHECK-RS32: void @argInt5([5 x i32] %s.coerce)
// CHECK-RS64: void @argInt5(%struct.sInt5* %s)
void argInt5(sInt5 s) {}

// CHECK-RS32: void @argLong2Char([3 x i64] %s.coerce)
// CHECK-RS64: void @argLong2Char(%struct.sLong2Char* %s)
void argLong2Char(sLong2Char s) {}

// =============================================================================
// aggregate return value <= 64 bytes: returned on stack for both 32-bit and
// 64-bit RenderScript
// =============================================================================

// CHECK-RS32: void @retInt5(%struct.sInt5* noalias sret(%struct.sInt5) align 4 %agg.result)
// CHECK-RS64: void @retInt5(%struct.sInt5* noalias sret(%struct.sInt5) align 4 %agg.result)
sInt5 retInt5() { sInt5 r; return r;}

// CHECK-RS32: void @retLong2Char(%struct.sLong2Char* noalias sret(%struct.sLong2Char) align 8 %agg.result)
// CHECK-RS64: void @retLong2Char(%struct.sLong2Char* noalias sret(%struct.sLong2Char) align 8 %agg.result)
sLong2Char retLong2Char() { sLong2Char r; return r;}

// =============================================================================
// aggregate parameters and return values > 64 bytes: passed and returned on the
// stack for both 32-bit and 64-bit RenderScript
// =============================================================================

typedef struct {long l1, l2, l3, l4, l5, l6, l7, l8, l9; } sLong9;

// CHECK-RS32: void @argLong9(%struct.sLong9* byval(%struct.sLong9) align 8 %s)
// CHECK-RS64: void @argLong9(%struct.sLong9* %s)
void argLong9(sLong9 s) {}

// CHECK-RS32: void @retLong9(%struct.sLong9* noalias sret(%struct.sLong9) align 8 %agg.result)
// CHECK-RS64: void @retLong9(%struct.sLong9* noalias sret(%struct.sLong9) align 8 %agg.result)
sLong9 retLong9() { sLong9 r; return r; }
