// RUN: %clang_cc1 -triple x86_64-pc-windows-msvc -std=c++11 -emit-llvm \
// RUN:   -o - %s | FileCheck %s
// RUN: %clang_cc1 -triple arm64ec-pc-windows-msvc -std=c++11 -emit-llvm \
// RUN:   -o - %s | FileCheck %s

enum class U8 : unsigned char {};
enum class S8 : signed char {};
enum class U16 : unsigned short {};
enum class S16 : short {};
enum class U32 : unsigned int {};
enum ClassicU8 : unsigned char {};

extern "C" void variadic(int, ...);

// Fixed parameters should continue to be passed without extension.
// CHECK-LABEL: define dso_local void @fixed(i8 noundef %{{[^)]+}})
extern "C" void fixed(U8) {}

// Named parameters of variadic functions are still fixed arguments.
// CHECK-LABEL: define dso_local void @named_variadic(i8 noundef %{{[^,]+}}, ...)
extern "C" void named_variadic(U8, ...) {}

// CHECK-LABEL: define dso_local void @test(
// CHECK: call void (i32, ...) @variadic(
// CHECK-SAME: i32 noundef 0,
// Scoped enums do not undergo the default argument integer promotions, so they
// retain their i8/i16 IR types and use extension attributes for ABI widening.
// CHECK-SAME: i8 noundef zeroext %{{[^,]+}}, i8 noundef signext %{{[^,]+}},
// CHECK-SAME: i16 noundef zeroext %{{[^,]+}}, i16 noundef signext %{{[^,]+}},
// Regular integer types and unscoped enums do undergo integer promotion (for
// example, unsigned char to int), so those arguments are passed as i32.
// CHECK-SAME: i32 noundef %{{[^,]+}}, i32 noundef %{{[^,]+}}, i32 noundef %{{[^)]+}})
extern "C" void test(U8 u8, S8 s8, U16 u16, S16 s16, U32 u32,
                     ClassicU8 classic_u8, unsigned char plain_u8) {
  variadic(0, u8, s8, u16, s16, u32, classic_u8, plain_u8);
}
