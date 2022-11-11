// RUN: %clang_cc1 -w -fblocks -ffreestanding -triple i386-pc-linux-gnu -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -w -fblocks -ffreestanding -triple i386-pc-linux-gnu -target-feature +avx -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -w -fblocks -ffreestanding -triple i386-pc-linux-gnu -target-feature +avx512f -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -w -fblocks -ffreestanding -triple i386-pc-linux-mingw -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -w -fblocks -ffreestanding -triple i386-pc-linux-mingw -target-feature +avx -emit-llvm -o - %s | FileCheck %s
// RUN: %clang_cc1 -w -fblocks -ffreestanding -triple i386-pc-linux-mingw -target-feature +avx512f -emit-llvm -o - %s | FileCheck %s

#include <immintrin.h>

// CHECK-LABEL: define dso_local void @testm128
// CHECK-LABEL: %argp.cur = load ptr, ptr %args, align 4
// CHECK-NEXT:  %0 = ptrtoint ptr %argp.cur to i32
// CHECK-NEXT:  %1 = add i32 %0, 15
// CHECK-NEXT:  %2 = and i32 %1, -16
// CHECK-NEXT:  %argp.cur.aligned = inttoptr i32 %2 to ptr
void testm128(int argCount, ...) {
  __m128 res;
  __builtin_va_list args;
  __builtin_va_start(args, argCount);
  res = __builtin_va_arg(args, __m128);
  __builtin_va_end(args);
}

// CHECK-LABEL: define dso_local void @testm256
// CHECK-LABEL: %argp.cur = load ptr, ptr %args, align 4
// CHECK-NEXT:  %0 = ptrtoint ptr %argp.cur to i32
// CHECK-NEXT:  %1 = add i32 %0, 31
// CHECK-NEXT:  %2 = and i32 %1, -32
// CHECK-NEXT:  %argp.cur.aligned = inttoptr i32 %2 to ptr
void testm256(int argCount, ...) {
  __m256 res;
  __builtin_va_list args;
  __builtin_va_start(args, argCount);
  res = __builtin_va_arg(args, __m256);
  __builtin_va_end(args);
}

// CHECK-LABEL: define dso_local void @testm512
// CHECK-LABEL: %argp.cur = load ptr, ptr %args, align 4
// CHECK-NEXT:  %0 = ptrtoint ptr %argp.cur to i32
// CHECK-NEXT:  %1 = add i32 %0, 63
// CHECK-NEXT:  %2 = and i32 %1, -64
// CHECK-NEXT:  %argp.cur.aligned = inttoptr i32 %2 to ptr
void testm512(int argCount, ...) {
  __m512 res;
  __builtin_va_list args;
  __builtin_va_start(args, argCount);
  res = __builtin_va_arg(args, __m512);
  __builtin_va_end(args);
}

// CHECK-LABEL: define dso_local void @testPastArguments
// CHECK: call void (i32, ...) @testm128(i32 noundef 1, <4 x float> noundef %0)
// CHECK: call void (i32, ...) @testm256(i32 noundef 1, <8 x float> noundef %1)
// CHECK: call void (i32, ...) @testm512(i32 noundef 1, <16 x float> noundef %2)
void testPastArguments(void) {
  __m128 a;
  __m256 b;
  __m512 c;
  testm128(1, a);
  testm256(1, b);
  testm512(1, c);
}
