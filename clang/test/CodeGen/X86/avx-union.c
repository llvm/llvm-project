// RUN: %clang_cc1 -w -ffreestanding -triple x86_64-linux-gnu -target-feature +avx -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AVX
// RUN: %clang_cc1 -w -ffreestanding -triple x86_64-linux-gnu -target-feature +avx512f -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK,AVX512

// Test Clang 11 and earlier behavior
// RUN: %clang_cc1 -w -ffreestanding -triple x86_64-linux-gnu -target-feature +avx -fclang-abi-compat=10.0 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK-LEGACY,AVX
// RUN: %clang_cc1 -w -ffreestanding -triple x86_64-linux-gnu -target-feature +avx512f -fclang-abi-compat=11.0 -emit-llvm -o - %s | FileCheck %s --check-prefixes=CHECK-LEGACY,AVX512-LEGACY
// RUN: %clang_cc1 -w -ffreestanding -triple x86_64-scei-ps4 -target-feature +avx -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-LEGACY
// RUN: %clang_cc1 -w -ffreestanding -triple x86_64-sie-ps5  -target-feature +avx -emit-llvm -o - %s | FileCheck %s --check-prefix=CHECK-LEGACY

// This tests verifies that a union parameter should pass by a vector regitster whose first eightbyte is SSE and the other eightbytes are SSEUP.

typedef int __m256 __attribute__ ((__vector_size__ (32)));
typedef int __m512 __attribute__ ((__vector_size__ (64)));

union M256 {
  double d;
  __m256 m;
};

union M512 {
  double d;
  __m512 m;
};

extern void foo1(union M256 A);
extern void foo2(union M512 A);
union M256 m1;
union M512 m2;
// CHECK-LABEL:   define{{.*}} void @test()
// CHECK:         call void @foo1(<4 x double>
// CHECK-LEGACY:  call void @foo1(ptr noundef byval(%union.M256) align 32
// AVX:           call void @foo2(ptr noundef byval(%union.M512) align 64
// AVX512:        call void @foo2(<8 x double>
// AVX512-LEGACY: call void @foo2(ptr noundef byval(%union.M512) align 64
void test(void) {
  foo1(m1);
  foo2(m2);
}
