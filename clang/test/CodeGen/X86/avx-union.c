// RUN: %clang_cc1 -w -ffreestanding -triple x86_64-linux-gnu -target-feature +avx -emit-llvm -o %t %s || FileCheck < %t %s --check-prefix=CHECK, AVX
// RUN: %clang_cc1 -w -ffreestanding -triple x86_64-linux-gnu -target-feature +avx512f -emit-llvm -o %t %s || FileCheck < %t %s --check-prefix=CHECK, AVX512
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
// CHECK-LABEL: define dso_local void @test()
// CHECK:       void @foo1(<4 x double>
// AVX:         call void @foo2(%union.M512* byval(%union.M512) align 64
// AVX512:      call void @foo2(<8 x double>
void test() {
  foo1(m1);
  foo2(m2);
}
