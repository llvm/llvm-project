// Test crc32 target attribute on x86

// RUN: %clang_cc1 -triple i386-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s
// RUN: %clang_cc1 -triple x86_64-unknown-linux-gnu -emit-llvm %s -o - | FileCheck %s

// CHECK: define{{.*}} i32 @test1({{.*}}) [[TEST1_ATTRS:#[0-9]+]]
// CHECK: call i32 @llvm.x86.sse42.crc32.32.8(i32 %{{.*}}, i8 %{{.*}})

#define __MM_MALLOC_H

#include <x86intrin.h>

unsigned int __attribute__((target("crc32"))) test1(unsigned int CRC, unsigned char V) {
  return __builtin_ia32_crc32qi(CRC, V);
}

// CHECK: define{{.*}} i32 @test2({{.*}}) [[GPR_ONLY_ATTRS:#[0-9]+]]
// CHECK: call i32 @llvm.x86.sse42.crc32.32.8(i32 %{{.*}}, i8 %{{.*}})
unsigned int __attribute__((target("general-regs-only,crc32"))) test2(unsigned int CRC, unsigned char V) {
  return __builtin_ia32_crc32qi(CRC, V);
}

// CHECK: define{{.*}} i32 @test3({{.*}}) [[GPR_ONLY_ATTRS:#[0-9]+]]
// CHECK: call i32 @llvm.x86.sse42.crc32.32.8(i32 %{{.*}}, i8 %{{.*}})
unsigned int __attribute__((target("crc32,general-regs-only"))) test3(unsigned int CRC, unsigned char V) {
  return __builtin_ia32_crc32qi(CRC, V);
}

// CHECK: define{{.*}} i32 @test4({{.*}}) [[TEST4_ATTRS:#[0-9]+]]
// CHECK: call i32 @llvm.x86.sse42.crc32.32.8(i32 %{{.*}}, i8 %{{.*}})
unsigned int __attribute__((target("sse4.2"))) test4(unsigned int CRC, unsigned char V) {
  return __builtin_ia32_crc32qi(CRC, V);
}

// CHECK: define{{.*}} i32 @test5({{.*}}) [[GPR_ONLY_ATTRS:#[0-9]+]]
// CHECK: call i32 @llvm.x86.sse42.crc32.32.8(i32 %{{.*}}, i8 %{{.*}})
unsigned int __attribute__((target("sse4.2,general-regs-only,crc32"))) test5(unsigned int CRC, unsigned char V) {
  return __builtin_ia32_crc32qi(CRC, V);
}

// CHECK: define{{.*}} i32 @test6({{.*}}) [[TEST4_ATTRS:#[0-9]+]]
// CHECK: call i32 @llvm.x86.sse42.pcmpestria128(<16 x i8> %{{.*}}, i32 %{{.*}}, <16 x i8> %{{.*}}, i32 %{{.*}}, i8 7)
int __attribute__((target("sse4.2,no-crc32,crc32"))) test6(__m128i A, int LA, __m128i B, int LB) {
  return _mm_cmpestra(A, LA, B, LB, 7);
}

// CHECK: define{{.*}} i32 @test7({{.*}}) [[TEST4_ATTRS:#[0-9]+]]
// CHECK: call i32 @llvm.x86.sse42.pcmpestria128(<16 x i8> %{{.*}}, i32 %{{.*}}, <16 x i8> %{{.*}}, i32 %{{.*}}, i8 7)
int __attribute__((target("no-crc32,crc32,sse4.2"))) test7(__m128i A, int LA, __m128i B, int LB) {
  return _mm_cmpestra(A, LA, B, LB, 7);
}

// CHECK: attributes [[TEST1_ATTRS]] = { {{.*}} "target-features"="{{.*}}+crc32{{.*}}"
// CHECK: attributes [[GPR_ONLY_ATTRS]] = { {{.*}} "target-features"="{{.*}}+crc32{{.*}}-avx{{.*}}-avx2{{.*}}-avx512f{{.*}}-sse{{.*}}-sse2{{.*}}-ssse3{{.*}}-x87{{.*}}"
// CHECK: attributes [[TEST4_ATTRS]] = { {{.*}} "target-features"="{{.*}}+crc32{{.*}}+sse4.2{{.*}}"
