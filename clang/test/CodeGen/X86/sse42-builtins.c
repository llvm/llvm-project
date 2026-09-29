// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.2 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.2 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.2 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.2 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.2 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.2 -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.2 -fno-signed-char -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.2 -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.2 -fno-signed-char -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.2 -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.2 -fno-signed-char -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.2 -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.2 -fno-signed-char -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK


#include <immintrin.h>
#include "builtin_test_helpers.h"

// NOTE: This should match the tests in llvm/test/CodeGen/X86/sse42-intrinsics-fast-isel.ll

int test_mm_cmpestra(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestra
  // CHECK: call {{.*}}i32 @llvm.x86.sse42.pcmpestria128(<16 x i8> %{{.*}}, i32 %{{.*}}, <16 x i8> %{{.*}}, i32 %{{.*}}, i8 7)
  return _mm_cmpestra(A, LA, B, LB, 7);
}

int test_mm_cmpestrc(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestrc
  // CHECK: call {{.*}}i32 @llvm.x86.sse42.pcmpestric128(<16 x i8> %{{.*}}, i32 %{{.*}}, <16 x i8> %{{.*}}, i32 %{{.*}}, i8 7)
  return _mm_cmpestrc(A, LA, B, LB, 7);
}

int test_mm_cmpestri(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestri
  // CHECK: call {{.*}}i32 @llvm.x86.sse42.pcmpestri128(<16 x i8> %{{.*}}, i32 %{{.*}}, <16 x i8> %{{.*}}, i32 %{{.*}}, i8 7)
  return _mm_cmpestri(A, LA, B, LB, 7);
}

__m128i test_mm_cmpestrm(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestrm
  // CHECK: call <16 x i8> @llvm.x86.sse42.pcmpestrm128(<16 x i8> %{{.*}}, i32 %{{.*}}, <16 x i8> %{{.*}}, i32 %{{.*}}, i8 7)
  return _mm_cmpestrm(A, LA, B, LB, 7);
}

int test_mm_cmpestro(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestro
  // CHECK: call {{.*}}i32 @llvm.x86.sse42.pcmpestrio128(<16 x i8> %{{.*}}, i32 %{{.*}}, <16 x i8> %{{.*}}, i32 %{{.*}}, i8 7)
  return _mm_cmpestro(A, LA, B, LB, 7);
}

int test_mm_cmpestrs(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestrs
  // CHECK: call {{.*}}i32 @llvm.x86.sse42.pcmpestris128(<16 x i8> %{{.*}}, i32 %{{.*}}, <16 x i8> %{{.*}}, i32 %{{.*}}, i8 7)
  return _mm_cmpestrs(A, LA, B, LB, 7);
}

int test_mm_cmpestrz(__m128i A, int LA, __m128i B, int LB) {
  // CHECK-LABEL: test_mm_cmpestrz
  // CHECK: call {{.*}}i32 @llvm.x86.sse42.pcmpestriz128(<16 x i8> %{{.*}}, i32 %{{.*}}, <16 x i8> %{{.*}}, i32 %{{.*}}, i8 7)
  return _mm_cmpestrz(A, LA, B, LB, 7);
}

__m128i test_mm_cmpgt_epi64(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpgt_epi64
  // CHECK: icmp sgt <2 x i64>
  return _mm_cmpgt_epi64(A, B);
}
TEST_CONSTEXPR(match_v2di(_mm_cmpgt_epi64((__m128i)(__v2di){+1, -8}, (__m128i)(__v2di){-10, -8}), -1, 0));

int test_mm_cmpistra(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistra
  // CHECK: call {{.*}}i32 @llvm.x86.sse42.pcmpistria128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 7)
  return _mm_cmpistra(A, B, 7);
}

int test_mm_cmpistrc(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistrc
  // CHECK: call {{.*}}i32 @llvm.x86.sse42.pcmpistric128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 7)
  return _mm_cmpistrc(A, B, 7);
}

int test_mm_cmpistri(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistri
  // CHECK: call {{.*}}i32 @llvm.x86.sse42.pcmpistri128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 7)
  return _mm_cmpistri(A, B, 7);
}

__m128i test_mm_cmpistrm(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistrm
  // CHECK: call <16 x i8> @llvm.x86.sse42.pcmpistrm128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 7)
  return _mm_cmpistrm(A, B, 7);
}

int test_mm_cmpistro(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistro
  // CHECK: call {{.*}}i32 @llvm.x86.sse42.pcmpistrio128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 7)
  return _mm_cmpistro(A, B, 7);
}

int test_mm_cmpistrs(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistrs
  // CHECK: call {{.*}}i32 @llvm.x86.sse42.pcmpistris128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 7)
  return _mm_cmpistrs(A, B, 7);
}

int test_mm_cmpistrz(__m128i A, __m128i B) {
  // CHECK-LABEL: test_mm_cmpistrz
  // CHECK: call {{.*}}i32 @llvm.x86.sse42.pcmpistriz128(<16 x i8> %{{.*}}, <16 x i8> %{{.*}}, i8 7)
  return _mm_cmpistrz(A, B, 7);
}

unsigned int test_mm_crc32_u8(unsigned int CRC, unsigned char V) {
  // CHECK-LABEL: test_mm_crc32_u8
  // CHECK: call {{.*}}i32 @llvm.x86.sse42.crc32.32.8(i32 %{{.*}}, i8 %{{.*}})
  return _mm_crc32_u8(CRC, V);
}
TEST_CONSTEXPR(_mm_crc32_u8(0x02a24bf8, 0xba) == 0xa042cf00);
TEST_CONSTEXPR(_mm_crc32_u8(0x0cd5e10f, 0xe7) == 0x69e52534);
TEST_CONSTEXPR(_mm_crc32_u8(0x50e739b8, 0x2e) == 0xb459fcc6);
TEST_CONSTEXPR(_mm_crc32_u8(0x3c2db116, 0x3d) == 0xb9080854);

unsigned int test_mm_crc32_u16(unsigned int CRC, unsigned short V) {
  // CHECK-LABEL: test_mm_crc32_u16
  // CHECK: call {{.*}}i32 @llvm.x86.sse42.crc32.32.16(i32 %{{.*}}, i16 %{{.*}})
  return _mm_crc32_u16(CRC, V);
}
TEST_CONSTEXPR(_mm_crc32_u16(0x02a24bf8, 0xd4ba) == 0x14e9347b);
TEST_CONSTEXPR(_mm_crc32_u16(0x0cd5e10f, 0x42e7) == 0x575056c0);
TEST_CONSTEXPR(_mm_crc32_u16(0x50e739b8, 0x382e) == 0x5fa289ae);
TEST_CONSTEXPR(_mm_crc32_u16(0x3c2db116, 0x7f3d) == 0xb98d2ded);

unsigned int test_mm_crc32_u32(unsigned int CRC, unsigned int V) {
  // CHECK-LABEL: test_mm_crc32_u32
  // CHECK: call {{.*}}i32 @llvm.x86.sse42.crc32.32.32(i32 %{{.*}}, i32 %{{.*}})
  return _mm_crc32_u32(CRC, V);
}
TEST_CONSTEXPR(_mm_crc32_u32(0x02a24bf8, 0xf37dd4ba) == 0x7e8807f4);
TEST_CONSTEXPR(_mm_crc32_u32(0x0cd5e10f, 0x832b42e7) == 0x348e1639);
TEST_CONSTEXPR(_mm_crc32_u32(0x50e739b8, 0xcefb382e) == 0x084be8db);
TEST_CONSTEXPR(_mm_crc32_u32(0x3c2db116, 0xd3947f3d) == 0x6ef9e697);

#ifdef __x86_64__
unsigned long long test_mm_crc32_u64(unsigned long long CRC, unsigned long long V) {
  // X64-LABEL: test_mm_crc32_u64
  // X64: call {{.*}}i64 @llvm.x86.sse42.crc32.64.64(i64 %{{.*}}, i64 %{{.*}})
  return _mm_crc32_u64(CRC, V);
}
TEST_CONSTEXPR(_mm_crc32_u64(0x9f65239602a24bf8, 0x894a58bff37dd4ba) == 0x00000000c093154a);
TEST_CONSTEXPR(_mm_crc32_u64(0x06ef97970cd5e10f, 0x24334e2e832b42e7) == 0x00000000141ddf67);
TEST_CONSTEXPR(_mm_crc32_u64(0x5024a45450e739b8, 0x289ee1b7cefb382e) == 0x000000002a710fcb);
TEST_CONSTEXPR(_mm_crc32_u64(0xcbc89e1c3c2db116, 0xa89143dad3947f3d) == 0x0000000052dd3aeb);
#endif
