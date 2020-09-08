// RUN: %clang_cc1 %s -triple x86_64-unknown-linux-gnu -emit-llvm -target-feature +avx512f -o - | FileCheck %s

typedef long long __m128i __attribute__ ((vector_size (16)));
typedef long long __m256i __attribute__ ((vector_size (32)));
typedef long long __m512i __attribute__ ((vector_size (64)));

// CHECK: define <2 x i64> @testXMMout(<2 x i64>* %p) #0
__m128i testXMMout(__m128i *p) {
  __m128i xmm0;
  __asm__("vmovdqu %1, %0" :"=v"(xmm0) : "m"(*(__m128i*)p));
  return xmm0;
}

// CHECK: define <4 x i64> @testYMMout(<4 x i64>* %p) #1
__m256i testYMMout(__m256i *p) {
  __m256i ymm0;
  __asm__("vmovdqu %1, %0" :"=v"(ymm0) : "m"(*(__m256i*)p));
  return ymm0;
}

// CHECK: define <8 x i64> @testZMMout(<8 x i64>* %p) #2
__m512i testZMMout(__m512i *p) {
  __m512i zmm0;
  __asm__("vmovdqu64 %1, %0" :"=v"(zmm0) : "m"(*(__m512i*)p));
  return zmm0;
}

// CHECK: define void @testXMMin(<2 x i64> %xmm0, <2 x i64>* %p) #0
void testXMMin(__m128i xmm0, __m128i *p) {
  __asm__("vmovdqu %0, %1" : : "v"(xmm0), "m"(*(__m128i*)p));
}

// CHECK: define void @testYMMin(<4 x i64> %ymm0, <4 x i64>* %p) #1
void testYMMin(__m256i ymm0, __m256i *p) {
  __asm__("vmovdqu %0, %1" : : "v"(ymm0), "m"(*(__m256i*)p));
}

// CHECK: define void @testZMMin(<8 x i64> %zmm0, <8 x i64>* %p) #2
void testZMMin(__m512i zmm0, __m512i *p) {
  __asm__("vmovdqu64 %0, %1" : : "v"(zmm0), "m"(*(__m512i*)p));
}

// CHECK: attributes #0 = {{.*}}"min-legal-vector-width"="128"
// CHECK: attributes #1 = {{.*}}"min-legal-vector-width"="256"
// CHECK: attributes #2 = {{.*}}"min-legal-vector-width"="512"
