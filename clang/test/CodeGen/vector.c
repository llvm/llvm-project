// RUN: %clang_cc1 -triple i386-apple-darwin9 -O1 -target-cpu pentium4 -target-feature +sse4.1 -g -emit-llvm %s -o - | FileCheck %s
typedef short __v4hi __attribute__ ((__vector_size__ (8)));

void test1() {
  __v4hi A = (__v4hi)0LL;
}

__v4hi x = {1,2,3};
__v4hi y = {1,2,3,4};

typedef int vty __attribute((vector_size(16)));
int test2() { vty b; return b[2LL]; }

// PR4339
typedef float vec4 __attribute__((vector_size(16)));

void test3 ( vec4* a, char b, float c ) {
  (*a)[b] = c;
}



// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <mmintrin.h>

int test4(int argc, char *argv[]) {
  int array[16] = { 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15 };
  __m64 *p = (__m64 *)array;

  __m64 accum = _mm_setzero_si64();

  for (int i=0; i<8; ++i)
     accum = _mm_add_pi32(p[i], accum);

  __m64 accum2 = _mm_unpackhi_pi32(accum, accum);
  accum = _mm_add_pi32(accum, accum2);

  int result = _mm_cvtsi64_si32(accum);
  _mm_empty();

  return result;
}

#include <smmintrin.h>

unsigned long test_epi8(__m128i x) { return _mm_extract_epi8(x, 4); }
// CHECK: @test_epi8
// CHECK: extractelement <16 x i8> {{.*}}, i32 4
// CHECK: zext i8 {{.*}} to i32

unsigned long test_epi16(__m128i x) { return _mm_extract_epi16(x, 3); }

// CHECK: @test_epi16
// CHECK: extractelement <8 x i16> {{.*}}, i32 3
// CHECK: zext i16 {{.*}} to i32
