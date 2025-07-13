// RUN: %clang_cc1 -emit-llvm -triple i386 -target-feature +sse2 %s -o - | FileCheck %s
#include <mmintrin.h>

void shift(__m64 a, __m64 b, int c) {
  // CHECK: <8 x i16> @llvm.x86.sse2.pslli.w(<8 x i16> %{{.*}}, i32 {{.*}})
  _mm_slli_pi16(a, c);
  // CHECK: <4 x i32> @llvm.x86.sse2.pslli.d(<4 x i32> %{{.*}}, i32 {{.*}})
  _mm_slli_pi32(a, c);
  // CHECK: <2 x i64> @llvm.x86.sse2.pslli.q(<2 x i64> %{{.*}}, i32 {{.*}})
  _mm_slli_si64(a, c);

  // CHECK: <8 x i16> @llvm.x86.sse2.psrli.w(<8 x i16> %{{.*}}, i32 {{.*}})
  _mm_srli_pi16(a, c);
  // CHECK: <4 x i32> @llvm.x86.sse2.psrli.d(<4 x i32> %{{.*}}, i32 {{.*}})
  _mm_srli_pi32(a, c);
  // CHECK: <2 x i64> @llvm.x86.sse2.psrli.q(<2 x i64> %{{.*}}, i32 {{.*}})
  _mm_srli_si64(a, c);

  // CHECK: <8 x i16> @llvm.x86.sse2.psrai.w(<8 x i16> %{{.*}}, i32 {{.*}})
  _mm_srai_pi16(a, c);
  // CHECK: <4 x i32> @llvm.x86.sse2.psrai.d(<4 x i32> %{{.*}}, i32 {{.*}})
  _mm_srai_pi32(a, c);
}
