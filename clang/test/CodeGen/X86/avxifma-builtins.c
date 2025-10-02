// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avxifma -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=i386-apple-darwin -target-feature +avxifma -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avxifma -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=i386-apple-darwin -target-feature +avxifma -emit-llvm -o - -Wall -Werror | FileCheck %s

// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avxifma -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=i386-apple-darwin -target-feature +avxifma -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avxifma -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -ffreestanding %s -triple=i386-apple-darwin -target-feature +avxifma -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s


#include <immintrin.h>

__m128i test_mm_madd52hi_epu64(__m128i __X, __m128i __Y, __m128i __Z) {
// CHECK-LABEL: test_mm_madd52hi_epu64
// CHECK:    call {{.*}}<2 x i64> @llvm.x86.avx512.vpmadd52h.uq.128(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_madd52hi_epu64(__X, __Y, __Z);
}

__m256i test_mm256_madd52hi_epu64(__m256i __X, __m256i __Y, __m256i __Z) {
// CHECK-LABEL: test_mm256_madd52hi_epu64
// CHECK:    call {{.*}}<4 x i64> @llvm.x86.avx512.vpmadd52h.uq.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_madd52hi_epu64(__X, __Y, __Z);
}

__m128i test_mm_madd52lo_epu64(__m128i __X, __m128i __Y, __m128i __Z) {
// CHECK-LABEL: test_mm_madd52lo_epu64
// CHECK:    call {{.*}}<2 x i64> @llvm.x86.avx512.vpmadd52l.uq.128(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_madd52lo_epu64(__X, __Y, __Z);
}

__m256i test_mm256_madd52lo_epu64(__m256i __X, __m256i __Y, __m256i __Z) {
// CHECK-LABEL: test_mm256_madd52lo_epu64
// CHECK:    call {{.*}}<4 x i64> @llvm.x86.avx512.vpmadd52l.uq.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_madd52lo_epu64(__X, __Y, __Z);
}

__m128i test_mm_madd52hi_avx_epu64(__m128i __X, __m128i __Y, __m128i __Z) {
// CHECK-LABEL: test_mm_madd52hi_avx_epu64
// CHECK:    call {{.*}}<2 x i64> @llvm.x86.avx512.vpmadd52h.uq.128(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_madd52hi_avx_epu64(__X, __Y, __Z);
}

__m256i test_mm256_madd52hi_avx_epu64(__m256i __X, __m256i __Y, __m256i __Z) {
// CHECK-LABEL: test_mm256_madd52hi_avx_epu64
// CHECK:    call {{.*}}<4 x i64> @llvm.x86.avx512.vpmadd52h.uq.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_madd52hi_avx_epu64(__X, __Y, __Z);
}

__m128i test_mm_madd52lo_avx_epu64(__m128i __X, __m128i __Y, __m128i __Z) {
// CHECK-LABEL: test_mm_madd52lo_avx_epu64
// CHECK:    call {{.*}}<2 x i64> @llvm.x86.avx512.vpmadd52l.uq.128(<2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}})
  return _mm_madd52lo_avx_epu64(__X, __Y, __Z);
}

__m256i test_mm256_madd52lo_avx_epu64(__m256i __X, __m256i __Y, __m256i __Z) {
// CHECK-LABEL: test_mm256_madd52lo_avx_epu64
// CHECK:    call {{.*}}<4 x i64> @llvm.x86.avx512.vpmadd52l.uq.256(<4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}})
  return _mm256_madd52lo_avx_epu64(__X, __Y, __Z);
}
