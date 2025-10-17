// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vl -target-feature +avx512cd -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vl -target-feature +avx512cd -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vl -target-feature +avx512cd -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vl -target-feature +avx512cd -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vl -target-feature +avx512cd -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vl -target-feature +avx512cd -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512vl -target-feature +avx512cd -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512vl -target-feature +avx512cd -emit-llvm -o - -Wall -Werror -fexperimental-new-constant-interpreter | FileCheck %s


#include <immintrin.h>
#include "builtin_test_helpers.h"

__m128i test_mm_broadcastmb_epi64(__m128i a,__m128i b) {
  // CHECK-LABEL: test_mm_broadcastmb_epi64
  // CHECK: icmp eq <4 x i32> %{{.*}}, %{{.*}}
  // CHECK: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: zext i8 %{{.*}} to i64
  // CHECK: insertelement <2 x i64> poison, i64 %{{.*}}, i32 0
  // CHECK: insertelement <2 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  return _mm_broadcastmb_epi64(_mm_cmpeq_epi32_mask (a, b)); 
}
TEST_CONSTEXPR(match_v2du(_mm_broadcastmb_epi64((__mmask8)(76)), 76, 76));

__m256i test_mm256_broadcastmb_epi64(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_broadcastmb_epi64
  // CHECK: icmp eq <4 x i64> %{{.*}}, %{{.*}}
  // CHECK: shufflevector <4 x i1> %{{.*}}, <4 x i1> zeroinitializer, <8 x i32> <i32 0, i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7>
  // CHECK: zext i8 %{{.*}} to i64
  // CHECK: insertelement <4 x i64> poison, i64 %{{.*}}, i32 0
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 2
  // CHECK: insertelement <4 x i64> %{{.*}}, i64 %{{.*}}, i32 3
  return _mm256_broadcastmb_epi64(_mm256_cmpeq_epi64_mask ( a, b)); 
}
TEST_CONSTEXPR(match_v4di(_mm256_broadcastmb_epi64((__mmask8)(67)), 67, 67, 67, 67));

__m128i test_mm_broadcastmw_epi32(__m512i a, __m512i b) {
  // CHECK-LABEL: test_mm_broadcastmw_epi32
  // CHECK: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: zext i16 %{{.*}} to i32
  // CHECK: insertelement <4 x i32> poison, i32 %{{.*}}, i32 0
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 1
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 2
  // CHECK: insertelement <4 x i32> %{{.*}}, i32 %{{.*}}, i32 3
  return _mm_broadcastmw_epi32(_mm512_cmpeq_epi32_mask ( a, b));
}
TEST_CONSTEXPR(match_v4su(_mm_broadcastmw_epi32((__mmask16)(0xbabe)), 0xbabe, 0xbabe, 0xbabe, 0xbabe));

__m256i test_mm256_broadcastmw_epi32(__m512i a, __m512i b) {
  // CHECK-LABEL: test_mm256_broadcastmw_epi32
  // CHECK: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: zext i16 %{{.*}} to i32
  // CHECK: insertelement <8 x i32> poison, i32 %{{.*}}, i32 0
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 1
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 2
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 3
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 4
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 5
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 6
  // CHECK: insertelement <8 x i32> %{{.*}}, i32 %{{.*}}, i32 7
  return _mm256_broadcastmw_epi32(_mm512_cmpeq_epi32_mask ( a, b)); 
}
TEST_CONSTEXPR(match_v8si(_mm256_broadcastmw_epi32((__mmask16)(0xcafe)), 0xcafe,0xcafe,0xcafe,0xcafe, 0xcafe,0xcafe,0xcafe,0xcafe));

__m128i test_mm_conflict_epi64(__m128i __A) {
  // CHECK-LABEL: test_mm_conflict_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.avx512.conflict.q.128(<2 x i64> %{{.*}})
  return _mm_conflict_epi64(__A);
}

TEST_CONSTEXPR(match_v2di(_mm_conflict_epi64((__m128i)(__v2di){1, 2}), 0, 0));
TEST_CONSTEXPR(match_v2di(_mm_conflict_epi64((__m128i)(__v2di){5, 5}), 0, 1));

__m128i test_mm_mask_conflict_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_conflict_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.avx512.conflict.q.128(<2 x i64> %{{.*}})
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_conflict_epi64(__W, __U, __A);
}

TEST_CONSTEXPR(match_v2di(_mm_mask_conflict_epi64((__m128i)(__v2di){0xFF, 0xFF}, 0x2, (__m128i)(__v2di){5, 5}), 0xFF, 1));

__m128i test_mm_maskz_conflict_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_conflict_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.x86.avx512.conflict.q.128(<2 x i64> %{{.*}})
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_conflict_epi64(__U, __A);
}

TEST_CONSTEXPR(match_v2di(_mm_maskz_conflict_epi64(0x2, (__m128i)(__v2di){5, 5}), 0, 1));

__m256i test_mm256_conflict_epi64(__m256i __A) {
  // CHECK-LABEL: test_mm256_conflict_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx512.conflict.q.256(<4 x i64> %{{.*}})
  return _mm256_conflict_epi64(__A);
}

TEST_CONSTEXPR(match_v4di(_mm256_conflict_epi64((__m256i)(__v4di){1, 2, 1, 3}), 0, 0, 1, 0));
TEST_CONSTEXPR(match_v4di(_mm256_conflict_epi64((__m256i)(__v4di){7, 7, 7, 7}), 0, 1, 3, 7));
TEST_CONSTEXPR(match_v4di(_mm256_conflict_epi64((__m256i)(__v4di){1, 2, 3, 4}), 0, 0, 0, 0));

__m256i test_mm256_mask_conflict_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_conflict_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx512.conflict.q.256(<4 x i64> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_conflict_epi64(__W, __U, __A);
}

TEST_CONSTEXPR(match_v4di(_mm256_mask_conflict_epi64((__m256i)(__v4di){0xFF, 0xFF, 0xFF, 0xFF}, 0x5, (__m256i)(__v4di){1, 2, 1, 3}), 0, 0xFF, 1, 0xFF));

__m256i test_mm256_maskz_conflict_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_maskz_conflict_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.x86.avx512.conflict.q.256(<4 x i64> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_conflict_epi64(__U, __A);
}

TEST_CONSTEXPR(match_v4di(_mm256_maskz_conflict_epi64(0x5, (__m256i)(__v4di){1, 2, 1, 3}), 0, 0, 1, 0));

__m128i test_mm_conflict_epi32(__m128i __A) {
  // CHECK-LABEL: test_mm_conflict_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.conflict.d.128(<4 x i32> %{{.*}})
  return _mm_conflict_epi32(__A);
}

TEST_CONSTEXPR(match_v4si(_mm_conflict_epi32((__m128i)(__v4si){1, 2, 1, 3}), 0, 0, 1, 0));
TEST_CONSTEXPR(match_v4si(_mm_conflict_epi32((__m128i)(__v4si){3, 3, 3, 3}), 0, 1, 3, 7));
TEST_CONSTEXPR(match_v4si(_mm_conflict_epi32((__m128i)(__v4si){1, 2, 3, 4}), 0, 0, 0, 0));

__m128i test_mm_mask_conflict_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_conflict_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.conflict.d.128(<4 x i32> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_conflict_epi32(__W, __U, __A);
}

TEST_CONSTEXPR(match_v4si(_mm_mask_conflict_epi32((__m128i)(__v4si){0xFF, 0xFF, 0xFF, 0xFF}, 0x5, (__m128i)(__v4si){1, 2, 1, 3}), 0, 0xFF, 1, 0xFF));

__m128i test_mm_maskz_conflict_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_conflict_epi32
  // CHECK: call <4 x i32> @llvm.x86.avx512.conflict.d.128(<4 x i32> %{{.*}})
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_conflict_epi32(__U, __A);
}

TEST_CONSTEXPR(match_v4si(_mm_maskz_conflict_epi32(0x5, (__m128i)(__v4si){1, 2, 1, 3}), 0, 0, 1, 0));

__m256i test_mm256_conflict_epi32(__m256i __A) {
  // CHECK-LABEL: test_mm256_conflict_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.conflict.d.256(<8 x i32> %{{.*}})
  return _mm256_conflict_epi32(__A);
}

TEST_CONSTEXPR(match_v8si(_mm256_conflict_epi32((__m256i)(__v8si){1, 2, 1, 3, 2, 4, 1, 5}), 0, 0, 1, 0, 2, 0, 5, 0));
TEST_CONSTEXPR(match_v8si(_mm256_conflict_epi32((__m256i)(__v8si){4, 4, 4, 4, 4, 4, 4, 4}), 0, 1, 3, 7, 15, 31, 63, 127));
TEST_CONSTEXPR(match_v8si(_mm256_conflict_epi32((__m256i)(__v8si){1, 2, 3, 4, 5, 6, 7, 8}), 0, 0, 0, 0, 0, 0, 0, 0));

__m256i test_mm256_mask_conflict_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_conflict_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.conflict.d.256(<8 x i32> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_conflict_epi32(__W, __U, __A);
}

TEST_CONSTEXPR(match_v8si(_mm256_mask_conflict_epi32((__m256i)(__v8si){0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF}, /*0101 0101=*/0x55, (__m256i)(__v8si){1, 2, 1, 3, 2, 4, 1, 5}), 0, 0xFF, 1, 0xFF, 2, 0xFF, 5, 0xFF));

__m256i test_mm256_maskz_conflict_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_maskz_conflict_epi32
  // CHECK: call <8 x i32> @llvm.x86.avx512.conflict.d.256(<8 x i32> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_conflict_epi32(__U, __A);
}

TEST_CONSTEXPR(match_v8si(_mm256_maskz_conflict_epi32(0x55, (__m256i)(__v8si){1, 2, 1, 3, 2, 4, 1, 5}), 0, 0, 1, 0, 2, 0, 5, 0));

__m128i test_mm_lzcnt_epi32(__m128i __A) {
  // CHECK-LABEL: test_mm_lzcnt_epi32
  // CHECK: call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %{{.*}}, i1 true)
  // CHECK: [[ISZERO:%.+]] = icmp eq <4 x i32> %{{.*}}, zeroinitializer
  // CHECK: select <4 x i1> [[ISZERO]], <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_lzcnt_epi32(__A); 
}

TEST_CONSTEXPR(match_v4si(_mm_lzcnt_epi32((__m128i)(__v4si){8, 16, 32, 64}), 28, 27, 26, 25));
TEST_CONSTEXPR(match_v4si(_mm_lzcnt_epi32((__m128i)(__v4si){0, 0, 0, 0}), 32, 32, 32, 32));

__m128i test_mm_mask_lzcnt_epi32(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_lzcnt_epi32
  // CHECK: call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %{{.*}}, i1 true)
  // CHECK: [[ISZERO:%.+]] = icmp eq <4 x i32> %{{.*}}, zeroinitializer
  // CHECK: select <4 x i1> [[ISZERO]], <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_mask_lzcnt_epi32(__W, __U, __A); 
}

TEST_CONSTEXPR(match_v4si(_mm_mask_lzcnt_epi32(_mm_set1_epi32(32), /*0000 0101=*/0x5, (__m128i)(__v4si){8, 16, 32, 64}), 28, 32, 26, 32));

__m128i test_mm_maskz_lzcnt_epi32(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_lzcnt_epi32
  // CHECK: call <4 x i32> @llvm.ctlz.v4i32(<4 x i32> %{{.*}}, i1 true)
  // CHECK: [[ISZERO:%.+]] = icmp eq <4 x i32> %{{.*}}, zeroinitializer
  // CHECK: select <4 x i1> [[ISZERO]], <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> %{{.*}}
  return _mm_maskz_lzcnt_epi32(__U, __A); 
}

TEST_CONSTEXPR(match_v4si(_mm_maskz_lzcnt_epi32(/*0000 0101=*/0x5, (__m128i)(__v4si){8, 16, 32, 64}), 28, 0, 26, 0));

__m256i test_mm256_lzcnt_epi32(__m256i __A) {
  // CHECK-LABEL: test_mm256_lzcnt_epi32
  // CHECK: call <8 x i32> @llvm.ctlz.v8i32(<8 x i32> %{{.*}}, i1 true)
  // CHECK: [[ISZERO:%.+]] = icmp eq <8 x i32> %{{.*}}, zeroinitializer
  // CHECK: select <8 x i1> [[ISZERO]], <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_lzcnt_epi32(__A); 
}

TEST_CONSTEXPR(match_v8si(_mm256_lzcnt_epi32((__m256i)(__v8si){1, 2, 4, 8, 16, 32, 64, 128}), 31, 30, 29, 28, 27, 26, 25, 24));
TEST_CONSTEXPR(match_v8si(_mm256_lzcnt_epi32((__m256i)(__v8si){0, 0, 0, 0, 0, 0, 0, 0}), 32, 32, 32, 32, 32, 32, 32, 32));

__m256i test_mm256_mask_lzcnt_epi32(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_lzcnt_epi32
  // CHECK: call <8 x i32> @llvm.ctlz.v8i32(<8 x i32> %{{.*}}, i1 true)
  // CHECK: [[ISZERO:%.+]] = icmp eq <8 x i32> %{{.*}}, zeroinitializer
  // CHECK: select <8 x i1> [[ISZERO]], <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_mask_lzcnt_epi32(__W, __U, __A); 
}

TEST_CONSTEXPR(match_v8si(_mm256_mask_lzcnt_epi32(_mm256_set1_epi32(32), /*0101 0101=*/0x55, (__m256i)(__v8si){1, 2, 4, 8, 16, 32, 64, 128}), 31, 32, 29, 32, 27, 32, 25, 32));

__m256i test_mm256_maskz_lzcnt_epi32(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_maskz_lzcnt_epi32
  // CHECK: call <8 x i32> @llvm.ctlz.v8i32(<8 x i32> %{{.*}}, i1 true)
  // CHECK: [[ISZERO:%.+]] = icmp eq <8 x i32> %{{.*}}, zeroinitializer
  // CHECK: select <8 x i1> [[ISZERO]], <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> %{{.*}}
  return _mm256_maskz_lzcnt_epi32(__U, __A); 
}

TEST_CONSTEXPR(match_v8si(_mm256_maskz_lzcnt_epi32(/*0101 0101=*/0x55, (__m256i)(__v8si){1, 2, 4, 8, 16, 32, 64, 128}), 31, 0, 29, 0, 27, 0, 25, 0));

__m128i test_mm_lzcnt_epi64(__m128i __A) {
  // CHECK-LABEL: test_mm_lzcnt_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.ctlz.v2i64(<2 x i64> %{{.*}}, i1 true)
  // CHECK: [[ISZERO:%.+]] = icmp eq <2 x i64> %{{.*}}, zeroinitializer
  // CHECK: select <2 x i1> [[ISZERO]], <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_lzcnt_epi64(__A); 
}

TEST_CONSTEXPR(match_v2di(_mm_lzcnt_epi64((__m128i)(__v2di){1, 2}), 63, 62));
TEST_CONSTEXPR(match_v2di(_mm_lzcnt_epi64((__m128i)(__v2di){0, 0}), 64, 64));

__m128i test_mm_mask_lzcnt_epi64(__m128i __W, __mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_mask_lzcnt_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.ctlz.v2i64(<2 x i64> %{{.*}}, i1 true)
  // CHECK: [[ISZERO:%.+]] = icmp eq <2 x i64> %{{.*}}, zeroinitializer
  // CHECK: select <2 x i1> [[ISZERO]], <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_mask_lzcnt_epi64(__W, __U, __A); 
}

TEST_CONSTEXPR(match_v2di(_mm_mask_lzcnt_epi64(_mm_set1_epi64x((long long)64), /*0000 0010=*/0x2, (__m128i)(__v2di){1, 2}), 64, 62));

__m128i test_mm_maskz_lzcnt_epi64(__mmask8 __U, __m128i __A) {
  // CHECK-LABEL: test_mm_maskz_lzcnt_epi64
  // CHECK: call {{.*}}<2 x i64> @llvm.ctlz.v2i64(<2 x i64> %{{.*}}, i1 true)
  // CHECK: [[ISZERO:%.+]] = icmp eq <2 x i64> %{{.*}}, zeroinitializer
  // CHECK: select <2 x i1> [[ISZERO]], <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  // CHECK: select <2 x i1> %{{.*}}, <2 x i64> %{{.*}}, <2 x i64> %{{.*}}
  return _mm_maskz_lzcnt_epi64(__U, __A); 
}

TEST_CONSTEXPR(match_v2di(_mm_maskz_lzcnt_epi64(/*0000 0010=*/0x2, (__m128i)(__v2di){1, 2}), 0, 62));

__m256i test_mm256_lzcnt_epi64(__m256i __A) {
  // CHECK-LABEL: test_mm256_lzcnt_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.ctlz.v4i64(<4 x i64> %{{.*}}, i1 true)
  // CHECK: [[ISZERO:%.+]] = icmp eq <4 x i64> %{{.*}}, zeroinitializer
  // CHECK: select <4 x i1> [[ISZERO]], <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_lzcnt_epi64(__A); 
}

TEST_CONSTEXPR(match_v4di(_mm256_lzcnt_epi64((__m256i)(__v4di){1, 2, 4, 8}), 63, 62, 61, 60));
TEST_CONSTEXPR(match_v4di(_mm256_lzcnt_epi64((__m256i)(__v4di){0, 0, 0, 0}), 64, 64, 64, 64));

__m256i test_mm256_mask_lzcnt_epi64(__m256i __W, __mmask8 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_mask_lzcnt_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.ctlz.v4i64(<4 x i64> %{{.*}}, i1 true)
  // CHECK: [[ISZERO:%.+]] = icmp eq <4 x i64> %{{.*}}, zeroinitializer
  // CHECK: select <4 x i1> [[ISZERO]], <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_mask_lzcnt_epi64(__W, __U, __A); 
}

TEST_CONSTEXPR(match_v4di(_mm256_mask_lzcnt_epi64(_mm256_set1_epi64x((long long) 64), /*0000 0110=*/0x6, (__m256i)(__v4di){1, 2, 4, 8}), 64, 62, 61, 64));

__m256i test_mm256_maskz_lzcnt_epi64(__mmask8 __U, __m256i __A) {
  // CHECK-LABEL: test_mm256_maskz_lzcnt_epi64
  // CHECK: call {{.*}}<4 x i64> @llvm.ctlz.v4i64(<4 x i64> %{{.*}}, i1 true)
  // CHECK: [[ISZERO:%.+]] = icmp eq <4 x i64> %{{.*}}, zeroinitializer
  // CHECK: select <4 x i1> [[ISZERO]], <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  // CHECK: select <4 x i1> %{{.*}}, <4 x i64> %{{.*}}, <4 x i64> %{{.*}}
  return _mm256_maskz_lzcnt_epi64(__U, __A); 
}

TEST_CONSTEXPR(match_v4di(_mm256_maskz_lzcnt_epi64(/*0000 0011*/0x3, (__m256i)(__v4di){1, 2, 4, 8}), 63, 62, 0, 0));
