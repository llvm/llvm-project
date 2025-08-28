// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512cd -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512cd -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx512cd -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +avx512cd -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <immintrin.h>
#include "builtin_test_helpers.h"

__m512i test_mm512_conflict_epi64(__m512i __A) {
  // CHECK-LABEL: test_mm512_conflict_epi64
  // CHECK: call {{.*}}<8 x i64> @llvm.x86.avx512.conflict.q.512(<8 x i64> %{{.*}})
  return _mm512_conflict_epi64(__A); 
}
__m512i test_mm512_mask_conflict_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_conflict_epi64
  // CHECK: call {{.*}}<8 x i64> @llvm.x86.avx512.conflict.q.512(<8 x i64> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_conflict_epi64(__W,__U,__A); 
}
__m512i test_mm512_maskz_conflict_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_conflict_epi64
  // CHECK: call {{.*}}<8 x i64> @llvm.x86.avx512.conflict.q.512(<8 x i64> %{{.*}})
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_conflict_epi64(__U,__A); 
}
__m512i test_mm512_conflict_epi32(__m512i __A) {
  // CHECK-LABEL: test_mm512_conflict_epi32
  // CHECK: call <16 x i32> @llvm.x86.avx512.conflict.d.512(<16 x i32> %{{.*}})
  return _mm512_conflict_epi32(__A); 
}
__m512i test_mm512_mask_conflict_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_conflict_epi32
  // CHECK: call <16 x i32> @llvm.x86.avx512.conflict.d.512(<16 x i32> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_conflict_epi32(__W,__U,__A); 
}
__m512i test_mm512_maskz_conflict_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_conflict_epi32
  // CHECK: call <16 x i32> @llvm.x86.avx512.conflict.d.512(<16 x i32> %{{.*}})
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_conflict_epi32(__U,__A); 
}
__m512i test_mm512_lzcnt_epi32(__m512i __A) {
  // CHECK-LABEL: test_mm512_lzcnt_epi32
  // CHECK: call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %{{.*}}, i1 true)
  // CHECK: [[ISZERO:%.+]] = icmp eq <16 x i32> %{{.*}}, zeroinitializer
  // CHECK: select <16 x i1> [[ISZERO]], <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_lzcnt_epi32(__A); 
}

TEST_CONSTEXPR(match_v16si(_mm512_lzcnt_epi32((__m512i)(__v16si){1, 2, 4, 8, 16, 32, 64, 128, 3, 5, 6, 7, 9, 10, 11, 12}), 31, 30, 29, 28, 27, 26, 25, 24, 30, 29, 29, 29, 28, 28, 28, 28));
TEST_CONSTEXPR(match_v16si(_mm512_lzcnt_epi32((__m512i)(__v16si){0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}), 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32));

__m512i test_mm512_mask_lzcnt_epi32(__m512i __W, __mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_lzcnt_epi32
  // CHECK: call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %{{.*}}, i1 true)
  // CHECK: [[ISZERO:%.+]] = icmp eq <16 x i32> %{{.*}}, zeroinitializer
  // CHECK: select <16 x i1> [[ISZERO]], <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_mask_lzcnt_epi32(__W,__U,__A); 
}

TEST_CONSTEXPR(match_v16si(_mm512_mask_lzcnt_epi32(_mm512_set1_epi32(32), /*1010 1100 1010 1101=*/0xacad, (__m512i)(__v16si){1, 2, 4, 8, 16, 32, 64, 128, 3, 5, 6, 7, 9, 10, 11, 12}), 31, 32, 29, 28, 32, 26, 32, 24, 32, 32, 29, 29, 32, 28, 32, 28));

__m512i test_mm512_maskz_lzcnt_epi32(__mmask16 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_lzcnt_epi32
  // CHECK: call <16 x i32> @llvm.ctlz.v16i32(<16 x i32> %{{.*}}, i1 true)
  // CHECK: [[ISZERO:%.+]] = icmp eq <16 x i32> %{{.*}}, zeroinitializer
  // CHECK: select <16 x i1> [[ISZERO]], <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  // CHECK: select <16 x i1> %{{.*}}, <16 x i32> %{{.*}}, <16 x i32> %{{.*}}
  return _mm512_maskz_lzcnt_epi32(__U,__A); 
}

TEST_CONSTEXPR(match_v16si(_mm512_maskz_lzcnt_epi32(/*1010 1100 1010 1101=*/0xacad, (__m512i)(__v16si){1, 2, 4, 8, 16, 32, 64, 128, 3, 5, 6, 7, 9, 10, 11, 12}), 31, 0, 29, 28, 0, 26, 0, 24, 0, 0, 29, 29, 0, 28, 0, 28));

__m512i test_mm512_lzcnt_epi64(__m512i __A) {
  // CHECK-LABEL: test_mm512_lzcnt_epi64
  // CHECK: call {{.*}}<8 x i64> @llvm.ctlz.v8i64(<8 x i64> %{{.*}}, i1 true)
  // CHECK: [[ISZERO:%.+]] = icmp eq <8 x i64> %{{.*}}, zeroinitializer
  // CHECK: select <8 x i1> [[ISZERO]], <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_lzcnt_epi64(__A); 
}

TEST_CONSTEXPR(match_v8di(_mm512_lzcnt_epi64((__m512i)(__v8di){1, 2, 4, 8, 16, 32, 64, 128}), 63, 62, 61, 60, 59, 58, 57, 56));
TEST_CONSTEXPR(match_v8di(_mm512_lzcnt_epi64((__m512i)(__v8di){0, 0, 0, 0, 0, 0, 0, 0}), 64, 64, 64, 64, 64, 64, 64, 64));

__m512i test_mm512_mask_lzcnt_epi64(__m512i __W, __mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_mask_lzcnt_epi64
  // CHECK: call {{.*}}<8 x i64> @llvm.ctlz.v8i64(<8 x i64> %{{.*}}, i1 true)
  // CHECK: [[ISZERO:%.+]] = icmp eq <8 x i64> %{{.*}}, zeroinitializer
  // CHECK: select <8 x i1> [[ISZERO]], <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_mask_lzcnt_epi64(__W,__U,__A); 
}

TEST_CONSTEXPR(match_v8di(_mm512_mask_lzcnt_epi64(_mm512_set1_epi64((long long) 64), /*0101 0111=*/0x57, (__m512i)(__v8di){1, 2, 4, 8, 16, 32, 64, 128}), 63, 62, 61, 64, 59, 64, 57, 64));

__m512i test_mm512_maskz_lzcnt_epi64(__mmask8 __U, __m512i __A) {
  // CHECK-LABEL: test_mm512_maskz_lzcnt_epi64
  // CHECK: call {{.*}}<8 x i64> @llvm.ctlz.v8i64(<8 x i64> %{{.*}}, i1 true)
  // CHECK: [[ISZERO:%.+]] = icmp eq <8 x i64> %{{.*}}, zeroinitializer
  // CHECK: select <8 x i1> [[ISZERO]], <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  // CHECK: select <8 x i1> %{{.*}}, <8 x i64> %{{.*}}, <8 x i64> %{{.*}}
  return _mm512_maskz_lzcnt_epi64(__U,__A); 
}

TEST_CONSTEXPR(match_v8di(_mm512_maskz_lzcnt_epi64(/*0101 0111=*/0x57, (__m512i)(__v8di){1, 2, 4, 8, 16, 32, 64, 128}), 63, 62, 61, 0, 59, 0, 57, 0));

__m512i test_mm512_broadcastmb_epi64(__m512i a, __m512i b) {
  // CHECK-LABEL: test_mm512_broadcastmb_epi64
  // CHECK: icmp eq <8 x i64> %{{.*}}, %{{.*}}
  // CHECK: zext i8 %{{.*}} to i64
  // CHECK: insertelement <8 x i64> poison, i64 %{{.*}}, i32 0
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 1
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 2
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 3
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 4
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 5
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 6
  // CHECK: insertelement <8 x i64> %{{.*}}, i64 %{{.*}}, i32 7
  return _mm512_broadcastmb_epi64(_mm512_cmpeq_epu64_mask ( a, b)); 
}

__m512i test_mm512_broadcastmw_epi32(__m512i a, __m512i b) {
  // CHECK-LABEL: test_mm512_broadcastmw_epi32
  // CHECK: icmp eq <16 x i32> %{{.*}}, %{{.*}}
  // CHECK: zext i16 %{{.*}} to i32
  // CHECK: insertelement <16 x i32> poison, i32 %{{.*}}
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}
  // CHECK: insertelement <16 x i32> %{{.*}}, i32 %{{.*}}
  return _mm512_broadcastmw_epi32(_mm512_cmpeq_epi32_mask ( a, b)); 
}
