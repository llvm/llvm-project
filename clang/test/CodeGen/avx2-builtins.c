// REQUIRES: x86-registered-target

// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +avx2 -emit-llvm -o - -Werror | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +avx2 -fno-signed-char -emit-llvm -o - -Werror | FileCheck %s
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +avx2 -S -o - -Werror | FileCheck %s --check-prefix=CHECK-ASM
// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin -target-feature +avx2 -fno-signed-char -S -o - -Werror | FileCheck %s --check-prefix=CHECK-ASM

// Don't include mm_malloc.h, it's system specific.
#define __MM_MALLOC_H

#include <immintrin.h>

__m256i test_mm256_mpsadbw_epu8(__m256i x, __m256i y) {
  // CHECK: @llvm.x86.avx2.mpsadbw({{.*}}, {{.*}}, i8 3)
  // CHECK-ASM: vmpsadbw $3, %ymm{{.*}}
  return _mm256_mpsadbw_epu8(x, y, 3);
}

__m256i test_mm256_sad_epu8(__m256i x, __m256i y) {
  // CHECK: @llvm.x86.avx2.psad.bw
  // CHECK-ASM: vpsadbw %ymm{{.*}}
  return _mm256_sad_epu8(x, y);
}

__m256i test_mm256_abs_epi8(__m256i a) {
  // CHECK: @llvm.x86.avx2.pabs.b
  // CHECK-ASM: vpabsb %ymm{{.*}}
  return _mm256_abs_epi8(a);
}

__m256i test_mm256_abs_epi16(__m256i a) {
  // CHECK: @llvm.x86.avx2.pabs.w
  // CHECK-ASM: vpabsw %ymm{{.*}}
  return _mm256_abs_epi16(a);
}

__m256i test_mm256_abs_epi32(__m256i a) {
  // CHECK: @llvm.x86.avx2.pabs.d
  // CHECK-ASM: vpabsd %ymm{{.*}}
  return _mm256_abs_epi32(a);
}

__m256i test_mm256_packs_epi16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.packsswb
  // CHECK-ASM: vpacksswb %ymm{{.*}}
  return _mm256_packs_epi16(a, b);
}

__m256i test_mm256_packs_epi32(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.packssdw
  // CHECK-ASM: vpackssdw %ymm{{.*}}
  return _mm256_packs_epi32(a, b);
}

__m256i test_mm256_packs_epu16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.packuswb
  // CHECK-ASM: vpackuswb %ymm{{.*}}
  return _mm256_packus_epi16(a, b);
}

__m256i test_mm256_packs_epu32(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.packusdw
  // CHECK-ASM: vpackusdw %ymm{{.*}}
  return _mm256_packus_epi32(a, b);
}

__m256i test_mm256_add_epi8(__m256i a, __m256i b) {
  // CHECK: add <32 x i8>
  // CHECK-ASM: vpaddb %ymm{{.*}}
  return _mm256_add_epi8(a, b);
}

__m256i test_mm256_add_epi16(__m256i a, __m256i b) {
  // CHECK: add <16 x i16>
  // CHECK-ASM: vpaddw %ymm{{.*}}
  return _mm256_add_epi16(a, b);
}

__m256i test_mm256_add_epi32(__m256i a, __m256i b) {
  // CHECK: add <8 x i32>
  // CHECK-ASM: vpaddd %ymm{{.*}}
  return _mm256_add_epi32(a, b);
}

__m256i test_mm256_add_epi64(__m256i a, __m256i b) {
  // CHECK: add <4 x i64>
  // CHECK-ASM: vpaddq {{.*}}, %ymm{{.*}}
  return _mm256_add_epi64(a, b);
}

__m256i test_mm256_adds_epi8(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.padds.b
  // CHECK-ASM: vpaddsb %ymm{{.*}}
  return _mm256_adds_epi8(a, b);
}

__m256i test_mm256_adds_epi16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.padds.w
  // CHECK-ASM: vpaddsw %ymm{{.*}}
  return _mm256_adds_epi16(a, b);
}

__m256i test_mm256_adds_epu8(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.paddus.b
  // CHECK-ASM: vpaddusb %ymm{{.*}}
  return _mm256_adds_epu8(a, b);
}

__m256i test_mm256_adds_epu16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.paddus.w
  // CHECK-ASM: vpaddusw %ymm{{.*}}
  return _mm256_adds_epu16(a, b);
}

__m256i test_mm256_alignr_epi8(__m256i a, __m256i b) {
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49>
  // CHECK-ASM: vpalignr $2, %ymm{{.*}}
  return _mm256_alignr_epi8(a, b, 2);
}

__m256i test2_mm256_alignr_epi8(__m256i a, __m256i b) {
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> zeroinitializer, <32 x i32> <i32 1, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 17, i32 18, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48>
  // CHECK-ASM: vpsrldq $1, %ymm{{.*}}
  return _mm256_alignr_epi8(a, b, 17);
}

__m256i test_mm256_sub_epi8(__m256i a, __m256i b) {
  // CHECK: sub <32 x i8>
  // CHECK-ASM: vpsubb %ymm{{.*}}
  return _mm256_sub_epi8(a, b);
}

__m256i test_mm256_sub_epi16(__m256i a, __m256i b) {
  // CHECK: sub <16 x i16>
  // CHECK-ASM: vpsubw %ymm{{.*}}
  return _mm256_sub_epi16(a, b);
}

__m256i test_mm256_sub_epi32(__m256i a, __m256i b) {
  // CHECK: sub <8 x i32>
  // CHECK-ASM: vpsubd %ymm{{.*}}
  return _mm256_sub_epi32(a, b);
}

__m256i test_mm256_sub_epi64(__m256i a, __m256i b) {
  // CHECK: sub <4 x i64>
  // CHECK-ASM: vpsubq {{.*}}, %ymm{{.*}}
  return _mm256_sub_epi64(a, b);
}

__m256i test_mm256_subs_epi8(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.psubs.b
  // CHECK-ASM: vpsubsb %ymm{{.*}}
  return _mm256_subs_epi8(a, b);
}

__m256i test_mm256_subs_epi16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.psubs.w
  // CHECK-ASM: vpsubsw %ymm{{.*}}
  return _mm256_subs_epi16(a, b);
}

__m256i test_mm256_subs_epu8(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.psubus.b
  // CHECK-ASM: vpsubusb %ymm{{.*}}
  return _mm256_subs_epu8(a, b);
}

__m256i test_mm256_subs_epu16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.psubus.w
  // CHECK-ASM: vpsubusw %ymm{{.*}}
  return _mm256_subs_epu16(a, b);
}

__m256i test_mm256_and_si256(__m256i a, __m256i b) {
  // CHECK: and <4 x i64>
  // CHECK-ASM: vandps {{.*}}, %ymm{{.*}}
  return _mm256_and_si256(a, b);
}

__m256i test_mm256_andnot_si256(__m256i a, __m256i b) {
  // CHECK: xor <4 x i64>
  // CHECK: and <4 x i64>

  // Note that, at -O0, we generate the expansion instead of matching vpandn.
  // CHECK-ASM:      vpcmpeqd [[ALLONES:%ymm[0-9]+]], [[ALLONES]], [[ALLONES]]
  // CHECK-ASM-NEXT: vpxor [[ALLONES]], %ymm{{.*}}, [[NOT:%ymm[0-9]+]]
  // CHECK-ASM-NEXT: vandps {{.*}}, [[NOT]], %ymm{{.*}}
  return _mm256_andnot_si256(a, b);
}

__m256i test_mm256_or_si256(__m256i a, __m256i b) {
  // CHECK: or <4 x i64>
  // CHECK-ASM: vorps {{.*}}, %ymm{{.*}}
  return _mm256_or_si256(a, b);
}

__m256i test_mm256_xor_si256(__m256i a, __m256i b) {
  // CHECK: xor <4 x i64>
  // CHECK-ASM: vxorps {{.*}}, %ymm{{.*}}
  return _mm256_xor_si256(a, b);
}

__m256i test_mm256_avg_epu8(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pavg.b
  // CHECK-ASM: vpavgb %ymm{{.*}}
  return _mm256_avg_epu8(a, b);
}

__m256i test_mm256_avg_epu16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pavg.w
  // CHECK-ASM: vpavgw %ymm{{.*}}
  return _mm256_avg_epu16(a, b);
}

__m256i test_mm256_blendv_epi8(__m256i a, __m256i b, __m256i m) {
  // CHECK: @llvm.x86.avx2.pblendvb
  // CHECK-ASM: vpblendvb %ymm{{.*}}
  return _mm256_blendv_epi8(a, b, m);
}

// FIXME: We should also lower the __builtin_ia32_pblendw128 (and similar)
// functions to this IR. In the future we could delete the corresponding
// intrinsic in LLVM if it's not being used anymore.
__m256i test_mm256_blend_epi16(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_blend_epi16
  // CHECK-NOT: @llvm.x86.avx2.pblendw
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 0, i32 17, i32 2, i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 25, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15>
  // CHECK-ASM: vpblendw $2, %ymm{{.*}}
  return _mm256_blend_epi16(a, b, 2);
}

__m256i test_mm256_cmpeq_epi8(__m256i a, __m256i b) {
  // CHECK: icmp eq <32 x i8>
  // CHECK-ASM: vpcmpeqb %ymm{{.*}}
  return _mm256_cmpeq_epi8(a, b);
}

__m256i test_mm256_cmpeq_epi16(__m256i a, __m256i b) {
  // CHECK: icmp eq <16 x i16>
  // CHECK-ASM: vpcmpeqw %ymm{{.*}}
  return _mm256_cmpeq_epi16(a, b);
}

__m256i test_mm256_cmpeq_epi32(__m256i a, __m256i b) {
  // CHECK: icmp eq <8 x i32>
  // CHECK-ASM: vpcmpeqd %ymm{{.*}}
  return _mm256_cmpeq_epi32(a, b);
}

__m256i test_mm256_cmpeq_epi64(__m256i a, __m256i b) {
  // CHECK: icmp eq <4 x i64>
  // CHECK-ASM: vpcmpeqq %ymm{{.*}}
  return _mm256_cmpeq_epi64(a, b);
}

__m256i test_mm256_cmpgt_epi8(__m256i a, __m256i b) {
  // CHECK: icmp sgt <32 x i8>
  // CHECK-ASM: vpcmpgtb %ymm{{.*}}
  return _mm256_cmpgt_epi8(a, b);
}

__m256i test_mm256_cmpgt_epi16(__m256i a, __m256i b) {
  // CHECK: icmp sgt <16 x i16>
  // CHECK-ASM: vpcmpgtw %ymm{{.*}}
  return _mm256_cmpgt_epi16(a, b);
}

__m256i test_mm256_cmpgt_epi32(__m256i a, __m256i b) {
  // CHECK: icmp sgt <8 x i32>
  // CHECK-ASM: vpcmpgtd %ymm{{.*}}
  return _mm256_cmpgt_epi32(a, b);
}

__m256i test_mm256_cmpgt_epi64(__m256i a, __m256i b) {
  // CHECK: icmp sgt <4 x i64>
  // CHECK-ASM: vpcmpgtq %ymm{{.*}}
  return _mm256_cmpgt_epi64(a, b);
}

__m256i test_mm256_hadd_epi16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.phadd.w
  // CHECK-ASM: vphaddw %ymm{{.*}}
  return _mm256_hadd_epi16(a, b);
}

__m256i test_mm256_hadd_epi32(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.phadd.d
  // CHECK-ASM: vphaddd %ymm{{.*}}
  return _mm256_hadd_epi32(a, b);
}

__m256i test_mm256_hadds_epi16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.phadd.sw
  // CHECK-ASM: vphaddsw %ymm{{.*}}
  return _mm256_hadds_epi16(a, b);
}

__m256i test_mm256_hsub_epi16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.phsub.w
  // CHECK-ASM: vphsubw %ymm{{.*}}
  return _mm256_hsub_epi16(a, b);
}

__m256i test_mm256_hsub_epi32(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.phsub.d
  // CHECK-ASM: vphsubd %ymm{{.*}}
  return _mm256_hsub_epi32(a, b);
}

__m256i test_mm256_hsubs_epi16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.phsub.sw
  // CHECK-ASM: vphsubsw %ymm{{.*}}
  return _mm256_hsubs_epi16(a, b);
}

__m256i test_mm256_maddubs_epi16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pmadd.ub.sw
  // CHECK-ASM: vpmaddubsw %ymm{{.*}}
  return _mm256_maddubs_epi16(a, b);
}

__m256i test_mm256_madd_epi16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pmadd.wd
  // CHECK-ASM: vpmaddwd %ymm{{.*}}
  return _mm256_madd_epi16(a, b);
}

__m256i test_mm256_max_epi8(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pmaxs.b
  // CHECK-ASM: vpmaxsb %ymm{{.*}}
  return _mm256_max_epi8(a, b);
}

__m256i test_mm256_max_epi16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pmaxs.w
  // CHECK-ASM: vpmaxsw %ymm{{.*}}
  return _mm256_max_epi16(a, b);
}

__m256i test_mm256_max_epi32(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pmaxs.d
  // CHECK-ASM: vpmaxsd %ymm{{.*}}
  return _mm256_max_epi32(a, b);
}

__m256i test_mm256_max_epu8(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pmaxu.b
  // CHECK-ASM: vpmaxub %ymm{{.*}}
  return _mm256_max_epu8(a, b);
}

__m256i test_mm256_max_epu16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pmaxu.w
  // CHECK-ASM: vpmaxuw %ymm{{.*}}
  return _mm256_max_epu16(a, b);
}

__m256i test_mm256_max_epu32(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pmaxu.d
  // CHECK-ASM: vpmaxud %ymm{{.*}}
  return _mm256_max_epu32(a, b);
}

__m256i test_mm256_min_epi8(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pmins.b
  // CHECK-ASM: vpminsb %ymm{{.*}}
  return _mm256_min_epi8(a, b);
}

__m256i test_mm256_min_epi16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pmins.w
  // CHECK-ASM: vpminsw %ymm{{.*}}
  return _mm256_min_epi16(a, b);
}

__m256i test_mm256_min_epi32(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pmins.d
  // CHECK-ASM: vpminsd %ymm{{.*}}
  return _mm256_min_epi32(a, b);
}

__m256i test_mm256_min_epu8(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pminu.b
  // CHECK-ASM: vpminub %ymm{{.*}}
  return _mm256_min_epu8(a, b);
}

__m256i test_mm256_min_epu16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pminu.w
  // CHECK-ASM: vpminuw %ymm{{.*}}
  return _mm256_min_epu16(a, b);
}

__m256i test_mm256_min_epu32(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pminu.d
  // CHECK-ASM: vpminud %ymm{{.*}}
  return _mm256_min_epu32(a, b);
}

int test_mm256_movemask_epi8(__m256i a) {
  // CHECK: @llvm.x86.avx2.pmovmskb
  // CHECK-ASM: vpmovmskb %ymm{{.*}}
  return _mm256_movemask_epi8(a);
}

__m256i test_mm256_cvtepi8_epi16(__m128i a) {
  // CHECK: @llvm.x86.avx2.pmovsxbw
  // CHECK-ASM: vpmovsxbw %xmm{{.*}}, %ymm{{.*}}
  return _mm256_cvtepi8_epi16(a);
}

__m256i test_mm256_cvtepi8_epi32(__m128i a) {
  // CHECK: @llvm.x86.avx2.pmovsxbd
  // CHECK-ASM: vpmovsxbd %xmm{{.*}}, %ymm{{.*}}
  return _mm256_cvtepi8_epi32(a);
}

__m256i test_mm256_cvtepi8_epi64(__m128i a) {
  // CHECK: @llvm.x86.avx2.pmovsxbq
  // CHECK-ASM: vpmovsxbq %xmm{{.*}}, %ymm{{.*}}
  return _mm256_cvtepi8_epi64(a);
}

__m256i test_mm256_cvtepi16_epi32(__m128i a) {
  // CHECK: @llvm.x86.avx2.pmovsxwd
  // CHECK-ASM: vpmovsxwd %xmm{{.*}}, %ymm{{.*}}
  return _mm256_cvtepi16_epi32(a);
}

__m256i test_mm256_cvtepi16_epi64(__m128i a) {
  // CHECK: @llvm.x86.avx2.pmovsxwq
  // CHECK-ASM: vpmovsxwq %xmm{{.*}}, %ymm{{.*}}
  return _mm256_cvtepi16_epi64(a);
}

__m256i test_mm256_cvtepi32_epi64(__m128i a) {
  // CHECK: @llvm.x86.avx2.pmovsxdq
  // CHECK-ASM: vpmovsxdq %xmm{{.*}}, %ymm{{.*}}
  return _mm256_cvtepi32_epi64(a);
}

__m256i test_mm256_cvtepu8_epi16(__m128i a) {
  // CHECK: @llvm.x86.avx2.pmovzxbw
  // CHECK-ASM: vpmovzxbw %xmm{{.*}}, %ymm{{.*}}
  return _mm256_cvtepu8_epi16(a);
}

__m256i test_mm256_cvtepu8_epi32(__m128i a) {
  // CHECK: @llvm.x86.avx2.pmovzxbd
  // CHECK-ASM: vpmovzxbd %xmm{{.*}}, %ymm{{.*}}
  return _mm256_cvtepu8_epi32(a);
}

__m256i test_mm256_cvtepu8_epi64(__m128i a) {
  // CHECK: @llvm.x86.avx2.pmovzxbq
  // CHECK-ASM: vpmovzxbq %xmm{{.*}}, %ymm{{.*}}
  return _mm256_cvtepu8_epi64(a);
}

__m256i test_mm256_cvtepu16_epi32(__m128i a) {
  // CHECK: @llvm.x86.avx2.pmovzxwd
  // CHECK-ASM: vpmovzxwd %xmm{{.*}}, %ymm{{.*}}
  return _mm256_cvtepu16_epi32(a);
}

__m256i test_mm256_cvtepu16_epi64(__m128i a) {
  // CHECK: @llvm.x86.avx2.pmovzxwq
  // CHECK-ASM: vpmovzxwq %xmm{{.*}}, %ymm{{.*}}
  return _mm256_cvtepu16_epi64(a);
}

__m256i test_mm256_cvtepu32_epi64(__m128i a) {
  // CHECK: @llvm.x86.avx2.pmovzxdq
  // CHECK-ASM: vpmovzxdq %xmm{{.*}}, %ymm{{.*}}
  return _mm256_cvtepu32_epi64(a);
}

__m256i test_mm256_mul_epi32(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pmul.dq
  // CHECK-ASM: vpmuldq %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_mul_epi32(a, b);
}

__m256i test_mm256_mulhrs_epi16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pmul.hr.sw
  // CHECK-ASM: vpmulhrsw %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_mulhrs_epi16(a, b);
}

__m256i test_mm256_mulhi_epu16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pmulhu.w
  // CHECK-ASM: vpmulhuw %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_mulhi_epu16(a, b);
}

__m256i test_mm256_mulhi_epi16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pmulh.w
  // CHECK-ASM: vpmulhw %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_mulhi_epi16(a, b);
}

__m256i test_mm256_mullo_epi16(__m256i a, __m256i b) {
  // CHECK: mul <16 x i16>
  // CHECK-ASM: vpmullw %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_mullo_epi16(a, b);
}

__m256i test_mm256_mullo_epi32(__m256i a, __m256i b) {
  // CHECK: mul <8 x i32>
  // CHECK-ASM: vpmulld %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_mullo_epi32(a, b);
}

__m256i test_mm256_mul_epu32(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pmulu.dq
  // CHECK-ASM: vpmuludq %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_mul_epu32(a, b);
}

__m256i test_mm256_shuffle_epi8(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.pshuf.b
  // CHECK-ASM: vpshufb %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_shuffle_epi8(a, b);
}

__m256i test_mm256_shuffle_epi32(__m256i a) {
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 3, i32 3, i32 0, i32 0, i32 7, i32 7, i32 4, i32 4>
  // CHECK-ASM: vpshufd $15, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_shuffle_epi32(a, 15);
}

__m256i test_mm256_shufflehi_epi16(__m256i a) {
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 0, i32 1, i32 2, i32 3, i32 7, i32 6, i32 6, i32 5, i32 8, i32 9, i32 10, i32 11, i32 15, i32 14, i32 14, i32 13>
  // CHECK-ASM: vpshufhw $107, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_shufflehi_epi16(a, 107);
}

__m256i test_mm256_shufflelo_epi16(__m256i a) {
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 3, i32 0, i32 1, i32 1, i32 4, i32 5, i32 6, i32 7, i32 11, i32 8, i32 9, i32 9, i32 12, i32 13, i32 14, i32 15>
  // CHECK-ASM: vpshuflw $83, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_shufflelo_epi16(a, 83);
}

__m256i test_mm256_sign_epi8(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.psign.b
  // CHECK-ASM: vpsignb %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_sign_epi8(a, b);
}

__m256i test_mm256_sign_epi16(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.psign.w
  // CHECK-ASM: vpsignw %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_sign_epi16(a, b);
}

__m256i test_mm256_sign_epi32(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.psign.d
  // CHECK-ASM: vpsignd %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_sign_epi32(a, b);
}

__m256i test_mm256_slli_si256(__m256i a) {
  // CHECK: shufflevector <32 x i8> zeroinitializer, <32 x i8> %{{.*}}, <32 x i32> <i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 29, i32 30, i32 31, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60>
  // CHECK-ASM: vpslldq $3, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_slli_si256(a, 3);
}

__m256i test_mm256_bslli_epi128(__m256i a) {
  // CHECK: shufflevector <32 x i8> zeroinitializer, <32 x i8> %{{.*}}, <32 x i32> <i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 35, i32 36, i32 37, i32 38, i32 39, i32 40, i32 41, i32 42, i32 43, i32 44, i32 29, i32 30, i32 31, i32 48, i32 49, i32 50, i32 51, i32 52, i32 53, i32 54, i32 55, i32 56, i32 57, i32 58, i32 59, i32 60>
  // CHECK-ASM: vpslldq $3, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_bslli_epi128(a, 3);
}

__m256i test_mm256_slli_epi16(__m256i a) {
  // CHECK: @llvm.x86.avx2.pslli.w
  // CHECK-ASM: vpsllw {{\$3|%xmm[0-9]+}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_slli_epi16(a, 3);
}

__m256i test_mm256_sll_epi16(__m256i a, __m128i b) {
  // CHECK: @llvm.x86.avx2.psll.w
  // CHECK-ASM: vpsllw %xmm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_sll_epi16(a, b);
}

__m256i test_mm256_slli_epi32(__m256i a) {
  // CHECK: @llvm.x86.avx2.pslli.d
  // CHECK-ASM: vpslld {{\$3|%xmm[0-9]+}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_slli_epi32(a, 3);
}

__m256i test_mm256_sll_epi32(__m256i a, __m128i b) {
  // CHECK: @llvm.x86.avx2.psll.d
  // CHECK-ASM: vpslld %xmm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_sll_epi32(a, b);
}

__m256i test_mm256_slli_epi64(__m256i a) {
  // CHECK: @llvm.x86.avx2.pslli.q
  // CHECK-ASM: vpsllq %xmm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_slli_epi64(a, 3);
}

__m256i test_mm256_sll_epi64(__m256i a, __m128i b) {
  // CHECK: @llvm.x86.avx2.psll.q
  // CHECK-ASM: vpsllq %xmm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_sll_epi64(a, b);
}

__m256i test_mm256_srai_epi16(__m256i a) {
  // CHECK: @llvm.x86.avx2.psrai.w
  // CHECK-ASM: vpsraw {{\$3|%xmm[0-9]+}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_srai_epi16(a, 3);
}

__m256i test_mm256_sra_epi16(__m256i a, __m128i b) {
  // CHECK: @llvm.x86.avx2.psra.w
  // CHECK-ASM: vpsraw %xmm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_sra_epi16(a, b);
}

__m256i test_mm256_srai_epi32(__m256i a) {
  // CHECK: @llvm.x86.avx2.psrai.d
  // CHECK-ASM: vpsrad {{\$3|%xmm[0-9]+}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_srai_epi32(a, 3);
}

__m256i test_mm256_sra_epi32(__m256i a, __m128i b) {
  // CHECK: @llvm.x86.avx2.psra.d
  // CHECK-ASM: vpsrad %xmm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_sra_epi32(a, b);
}

__m256i test_mm256_srli_si256(__m256i a) {
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> zeroinitializer, <32 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49, i32 50>
  // CHECK-ASM: vpsrldq $3, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_srli_si256(a, 3);
}

__m256i test_mm256_bsrli_epi128(__m256i a) {
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> zeroinitializer, <32 x i32> <i32 3, i32 4, i32 5, i32 6, i32 7, i32 8, i32 9, i32 10, i32 11, i32 12, i32 13, i32 14, i32 15, i32 32, i32 33, i32 34, i32 19, i32 20, i32 21, i32 22, i32 23, i32 24, i32 25, i32 26, i32 27, i32 28, i32 29, i32 30, i32 31, i32 48, i32 49, i32 50>
  // CHECK-ASM: vpsrldq $3, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_bsrli_epi128(a, 3);
}

__m256i test_mm256_srli_epi16(__m256i a) {
  // CHECK: @llvm.x86.avx2.psrli.w
  // CHECK-ASM: vpsrlw {{\$3|%xmm[0-9]+}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_srli_epi16(a, 3);
}

__m256i test_mm256_srl_epi16(__m256i a, __m128i b) {
  // CHECK: @llvm.x86.avx2.psrl.w
  // CHECK-ASM: vpsrlw %xmm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_srl_epi16(a, b);
}

__m256i test_mm256_srli_epi32(__m256i a) {
  // CHECK: @llvm.x86.avx2.psrli.d
  // CHECK-ASM: vpsrld {{\$3|%xmm[0-9]+}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_srli_epi32(a, 3);
}

__m256i test_mm256_srl_epi32(__m256i a, __m128i b) {
  // CHECK: @llvm.x86.avx2.psrl.d
  // CHECK-ASM: vpsrld %xmm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_srl_epi32(a, b);
}

__m256i test_mm256_srli_epi64(__m256i a) {
  // CHECK: @llvm.x86.avx2.psrli.q
  // CHECK-ASM: vpsrlq %xmm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_srli_epi64(a, 3);
}

__m256i test_mm256_srl_epi64(__m256i a, __m128i b) {
  // CHECK: @llvm.x86.avx2.psrl.q
  // CHECK-ASM: vpsrlq %xmm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_srl_epi64(a, b);
}

__m256i test_mm256_unpackhi_epi8(__m256i a, __m256i b) {
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 8, i32 40, i32 9, i32 41, i32 10, i32 42, i32 11, i32 43, i32 12, i32 44, i32 13, i32 45, i32 14, i32 46, i32 15, i32 47, i32 24, i32 56, i32 25, i32 57, i32 26, i32 58, i32 27, i32 59, i32 28, i32 60, i32 29, i32 61, i32 30, i32 62, i32 31, i32 63>
  // CHECK-ASM: vpunpckhbw %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_unpackhi_epi8(a, b);
}

__m256i test_mm256_unpackhi_epi16(__m256i a, __m256i b) {
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 4, i32 20, i32 5, i32 21, i32 6, i32 22, i32 7, i32 23, i32 12, i32 28, i32 13, i32 29, i32 14, i32 30, i32 15, i32 31>
  // CHECK-ASM: vpunpckhwd %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_unpackhi_epi16(a, b);
}

__m256i test_mm256_unpackhi_epi32(__m256i a, __m256i b) {
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 2, i32 10, i32 3, i32 11, i32 6, i32 14, i32 7, i32 15>
  // CHECK-ASM: vpunpckhdq %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_unpackhi_epi32(a, b);
}

__m256i test_mm256_unpackhi_epi64(__m256i a, __m256i b) {
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 1, i32 5, i32 3, i32 7>
  // CHECK-ASM: vpunpckhqdq %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_unpackhi_epi64(a, b);
}

__m256i test_mm256_unpacklo_epi8(__m256i a, __m256i b) {
  // CHECK: shufflevector <32 x i8> %{{.*}}, <32 x i8> %{{.*}}, <32 x i32> <i32 0, i32 32, i32 1, i32 33, i32 2, i32 34, i32 3, i32 35, i32 4, i32 36, i32 5, i32 37, i32 6, i32 38, i32 7, i32 39, i32 16, i32 48, i32 17, i32 49, i32 18, i32 50, i32 19, i32 51, i32 20, i32 52, i32 21, i32 53, i32 22, i32 54, i32 23, i32 55>
  // CHECK-ASM: vpunpcklbw %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_unpacklo_epi8(a, b);
}

__m256i test_mm256_unpacklo_epi16(__m256i a, __m256i b) {
  // CHECK: shufflevector <16 x i16> %{{.*}}, <16 x i16> %{{.*}}, <16 x i32> <i32 0, i32 16, i32 1, i32 17, i32 2, i32 18, i32 3, i32 19, i32 8, i32 24, i32 9, i32 25, i32 10, i32 26, i32 11, i32 27>
  // CHECK-ASM: vpunpcklwd %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_unpacklo_epi16(a, b);
}

__m256i test_mm256_unpacklo_epi32(__m256i a, __m256i b) {
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 0, i32 8, i32 1, i32 9, i32 4, i32 12, i32 5, i32 13>
  // CHECK-ASM: vpunpckldq %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_unpacklo_epi32(a, b);
}

__m256i test_mm256_unpacklo_epi64(__m256i a, __m256i b) {
  // CHECK: shufflevector <4 x i64> %{{.*}}, <4 x i64> %{{.*}}, <4 x i32> <i32 0, i32 4, i32 2, i32 6>
  // CHECK-ASM: vpunpcklqdq %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_unpacklo_epi64(a, b);
}

__m256i test_mm256_stream_load_si256(__m256i const *a) {
  // CHECK: @llvm.x86.avx2.movntdqa
  // CHECK-ASM: vmovntdqa (%rdi), %ymm{{.*}}
  return _mm256_stream_load_si256(a);
}

__m128 test_mm_broadcastss_ps(__m128 a) {
  // CHECK-LABEL: test_mm_broadcastss_ps
  // CHECK-NOT: @llvm.x86.avx2.vbroadcast.ss.ps
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK-ASM: vbroadcastss %xmm{{.*}}, %xmm{{.*}}
  return _mm_broadcastss_ps(a);
}

__m128d test_mm_broadcastsd_pd(__m128d a) {
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <2 x i32> zeroinitializer
  // CHECK-ASM: vmovddup %xmm{{.*}}, %xmm{{.*}}
  return _mm_broadcastsd_pd(a);
}

__m256 test_mm256_broadcastss_ps(__m128 a) {
  // CHECK-LABEL: test_mm256_broadcastss_ps
  // CHECK-NOT: @llvm.x86.avx2.vbroadcast.ss.ps.256
  // CHECK: shufflevector <4 x float> %{{.*}}, <4 x float> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK-ASM: vbroadcastss %xmm{{.*}}, %ymm{{.*}}
  return _mm256_broadcastss_ps(a);
}

__m256d test_mm256_broadcastsd_pd(__m128d a) {
  // CHECK-LABEL: test_mm256_broadcastsd_pd
  // CHECK-NOT: @llvm.x86.avx2.vbroadcast.sd.pd.256
  // CHECK: shufflevector <2 x double> %{{.*}}, <2 x double> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK-ASM: vbroadcastsd %xmm{{.*}}, %ymm{{.*}}
  return _mm256_broadcastsd_pd(a);
}

__m256i test_mm256_broadcastsi128_si256(__m128i a) {
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> <i32 0, i32 1, i32 0, i32 1>
  // CHECK-ASM: vinserti128 $1, %xmm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_broadcastsi128_si256(a);
}

__m128i test_mm_blend_epi32(__m128i a, __m128i b) {
  // CHECK-LABEL: test_mm_blend_epi32
  // CHECK-NOT: @llvm.x86.avx2.pblendd.128
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> <i32 4, i32 1, i32 6, i32 3>
  // CHECK-ASM: vpblendd $10, %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_blend_epi32(a, b, 0x35);
}

__m256i test_mm256_blend_epi32(__m256i a, __m256i b) {
  // CHECK-LABEL: test_mm256_blend_epi32
  // CHECK-NOT: @llvm.x86.avx2.pblendd.256
  // CHECK: shufflevector <8 x i32> %{{.*}}, <8 x i32> %{{.*}}, <8 x i32> <i32 8, i32 1, i32 10, i32 3, i32 12, i32 13, i32 6, i32 7>
  // CHECK-ASM: vpblendd $202, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_blend_epi32(a, b, 0x35);
}

__m256i test_mm256_broadcastb_epi8(__m128i a) {
  // CHECK-LABEL: test_mm256_broadcastb_epi8
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastb.256
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <32 x i32> zeroinitializer
  // CHECK-ASM: vpbroadcastb %xmm{{.*}}, %ymm{{.*}}
  return _mm256_broadcastb_epi8(a);
}

__m256i test_mm256_broadcastw_epi16(__m128i a) {
  // CHECK-LABEL: test_mm256_broadcastw_epi16
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastw.256
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <16 x i32> zeroinitializer
  // CHECK-ASM: vpbroadcastw %xmm{{.*}}, %ymm{{.*}}
  return _mm256_broadcastw_epi16(a);
}

__m256i test_mm256_broadcastd_epi32(__m128i a) {
  // CHECK-LABEL: test_mm256_broadcastd_epi32
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastd.256
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK-ASM: vpbroadcastd %xmm{{.*}}, %ymm{{.*}}
  return _mm256_broadcastd_epi32(a);
}

__m256i test_mm256_broadcastq_epi64(__m128i a) {
  // CHECK-LABEL: test_mm256_broadcastq_epi64
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastq.256
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK-ASM: vpbroadcastq %xmm{{.*}}, %ymm{{.*}}
  return _mm256_broadcastq_epi64(a);
}

__m128i test_mm_broadcastb_epi8(__m128i a) {
  // CHECK-LABEL: test_mm_broadcastb_epi8
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastb.128
  // CHECK: shufflevector <16 x i8> %{{.*}}, <16 x i8> %{{.*}}, <16 x i32> zeroinitializer
  // CHECK-ASM: vpbroadcastb %xmm{{.*}}, %xmm{{.*}}
  return _mm_broadcastb_epi8(a);
}

__m128i test_mm_broadcastw_epi16(__m128i a) {
  // CHECK-LABEL: test_mm_broadcastw_epi16
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastw.128
  // CHECK: shufflevector <8 x i16> %{{.*}}, <8 x i16> %{{.*}}, <8 x i32> zeroinitializer
  // CHECK-ASM: vpbroadcastw %xmm{{.*}}, %xmm{{.*}}
  return _mm_broadcastw_epi16(a);
}

__m128i test_mm_broadcastd_epi32(__m128i a) {
  // CHECK-LABEL: test_mm_broadcastd_epi32
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastd.128
  // CHECK: shufflevector <4 x i32> %{{.*}}, <4 x i32> %{{.*}}, <4 x i32> zeroinitializer
  // CHECK-ASM: vpbroadcastd %xmm{{.*}}, %xmm{{.*}}
  return _mm_broadcastd_epi32(a);
}

__m128i test_mm_broadcastq_epi64(__m128i a) {
  // CHECK-LABEL: test_mm_broadcastq_epi64
  // CHECK-NOT: @llvm.x86.avx2.pbroadcastq.128
  // CHECK: shufflevector <2 x i64> %{{.*}}, <2 x i64> %{{.*}}, <2 x i32> zeroinitializer
  // CHECK-ASM: vpbroadcastq %xmm{{.*}}, %xmm{{.*}}
  return _mm_broadcastq_epi64(a);
}

__m256i test_mm256_permutevar8x32_epi32(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.permd
  // CHECK-ASM: vpermd %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_permutevar8x32_epi32(a, b);
}

__m256d test_mm256_permute4x64_pd(__m256d a) {
  // CHECK: shufflevector{{.*}}<i32 1, i32 2, i32 1, i32 0>
  // CHECK-ASM: vpermpd $25, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_permute4x64_pd(a, 25);
}

__m256 test_mm256_permutevar8x32_ps(__m256 a, __m256i b) {
  // CHECK: @llvm.x86.avx2.permps
  // CHECK-ASM: vpermps %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_permutevar8x32_ps(a, b);
}

__m256i test_mm256_permute4x64_epi64(__m256i a) {
  // CHECK: shufflevector{{.*}}<i32 3, i32 0, i32 2, i32 0>
  // CHECK-ASM: vpermq $35, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_permute4x64_epi64(a, 35);
}

__m256i test_mm256_permute2x128_si256(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.vperm2i128
  // CHECK-ASM: vperm2i128 $49, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_permute2x128_si256(a, b, 0x31);
}

__m128i test_mm256_extracti128_si256_0(__m256i a) {
  // CHECK-LABEL: @test_mm256_extracti128_si256_0
  // CHECK: shufflevector{{.*}}<i32 0, i32 1>

  // Note that we just match an XMM copy: vextracti128 $0 is a little overkill.
  // CHECK-ASM: vmovdqa {{.*}}, %xmm0
  return _mm256_extracti128_si256(a, 0);
}

__m128i test_mm256_extracti128_si256_1(__m256i a) {
  // CHECK-LABEL: @test_mm256_extracti128_si256_1
  // CHECK: shufflevector{{.*}}<i32 2, i32 3>
  // CHECK-ASM: vextracti128 $1, %ymm{{.*}}, %xmm{{.*}}
  return _mm256_extracti128_si256(a, 1);
}

// Immediate should be truncated to one bit.
__m128i test_mm256_extracti128_si256_2(__m256i a) {
  // CHECK-LABEL: @test_mm256_extracti128_si256_2
  // CHECK: shufflevector{{.*}}<i32 0, i32 1>

  // Same as extracti128 $0.
  // CHECK-ASM: vmovdqa {{.*}}, %xmm0
  return _mm256_extracti128_si256(a, 2);
}

__m256i test_mm256_inserti128_si256_0(__m256i a, __m128i b) {
  // CHECK-LABEL: @test_mm256_inserti128_si256_0
  // CHECK: shufflevector{{.*}}<i32 4, i32 5, i32 2, i32 3>
  // CHECK-ASM: vpblendd $240, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_inserti128_si256(a, b, 0);
}

__m256i test_mm256_inserti128_si256_1(__m256i a, __m128i b) {
  // CHECK-LABEL: @test_mm256_inserti128_si256_1
  // CHECK: shufflevector{{.*}}<i32 0, i32 1, i32 4, i32 5>
  // CHECK-ASM: vinserti128 $1, %xmm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_inserti128_si256(a, b, 1);
}

// Immediate should be truncated to one bit.
__m256i test_mm256_inserti128_si256_2(__m256i a, __m128i b) {
  // CHECK-LABEL: @test_mm256_inserti128_si256_2
  // CHECK: shufflevector{{.*}}<i32 4, i32 5, i32 2, i32 3>
  // CHECK-ASM: vpblendd $240, %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_inserti128_si256(a, b, 2);
}

__m256i test_mm256_maskload_epi32(int const *a, __m256i m) {
  // CHECK: @llvm.x86.avx2.maskload.d.256
  // CHECK-ASM: vpmaskmovd (%rdi), %ymm{{.*}}, %ymm{{.*}}
  return _mm256_maskload_epi32(a, m);
}

__m256i test_mm256_maskload_epi64(long long const *a, __m256i m) {
  // CHECK: @llvm.x86.avx2.maskload.q.256
  // CHECK-ASM: vpmaskmovq (%rdi), %ymm{{.*}}, %ymm{{.*}}
  return _mm256_maskload_epi64(a, m);
}

__m128i test_mm_maskload_epi32(int const *a, __m128i m) {
  // CHECK: @llvm.x86.avx2.maskload.d
  // CHECK-ASM: vpmaskmovd (%rdi), %xmm{{.*}}, %xmm{{.*}}
  return _mm_maskload_epi32(a, m);
}

__m128i test_mm_maskload_epi64(long long const *a, __m128i m) {
  // CHECK: @llvm.x86.avx2.maskload.q
  // CHECK-ASM: vpmaskmovq (%rdi), %xmm{{.*}}, %xmm{{.*}}
  return _mm_maskload_epi64(a, m);
}

void test_mm256_maskstore_epi32(int *a, __m256i m, __m256i b) {
  // CHECK: @llvm.x86.avx2.maskstore.d.256
  // CHECK-ASM: vpmaskmovd %ymm{{.*}}, %ymm{{.*}}, (%r{{.*}})
  _mm256_maskstore_epi32(a, m, b);
}

void test_mm256_maskstore_epi64(long long *a, __m256i m, __m256i b) {
  // CHECK: @llvm.x86.avx2.maskstore.q.256
  // CHECK-ASM: vpmaskmovq %ymm{{.*}}, %ymm{{.*}}, (%r{{.*}})
  _mm256_maskstore_epi64(a, m, b);
}

void test_mm_maskstore_epi32(int *a, __m128i m, __m128i b) {
  // CHECK: @llvm.x86.avx2.maskstore.d
  // CHECK-ASM: vpmaskmovd %xmm{{.*}}, %xmm{{.*}}, (%r{{.*}})
  _mm_maskstore_epi32(a, m, b);
}

void test_mm_maskstore_epi64(long long *a, __m128i m, __m128i b) {
  // CHECK: @llvm.x86.avx2.maskstore.q
  // CHECK-ASM: vpmaskmovq %xmm{{.*}}, %xmm{{.*}}, (%r{{.*}})
  _mm_maskstore_epi64(a, m, b);
}

__m256i test_mm256_sllv_epi32(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.psllv.d.256
  // CHECK-ASM: vpsllvd %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_sllv_epi32(a, b);
}

__m128i test_mm_sllv_epi32(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.avx2.psllv.d
  // CHECK-ASM: vpsllvd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_sllv_epi32(a, b);
}

__m256i test_mm256_sllv_epi64(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.psllv.q.256
  // CHECK-ASM: vpsllvq %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_sllv_epi64(a, b);
}

__m128i test_mm_sllv_epi64(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.avx2.psllv.q
  // CHECK-ASM: vpsllvq %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_sllv_epi64(a, b);
}

__m256i test_mm256_srav_epi32(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.psrav.d.256
  // CHECK-ASM: vpsravd %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_srav_epi32(a, b);
}

__m128i test_mm_srav_epi32(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.avx2.psrav.d
  // CHECK-ASM: vpsravd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_srav_epi32(a, b);
}

__m256i test_mm256_srlv_epi32(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.psrlv.d.256
  // CHECK-ASM: vpsrlvd %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_srlv_epi32(a, b);
}

__m128i test_mm_srlv_epi32(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.avx2.psrlv.d
  // CHECK-ASM: vpsrlvd %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_srlv_epi32(a, b);
}

__m256i test_mm256_srlv_epi64(__m256i a, __m256i b) {
  // CHECK: @llvm.x86.avx2.psrlv.q.256
  // CHECK-ASM: vpsrlvq %ymm{{.*}}, %ymm{{.*}}, %ymm{{.*}}
  return _mm256_srlv_epi64(a, b);
}

__m128i test_mm_srlv_epi64(__m128i a, __m128i b) {
  // CHECK: @llvm.x86.avx2.psrlv.q
  // CHECK-ASM: vpsrlvq %xmm{{.*}}, %xmm{{.*}}, %xmm{{.*}}
  return _mm_srlv_epi64(a, b);
}

__m128d test_mm_mask_i32gather_pd(__m128d a, double const *b, __m128i c,
                                  __m128d d) {
  // CHECK: @llvm.x86.avx2.gather.d.pd
  // CHECK-ASM: vgatherdpd %xmm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %xmm{{.*}}
  return _mm_mask_i32gather_pd(a, b, c, d, 2);
}

__m256d test_mm256_mask_i32gather_pd(__m256d a, double const *b, __m128i c,
                                      __m256d d) {
  // CHECK: @llvm.x86.avx2.gather.d.pd.256
  // CHECK-ASM: vgatherdpd %ymm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %ymm{{.*}}
  return _mm256_mask_i32gather_pd(a, b, c, d, 2);
}

__m128d test_mm_mask_i64gather_pd(__m128d a, double const *b, __m128i c,
                                  __m128d d) {
  // CHECK: @llvm.x86.avx2.gather.q.pd
  // CHECK-ASM: vgatherqpd %xmm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %xmm{{.*}}
  return _mm_mask_i64gather_pd(a, b, c, d, 2);
}

__m256d test_mm256_mask_i64gather_pd(__m256d a, double const *b, __m256i c,
                                      __m256d d) {
  // CHECK: @llvm.x86.avx2.gather.q.pd.256
  // CHECK-ASM: vgatherqpd %ymm{{.*}}, (%r{{.*}},%ymm{{.*}},2), %ymm{{.*}}
  return _mm256_mask_i64gather_pd(a, b, c, d, 2);
}

__m128 test_mm_mask_i32gather_ps(__m128 a, float const *b, __m128i c,
                                 __m128 d) {
  // CHECK: @llvm.x86.avx2.gather.d.ps
  // CHECK-ASM: vgatherdps %xmm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %xmm{{.*}}
  return _mm_mask_i32gather_ps(a, b, c, d, 2);
}

__m256 test_mm256_mask_i32gather_ps(__m256 a, float const *b, __m256i c,
                                     __m256 d) {
  // CHECK: @llvm.x86.avx2.gather.d.ps.256
  // CHECK-ASM: vgatherdps %ymm{{.*}}, (%r{{.*}},%ymm{{.*}},2), %ymm{{.*}}
  return _mm256_mask_i32gather_ps(a, b, c, d, 2);
}

__m128 test_mm_mask_i64gather_ps(__m128 a, float const *b, __m128i c,
                                 __m128 d) {
  // CHECK: @llvm.x86.avx2.gather.q.ps
  // CHECK-ASM: vgatherqps %xmm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %xmm{{.*}}
  return _mm_mask_i64gather_ps(a, b, c, d, 2);
}

__m128 test_mm256_mask_i64gather_ps(__m128 a, float const *b, __m256i c,
                                    __m128 d) {
  // CHECK: @llvm.x86.avx2.gather.q.ps.256
  // CHECK-ASM: vgatherqps %xmm{{.*}}, (%r{{.*}},%ymm{{.*}},2), %xmm{{.*}}
  return _mm256_mask_i64gather_ps(a, b, c, d, 2);
}

__m128i test_mm_mask_i32gather_epi32(__m128i a, int const *b, __m128i c,
                                     __m128i d) {
  // CHECK: @llvm.x86.avx2.gather.d.d
  // CHECK-ASM: vpgatherdd %xmm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %xmm{{.*}}
  return _mm_mask_i32gather_epi32(a, b, c, d, 2);
}

__m256i test_mm256_mask_i32gather_epi32(__m256i a, int const *b, __m256i c,
                                        __m256i d) {
  // CHECK: @llvm.x86.avx2.gather.d.d.256
  // CHECK-ASM: vpgatherdd %ymm{{.*}}, (%r{{.*}},%ymm{{.*}},2), %ymm{{.*}}
  return _mm256_mask_i32gather_epi32(a, b, c, d, 2);
}

__m128i test_mm_mask_i64gather_epi32(__m128i a, int const *b, __m128i c,
                                     __m128i d) {
  // CHECK: @llvm.x86.avx2.gather.q.d
  // CHECK-ASM: vpgatherqd %xmm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %xmm{{.*}}
  return _mm_mask_i64gather_epi32(a, b, c, d, 2);
}

__m128i test_mm256_mask_i64gather_epi32(__m128i a, int const *b, __m256i c,
                                        __m128i d) {
  // CHECK: @llvm.x86.avx2.gather.q.d.256
  // CHECK-ASM: vpgatherqd %xmm{{.*}}, (%r{{.*}},%ymm{{.*}},2), %xmm{{.*}}
  return _mm256_mask_i64gather_epi32(a, b, c, d, 2);
}

__m128i test_mm_mask_i32gather_epi64(__m128i a, long long const *b, __m128i c,
                                     __m128i d) {
  // CHECK: @llvm.x86.avx2.gather.d.q
  // CHECK-ASM: vpgatherdq %xmm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %xmm{{.*}}
  return _mm_mask_i32gather_epi64(a, b, c, d, 2);
}

__m256i test_mm256_mask_i32gather_epi64(__m256i a, long long const *b, __m128i c,
                                        __m256i d) {
  // CHECK: @llvm.x86.avx2.gather.d.q.256
  // CHECK-ASM: vpgatherdq %ymm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %ymm{{.*}}
  return _mm256_mask_i32gather_epi64(a, b, c, d, 2);
}

__m128i test_mm_mask_i64gather_epi64(__m128i a, long long const *b, __m128i c,
                                     __m128i d) {
  // CHECK: @llvm.x86.avx2.gather.q.q
  // CHECK-ASM: vpgatherqq %xmm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %xmm{{.*}}
  return _mm_mask_i64gather_epi64(a, b, c, d, 2);
}

__m256i test_mm256_mask_i64gather_epi64(__m256i a, long long const *b, __m256i c,
                                        __m256i d) {
  // CHECK: @llvm.x86.avx2.gather.q.q.256
  // CHECK-ASM: vpgatherqq %ymm{{.*}}, (%r{{.*}},%ymm{{.*}},2), %ymm{{.*}}
  return _mm256_mask_i64gather_epi64(a, b, c, d, 2);
}

__m128d test_mm_i32gather_pd(double const *b, __m128i c) {
  // CHECK: @llvm.x86.avx2.gather.d.pd
  // CHECK-ASM: vgatherdpd %xmm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %xmm{{.*}}
  return _mm_i32gather_pd(b, c, 2);
}

__m256d test_mm256_i32gather_pd(double const *b, __m128i c) {
  // CHECK: @llvm.x86.avx2.gather.d.pd.256
  // CHECK-ASM: vgatherdpd %ymm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %ymm{{.*}}
  return _mm256_i32gather_pd(b, c, 2);
}

__m128d test_mm_i64gather_pd(double const *b, __m128i c) {
  // CHECK: @llvm.x86.avx2.gather.q.pd
  // CHECK-ASM: vgatherqpd %xmm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %xmm{{.*}}
  return _mm_i64gather_pd(b, c, 2);
}

__m256d test_mm256_i64gather_pd(double const *b, __m256i c) {
  // CHECK: @llvm.x86.avx2.gather.q.pd.256
  // CHECK-ASM: vgatherqpd %ymm{{.*}}, (%r{{.*}},%ymm{{.*}},2), %ymm{{.*}}
  return _mm256_i64gather_pd(b, c, 2);
}

__m128 test_mm_i32gather_ps(float const *b, __m128i c) {
  // CHECK: @llvm.x86.avx2.gather.d.ps
  // CHECK-ASM: vgatherdps %xmm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %xmm{{.*}}
  return _mm_i32gather_ps(b, c, 2);
}

__m256 test_mm256_i32gather_ps(float const *b, __m256i c) {
  // CHECK: @llvm.x86.avx2.gather.d.ps.256
  // CHECK-ASM: vgatherdps %ymm{{.*}}, (%r{{.*}},%ymm{{.*}},2), %ymm{{.*}}
  return _mm256_i32gather_ps(b, c, 2);
}

__m128 test_mm_i64gather_ps(float const *b, __m128i c) {
  // CHECK: @llvm.x86.avx2.gather.q.ps
  // CHECK-ASM: vgatherqps %xmm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %xmm{{.*}}
  return _mm_i64gather_ps(b, c, 2);
}

__m128 test_mm256_i64gather_ps(float const *b, __m256i c) {
  // CHECK: @llvm.x86.avx2.gather.q.ps.256
  // CHECK-ASM: vgatherqps %xmm{{.*}}, (%r{{.*}},%ymm{{.*}},2), %xmm{{.*}}
  return _mm256_i64gather_ps(b, c, 2);
}

__m128i test_mm_i32gather_epi32(int const *b, __m128i c) {
  // CHECK: @llvm.x86.avx2.gather.d.d
  // CHECK-ASM: vpgatherdd %xmm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %xmm{{.*}}
  return _mm_i32gather_epi32(b, c, 2);
}

__m256i test_mm256_i32gather_epi32(int const *b, __m256i c) {
  // CHECK: @llvm.x86.avx2.gather.d.d.256
  // CHECK-ASM: vpgatherdd %ymm{{.*}}, (%r{{.*}},%ymm{{.*}},2), %ymm{{.*}}
  return _mm256_i32gather_epi32(b, c, 2);
}

__m128i test_mm_i64gather_epi32(int const *b, __m128i c) {
  // CHECK: @llvm.x86.avx2.gather.q.d
  // CHECK-ASM: vpgatherqd %xmm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %xmm{{.*}}
  return _mm_i64gather_epi32(b, c, 2);
}

__m128i test_mm256_i64gather_epi32(int const *b, __m256i c) {
  // CHECK: @llvm.x86.avx2.gather.q.d.256
  // CHECK-ASM: vpgatherqd %xmm{{.*}}, (%r{{.*}},%ymm{{.*}},2), %xmm{{.*}}
  return _mm256_i64gather_epi32(b, c, 2);
}

__m128i test_mm_i32gather_epi64(long long const *b, __m128i c) {
  // CHECK: @llvm.x86.avx2.gather.d.q
  // CHECK-ASM: vpgatherdq %xmm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %xmm{{.*}}
  return _mm_i32gather_epi64(b, c, 2);
}

__m256i test_mm256_i32gather_epi64(long long const *b, __m128i c) {
  // CHECK: @llvm.x86.avx2.gather.d.q.256
  // CHECK-ASM: vpgatherdq %ymm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %ymm{{.*}}
  return _mm256_i32gather_epi64(b, c, 2);
}

__m128i test_mm_i64gather_epi64(long long const *b, __m128i c) {
  // CHECK: @llvm.x86.avx2.gather.q.q
  // CHECK-ASM: vpgatherqq %xmm{{.*}}, (%r{{.*}},%xmm{{.*}},2), %xmm{{.*}}
  return _mm_i64gather_epi64(b, c, 2);
}

__m256i test_mm256_i64gather_epi64(long long const *b, __m256i c) {
  // CHECK: @llvm.x86.avx2.gather.q.q.256
  // CHECK-ASM: vpgatherqq %ymm{{.*}}, (%r{{.*}},%ymm{{.*}},2), %ymm{{.*}}
  return _mm256_i64gather_epi64(b, c, 2);
}
