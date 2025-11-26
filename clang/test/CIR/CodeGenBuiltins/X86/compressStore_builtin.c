// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-cir -o %t.cir -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -fclangir -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512f -target-feature +avx512vl -emit-llvm -o %t.ll -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=OGCG --input-file=%t.ll %s

#include <immintrin.h>

void test_compress_store(void *__P, __mmask8 __U, __m128d __A) {
  // CIR-LABEL: test_compress_store
  // CIR: cir.call_llvm_intrinsic "masked_compressstore"
  // LLVM-LABEL: @test_compress_store
  // LLVM: @llvm.x86.avx512.mask.compress.store
  // OGCG-LABEL: @test_compress_store
  // OGCG: @llvm.x86.avx512.mask.compress.store
  return _mm_mask_compressstoreu_pd(__P, __U, __A);
}