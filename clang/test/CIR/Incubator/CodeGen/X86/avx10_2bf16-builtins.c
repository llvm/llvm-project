// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2 -target-feature +avx10.2-256 -fclangir -emit-cir -o %t.cir -Wno-invalid-feature-combination -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2 -target-feature +avx10.2-256 -fclangir -emit-llvm -o %t.ll -Wno-invalid-feature-combination -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

#include <immintrin.h>

__m128bh test_mm_undefined_pbh(void) {
  // CIR-LABEL: _mm_undefined_pbh
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<!cir.double x 2>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<!cir.double x 2> -> !cir.vector<!cir.bf16 x 8>
  // CIR: cir.return %{{.*}} : !cir.vector<!cir.bf16 x 8>

  // LLVM-LABEL: @test_mm_undefined_pbh
  // LLVM: store <8 x bfloat> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM: %{{.*}} = load <8 x bfloat>, ptr %[[A]], align 16
  // LLVM: ret <8 x bfloat> %{{.*}}
  return _mm_undefined_pbh();
}

__m256bh test_mm256_undefined_pbh(void) {
  // CIR-LABEL: _mm256_undefined_pbh
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<!cir.double x 4>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<!cir.double x 4> -> !cir.vector<!cir.bf16 x 16>
  // CIR: cir.return %{{.*}} : !cir.vector<!cir.bf16 x 16>

  // LLVM-LABEL: @test_mm256_undefined_pbh
  // LLVM: store <16 x bfloat> zeroinitializer, ptr %[[A:.*]], align 32
  // LLVM: %{{.*}} = load <16 x bfloat>, ptr %[[A]], align 32
  // LLVM: ret <16 x bfloat> %{{.*}}
  return _mm256_undefined_pbh();
}

void test_mm_mask_store_sbh(void *__P, __mmask8 __U, __m128bh __A) {
  // CIR-LABEL: _mm_mask_store_sbh
  // CIR: cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.bf16 x 8>, !cir.ptr<!cir.vector<!cir.bf16 x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: @test_mm_mask_store_sbh
  // LLVM: call void @llvm.masked.store.v8bf16.p0(<8 x bfloat> %{{.*}}, ptr elementtype(<8 x bfloat>) align 1 %{{.*}}, <8 x i1> %{{.*}})
  _mm_mask_store_sbh(__P, __U, __A);
}

__m128bh test_mm_load_sbh(void const *A) {
  // CIR-LABEL: _mm_load_sbh
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.bf16 x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.bf16 x 8>) -> !cir.vector<!cir.bf16 x 8> 

  // LLVM-LABEL: @test_mm_load_sbh
  // NOTE: OG represents the mask using a bitcast from splat (i8 1), see IR-differences #1767
  // LLVM: %{{.*}} = call <8 x bfloat> @llvm.masked.load.v8bf16.p0(ptr elementtype(<8 x bfloat>) align 1 %{{.*}}, <8 x i1> <i1 true, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false, i1 false>, <8 x bfloat> %{{.*}})
  return _mm_load_sbh(A);
}

__m128bh test_mm_mask_load_sbh(__m128bh __A, __mmask8 __U, const void *__W) {
  // CIR-LABEL: _mm_mask_load_sbh
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.bf16 x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.bf16 x 8>) -> !cir.vector<!cir.bf16 x 8>

  // LLVM-LABEL: @test_mm_mask_load_sbh
  // LLVM: %{{.*}} = call <8 x bfloat> @llvm.masked.load.v8bf16.p0(ptr elementtype(<8 x bfloat>) align 1 %{{.*}}, <8 x i1> %{{.*}}, <8 x bfloat> %{{.*}})
  return _mm_mask_load_sbh(__A, __U, __W);
}
