// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512fp16 -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx512fp16 -fclangir -emit-llvm -o %t.ll  -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s


#include <immintrin.h>

__m128h test_mm_undefined_ph(void) {
  // CIR-LABEL: _mm_undefined_ph
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<!cir.double x 2>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<!cir.double x 2> -> !cir.vector<!cir.f16 x 8>
  // CIR: cir.return %{{.*}} : !cir.vector<!cir.f16 x 8>

  // LLVM-LABEL: @test_mm_undefined_ph
  // LLVM: store <8 x half> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM: %{{.*}} = load <8 x half>, ptr %[[A]], align 16
  // LLVM: ret <8 x half> %{{.*}}
    return _mm_undefined_ph();
}

__m256h test_mm256_undefined_ph(void) {
  // CIR-LABEL: _mm256_undefined_ph
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<!cir.double x 4>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<!cir.double x 4> -> !cir.vector<!cir.f16 x 16>
  // CIR: cir.return %{{.*}} : !cir.vector<!cir.f16 x 16>

  // LLVM-LABEL: @test_mm256_undefined_ph
  // LLVM: store <16 x half> zeroinitializer, ptr %[[A:.*]], align 32
  // LLVM: %{{.*}} = load <16 x half>, ptr %[[A]], align 32
  // LLVM: ret <16 x half> %{{.*}}
  return _mm256_undefined_ph();
}

__m512h test_mm512_undefined_ph(void) {
  // CIR-LABEL: _mm512_undefined_ph
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<!cir.double x 8>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<!cir.double x 8> -> !cir.vector<!cir.f16 x 32>
  // CIR: cir.return %{{.*}} : !cir.vector<!cir.f16 x 32>

  // LLVM-LABEL: @test_mm512_undefined_ph
  // LLVM: store <32 x half> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <32 x half>, ptr %[[A]], align 64
  // LLVM: ret <32 x half> %{{.*}}
  return _mm512_undefined_ph();
}

void test_mm_mask_store_sh(void *__P, __mmask8 __U, __m128h __A) {
  // CIR-LABEL: _mm_mask_store_sh
  // CIR: cir.llvm.intrinsic "masked.store" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.vector<!cir.f16 x 8>, !cir.ptr<!cir.vector<!cir.f16 x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>) -> !void

  // LLVM-LABEL: @test_mm_mask_store_sh
  // LLVM: call void @llvm.masked.store.v8f16.p0(<8 x half> %{{.*}}, ptr elementtype(<8 x half>) align 1 %{{.*}}, <8 x i1> %{{.*}})
  _mm_mask_store_sh(__P, __U, __A);
}

__m128h test_mm_mask_load_sh(__m128h __A, __mmask8 __U, const void *__W) {
  // CIR-LABEL: _mm_mask_load_sh
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.f16 x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.f16 x 8>) -> !cir.vector<!cir.f16 x 8>

  // LLVM-LABEL: @test_mm_mask_load_sh
  // LLVM: %{{.*}} = call <8 x half> @llvm.masked.load.v8f16.p0(ptr elementtype(<8 x half>) align 1 %{{.*}}, <8 x i1> %{{.*}}, <8 x half> %{{.*}})
  return _mm_mask_load_sh(__A, __U, __W);
}

__m128h test_mm_maskz_load_sh(__mmask8 __U, const void *__W) {
  // CIR-LABEL: _mm_maskz_load_sh
  // CIR: %{{.*}} = cir.llvm.intrinsic "masked.load" %{{.*}}, %{{.*}}, %{{.*}}, %{{.*}} : (!cir.ptr<!cir.vector<!cir.f16 x 8>>, !u32i, !cir.vector<!cir.int<s, 1> x 8>, !cir.vector<!cir.f16 x 8>) -> !cir.vector<!cir.f16 x 8>

  // LLVM-LABEL: @test_mm_maskz_load_sh
  // LLVM: %{{.*}} = call <8 x half> @llvm.masked.load.v8f16.p0(ptr elementtype(<8 x half>) align 1 %{{.*}}, <8 x i1> %{{.*}}, <8 x half> %{{.*}})
  return _mm_maskz_load_sh(__U, __W);
}
