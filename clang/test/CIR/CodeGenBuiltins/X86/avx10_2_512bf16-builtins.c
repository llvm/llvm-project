// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2-512 -fclangir -emit-cir -o %t.cir -Wno-invalid-feature-combination -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2-512 -fclangir -emit-llvm -o %t.ll -Wno-invalid-feature-combination -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2 -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2 -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG

#include <immintrin.h>

__m512bh test_mm512_undefined_pbh(void) {
  // CIR-LABEL: _mm512_undefined_pbh
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<8 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<8 x !cir.double> -> !cir.vector<32 x !cir.bf16>
  // CIR: cir.return %{{.*}} : !cir.vector<32 x !cir.bf16>

  // CIR-LABEL: cir.func {{.*}}test_mm512_undefined_pbh
  // CIR: call @_mm512_undefined_pbh

  // LLVM-LABEL: test_mm512_undefined_pbh
  // LLVM: store <32 x bfloat> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <32 x bfloat>, ptr %[[A]], align 64
  // LLVM: ret <32 x bfloat> %{{.*}}

  // OGCG-LABEL: test_mm512_undefined_pbh
  // OGCG: ret <32 x bfloat> zeroinitializer
  return _mm512_undefined_pbh();
}

__mmask32 test_mm512_mask_fpclass_pbh_mask(__mmask32 __U, __m512bh __A) {
  // CIR-LABEL: _mm512_mask_fpclass_pbh_mask
  // CIR: %[[A:.*]] = cir.call_llvm_intrinsic "x86.avx10.fpclass.bf16.512"
  // CIR: %[[B:.*]] = cir.cast bitcast {{.*}} : !u32i -> !cir.vector<32 x !cir.int<s, 1>>
  // CIR: %[[C:.*]] = cir.binop(and, %[[A]], %[[B]]) : !cir.vector<32 x !cir.int<s, 1>>
  // CIR: cir.cast bitcast %[[C]] : !cir.vector<32 x !cir.int<s, 1>> -> !u32i

  // LLVM-LABEL: test_mm512_mask_fpclass_pbh_mask
  // LLVM: %[[A:.*]] = call <32 x i1> @llvm.x86.avx10.fpclass.bf16.512
  // LLVM: %[[B:.*]] = bitcast i32 {{.*}} to <32 x i1>
  // LLVM: %[[C:.*]] = and <32 x i1> %[[A]], %[[B]]
  // LLVM: bitcast <32 x i1> %[[C]] to i32

  // OGCG-LABEL: test_mm512_mask_fpclass_pbh_mask
  // OGCG: %[[A:.*]] = call <32 x i1> @llvm.x86.avx10.fpclass.bf16.512
  // OGCG: %[[B:.*]] = bitcast i32 {{.*}} to <32 x i1>
  // OGCG: %[[C:.*]] = and <32 x i1> %[[A]], %[[B]]
  // OGCG: bitcast <32 x i1> %[[C]] to i32
  return _mm512_mask_fpclass_pbh_mask(__U, __A, 4);
}

__mmask32 test_mm512_fpclass_pbh_mask(__m512bh __A) {
  // CIR-LABEL: _mm512_fpclass_pbh_mask
  // CIR: %[[A:.*]] = cir.call_llvm_intrinsic "x86.avx10.fpclass.bf16.512"
  // CIR: cir.cast bitcast %[[A]] : !cir.vector<32 x !cir.int<s, 1>> -> !u32i

  // LLVM-LABEL: test_mm512_fpclass_pbh_mask
  // LLVM: %[[A:.*]] = call <32 x i1> @llvm.x86.avx10.fpclass.bf16.512
  // LLVM: bitcast <32 x i1> %[[A]] to i32

  // OGCG-LABEL: test_mm512_fpclass_pbh_mask
  // OGCG: %[[A:.*]] = call <32 x i1> @llvm.x86.avx10.fpclass.bf16.512
  // OGCG: bitcast <32 x i1> %[[A]] to i32
  return _mm512_fpclass_pbh_mask(__A, 4);
}
