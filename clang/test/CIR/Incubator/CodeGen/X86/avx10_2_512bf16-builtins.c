// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2-512 -fclangir -emit-cir -o %t.cir -Wno-invalid-feature-combination -Wall -Werror -Wsign-conversion 
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx10.2-512 -fclangir -emit-llvm -o %t.ll -Wno-invalid-feature-combination -Wall -Werror -Wsign-conversion
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

#include <immintrin.h>

__m512bh test_mm512_undefined_pbh(void) {

  // CIR-LABEL: _mm512_undefined_pbh
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<!cir.double x 8>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<!cir.double x 8> -> !cir.vector<!cir.bf16 x 32>
  // CIR: cir.return %{{.*}} : !cir.vector<!cir.bf16 x 32>

  // LLVM-LABEL: test_mm512_undefined_pbh
  // LLVM: store <32 x bfloat> zeroinitializer, ptr %[[A:.*]], align 64
  // LLVM: %{{.*}} = load <32 x bfloat>, ptr %[[A]], align 64
  // LLVM: ret <32 x bfloat> %{{.*}}
  return _mm512_undefined_pbh();
}
