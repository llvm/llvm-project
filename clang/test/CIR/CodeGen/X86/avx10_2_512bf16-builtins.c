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
