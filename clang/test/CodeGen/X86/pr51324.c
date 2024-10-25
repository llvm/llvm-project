// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <immintrin.h>

// Make sure brackets work after macro intrinsics.
float pr51324(__m128 a) {
  // CHECK-LABEL: pr51324
  // CHECK: call <4 x float> @llvm.x86.sse41.round.ps(<4 x float> %{{.*}}, i32 0)
  // CHECK: extractelement <4 x float> %{{.*}}, i32 0
  return _mm_round_ps(a, 0)[0];
}
