// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK


#include <immintrin.h>
#include "builtin_test_helpers.h"

__m128d test_mm_round_pd(__m128d x) {
  // CHECK-LABEL: test_mm_round_pd
  // CHECK: %{{.*}} = call <2 x double> @llvm.experimental.constrained.nearbyint.v2f64(<2 x double> %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  return _mm_round_pd(x, 8);
}

__m128 test_mm_round_ps(__m128 x) {
  // CHECK-LABEL: test_mm_round_ps
  // CHECK: %{{.*}} = call <4 x float> @llvm.experimental.constrained.nearbyint.v4f32(<4 x float> %{{.*}}, metadata !"round.dynamic", metadata !"fpexcept.ignore")
  return _mm_round_ps(x, 8);
}

__m128d test_mm_round_sd(__m128d x, __m128d y) {
  // CHECK-LABEL: test_mm_round_sd
  // CHECK: %[[A:.*]] = extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: %[[B:.*]] = call double @llvm.experimental.constrained.nearbyint.f64(double %[[A:.*]], metadata !"round.dynamic", metadata !"fpexcept.ignore")
  // CHECK: %{{.*}} = insertelement <2 x double> %0, double %[[B:.*]], i32 0
  return _mm_round_sd(x, y, 8);
}

__m128 test_mm_round_ss(__m128 x, __m128 y) {
  // CHECK-LABEL: test_mm_round_ss
  // CHECK: %[[A:.*]] = extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: %[[B:.*]] = call float @llvm.experimental.constrained.nearbyint.f32(float %[[A:.*]], metadata !"round.dynamic", metadata !"fpexcept.ignore") 
  // CHECK: %{{.*}} = insertelement <4 x float> %0, float %[[B:.*]], i32 0
  return _mm_round_ss(x, y, 8);
}