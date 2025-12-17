// RUN: %clang_cc1 -x c -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=CHECK

// RUN: %clang_cc1 -x c -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK
// RUN: %clang_cc1 -x c++ -frounding-math -flax-vector-conversions=none -ffreestanding %s -triple=i386-apple-darwin -target-feature +sse4.1 -fno-signed-char -emit-llvm -o - -Wall -Werror  -fexperimental-new-constant-interpreter | FileCheck %s --check-prefixes=CHECK


#include <immintrin.h>

__m128d test_mm_round_pd_roundeven(__m128d x) {
  // CHECK-LABEL: test_mm_round_pd_roundeven
  // CHECK: %{{.*}} = call <2 x double> @llvm.experimental.constrained.roundeven.v2f64(<2 x double> %{{.*}}, metadata !"fpexcept.ignore")
  return _mm_round_pd(x, 0);
}

__m128 test_mm_round_ps_floor(__m128 x) {
  // CHECK-LABEL: test_mm_round_ps_floor
  // CHECK: %{{.*}} = call <4 x float> @llvm.experimental.constrained.floor.v4f32(<4 x float> %{{.*}}, metadata !"fpexcept.ignore")
  return _mm_round_ps(x, 1);
}

__m128d test_mm_round_sd_ceil(__m128d x, __m128d y) {
  // CHECK-LABEL: test_mm_round_sd_ceil
  // CHECK: %[[A:.*]] = extractelement <2 x double> %{{.*}}, i32 0
  // CHECK: %[[B:.*]] = call double @llvm.experimental.constrained.ceil.f64(double %[[A:.*]], metadata !"fpexcept.ignore")
  // CHECK: %{{.*}} = insertelement <2 x double> %0, double %[[B:.*]], i32 0
  return _mm_round_sd(x, y, 2);
}

__m128 test_mm_round_ss_trunc(__m128 x, __m128 y) {
  // CHECK-LABEL: test_mm_round_ss_trunc
  // CHECK: %[[A:.*]] = extractelement <4 x float> %{{.*}}, i32 0
  // CHECK: %[[B:.*]] = call float @llvm.experimental.constrained.trunc.f32(float %[[A:.*]], metadata !"fpexcept.ignore") 
  // CHECK: %{{.*}} = insertelement <4 x float> %0, float %[[B:.*]], i32 0
  return _mm_round_ss(x, y, 3);
}