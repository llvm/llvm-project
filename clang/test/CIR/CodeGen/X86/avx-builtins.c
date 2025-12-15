// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +avx -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +avx -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s --check-prefixes=OGCG

// This test mimics clang/test/CodeGen/X86/avx-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

__m256 test_mm256_undefined_ps(void) {
  // CIR-LABEL: _mm256_undefined_ps
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<4 x !cir.double> -> !cir.vector<8 x !cir.float>
  // CIR: cir.return %{{.*}} : !cir.vector<8 x !cir.float>

  // LLVM-LABEL: test_mm256_undefined_ps
  // LLVM: store <8 x float> zeroinitializer, ptr %[[A:.*]], align 32
  // LLVM: %{{.*}} = load <8 x float>, ptr %[[A]], align 32
  // LLVM: ret <8 x float> %{{.*}}

  // OGCG-LABEL: test_mm256_undefined_ps
  // OGCG: ret <8 x float> zeroinitializer
  return _mm256_undefined_ps();
}

__m256d test_mm256_undefined_pd(void) {
  // CIR-LABEL: _mm256_undefined_pd
  // CIR: %{{.*}} = cir.const #cir.zero : !cir.vector<4 x !cir.double>
  // CIR: cir.return %{{.*}} : !cir.vector<4 x !cir.double>

  // LLVM-LABEL: test_mm256_undefined_pd
  // LLVM: store <4 x double> zeroinitializer, ptr %[[A:.*]], align 32
  // LLVM: %{{.*}} = load <4 x double>, ptr %[[A]], align 32
  // LLVM: ret <4 x double> %{{.*}}

  // OGCG-LABEL: test_mm256_undefined_pd
  // OGCG: ret <4 x double> zeroinitializer
  return _mm256_undefined_pd();
}

__m256i test_mm256_undefined_si256(void) {
  // CIR-LABEL: _mm256_undefined_si256
  // CIR: %[[A:.*]] = cir.const #cir.zero : !cir.vector<4 x !cir.double>
  // CIR: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<4 x !cir.double> -> !cir.vector<4 x !s64i>
  // CIR: cir.return %{{.*}} : !cir.vector<4 x !s64i>

  // LLVM-LABEL: test_mm256_undefined_si256
  // LLVM: store <4 x i64> zeroinitializer, ptr %[[A:.*]], align 32
  // LLVM: %{{.*}} = load <4 x i64>, ptr %[[A]], align 32
  // LLVM: ret <4 x i64> %{{.*}}

  // OGCG-LABEL: test_mm256_undefined_si256
  // OGCG: ret <4 x i64> zeroinitializer
  return _mm256_undefined_si256();
}