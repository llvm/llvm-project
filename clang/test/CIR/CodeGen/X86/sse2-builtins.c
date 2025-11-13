// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR-CHECK,CIR-X64 --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefixes=CIR-CHECK,CIR-X64 --input-file=%t.cir %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM-CHECK,LLVM-X64 --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +sse2 -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM-CHECK,LLVM-X64 --input-file=%t.ll %s

// This test mimics clang/test/CodeGen/X86/sse2-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

__m128d test_mm_undefined_pd(void) {
  // CIR-X64-LABEL: _mm_undefined_pd
  // CIR-X64: %{{.*}} = cir.const #cir.zero : !cir.vector<2 x !cir.double>
  // CIR-X64: cir.return %{{.*}} : !cir.vector<2 x !cir.double>

  // LLVM-X64-LABEL: test_mm_undefined_pd
  // LLVM-X64: store <2 x double> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM-X64: %{{.*}} = load <2 x double>, ptr %[[A]], align 16
  // LLVM-X64: ret <2 x double> %{{.*}}
  return _mm_undefined_pd();
}

__m128i test_mm_undefined_si128(void) {
  // CIR-LABEL: _mm_undefined_si128
  // CIR-CHECK: %[[A:.*]] = cir.const #cir.zero : !cir.vector<2 x !cir.double>
  // CIR-CHECK: %{{.*}} = cir.cast bitcast %[[A]] : !cir.vector<2 x !cir.double> -> !cir.vector<2 x !s64i>
  // CIR-CHECK: cir.return %{{.*}} : !cir.vector<2 x !s64i>

  // LLVM-CHECK-LABEL: test_mm_undefined_si128
  // LLVM-CHECK: store <2 x i64> zeroinitializer, ptr %[[A:.*]], align 16
  // LLVM-CHECK: %{{.*}} = load <2 x i64>, ptr %[[A]], align 16
  // LLVM-CHECK: ret <2 x i64> %{{.*}}
  return _mm_undefined_si128();
}
