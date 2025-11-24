// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -emit-cir -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -emit-cir -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fclangir -emit-llvm -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -fclangir -emit-llvm -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -emit-cir -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -emit-cir -o %t.cir
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fclangir -emit-llvm -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -fclangir -emit-llvm -o %t.ll
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +xop -fno-signed-char -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG

#include <x86intrin.h>

// This test mimics clang/test/CodeGen/X86/xop-builtins.c, which eventually
// CIR shall be able to support fully.

__m128i test_mm_roti_epi8(__m128i a) {
  // CIR-LABEL: test_mm_roti_epi8
  // CIR: {{%.*}} = cir.vec.splat {{%.*}} : !{{[us]}}8i, !cir.vector<16 x !{{[us]}}8i> 
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "fshl" {{.*}} : (!cir.vector<16 x !{{[su]}}8i>, !cir.vector<16 x !{{[su]}}8i>, !cir.vector<16 x !{{[su]}}8i>) -> !cir.vector<16 x !{{[su]}}8i> 
  // LLVM-LABEL: test_mm_roti_epi8
  // LLVM: %[[CASTED_VAR:.*]] = bitcast <2 x i64> {{%.*}} to <16 x i8>
  // LLVM: {{%.*}} = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %[[CASTED_VAR]], <16 x i8> %[[CASTED_VAR]], <16 x i8> splat (i8 1))
  // OGCG-LABEL: test_mm_roti_epi8
  // OGCG: %[[CASTED_VAR:.*]] = bitcast <2 x i64> {{%.*}} to <16 x i8>
  // OGCG: {{%.*}} = call <16 x i8> @llvm.fshl.v16i8(<16 x i8> %[[CASTED_VAR]], <16 x i8> %[[CASTED_VAR]], <16 x i8> splat (i8 1))
  return _mm_roti_epi8(a, 1);
 }
