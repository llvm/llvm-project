// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +ssse3 -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR-CHECK --implicit-check-not=x86mmx --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +ssse3 -fno-signed-char -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR-CHECK --implicit-check-not=x86mmx --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +ssse3 -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefix=LLVM-CHECK --implicit-check-not=x86mmx --input-file=%t.ll %s
// RUN: %clang_cc1 -x c -flax-vector-conversions=none -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +ssse3 -fno-signed-char -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefix=LLVM-CHECK --implicit-check-not=x86mmx --input-file=%t.ll %s

// This test mimics clang/test/CodeGen/X86/mmx-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

int test_mm_extract_pi16(__m64 a) {

  // CIR-CHECK-LABEL: test_mm_extract_pi16
  // CIR-CHECK %{{.*}} = cir.vec.extract %{{.*}}[%{{.*}} : !u64i : !cir.vector<!s16i x 4>

  // LLVM-CHECK-LABEL: test_mm_extract_pi16
  // LLVM-CHECK: extractelement <4 x i16> %{{.*}}, i64 2
  return _mm_extract_pi16(a, 2);
}

__m64 test_mm_insert_pi16(__m64 a, int d) {

  // CIR-CHECK-LABEL: test_mm_insert_pi16
  // CIR-CHECK-LABEL: {{%.*}} = cir.vec.insert {{%.*}}, {{%.*}}[{{%.*}} : !u64i] : !cir.vector<!s16i x 4>

  // LLVM-CHECK-LABEL: test_mm_insert_pi16
  // LLVM-CHECK: insertelement <4 x i16>
  return _mm_insert_pi16(a, d, 2);
}
