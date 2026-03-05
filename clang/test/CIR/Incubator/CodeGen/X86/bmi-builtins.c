// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// This test mimics clang/test/CodeGen/X86/bmi-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

unsigned short test__tzcnt_u16(unsigned short __X) {
  // CIR-LABEL: __tzcnt_u16
  // LLVM-LABEL: __tzcnt_u16
  return __tzcnt_u16(__X);
  // CIR: {{%.*}} = cir.llvm.intrinsic "cttz" {{%.*}} : (!u16i, !cir.bool) -> !u16i
  // LLVM: i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
}

unsigned int test__tzcnt_u32(unsigned int __X) {
  // CIR-LABEL: __tzcnt_u32
  // LLVM-LABEL: __tzcnt_u32
  return __tzcnt_u32(__X);
  // CIR: {{%.*}} = cir.llvm.intrinsic "cttz" {{%.*}} : (!u32i, !cir.bool) -> !u32i
  // LLVM: i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
}

#ifdef __x86_64__
unsigned long long test__tzcnt_u64(unsigned long long __X) {
  // CIR-LABEL: __tzcnt_u64
  // LLVM-LABEL: __tzcnt_u64
  return __tzcnt_u64(__X);
  // CIR: {{%.*}} = cir.llvm.intrinsic "cttz" {{%.*}} : (!u64i, !cir.bool) -> !u64i
  // LLVM: i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
}
#endif
