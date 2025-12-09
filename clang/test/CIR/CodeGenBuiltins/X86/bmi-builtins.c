// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -ffreestanding -triple=x86_64-unknown-linux -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefix=OGCG
// RUN: %clang_cc1 -x c++ -ffreestanding -triple=x86_64-unknown-linux -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefix=OGCG

// This test mimics clang/test/CodeGen/X86/bmi-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

unsigned short test__tzcnt_u16(unsigned short __X) {
  // CIR-LABEL: __tzcnt_u16
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "cttz" {{%.*}} : (!u16i, !cir.bool) -> !u16i
  // LLVM-LABEL: __tzcnt_u16
  // LLVM: i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
  // OGCG-LABEL: __tzcnt_u16
  // OGCG: i16 @llvm.cttz.i16(i16 %{{.*}}, i1 false)
  return __tzcnt_u16(__X);
}

unsigned int test__tzcnt_u32(unsigned int __X) {
  // CIR-LABEL: __tzcnt_u32
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "cttz" {{%.*}} : (!u32i, !cir.bool) -> !u32i
  // LLVM-LABEL: __tzcnt_u32
  // LLVM: i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
  // OGCG-LABEL: __tzcnt_u32
  // OGCG: i32 @llvm.cttz.i32(i32 %{{.*}}, i1 false)
  return __tzcnt_u32(__X);
}

#ifdef __x86_64__
unsigned long long test__tzcnt_u64(unsigned long long __X) {
  // CIR-LABEL: __tzcnt_u64
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "cttz" {{%.*}} : (!u64i, !cir.bool) -> !u64i
  // LLVM-LABEL: __tzcnt_u64
  // LLVM: i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
  // OGCG-LABEL: __tzcnt_u64
  // OGCG: i64 @llvm.cttz.i64(i64 %{{.*}}, i1 false)
  return __tzcnt_u64(__X);
}
#endif
