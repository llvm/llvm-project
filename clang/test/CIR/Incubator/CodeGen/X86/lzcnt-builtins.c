// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-linux -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefix=LLVM --input-file=%t.ll %s

// This test mimics clang/test/CodeGen/X86/lzcnt-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

unsigned int test_lzcnt_u32(unsigned int __X)
{
  // CIR-LABEL: _lzcnt_u32
  // LLVM-LABEL: _lzcnt_u32
  return _lzcnt_u32(__X);
  // CIR: {{%.*}} = cir.llvm.intrinsic "ctlz" {{%.*}} : (!u32i, !cir.bool) -> !u32i
  // LLVM: @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
}

unsigned long long test__lzcnt_u64(unsigned long long __X)
{
  // CIR-LABEL: _lzcnt_u64
  // LLVM-LABEL: _lzcnt_u64
  return _lzcnt_u64(__X);
  // CIR: {{%.*}} = cir.llvm.intrinsic "ctlz" {{%.*}} : (!u64i, !cir.bool) -> !u64i
  // LLVM: @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
}
