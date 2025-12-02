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

// This test mimics clang/test/CodeGen/X86/lzcnt-builtins.c, which eventually
// CIR shall be able to support fully.

#include <immintrin.h>

unsigned int test__lzcnt16(unsigned short __X) {
  // CIR-LABEL: __lzcnt16
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "ctlz" {{%.*}} : (!u16i, !cir.bool) -> !u16i
  // LLVM-LABEL: __lzcnt16
  // LLVM: @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
  // OGCG-LABEL: __lzcnt16
  // OGCG: @llvm.ctlz.i16(i16 %{{.*}}, i1 false)
  return __lzcnt16(__X);
}

unsigned int test__lzcnt32(unsigned int __X) {
  // CIR-LABEL: __lzcnt32
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "ctlz" {{%.*}} : (!u32i, !cir.bool) -> !u32i
  // LLVM-LABEL: __lzcnt32
  // LLVM: @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
  // OGCG-LABEL: __lzcnt32
  // OGCG: @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
  return __lzcnt32(__X);
}

unsigned long long test__lzcnt64(unsigned long long __X) {
  // CIR-LABEL: __lzcnt64
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "ctlz" {{%.*}} : (!u64i, !cir.bool) -> !u64i
  // LLVM-LABEL: __lzcnt64
  // LLVM: @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
  // OGCG-LABEL: __lzcnt64
  // OGCG: @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
  return __lzcnt64(__X);
}

unsigned int test__lzcnt_u32(unsigned int __X) {
  // CIR-LABEL: _lzcnt_u32
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "ctlz" {{%.*}} : (!u32i, !cir.bool) -> !u32i
  // LLVM-LABEL: _lzcnt_u32
  // LLVM: @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
  // OGCG-LABEL: _lzcnt_u32
  // OGCG: @llvm.ctlz.i32(i32 %{{.*}}, i1 false)
  return _lzcnt_u32(__X);
}

unsigned long long test__lzcnt_u64(unsigned long long __X) {
  // CIR-LABEL: _lzcnt_u64
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "ctlz" {{%.*}} : (!u64i, !cir.bool) -> !u64i
  // LLVM-LABEL: _lzcnt_u64
  // LLVM: @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
  // OGCG-LABEL: _lzcnt_u64
  // OGCG: @llvm.ctlz.i64(i64 %{{.*}}, i1 false)
  return _lzcnt_u64(__X);
}
