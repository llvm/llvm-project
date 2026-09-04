// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -target-feature +rdrnd -target-feature +rdseed -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefixes=CIR,CIR-X64 --input-file=%t.cir %s
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64-unknown-linux -target-feature +rdrnd -target-feature +rdseed -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefixes=CIR,CIR-X64 --input-file=%t.cir %s

// RUN: %clang_cc1 -x c -ffreestanding -triple x86_64-unknown-linux -target-feature +rdrnd -target-feature +rdseed -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefixes=LLVM,LLVM-X64 --input-file=%t.ll %s
// RUN: %clang_cc1 -x c++ -ffreestanding -triple x86_64-unknown-linux -target-feature +rdrnd -target-feature +rdseed -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefixes=LLVM,LLVM-X64 --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -ffreestanding -triple=x86_64-unknown-linux -target-feature +rdrnd -target-feature +rdseed -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefixes=OGCG,OGCG-X64
// RUN: %clang_cc1 -x c++ -ffreestanding -triple=x86_64-unknown-linux -target-feature +rdrnd -target-feature +rdseed -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefixes=OGCG,OGCG-X64

// 32-bit tests for _rdrand64_step()
// RUN: %clang_cc1 -x c -ffreestanding -triple i386-unknown-linux -target-feature +rdrnd -target-feature +rdseed -Wno-implicit-function-declaration -fclangir -emit-cir -o %t.cir %s
// RUN: FileCheck --check-prefixes=CIR,CIR-X86 --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -ffreestanding -triple i386-unknown-linux -target-feature +rdrnd -target-feature +rdseed -Wno-implicit-function-declaration -fclangir -emit-llvm -o %t.ll %s
// RUN: FileCheck --check-prefixes=LLVM,LLVM-X86 --input-file=%t.ll %s
// RUN: %clang_cc1 -x c -ffreestanding -triple=i386-unknown-linux -target-feature +rdrnd -target-feature +rdseed -emit-llvm -Wall -Werror %s -o - | FileCheck %s -check-prefixes=OGCG,OGCG-X86

// This test mimics clang/test/CodeGen/X86/rdrand-builtins.c

#include <immintrin.h>

int test_rdrand16(unsigned short *p) {
  // CIR-LABEL: rdrand16
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "x86.rdrand.16"
  // CIR: {{%.*}} = cir.extract_member {{%.*}}[0]
  // CIR: cir.store {{%.*}}, {{%.*}} : !u16i, !cir.ptr<!u16i>
  // CIR: {{%.*}} = cir.extract_member {{%.*}}[1]
  // LLVM-LABEL: rdrand16
  // LLVM: call { i16, i32 } @llvm.x86.rdrand.16
  // LLVM: extractvalue { i16, i32 } {{%.*}}, 0
  // LLVM: store i16 {{%.*}}, ptr {{%.*}}, align 2
  // LLVM: extractvalue { i16, i32 } {{%.*}}, 1
  // OGCG-LABEL: rdrand16
  // OGCG: call { i16, i32 } @llvm.x86.rdrand.16
  // OGCG: extractvalue { i16, i32 } {{%.*}}, 0
  // OGCG: store i16 {{%.*}}, ptr {{%.*}}, align 2
  // OGCG: extractvalue { i16, i32 } {{%.*}}, 1
  return _rdrand16_step(p);
}

int test_rdrand32(unsigned *p) {
  // CIR-LABEL: rdrand32
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "x86.rdrand.32"
  // CIR: {{%.*}} = cir.extract_member {{%.*}}[0]
  // CIR: cir.store {{%.*}}, {{%.*}} : !u32i, !cir.ptr<!u32i>
  // CIR: {{%.*}} = cir.extract_member {{%.*}}[1]
  // LLVM-LABEL: rdrand32
  // LLVM: call { i32, i32 } @llvm.x86.rdrand.32
  // LLVM: extractvalue { i32, i32 } {{%.*}}, 0
  // LLVM: store i32 {{%.*}}, ptr {{%.*}}, align 4
  // LLVM: extractvalue { i32, i32 } {{%.*}}, 1
  // OGCG-LABEL: rdrand32
  // OGCG: call { i32, i32 } @llvm.x86.rdrand.32
  // OGCG: extractvalue { i32, i32 } {{%.*}}, 0
  // OGCG: store i32 {{%.*}}, ptr {{%.*}}, align 4
  // OGCG: extractvalue { i32, i32 } {{%.*}}, 1
  return _rdrand32_step(p);
}

int test_rdrand64(unsigned long long *p) {
  // CIR-LABEL: rdrand64
  // CIR-X64: {{%.*}} = cir.call_llvm_intrinsic "x86.rdrand.64"
  // CIR-X64: {{%.*}} = cir.extract_member {{%.*}}[0]
  // CIR-X64: cir.store {{%.*}}, {{%.*}} : !u64i, !cir.ptr<!u64i>
  // CIR-X64: {{%.*}} = cir.extract_member {{%.*}}[1]
  // LLVM-LABEL: rdrand64
  // LLVM-X64: call { i64, i32 } @llvm.x86.rdrand.64
  // LLVM-X64: extractvalue { i64, i32 } {{%.*}}, 0
  // LLVM-X64: store i64 {{%.*}}, ptr {{%.*}}, align 8
  // LLVM-X64: extractvalue { i64, i32 } {{%.*}}, 1
  // OGCG-LABEL: rdrand64
  // OGCG-X64: call { i64, i32 } @llvm.x86.rdrand.64
  // OGCG-X64: extractvalue { i64, i32 } {{%.*}}, 0
  // OGCG-X64: store i64 {{%.*}}, ptr {{%.*}}, align 8
  // OGCG-X64: extractvalue { i64, i32 } {{%.*}}, 1

  // CIR-X86: {{%.*}} = cir.call_llvm_intrinsic "x86.rdrand.32"
  // CIR-X86: {{%.*}} = cir.call_llvm_intrinsic "x86.rdrand.32"
  // LLVM-X86: call { i32, i32 } @llvm.x86.rdrand.32
  // LLVM-X86: call { i32, i32 } @llvm.x86.rdrand.32
  // OGCG-X86: call { i32, i32 } @llvm.x86.rdrand.32
  // OGCG-X86: call { i32, i32 } @llvm.x86.rdrand.32
  return _rdrand64_step(p);
}

int test_rdseed16(unsigned short *p) {
  // CIR-LABEL: rdseed16
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "x86.rdseed.16"
  // CIR: {{%.*}} = cir.extract_member {{%.*}}[0]
  // CIR: cir.store {{%.*}}, {{%.*}} : !u16i, !cir.ptr<!u16i>
  // CIR: {{%.*}} = cir.extract_member {{%.*}}[1]
  // LLVM-LABEL: rdseed16
  // LLVM: call { i16, i32 } @llvm.x86.rdseed.16
  // LLVM: extractvalue { i16, i32 } {{%.*}}, 0
  // LLVM: store i16 {{%.*}}, ptr {{%.*}}, align 2
  // LLVM: extractvalue { i16, i32 } {{%.*}}, 1
  // OGCG-LABEL: rdseed16
  // OGCG: call { i16, i32 } @llvm.x86.rdseed.16
  // OGCG: extractvalue { i16, i32 } {{%.*}}, 0
  // OGCG: store i16 {{%.*}}, ptr {{%.*}}, align 2
  // OGCG: extractvalue { i16, i32 } {{%.*}}, 1
  return _rdseed16_step(p);
}

int test_rdseed32(unsigned *p) {
  // CIR-LABEL: rdseed32
  // CIR: {{%.*}} = cir.call_llvm_intrinsic "x86.rdseed.32"
  // CIR: {{%.*}} = cir.extract_member {{%.*}}[0]
  // CIR: cir.store {{%.*}}, {{%.*}} : !u32i, !cir.ptr<!u32i>
  // CIR: {{%.*}} = cir.extract_member {{%.*}}[1]
  // LLVM-LABEL: rdseed32
  // LLVM: call { i32, i32 } @llvm.x86.rdseed.32
  // LLVM: extractvalue { i32, i32 } {{%.*}}, 0
  // LLVM: store i32 {{%.*}}, ptr {{%.*}}, align 4
  // LLVM: extractvalue { i32, i32 } {{%.*}}, 1
  // OGCG-LABEL: rdseed32
  // OGCG: call { i32, i32 } @llvm.x86.rdseed.32
  // OGCG: extractvalue { i32, i32 } {{%.*}}, 0
  // OGCG: store i32 {{%.*}}, ptr {{%.*}}, align 4
  // OGCG: extractvalue { i32, i32 } {{%.*}}, 1
  return _rdseed32_step(p);
}

#if __x86_64__
int test_rdseed64(unsigned long long *p) {
  // CIR-X64-LABEL: rdseed64
  // CIR-X64: {{%.*}} = cir.call_llvm_intrinsic "x86.rdseed.64"
  // CIR-X64: {{%.*}} = cir.extract_member {{%.*}}[0]
  // CIR-X64: cir.store {{%.*}}, {{%.*}} : !u64i, !cir.ptr<!u64i>
  // CIR-X64: {{%.*}} = cir.extract_member {{%.*}}[1]
  // LLVM-X64-LABEL: rdseed64
  // LLVM-X64: call { i64, i32 } @llvm.x86.rdseed.64
  // LLVM-X64: extractvalue { i64, i32 } {{%.*}}, 0
  // LLVM-X64: store i64 {{%.*}}, ptr {{%.*}}, align 8
  // LLVM-X64: extractvalue { i64, i32 } {{%.*}}, 1
  // OGCG-X64-LABEL: rdseed64
  // OGCG-X64: call { i64, i32 } @llvm.x86.rdseed.64
  // OGCG-X64: extractvalue { i64, i32 } {{%.*}}, 0
  // OGCG-X64: store i64 {{%.*}}, ptr {{%.*}}, align 8
  // OGCG-X64: extractvalue { i64, i32 } {{%.*}}, 1
  return _rdseed64_step(p);
}
#endif
