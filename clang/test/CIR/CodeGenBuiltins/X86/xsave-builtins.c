// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +xsave -target-feature +xsaveopt -target-feature +xsavec -target-feature +xsaves -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +xsave -target-feature +xsaveopt -target-feature +xsavec -target-feature +xsaves -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +xsave -target-feature +xsaveopt -target-feature +xsavec -target-feature +xsaves -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG

void test_xsave(void *p, unsigned long long m) {
  // CIR-LABEL: test_xsave
  // CIR: cir.call_llvm_intrinsic "x86.xsave"

  // LLVM-LABEL: test_xsave
  // LLVM: call void @llvm.x86.xsave

  // OGCG-LABEL: test_xsave
  // OGCG: call void @llvm.x86.xsave
  __builtin_ia32_xsave(p, m);
}

void test_xsave64(void *p, unsigned long long m) {
  // CIR-LABEL: test_xsave64
  // CIR: cir.call_llvm_intrinsic "x86.xsave64"

  // LLVM-LABEL: test_xsave64
  // LLVM: call void @llvm.x86.xsave64

  // OGCG-LABEL: test_xsave64
  // OGCG: call void @llvm.x86.xsave64
  __builtin_ia32_xsave64(p, m);
}

void test_xrstor(void *p, unsigned long long m) {
  // CIR-LABEL: test_xrstor
  // CIR: cir.call_llvm_intrinsic "x86.xrstor"

  // LLVM-LABEL: test_xrstor
  // LLVM: call void @llvm.x86.xrstor

  // OGCG-LABEL: test_xrstor
  // OGCG: call void @llvm.x86.xrstor
  __builtin_ia32_xrstor(p, m);
}

void test_xrstor64(void *p, unsigned long long m) {
  // CIR-LABEL: test_xrstor64
  // CIR: cir.call_llvm_intrinsic "x86.xrstor64"

  // LLVM-LABEL: test_xrstor64
  // LLVM: call void @llvm.x86.xrstor64

  // OGCG-LABEL: test_xrstor64
  // OGCG: call void @llvm.x86.xrstor64
  __builtin_ia32_xrstor64(p, m);
}

void test_xsaveopt(void *p, unsigned long long m) {
  // CIR-LABEL: test_xsaveopt
  // CIR: cir.call_llvm_intrinsic "x86.xsaveopt"

  // LLVM-LABEL: test_xsaveopt
  // LLVM: call void @llvm.x86.xsaveopt

  // OGCG-LABEL: test_xsaveopt
  // OGCG: call void @llvm.x86.xsaveopt
  __builtin_ia32_xsaveopt(p, m);
}

void test_xsaveopt64(void *p, unsigned long long m) {
  // CIR-LABEL: test_xsaveopt64
  // CIR: cir.call_llvm_intrinsic "x86.xsaveopt64"

  // LLVM-LABEL: test_xsaveopt64
  // LLVM: call void @llvm.x86.xsaveopt64

  // OGCG-LABEL: test_xsaveopt64
  // OGCG: call void @llvm.x86.xsaveopt64
  __builtin_ia32_xsaveopt64(p, m);
}

void test_xsavec(void *p, unsigned long long m) {
  // CIR-LABEL: test_xsavec
  // CIR: cir.call_llvm_intrinsic "x86.xsavec"

  // LLVM-LABEL: test_xsavec
  // LLVM: call void @llvm.x86.xsavec

  // OGCG-LABEL: test_xsavec
  // OGCG: call void @llvm.x86.xsavec
  __builtin_ia32_xsavec(p, m);
}

void test_xsavec64(void *p, unsigned long long m) {
  // CIR-LABEL: test_xsavec64
  // CIR: cir.call_llvm_intrinsic "x86.xsavec64"

  // LLVM-LABEL: test_xsavec64
  // LLVM: call void @llvm.x86.xsavec64

  // OGCG-LABEL: test_xsavec64
  // OGCG: call void @llvm.x86.xsavec64
  __builtin_ia32_xsavec64(p, m);
}

void test_xsaves(void *p, unsigned long long m) {
  // CIR-LABEL: test_xsaves
  // CIR: cir.call_llvm_intrinsic "x86.xsaves"

  // LLVM-LABEL: test_xsaves
  // LLVM: call void @llvm.x86.xsaves

  // OGCG-LABEL: test_xsaves
  // OGCG: call void @llvm.x86.xsaves
  __builtin_ia32_xsaves(p, m);
}

void test_xsaves64(void *p, unsigned long long m) {
  // CIR-LABEL: test_xsaves64
  // CIR: cir.call_llvm_intrinsic "x86.xsaves64"

  // LLVM-LABEL: test_xsaves64
  // LLVM: call void @llvm.x86.xsaves64

  // OGCG-LABEL: test_xsaves64
  // OGCG: call void @llvm.x86.xsaves64
  __builtin_ia32_xsaves64(p, m);
}

void test_xrstors(void *p, unsigned long long m) {
  // CIR-LABEL: test_xrstors
  // CIR: cir.call_llvm_intrinsic "x86.xrstors"

  // LLVM-LABEL: test_xrstors
  // LLVM: call void @llvm.x86.xrstors

  // OGCG-LABEL: test_xrstors
  // OGCG: call void @llvm.x86.xrstors
  __builtin_ia32_xrstors(p, m);
}

void test_xrstors64(void *p, unsigned long long m) {
  // CIR-LABEL: test_xrstors64
  // CIR: cir.call_llvm_intrinsic "x86.xrstors64"

  // LLVM-LABEL: test_xrstors64
  // LLVM: call void @llvm.x86.xrstors64

  // OGCG-LABEL: test_xrstors64
  // OGCG: call void @llvm.x86.xrstors64
  __builtin_ia32_xrstors64(p, m);
}

unsigned long long test_xgetbv(unsigned int a) {
  // CIR-LABEL: test_xgetbv
  // CIR: cir.call_llvm_intrinsic "x86.xgetbv"

  // LLVM-LABEL: test_xgetbv
  // LLVM: call i64 @llvm.x86.xgetbv

  // OGCG-LABEL: test_xgetbv
  // OGCG: call i64 @llvm.x86.xgetbv
  return __builtin_ia32_xgetbv(a);
}

void test_xsetbv(unsigned int a, unsigned long long m) {
  // CIR-LABEL: test_xsetbv
  // CIR: cir.call_llvm_intrinsic "x86.xsetbv"

  // LLVM-LABEL: test_xsetbv
  // LLVM: call void @llvm.x86.xsetbv

  // OGCG-LABEL: test_xsetbv
  // OGCG: call void @llvm.x86.xsetbv
  __builtin_ia32_xsetbv(a, m);
}

