// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +xsave -target-feature +xsaveopt -target-feature +xsavec -target-feature +xsaves -fclangir -emit-cir -o %t.cir -Wall -Werror
// RUN: FileCheck --check-prefix=CIR --input-file=%t.cir %s
// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +xsave -target-feature +xsaveopt -target-feature +xsavec -target-feature +xsaves -fclangir -emit-llvm -o %t.ll -Wall -Werror
// RUN: FileCheck --check-prefixes=LLVM --input-file=%t.ll %s

// RUN: %clang_cc1 -x c -ffreestanding %s -triple=x86_64-unknown-linux -target-feature +xsave -target-feature +xsaveopt -target-feature +xsavec -target-feature +xsaves -emit-llvm -o - -Wall -Werror | FileCheck %s -check-prefix=OGCG

void test_xsave(void *p, unsigned long long m) {
  // CIR-LABEL: test_xsave
  // CIR: [[P:%.*]] = cir.load {{.*}} : !cir.ptr<!cir.ptr<!void>>, !cir.ptr<!void>
  // CIR: [[M:%.*]] = cir.load {{.*}} : !cir.ptr<!u64i>, !u64i
  // CIR: [[CONST:%.*]] = cir.const #cir.int<32> : !s64i
  // CIR: [[SHIFT:%.*]] = cir.shift(right, [[M]] : !u64i, [[CONST]] : !s64i) -> !u64i
  // CIR: [[CAST1:%.*]] = cir.cast integral [[SHIFT]] : !u64i -> !s32i
  // CIR: [[CAST2:%.*]] = cir.cast integral [[M]] : !u64i -> !s32i
  // CIR: cir.call_llvm_intrinsic "x86.xsave" [[P]], [[CAST1]], [[CAST2]]

  // LLVM-LABEL: test_xsave
  // LLVM: [[LP:%.*]] = load ptr, ptr
  // LLVM: [[LM:%.*]] = load i64, ptr
  // LLVM: [[LSHIFT:%.*]] = lshr i64 [[LM]], 32
  // LLVM: [[LCAST1:%.*]] = trunc i64 [[LSHIFT]] to i32
  // LLVM: [[LCAST2:%.*]] = trunc i64 [[LM]] to i32
  // LLVM: call void @llvm.x86.xsave(ptr [[LP]], i32 [[LCAST1]], i32 [[LCAST2]])

  // OGCG-LABEL: test_xsave
  // OGCG: [[OP:%.*]] = load ptr, ptr
  // OGCG: [[OM:%.*]] = load i64, ptr
  // OGCG: [[OSHIFT:%.*]] = lshr i64 [[OM]], 32
  // OGCG: [[OCAST1:%.*]] = trunc i64 [[OSHIFT]] to i32
  // OGCG: [[OCAST2:%.*]] = trunc i64 [[OM]] to i32
  // OGCG: call void @llvm.x86.xsave(ptr [[OP]], i32 [[OCAST1]], i32 [[OCAST2]])
  __builtin_ia32_xsave(p, m);
}

// The following tests use the same pattern as test_xsave (load, shift, cast, cast, intrinsic call).
// Only the intrinsic name differs, so we just check the intrinsic call.

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

