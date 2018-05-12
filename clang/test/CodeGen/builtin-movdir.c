// RUN: %clang_cc1 -ffreestanding -Wall -pedantic -triple x86_64-unknown-unknown -target-feature +movdiri -target-feature +movdir64b %s -emit-llvm -o - | FileCheck %s --check-prefix=X86_64 --check-prefix=CHECK
// RUN: %clang_cc1 -ffreestanding -Wall -pedantic -triple i386-unknown-unknown -target-feature +movdiri -target-feature +movdir64b %s -emit-llvm -o - | FileCheck %s --check-prefix=X86 --check-prefix=CHECK

#include <x86intrin.h>
#include <stdint.h>

void test_directstore32(void *dst, uint32_t value) {
  // CHECK-LABEL: test_directstore32
  // CHECK: call void @llvm.x86.directstore32
  _directstoreu_u32(dst, value);
}

#ifdef __x86_64__

void test_directstore64(void *dst, uint64_t value) {
  // X86_64-LABEL: test_directstore64
  // X86_64: call void @llvm.x86.directstore64
  _directstoreu_u64(dst, value);
}

#endif

void test_dir64b(void *dst, const void *src) {
  // CHECK-LABEL: test_dir64b
  // CHECK: [[PTRINT1:%.+]] = ptrtoint
  // X86: [[MASKEDPTR1:%.+]] = and i32 [[PTRINT1]], 63
  // X86: [[MASKCOND1:%.+]] = icmp eq i32 [[MASKEDPTR1]], 0
  // X86_64: [[MASKEDPTR1:%.+]] = and i64 [[PTRINT1]], 63
  // X86_64: [[MASKCOND1:%.+]] = icmp eq i64 [[MASKEDPTR1]], 0
  // CHECK: call void @llvm.x86.movdir64b
  _movdir64b(dst, src);
}

// CHECK: declare void @llvm.x86.directstore32(i8*, i32)
// X86_64: declare void @llvm.x86.directstore64(i8*, i64)
// CHECK: declare void @llvm.x86.movdir64b(i8*, i8*)
