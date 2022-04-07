// RUN: %clang_cc1 -no-opaque-pointers -ffreestanding -Wall -pedantic -triple x86_64-unknown-unknown -target-feature +movdiri -target-feature +movdir64b %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK,X86_64
// RUN: %clang_cc1 -no-opaque-pointers -ffreestanding -Wall -pedantic -triple i386-unknown-unknown -target-feature +movdiri -target-feature +movdir64b %s -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK

#include <immintrin.h>
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
  // CHECK: call void @llvm.x86.movdir64b
  _movdir64b(dst, src);
}

// CHECK: declare void @llvm.x86.directstore32(i8*, i32)
// X86_64: declare void @llvm.x86.directstore64(i8*, i64)
// CHECK: declare void @llvm.x86.movdir64b(i8*, i8*)
