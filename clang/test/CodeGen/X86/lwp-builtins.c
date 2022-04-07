// RUN: %clang_cc1 -no-opaque-pointers -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +lwp -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <x86intrin.h>

void test_llwpcb(void *ptr) {
  // CHECK-LABEL: @test_llwpcb
  // CHECK: call void @llvm.x86.llwpcb(i8* %{{.*}})
  __llwpcb(ptr);
}

void* test_slwpcb(void) {
  // CHECK-LABEL: @test_slwpcb
  // CHECK: call i8* @llvm.x86.slwpcb()
  return __slwpcb();
}

unsigned char test_lwpins32(unsigned val2, unsigned val1) {
  // CHECK-LABEL: @test_lwpins32
  // CHECK: call i8 @llvm.x86.lwpins32(i32
  return __lwpins32(val2, val1, 0x01234);
}

unsigned char test_lwpins64(unsigned long long val2, unsigned val1) {
  // CHECK-LABEL: @test_lwpins64
  // CHECK: call i8 @llvm.x86.lwpins64(i64
  return __lwpins64(val2, val1, 0x56789);
}

void test_lwpval32(unsigned val2, unsigned val1) {
  // CHECK-LABEL: @test_lwpval32
  // CHECK: call void @llvm.x86.lwpval32(i32
  __lwpval32(val2, val1, 0x01234);
}

void test_lwpval64(unsigned long long val2, unsigned val1) {
  // CHECK-LABEL: @test_lwpval64
  // CHECK: call void @llvm.x86.lwpval64(i64
  __lwpval64(val2, val1, 0xABCDEF);
}
