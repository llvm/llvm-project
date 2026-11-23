// RUN: %clang_cc1 -ffreestanding -triple x86_64-unknown-unknown -target-feature +rdpid -emit-llvm -o - %s | FileCheck %s
// RUN: %if cir-enabled %{ %clang_cc1 -fclangir -ffreestanding -triple x86_64-unknown-unknown -target-feature +rdpid -emit-llvm -o - %s | FileCheck %s %}


#include <immintrin.h>

unsigned int test_rdpid_u32(void) {
// CHECK-LABEL: @test_rdpid_u32
// CHECK: call i32 @llvm.x86.rdpid
  return _rdpid_u32();
}
