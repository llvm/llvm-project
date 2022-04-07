// RUN: %clang_cc1 -no-opaque-pointers %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +invpcid -emit-llvm -o - -Wall -Werror -pedantic | FileCheck %s
// RUN: %clang_cc1 -no-opaque-pointers %s -ffreestanding -triple=i386-unknown-unknown -target-feature +invpcid -emit-llvm -o - -Wall -Werror -pedantic | FileCheck %s

#include <immintrin.h>

#include <stdint.h>

void test_invpcid(uint32_t type, void *descriptor) {
  //CHECK-LABEL: @test_invpcid
  //CHECK: call void @llvm.x86.invpcid(i32 %{{.*}}, i8* %{{.*}})
  _invpcid(type, descriptor);
}
