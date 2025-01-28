// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown  -target-feature +amx-fp8  \
// RUN: -emit-llvm -o - -Werror -pedantic | FileCheck %s
#include <immintrin.h>

void test_amx(void *data) {
  //CHECK-LABEL: @test_amx
  //CHECK: call void @llvm.x86.tdpbf8ps(i8 1, i8 2, i8 3)
  _tile_dpbf8ps(1, 2, 3);
}

void test_amx2(void *data) {
  //CHECK-LABEL: @test_amx2
  //CHECK: call void @llvm.x86.tdpbhf8ps(i8 1, i8 2, i8 3)
  _tile_dpbhf8ps(1, 2, 3);
}

void test_amx3(void *data) {
  //CHECK-LABEL: @test_amx3
  //CHECK: call void @llvm.x86.tdphbf8ps(i8 1, i8 2, i8 3)
  _tile_dphbf8ps(1, 2, 3);
}

void test_amx4(void *data) {
  //CHECK-LABEL: @test_amx4
  //CHECK: call void @llvm.x86.tdphf8ps(i8 1, i8 2, i8 3)
  _tile_dphf8ps(1, 2, 3);
}
