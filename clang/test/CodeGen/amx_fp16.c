// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown \
// RUN: -target-feature +amx-tile -target-feature +amx-int8 -target-feature +amx-bf16 -target-feature +amx-fp16 -emit-llvm -o - -Wall -Werror -pedantic \
// RUN: -Wno-gnu-statement-expression| FileCheck %s

#include <immintrin.h>
#include <stddef.h>
void test_tile_dpfp16ps(void) {
  // CHECK-LABEL: @test_tile_dpfp16ps
  // CHECK: call void @llvm.x86.tdpfp16ps(i8 1, i8 2, i8 3)
  _tile_dpfp16ps(1, 2, 3);
}
