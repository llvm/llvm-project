// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +amx-transpose \
// RUN: -target-feature +amx-bf16 -target-feature +amx-fp16 -target-feature +amx-complex \
// RUN: -target-feature +avx512f -emit-llvm -o - -Wall -Werror -pedantic -Wno-gnu-statement-expression| FileCheck %s

#include <immintrin.h>
#include <stddef.h>

void test_tile_2rpntlvwz0(const void *A, size_t B) {
  // CHECK-LABEL: @test_tile_2rpntlvwz0
  // CHECK: call void @llvm.x86.t2rpntlvwz0(i8 1, ptr %{{.*}}, i64 %{{.*}})
  _tile_2rpntlvwz0(1, A, B);
}

void test_tile_2rpntlvwz0t1(const void *A, size_t B) {
  // CHECK-LABEL: @test_tile_2rpntlvwz0t1
  // CHECK: call void @llvm.x86.t2rpntlvwz0t1(i8 1, ptr %{{.*}}, i64 %{{.*}})
  _tile_2rpntlvwz0t1(1, A, B);
}

void test_tile_2rpntlvwz1(const void *A, size_t B) {
  // CHECK-LABEL: @test_tile_2rpntlvwz1
  // CHECK: call void @llvm.x86.t2rpntlvwz1(i8 1, ptr %{{.*}}, i64 %{{.*}})
  _tile_2rpntlvwz1(1, A, B);
}

void test_tile_2rpntlvwz1t1(const void *A, size_t B) {
  // CHECK-LABEL: @test_tile_2rpntlvwz1t1
  // CHECK: call void @llvm.x86.t2rpntlvwz1t1(i8 1, ptr %{{.*}}, i64 %{{.*}})
  _tile_2rpntlvwz1t1(1, A, B);
}

void test_tile_transposed(void)
{
  // CHECK-LABEL: @test_tile_transposed
  // CHECK: call void @llvm.x86.ttransposed(i8 1, i8 2)
  _tile_transposed(1, 2);
}

void test_tile_tdpbf16ps(void)
{
  // CHECK-LABEL: @test_tile_tdpbf16ps
  // CHECK: call void @llvm.x86.ttdpbf16ps(i8 1, i8 2, i8 3)
  _tile_tdpbf16ps(1, 2, 3);
}

void test_tile_tdpfp16ps(void)
{
  // CHECK-LABEL: @test_tile_tdpfp16ps
  // CHECK: call void @llvm.x86.ttdpfp16ps(i8 4, i8 5, i8 6)
  _tile_tdpfp16ps(4, 5, 6);
}

void test_tile_tcmmimfp16ps(void) {
  // CHECK-LABEL: @test_tile_tcmmimfp16ps
  // CHECK: call void @llvm.x86.ttcmmimfp16ps(i8 1, i8 2, i8 3)
  _tile_tcmmimfp16ps(1, 2, 3);
}

void test_tile_tcmmrlfp16ps(void) {
  // CHECK-LABEL: @test_tile_tcmmrlfp16ps
  // CHECK: call void @llvm.x86.ttcmmrlfp16ps(i8 1, i8 2, i8 3)
  _tile_tcmmrlfp16ps(1, 2, 3);
}

void test_tile_conjtcmmimfp16ps(void) {
  // CHECK-LABEL: @test_tile_conjtcmmimfp16ps
  // CHECK: call void @llvm.x86.tconjtcmmimfp16ps(i8 1, i8 2, i8 3)
  _tile_conjtcmmimfp16ps(1, 2, 3);
}

void test_tile_conjtfp16(void) {
  // CHECK-LABEL: @test_tile_conjtfp16
  // CHECK: call void @llvm.x86.tconjtfp16(i8 1, i8 2)
  _tile_conjtfp16(1, 2);
}
