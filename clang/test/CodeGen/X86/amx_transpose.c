// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +amx-transpose \
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
