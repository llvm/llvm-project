// RUN: %clang_cc1 %s -flax-vector-conversions=none -ffreestanding -triple=x86_64-unknown-unknown -target-feature +movrs \
// RUN: -emit-llvm -o - -Wall -Werror -pedantic -Wno-gnu-statement-expression | FileCheck %s

#include <immintrin.h>
#include <stddef.h>

char test_movrs_si8(const char * __A) {
  // CHECK-LABEL: @test_movrs_si8(
  // CHECK: call i8 @llvm.x86.movrsqi(
  return _movrs_i8(__A);
}

short test_movrs_si16(const short * __A) {
  // CHECK-LABEL: @test_movrs_si16(
  // CHECK: call i16 @llvm.x86.movrshi(
  return _movrs_i16(__A);
}

int test_movrs_si32(const int * __A) {
  // CHECK-LABEL: @test_movrs_si32(
  // CHECK: call i32 @llvm.x86.movrssi(
  return _movrs_i32(__A);
}

long long test_movrs_si64(const long long * __A) {
  // CHECK-LABEL: @test_movrs_si64(
  // CHECK: call i64 @llvm.x86.movrsdi(
  return _movrs_i64(__A);
}

void test_m_prefetch_rs(void *p) {
  _m_prefetchrs(p);
  // CHECK-LABEL: define{{.*}} void @test_m_prefetch_rs
  // CHECK: call void @llvm.x86.prefetchrs({{.*}})
}
