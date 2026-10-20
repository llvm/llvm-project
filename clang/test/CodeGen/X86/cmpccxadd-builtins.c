// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +cmpccxadd \
// RUN: -emit-llvm -o - -Wall -Werror -pedantic -Wno-gnu-statement-expression | FileCheck %s

#include <immintrin.h>
#include <stddef.h>

int test_cmpbexadd32(void *__A, int __B, int __C) {
  // CHECK-LABEL: @test_cmpbexadd32(
  // CHECK: call i32 @llvm.x86.cmpccxadd32(ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 0)
  return _cmpccxadd_epi32(__A, __B, __C, _CMPCCX_O);
}

long long test_cmpbexadd64(void *__A, long long __B, long long __C) {
  // CHECK-LABEL: @test_cmpbexadd64(
  // CHECK: call i64 @llvm.x86.cmpccxadd64(ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 0)
  return _cmpccxadd_epi64(__A, __B, __C, _CMPCCX_O);
}

int test_cmpbxadd32(void *__A, int __B, int __C) {
  // CHECK-LABEL: @test_cmpbxadd32(
  // CHECK: call i32 @llvm.x86.cmpccxadd32(ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 1)
  return _cmpccxadd_epi32(__A, __B, __C, _CMPCCX_NO);
}

long long test_cmpbxadd64(void *__A, long long __B, long long __C) {
  // CHECK-LABEL: @test_cmpbxadd64(
  // CHECK: call i64 @llvm.x86.cmpccxadd64(ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 1)
  return _cmpccxadd_epi64(__A, __B, __C, _CMPCCX_NO);
}

int test_cmplexadd32(void *__A, int __B, int __C) {
  // CHECK-LABEL: @test_cmplexadd32(
  // CHECK: call i32 @llvm.x86.cmpccxadd32(ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 2)
  return _cmpccxadd_epi32(__A, __B, __C, _CMPCCX_B);
}

long long test_cmplexadd64(void *__A, long long __B, long long __C) {
  // CHECK-LABEL: @test_cmplexadd64(
  // CHECK: call i64 @llvm.x86.cmpccxadd64(ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 2)
  return _cmpccxadd_epi64(__A, __B, __C, _CMPCCX_B);
}

int test_cmplxadd32(void *__A, int __B, int __C) {
  // CHECK-LABEL: @test_cmplxadd32(
  // CHECK: call i32 @llvm.x86.cmpccxadd32(ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 3)
  return _cmpccxadd_epi32(__A, __B, __C, _CMPCCX_NB);
}

long long test_cmplxadd64(void *__A, long long __B, long long __C) {
  // CHECK-LABEL: @test_cmplxadd64(
  // CHECK: call i64 @llvm.x86.cmpccxadd64(ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 3)
  return _cmpccxadd_epi64(__A, __B, __C, _CMPCCX_NB);
}

int test_cmpaxadd32(void *__A, int __B, int __C) {
  // CHECK-LABEL: @test_cmpaxadd32(
  // CHECK: call i32 @llvm.x86.cmpccxadd32(ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 4)
  return _cmpccxadd_epi32(__A, __B, __C, _CMPCCX_Z);
}

long long test_cmpaxadd64(void *__A, long long __B, long long __C) {
  // CHECK-LABEL: @test_cmpaxadd64(
  // CHECK: call i64 @llvm.x86.cmpccxadd64(ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 4)
  return _cmpccxadd_epi64(__A, __B, __C, _CMPCCX_Z);
}

int test_cmpaexadd32(void *__A, int __B, int __C) {
  // CHECK-LABEL: @test_cmpaexadd32(
  // CHECK: call i32 @llvm.x86.cmpccxadd32(ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 5)
  return _cmpccxadd_epi32(__A, __B, __C, _CMPCCX_NZ);
}

long long test_cmpaexadd64(void *__A, long long __B, long long __C) {
  // CHECK-LABEL: @test_cmpaexadd64(
  // CHECK: call i64 @llvm.x86.cmpccxadd64(ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 5)
  return _cmpccxadd_epi64(__A, __B, __C, _CMPCCX_NZ);
}

int test_cmpgxadd32(void *__A, int __B, int __C) {
  // CHECK-LABEL: @test_cmpgxadd32(
  // CHECK: call i32 @llvm.x86.cmpccxadd32(ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 6)
  return _cmpccxadd_epi32(__A, __B, __C, _CMPCCX_BE);
}

long long test_cmpgxadd64(void *__A, long long __B, long long __C) {
  // CHECK-LABEL: @test_cmpgxadd64(
  // CHECK: call i64 @llvm.x86.cmpccxadd64(ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 6)
  return _cmpccxadd_epi64(__A, __B, __C, _CMPCCX_BE);
}

int test_cmpgexadd32(void *__A, int __B, int __C) {
  // CHECK-LABEL: @test_cmpgexadd32(
  // CHECK: call i32 @llvm.x86.cmpccxadd32(ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 7)
  return _cmpccxadd_epi32(__A, __B, __C, _CMPCCX_NBE);
}

long long test_cmpgexadd64(void *__A, long long __B, long long __C) {
  // CHECK-LABEL: @test_cmpgexadd64(
  // CHECK: call i64 @llvm.x86.cmpccxadd64(ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 7)
  return _cmpccxadd_epi64(__A, __B, __C, _CMPCCX_NBE);
}

int test_cmpnoxadd32(void *__A, int __B, int __C) {
  // CHECK-LABEL: @test_cmpnoxadd32(
  // CHECK: call i32 @llvm.x86.cmpccxadd32(ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 8)
  return _cmpccxadd_epi32(__A, __B, __C, _CMPCCX_S);
}

long long test_cmpnoxadd64(void *__A, long long __B, long long __C) {
  // CHECK-LABEL: @test_cmpnoxadd64(
  // CHECK: call i64 @llvm.x86.cmpccxadd64(ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 8)
  return _cmpccxadd_epi64(__A, __B, __C, _CMPCCX_S);
}

int test_cmpnpxadd32(void *__A, int __B, int __C) {
  // CHECK-LABEL: @test_cmpnpxadd32(
  // CHECK: call i32 @llvm.x86.cmpccxadd32(ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 9)
  return _cmpccxadd_epi32(__A, __B, __C, _CMPCCX_NS);
}

long long test_cmpnpxadd64(void *__A, long long __B, long long __C) {
  // CHECK-LABEL: @test_cmpnpxadd64(
  // CHECK: call i64 @llvm.x86.cmpccxadd64(ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 9)
  return _cmpccxadd_epi64(__A, __B, __C, _CMPCCX_NS);
}

int test_cmpnsxadd32(void *__A, int __B, int __C) {
  // CHECK-LABEL: @test_cmpnsxadd32(
  // CHECK: call i32 @llvm.x86.cmpccxadd32(ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 10)
  return _cmpccxadd_epi32(__A, __B, __C, _CMPCCX_P);
}

long long test_cmpnsxadd64(void *__A, long long __B, long long __C) {
  // CHECK-LABEL: @test_cmpnsxadd64(
  // CHECK: call i64 @llvm.x86.cmpccxadd64(ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 10)
  return _cmpccxadd_epi64(__A, __B, __C, _CMPCCX_P);
}

int test_cmpnexadd32(void *__A, int __B, int __C) {
  // CHECK-LABEL: @test_cmpnexadd32(
  // CHECK: call i32 @llvm.x86.cmpccxadd32(ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 11)
  return _cmpccxadd_epi32(__A, __B, __C, _CMPCCX_NP);
}

long long test_cmpnexadd64(void *__A, long long __B, long long __C) {
  // CHECK-LABEL: @test_cmpnexadd64(
  // CHECK: call i64 @llvm.x86.cmpccxadd64(ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 11)
  return _cmpccxadd_epi64(__A, __B, __C, _CMPCCX_NP);
}

int test_cmpoxadd32(void *__A, int __B, int __C) {
  // CHECK-LABEL: @test_cmpoxadd32(
  // CHECK: call i32 @llvm.x86.cmpccxadd32(ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 12)
  return _cmpccxadd_epi32(__A, __B, __C, _CMPCCX_L);
}

long long test_cmpoxadd64(void *__A, long long __B, long long __C) {
  // CHECK-LABEL: @test_cmpoxadd64(
  // CHECK: call i64 @llvm.x86.cmpccxadd64(ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 12)
  return _cmpccxadd_epi64(__A, __B, __C, _CMPCCX_L);
}

int test_cmppxadd32(void *__A, int __B, int __C) {
  // CHECK-LABEL: @test_cmppxadd32(
  // CHECK: call i32 @llvm.x86.cmpccxadd32(ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 13)
  return _cmpccxadd_epi32(__A, __B, __C, _CMPCCX_NL);
}

long long test_cmppxadd64(void *__A, long long __B, long long __C) {
  // CHECK-LABEL: @test_cmppxadd64(
  // CHECK: call i64 @llvm.x86.cmpccxadd64(ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 13)
  return _cmpccxadd_epi64(__A, __B, __C, _CMPCCX_NL);
}

int test_cmpsxadd32(void *__A, int __B, int __C) {
  // CHECK-LABEL: @test_cmpsxadd32(
  // CHECK: call i32 @llvm.x86.cmpccxadd32(ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 14)
  return _cmpccxadd_epi32(__A, __B, __C, _CMPCCX_LE);
}

long long test_cmpsxadd64(void *__A, long long __B, long long __C) {
  // CHECK-LABEL: @test_cmpsxadd64(
  // CHECK: call i64 @llvm.x86.cmpccxadd64(ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 14)
  return _cmpccxadd_epi64(__A, __B, __C, _CMPCCX_LE);
}

int test_cmpexadd32(void *__A, int __B, int __C) {
  // CHECK-LABEL: @test_cmpexadd32(
  // CHECK: call i32 @llvm.x86.cmpccxadd32(ptr %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 15)
  return _cmpccxadd_epi32(__A, __B, __C, _CMPCCX_NLE);
}

long long test_cmpexadd64(void *__A, long long __B, long long __C) {
  // CHECK-LABEL: @test_cmpexadd64(
  // CHECK: call i64 @llvm.x86.cmpccxadd64(ptr %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i32 15)
  return _cmpccxadd_epi64(__A, __B, __C, _CMPCCX_NLE);
}
