// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +raoint \
// RUN: -emit-llvm -o - -Wall -Werror -pedantic -Wno-gnu-statement-expression | FileCheck %s --check-prefixes=CHECK,X64
// RUN: %clang_cc1 %s -ffreestanding -triple=i686-unknown-unknown -target-feature +raoint \
// RUN: -emit-llvm -o - -Wall -Werror -pedantic -Wno-gnu-statement-expression | FileCheck %s --check-prefixes=CHECK

#include <stddef.h>
#include <x86gprintrin.h>

void test_aadd_i32(int *__A, int __B) {
  // CHECK-LABEL: @test_aadd_i32(
  // CHECK: call void @llvm.x86.aadd32(ptr %{{.*}}, i32 %{{.*}})
  _aadd_i32(__A, __B);
}

void test_aand_i32(int *__A, int __B) {
  // CHECK-LABEL: @test_aand_i32(
  // CHECK: call void @llvm.x86.aand32(ptr %{{.*}}, i32 %{{.*}})
  _aand_i32(__A, __B);
}

void test_aor_i32(int *__A, int __B) {
  // CHECK-LABEL: @test_aor_i32(
  // CHECK: call void @llvm.x86.aor32(ptr %{{.*}}, i32 %{{.*}})
  _aor_i32(__A, __B);
}

void test_axor_i32(int *__A, int __B) {
  // CHECK-LABEL: @test_axor_i32(
  // CHECK: call void @llvm.x86.axor32(ptr %{{.*}}, i32 %{{.*}})
  _axor_i32(__A, __B);
}

#ifdef __x86_64__
void test_aadd_i64(long long *__A, long long __B) {
  // X64-LABEL: @test_aadd_i64(
  // X64: call void @llvm.x86.aadd64(ptr %{{.*}}, i64 %{{.*}})
  _aadd_i64(__A, __B);
}

void test_aand_i64(long long *__A, long long __B) {
  // X64-LABEL: @test_aand_i64(
  // X64: call void @llvm.x86.aand64(ptr %{{.*}}, i64 %{{.*}})
  _aand_i64(__A, __B);
}

void test_aor_i64(long long *__A, long long __B) {
  // X64-LABEL: @test_aor_i64(
  // X64: call void @llvm.x86.aor64(ptr %{{.*}}, i64 %{{.*}})
  _aor_i64(__A, __B);
}

void test_axor_i64(long long *__A, long long __B) {
  // X64-LABEL: @test_axor_i64(
  // X64: call void @llvm.x86.axor64(ptr %{{.*}}, i64 %{{.*}})
  _axor_i64(__A, __B);
}
#endif
