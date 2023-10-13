// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +usermsr \
// RUN: -emit-llvm -o - -Wall -Werror -pedantic -Wno-gnu-statement-expression | FileCheck %s

#include <gprintrin.h>

unsigned long long test_urdmsr(unsigned long long __A) {
  // CHECK-LABEL: @test_urdmsr(
  // CHECK: call i64 @llvm.x86.urdmsr(
  return _urdmsr(__A);
}

unsigned long long test_urdmsr_const(unsigned long long __A) {
  // CHECK-LABEL: @test_urdmsr_const(
  // CHECK: call i64 @llvm.x86.urdmsr(
  return _urdmsr(123u);
}

void test_uwrmsr(unsigned long long __A, unsigned long long __B) {
  // CHECK-LABEL: @test_uwrmsr(
  // CHECK: call void @llvm.x86.uwrmsr(
  _uwrmsr(__A, __B);
}

void test_uwrmsr_const(unsigned long long __A, unsigned long long __B) {
  // CHECK-LABEL: @test_uwrmsr_const(
  // CHECK: call void @llvm.x86.uwrmsr(
  _uwrmsr(123u, __B);
}

