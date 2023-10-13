// RUN: %clang_cc1 %s -ffreestanding -triple=i386-unknown-unknown -target-feature +usermsr \
// RUN: -emit-llvm -fsyntax-only -verify

#include <gprintrin.h>

unsigned long long test_urdmsr(unsigned long long __A) {
  return _urdmsr(__A); // expected-error {{call to undeclared function '_urdmsr'}}
}

void test_uwrmsr(unsigned long long __A, unsigned long long __B) {
  // CHECK-LABEL: @test_uwrmsr(
  // CHECK: call void @llvm.x86.uwrmsr(
  _uwrmsr(__A, __B); // expected-error {{call to undeclared function '_uwrmsr'}}
}
