// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +wbnoinvd -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

void test_wbnoinvd(void) {
  //CHECK-LABEL: @test_wbnoinvd
  //CHECK: call void @llvm.x86.wbnoinvd()
  _wbnoinvd();
}
