// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -ffreestanding -triple=i386-unknown-unknown -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <x86intrin.h>

void test_wbinvd(void) {
  //CHECK-LABEL: @test_wbinvd
  //CHECK: call void @llvm.x86.wbinvd()
  _wbinvd();
}
