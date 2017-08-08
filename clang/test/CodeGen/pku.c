// RUN: %clang_cc1 -ffreestanding %s -triple=x86_64-apple-darwin -target-feature +pku -emit-llvm -o - -Wall -Werror | FileCheck %s


#include <immintrin.h>

unsigned int test_rdpkru_u32() {
  // CHECK-LABEL: @test_rdpkru_u32
  // CHECK: @llvm.x86.rdpkru
  return _rdpkru_u32(); 
}
void test_wrpkru(unsigned int __A) {
  // CHECK-LABEL: @test_wrpkru
  // CHECK: @llvm.x86.wrpkru
  _wrpkru(__A);
  return ;
}
