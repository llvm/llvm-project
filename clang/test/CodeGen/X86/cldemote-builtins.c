// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +cldemote -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -ffreestanding -triple=i386-unknown-unknown -target-feature +cldemote -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <immintrin.h>

void test_cldemote(const void *p) {
  //CHECK-LABEL: @test_cldemote
  //CHECK: call void @llvm.x86.cldemote(ptr %{{.*}})
  _cldemote(p);
  //CHECK: call void @llvm.x86.cldemote(ptr %{{.*}})
  _mm_cldemote(p);
}
