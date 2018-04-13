// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +cldemote -emit-llvm -o - -Wall -Werror | FileCheck %s
// RUN: %clang_cc1 %s -ffreestanding -triple=i386-unknown-unknown -target-feature +cldemote -emit-llvm -o - -Wall -Werror | FileCheck %s

#include <x86intrin.h>

void test_cldemote(const void *p) {
  //CHECK-LABEL: @test_cldemote
  //CHECK: call void @llvm.x86.cldemote(i8* %{{.*}})
  _cldemote(p);
}
