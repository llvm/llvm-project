// RUN: %clang_cc1 %s -ffreestanding -triple x86_64-unknown-unknown -target-feature +hreset -emit-llvm -o - | FileCheck %s
// RUN: %clang_cc1 %s -ffreestanding -triple i386-unknown-unknown -target-feature +hreset -emit-llvm -o - | FileCheck %s

#include <immintrin.h>

void test_hreset(int a)
{
// CHECK-LABEL: test_hreset
// CHECK: call void asm sideeffect "hreset $$0", "{ax},~{dirflag},~{fpsr},~{flags}"(i32 %{{[0-9]}})
  _hreset(a);
}
