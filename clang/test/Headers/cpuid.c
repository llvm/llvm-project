// RUN: %clang_cc1 %s -ffreestanding -triple x86_64-apple-darwin -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-64
// RUN: %clang_cc1 %s -ffreestanding -triple i386 -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-32

#include <cpuid.h>
#include <cpuid.h> // Make sure multiple inclusion protection works.

// CHECK-64: {{.*}} call { i32, i32, i32, i32 } asm "  xchgq  %rbx,${1:q}\0A cpuid\0A xchgq %rbx,${1:q}", "={ax},=r,={cx},={dx},0,~{dirflag},~{fpsr},~{flags}"(i32 %{{[a-z0-9]+}})
// CHECK-64: {{.*}} call { i32, i32, i32, i32 } asm "  xchgq  %rbx,${1:q}\0A  cpuid\0A  xchgq  %rbx,${1:q}", "={ax},=r,={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 %{{[a-z0-9]+}}, i32 %{{[a-z0-9]+}})

// CHECK-32: {{.*}} call { i32, i32, i32, i32 } asm "cpuid", "={ax},={bx},={cx},={dx},0,~{dirflag},~{fpsr},~{flags}"(i32 %{{[a-z0-9]+}})
// CHECK-32: {{.*}} call { i32, i32, i32, i32 } asm "cpuid", "={ax},={bx},={cx},={dx},0,2,~{dirflag},~{fpsr},~{flags}"(i32 %{{[a-z0-9]+}}, i32 %{{[a-z0-9]+}})

unsigned eax0, ebx0, ecx0, edx0;
unsigned eax1, ebx1, ecx1, edx1;

void test_cpuid(unsigned level, unsigned count) {
  __cpuid(level, eax1, ebx1, ecx1, edx1);
  __cpuid_count(level, count, eax0, ebx0, ecx0, edx0);
}
