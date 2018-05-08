// RUN: %clang_cc1 %s -ffreestanding -triple x86_64-unknown-unknown -target-feature +pconfig -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-64
// RUN: %clang_cc1 %s -ffreestanding -triple i386 -target-feature +pconfig -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-32

#include <x86intrin.h>
#include <stdint.h>
#include <stddef.h>

uint32_t test_pconfig(uint32_t leaf, size_t data[3]) {
// CHECK-64: call { i32, i64, i64, i64 } asm "pconfig", "={ax},={bx},={cx},={dx},{ax},{bx},{cx},{dx},~{cc},~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-32: call { i32, i32, i32, i32 } asm "pconfig", "={ax},={bx},={cx},={dx},{ax},{bx},{cx},{dx},~{cc},~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})
  return _pconfig_u32(leaf, data);
}
