// RUN: %clang_cc1 %s -ffreestanding -triple x86_64-unknown-unknown -target-feature +sgx -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-64
// RUN: %clang_cc1 %s -ffreestanding -triple i386 -target-feature +sgx -emit-llvm -o - | FileCheck %s --check-prefix=CHECK-32

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

uint32_t test_encls(uint32_t leaf, size_t data[3]) {
// CHECK-64: call { i32, i64, i64, i64 } asm "encls", "={ax},={bx},={cx},={dx},{ax},{bx},{cx},{dx},~{cc},~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-32: call { i32, i32, i32, i32 } asm "encls", "={ax},={bx},={cx},={dx},{ax},{bx},{cx},{dx},~{cc},~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})

  return _encls_u32(leaf, data);
}

uint32_t test_enclu(uint32_t leaf, size_t data[3]) {
// CHECK-64: call { i32, i64, i64, i64 } asm "enclu", "={ax},={bx},={cx},={dx},{ax},{bx},{cx},{dx},~{cc},~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-32: call { i32, i32, i32, i32 } asm "enclu", "={ax},={bx},={cx},={dx},{ax},{bx},{cx},{dx},~{cc},~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})

  return _enclu_u32(leaf, data);
}

uint32_t test_enclv(uint32_t leaf, size_t data[3]) {
// CHECK-64: call { i32, i64, i64, i64 } asm "enclv", "={ax},={bx},={cx},={dx},{ax},{bx},{cx},{dx},~{cc},~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}}, i64 %{{.*}}, i64 %{{.*}}, i64 %{{.*}})
// CHECK-32: call { i32, i32, i32, i32 } asm "enclv", "={ax},={bx},={cx},={dx},{ax},{bx},{cx},{dx},~{cc},~{dirflag},~{fpsr},~{flags}"(i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}}, i32 %{{.*}})

  return _enclv_u32(leaf, data);
}
