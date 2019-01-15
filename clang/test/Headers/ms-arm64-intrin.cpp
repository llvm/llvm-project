// REQUIRES: aarch64-registered-target

// RUN: %clang_cc1 -triple arm64-windows -O1 \
// RUN: -fms-compatibility -fms-compatibility-version=17.00 \
// RUN: -ffreestanding -fsyntax-only -Werror \
// RUN: -isystem %S/Inputs/include %s -S -o - -emit-llvm 2>&1 \
// RUN: | FileCheck %s

#include <intrin.h>

void check_nop() {
// CHECK: "nop"
  __nop();
}

unsigned short check_byteswap_ushort(unsigned short val) {
// CHECK: call i16 @llvm.bswap.i16(i16 %val)
  return _byteswap_ushort(val);
}

unsigned long check_byteswap_ulong(unsigned long val) {
// CHECK: call i32 @llvm.bswap.i32(i32 %val)
  return _byteswap_ulong(val);
}

unsigned __int64 check_byteswap_uint64(unsigned __int64 val) {
// CHECK: call i64 @llvm.bswap.i64(i64 %val)
  return _byteswap_uint64(val);
}
