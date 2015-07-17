// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=316
// XFAIL: android
//
// Test that kernel area is not sanitized on 32-bit machines.
//
// RUN: %clangxx_asan %s -o %t
// RUN: env ASAN_OPTIONS=$ASAN_OPTIONS:verbosity=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%kernel_bits
// RUN: env ASAN_OPTIONS=$ASAN_OPTIONS:verbosity=1:full_address_space=0 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%kernel_bits
// RUN: env ASAN_OPTIONS=$ASAN_OPTIONS:verbosity=1:full_address_space=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-kernel-64-bits
//
// CHECK-kernel-32-bits: || `[0x38{{0+}}, 0xb{{f+}}]` || HighMem    ||
// CHECK-kernel-32-bits: || `[0x27{{0+}}, 0x37{{f+}}]` || HighShadow ||
// CHECK-kernel-32-bits: || `[0x24{{0+}}, 0x26{{f+}}]` || ShadowGap  ||
//
// CHECK-kernel-64-bits: || `[0x4{{0+}}, 0x{{f+}}]` || HighMem    ||
// CHECK-kernel-64-bits: || `[0x28{{0+}}, 0x3{{f+}}]` || HighShadow ||
// CHECK-kernel-64-bits: || `[0x24{{0+}}, 0x27{{f+}}]` || ShadowGap  ||
//
// REQUIRES: asan-32-bits,i386-supported-target

int main() {
  return 0;
}

