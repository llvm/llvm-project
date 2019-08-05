// FIXME: https://code.google.com/p/address-sanitizer/issues/detail?id=316
// XFAIL: android
//
// Test that kernel area is not sanitized on 32-bit machines.
//
// RUN: %clangxx_asan %s -o %t
// RUN: %env_asan_opts=verbosity=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%kernel_bits
// RUN: %env_asan_opts=verbosity=1:full_address_space=0 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-%kernel_bits
// RUN: %env_asan_opts=verbosity=1:full_address_space=1 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-kernel-64-bits
//
// CHECK-kernel-32-bits: || `[0x38{{0+}}, 0xb{{f+}}]` || HighMem    ||
// CHECK-kernel-32-bits: || `[0x27{{0+}}, 0x37{{f+}}]` || HighShadow ||
// CHECK-kernel-32-bits: || `[0x24{{0+}}, 0x26{{f+}}]` || ShadowGap  ||
//
// CHECK-kernel-64-bits: || `[0x4{{0+}}, 0x{{f+}}]` || HighMem    ||
// CHECK-kernel-64-bits: || `[0x28{{0+}}, 0x3{{f+}}]` || HighShadow ||
// CHECK-kernel-64-bits: || `[0x24{{0+}}, 0x27{{f+}}]` || ShadowGap  ||
//
// REQUIRES: i386-target-arch, shadow-scale-3

int main() {
  return 0;
}
