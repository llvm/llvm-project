// RUN: %clang_asan -O0 %s -o %t
// RUN: %env_asan_opts=verbosity=0 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-PPC64-V0
// RUN: %env_asan_opts=verbosity=2 %run %t 2>&1 | FileCheck %s --check-prefix=CHECK-PPC64
// REQUIRES: powerpc64-target-arch

#include <stdio.h>

int main() {
// CHECK-PPC64: || `[{{0x200|0x180|0x0a0|0x040}}000000000, {{0x7ff|0x3ff|0x0ff}}fffffffff]` || HighMem    ||
// CHECK-PPC64: || `[{{0x140|0x130|0x034|0x028}}000000000, {{0x1ff|0x17f|0x09f|0x03f}}fffffffff]` || HighShadow ||
// CHECK-PPC64: || `[{{0x120|0x024|0x024}}000000000, {{0x12f|0x13f|0x033|0x027}}fffffffff]` || ShadowGap  ||
// CHECK-PPC64: || `[{{0x100|0x020}}000000000, {{0x11f|0x023}}fffffffff]`       || LowShadow  ||
// CHECK-PPC64: || `[0x000000000000, {{0x0ff|0x01f}}fffffffff]`       || LowMem     ||
//
  printf("ppc64 eyecatcher \n");
// CHECK-PPC64-V0: ppc64 eyecatcher

  return 0;
}

/*
 * Several different signatures noted.

Newer kernel: (Fedora starting with kernel version 4.?)
|| `[0x200000000000, 0x7fffffffffff]` || HighMem    ||
|| `[0x140000000000, 0x1fffffffffff]` || HighShadow ||
|| `[0x120000000000, 0x13ffffffffff]` || ShadowGap  ||
|| `[0x100000000000, 0x11ffffffffff]` || LowShadow  ||
|| `[0x000000000000, 0x0fffffffffff]` || LowMem     ||

Newer kernel: (Ubuntu starting with kernel version 4.?)
|| `[0x180000000000, 0x3fffffffffff]` || HighMem    ||
|| `[0x130000000000, 0x17ffffffffff]` || HighShadow ||
|| `[0x120000000000, 0x12ffffffffff]` || ShadowGap  ||
|| `[0x100000000000, 0x11ffffffffff]` || LowShadow  ||
|| `[0x000000000000, 0x0fffffffffff]` || LowMem     ||

Newish kernel: (64TB address range support, starting with kernel version 3.7)
|| `[0x0a0000000000, 0x3fffffffffff]` || HighMem    ||
|| `[0x034000000000, 0x09ffffffffff]` || HighShadow ||
|| `[0x024000000000, 0x033fffffffff]` || ShadowGap  ||
|| `[0x020000000000, 0x023fffffffff]` || LowShadow  ||
|| `[0x000000000000, 0x01ffffffffff]` || LowMem     ||

Oldish kernel:
|| `[0x040000000000, 0x0fffffffffff]` || HighMem    ||
|| `[0x028000000000, 0x03ffffffffff]` || HighShadow ||
|| `[0x024000000000, 0x027fffffffff]` || ShadowGap  ||
|| `[0x020000000000, 0x023fffffffff]` || LowShadow  ||
|| `[0x000000000000, 0x01ffffffffff]` || LowMem     ||
*/

