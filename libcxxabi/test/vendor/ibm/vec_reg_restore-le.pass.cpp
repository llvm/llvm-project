//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Check that the PowerPC vector registers are restored properly during
// unwinding.

// REQUIRES: target=powerpc{{(64)?}}le-unknown-linux-gnu
// UNSUPPORTED: no-exceptions

// Callee-saved VSR's 62 and 63 (vr30, vr31 respectively) are set to 16 bytes
// with values 1, 2 respectively in main. In order to ensure the two doublewords
// in each register are different, they are merged. Then they are reset to 16
// bytes with values 9 and 12 respectively in a callee and an exception is
// thrown. When catching an exception in main, the values in the two registers
// need to be the original ones (including the correct doubleword order).

#include <cassert>
#include <cstdlib>

int __attribute__((noinline)) test2(int i) {
  if (i > 3)
    throw i;
  srand(i);
  return rand();
}

int __attribute__((noinline)) test(int i) {
  // Clobber VS63 and VS62 in the function body.
  // Set VS63 to 16 bytes each with value 9
  asm volatile("vspltisb 31, 9" : : : "v31");

  // Set VS62 to 16 bytes each with value 12
  asm volatile("vspltisb 30, 12" : : : "v30");
  return test2(i);
}

#define cmpVS63(vec, result)                                                   \
  {                                                                            \
    vector unsigned char gbg;                                                  \
    asm volatile("vcmpequb. %[gbg], 31, %[veca];"                              \
                 "mfocrf %[res], 2;"                                           \
                 "rlwinm %[res], %[res], 25, 31, 31"                           \
                 : [res] "=r"(result), [gbg] "=v"(gbg)                         \
                 : [veca] "v"(vec)                                             \
                 : "cr6");                                                     \
  }

#define cmpVS62(vec, result)                                                   \
  {                                                                            \
    vector unsigned char gbg;                                                  \
    asm volatile("vcmpequb. %[gbg], 30, %[veca];"                              \
                 "mfocrf %[res], 2;"                                           \
                 "rlwinm %[res], %[res], 25, 31, 31"                           \
                 : [res] "=r"(result), [gbg] "=v"(gbg)                         \
                 : [veca] "v"(vec)                                             \
                 : "cr6");                                                     \
  }

int main(int, char **) {
  // Set VS63 to 16 bytes each with value 1.
  asm volatile("vspltisb 31, 1" : : : "v31");

  // Set VS62 to 16 bytes each with value 2.
  asm volatile("vspltisb 30, 2" : : : "v30");

  // Mix doublewords for both VS62 and VS63.
  asm volatile("xxmrghd 63, 63, 62");
  asm volatile("xxmrghd 62, 63, 62");

  vector unsigned long long expectedVS63Value = {0x202020202020202,
                                                 0x101010101010101};
  vector unsigned long long expectedVS62Value = {0x202020202020202,
                                                 0x101010101010101};
  try {
    test(4);
  } catch (int num) {
    // If the unwinder restores VS63 and VS62 correctly, they should contain
    // 0x01's and 0x02's respectively instead of 0x09's and 0x12's.
    bool isEqualVS63, isEqualVS62;
    cmpVS63(expectedVS63Value, isEqualVS63);
    cmpVS62(expectedVS62Value, isEqualVS62);
    assert(isEqualVS63 && isEqualVS62);
  }
  return 0;
}
